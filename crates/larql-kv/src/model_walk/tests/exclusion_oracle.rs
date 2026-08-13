//! **B1/B6** — is physical row excision equivalent to gathered
//! visibility, for continued model execution?
//!
//! ## What this oracle actually tests, exactly
//!
//! The candidate arm excises the span **once** and then decodes over the
//! compacted cache. The reference arm keeps the full cache as its
//! authority and materialises the gathered view **per step**: excise,
//! decode one token, splice the span back. During every decode both arms
//! present attention with exactly the same per-layer visible rows in the
//! same order at the same logical position — but they arrive there
//! through different machinery, and between steps the reference still
//! holds every original row.
//!
//! So the theorem under test is:
//!
//! > Compacting once and decoding N times is equivalent to compacting,
//! > decoding, and restoring N times — at the level of full-vocabulary
//! > logits through the real `StandardEngine` forward pass.
//!
//! A splice-offset error, a position-accounting error, or state that
//! silently depends on resident row count all break it. The `o6`
//! wrong-span control proves the oracle can detect a bad exclusion
//! rather than both arms ignoring the mechanism.
//!
//! ## What it does *not* test
//!
//! Not attention-level index-gather against a retained full cache during
//! the step itself. That needs a per-layer visibility parameter in
//! `KvDispatch::attention_step`, which does not exist and which should
//! not be added as a production masking feature just to be a test
//! oracle. The gather here is *materialised* rather than *indexed*; the
//! remaining gap is whether a hypothetical indexed gather would agree
//! with a materialised one, which `position_seam.rs` covers at dispatch
//! level for a single step.

use larql_inference::ffn::FfnBackend;
use larql_inference::hidden_to_raw_logits;
use larql_inference::kv_engine::PerLayerKvAccess;
use larql_inference::larql_models::test_fixtures::make_gemma3_narrow_window_test_weights;
use larql_inference::model::ModelWeights;
use ndarray::Array2;

use crate::engines::standard::StandardEngine;
use crate::KvEngine;

/// Source span planted in the prompt, well outside the 4-token window.
const SPAN: std::ops::Range<usize> = 4..8;
const PROMPT: [u32; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
/// Layer 5 is the only global layer in the 6-layer Gemma fixture.
const GLOBAL_LAYERS: [usize; 1] = [5];
/// Ordinary continuation tokens that do not name the planted span.
const NEUTRAL: [u32; 3] = [17, 18, 19];
/// Stand-in for the canonical record's rendering.
const RECORD: [u32; 5] = [20, 21, 22, 23, 24];

fn engine() -> StandardEngine {
    StandardEngine::with_backend(None, larql_inference::cpu_engine_backend())
}

fn ffn(weights: &ModelWeights) -> larql_compute::ffn::WeightFfn<'_> {
    larql_compute::ffn::WeightFfn { weights }
}

fn logits(weights: &ModelWeights, hidden: &Array2<f32>) -> Vec<f32> {
    hidden_to_raw_logits(weights, hidden)
}

/// Bit-exact comparison. Both arms run the same CPU attention over the
/// same rows, so anything short of bit identity is a real difference,
/// not noise — see the module docs before reaching for a tolerance.
fn assert_bit_identical(label: &str, a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "{label}: vocab size differs");
    let diverged: Vec<_> = a
        .iter()
        .zip(b.iter())
        .enumerate()
        .filter(|(_, (x, y))| x.to_bits() != y.to_bits())
        .map(|(i, (x, y))| (i, *x, *y))
        .collect();
    assert!(
        diverged.is_empty(),
        "{label}: {} of {} logits differ; first few {:?}",
        diverged.len(),
        a.len(),
        &diverged[..diverged.len().min(4)]
    );
}

/// Excise `span` from the named layers, returning the journal rows so
/// they can be spliced back.
fn excise(
    kv: &mut dyn PerLayerKvAccess,
    layers: &[usize],
    span: std::ops::Range<usize>,
) -> Vec<(usize, larql_inference::kv_engine::ExcisedKvRows)> {
    layers
        .iter()
        .map(|&l| {
            let rows = kv
                .excise_kv_rows(l, span.clone())
                .unwrap_or_else(|| panic!("layer {l} excision"));
            (l, rows)
        })
        .collect()
}

fn splice(
    kv: &mut dyn PerLayerKvAccess,
    at: usize,
    journal: &[(usize, larql_inference::kv_engine::ExcisedKvRows)],
) {
    for (layer, rows) in journal {
        assert!(kv.splice_kv_rows(*layer, at, rows), "layer {layer} splice");
    }
}

/// **Candidate arm.** Excise once, then decode every token over the
/// compacted cache.
fn candidate(
    weights: &ModelWeights,
    tokens: &[u32],
    span: std::ops::Range<usize>,
) -> Vec<Vec<f32>> {
    let mut e = engine();
    let backend = ffn(weights);
    e.prefill(weights, &backend, &PROMPT).unwrap();
    excise(
        e.per_layer_kv_mut().expect("per-layer"),
        &GLOBAL_LAYERS,
        span,
    );
    tokens
        .iter()
        .map(|&t| logits(weights, &e.decode_step(weights, &backend, t).unwrap()))
        .collect()
}

/// **Reference arm.** Keep the full cache; materialise the gathered view
/// for the duration of each decode step, then restore it.
fn gather_reference(
    weights: &ModelWeights,
    tokens: &[u32],
    span: std::ops::Range<usize>,
) -> Vec<Vec<f32>> {
    let mut e = engine();
    let backend = ffn(weights);
    e.prefill(weights, &backend, &PROMPT).unwrap();

    tokens
        .iter()
        .map(|&t| {
            let journal = excise(
                e.per_layer_kv_mut().expect("per-layer"),
                &GLOBAL_LAYERS,
                span.clone(),
            );
            let hidden = e.decode_step(weights, &backend, t).unwrap();
            splice(
                e.per_layer_kv_mut().expect("per-layer"),
                span.start,
                &journal,
            );
            logits(weights, &hidden)
        })
        .collect()
}

/// **O0** — an excise/splice round-trip is a no-op. Decoding after one
/// is bit-identical to never having done it.
///
/// This is the harness's own null hypothesis: if the round-trip itself
/// perturbed the cache, every later gate would be comparing two equally
/// corrupted arms and would still pass.
#[test]
fn o0_an_excise_splice_round_trip_changes_nothing() {
    let weights = make_gemma3_narrow_window_test_weights();
    let backend = ffn(&weights);

    let mut plain = engine();
    plain.prefill(&weights, &backend, &PROMPT).unwrap();
    let expected: Vec<_> = NEUTRAL
        .iter()
        .map(|&t| logits(&weights, &plain.decode_step(&weights, &backend, t).unwrap()))
        .collect();

    let mut e = engine();
    e.prefill(&weights, &backend, &PROMPT).unwrap();
    let journal = excise(e.per_layer_kv_mut().unwrap(), &GLOBAL_LAYERS, SPAN);
    splice(e.per_layer_kv_mut().unwrap(), SPAN.start, &journal);
    let actual: Vec<_> = NEUTRAL
        .iter()
        .map(|&t| logits(&weights, &e.decode_step(&weights, &backend, t).unwrap()))
        .collect();

    for (i, (a, b)) in expected.iter().zip(actual.iter()).enumerate() {
        assert_bit_identical(&format!("o0 step {i}"), a, b);
    }
}

/// **O1** — immediately after exclusion, before anything is appended,
/// the first token's full-vocabulary logits are bit-identical.
#[test]
fn o1_immediate_exclusion_logits_are_bit_identical() {
    let weights = make_gemma3_narrow_window_test_weights();
    let c = candidate(&weights, &NEUTRAL[..1], SPAN);
    let r = gather_reference(&weights, &NEUTRAL[..1], SPAN);
    assert_bit_identical("o1 first step", &r[0], &c[0]);
}

/// **O2** — parity survives recurrent K/V growth: three ordinary
/// continuation tokens, compared step by step.
#[test]
fn o2_neutral_continuation_stays_bit_identical() {
    let weights = make_gemma3_narrow_window_test_weights();
    let c = candidate(&weights, &NEUTRAL, SPAN);
    let r = gather_reference(&weights, &NEUTRAL, SPAN);
    for (i, (a, b)) in r.iter().zip(c.iter()).enumerate() {
        assert_bit_identical(&format!("o2 neutral step {i}"), a, b);
    }
}

/// **O3** — the promoted record's K/V is built identically under both
/// regimes. This is the append that matters: it happens while the source
/// is hidden, and its rows become the replacement.
#[test]
fn o3_record_encoding_stays_bit_identical() {
    let weights = make_gemma3_narrow_window_test_weights();
    let c = candidate(&weights, &RECORD, SPAN);
    let r = gather_reference(&weights, &RECORD, SPAN);
    for (i, (a, b)) in r.iter().zip(c.iter()).enumerate() {
        assert_bit_identical(&format!("o3 record token {i}"), a, b);
    }
}

/// **O6, the control.** A deliberately mismatched span must make the two
/// arms diverge. Without this the other gates could pass because neither
/// arm's exclusion does anything.
#[test]
fn o6_a_mismatched_span_makes_the_arms_diverge() {
    let weights = make_gemma3_narrow_window_test_weights();
    let matched = gather_reference(&weights, &NEUTRAL[..1], SPAN);
    let mismatched = candidate(&weights, &NEUTRAL[..1], 9..13);

    let differing = matched[0]
        .iter()
        .zip(mismatched[0].iter())
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert!(
        differing > 0,
        "excluding a different span must change the logits; if it does \
         not, the exclusion mechanism is not affecting execution and \
         every other oracle gate is vacuous"
    );
}

/// **O7** — both arms drive the caller's FFN backend identically. The
/// reference splices between steps; that must not add or reorder FFN
/// dispatches.
#[test]
fn o7_ffn_dispatch_sequences_match() {
    let weights = make_gemma3_narrow_window_test_weights();

    let candidate_calls = {
        let recorder = RecordingFfn::new(&weights);
        let mut e = engine();
        e.prefill(&weights, &recorder, &PROMPT).unwrap();
        excise(e.per_layer_kv_mut().unwrap(), &GLOBAL_LAYERS, SPAN);
        for &t in &NEUTRAL {
            e.decode_step(&weights, &recorder, t).unwrap();
        }
        recorder.calls()
    };

    let reference_calls = {
        let recorder = RecordingFfn::new(&weights);
        let mut e = engine();
        e.prefill(&weights, &recorder, &PROMPT).unwrap();
        for &t in &NEUTRAL {
            let journal = excise(e.per_layer_kv_mut().unwrap(), &GLOBAL_LAYERS, SPAN);
            e.decode_step(&weights, &recorder, t).unwrap();
            splice(e.per_layer_kv_mut().unwrap(), SPAN.start, &journal);
        }
        recorder.calls()
    };

    assert!(!candidate_calls.is_empty());
    assert_eq!(
        reference_calls, candidate_calls,
        "splicing between steps must not perturb FFN dispatch"
    );
}

/// **O8** — the per-layer extents relate as expected rather than being
/// equal: the reference retains the span, the candidate does not.
#[test]
fn o8_extents_differ_by_exactly_the_excluded_span() {
    let weights = make_gemma3_narrow_window_test_weights();
    let backend = ffn(&weights);

    let mut c = engine();
    c.prefill(&weights, &backend, &PROMPT).unwrap();
    excise(c.per_layer_kv_mut().unwrap(), &GLOBAL_LAYERS, SPAN);
    let candidate_extent = c.per_layer_kv_mut().unwrap().extent().unwrap();

    let mut r = engine();
    r.prefill(&weights, &backend, &PROMPT).unwrap();
    let reference_extent = r.per_layer_kv_mut().unwrap().extent().unwrap();

    assert_eq!(
        candidate_extent.logical_next_position, reference_extent.logical_next_position,
        "exclusion must not move the logical position"
    );
    for layer in 0..reference_extent.per_layer_resident_rows.len() {
        let expected = reference_extent.per_layer_resident_rows[layer]
            - if GLOBAL_LAYERS.contains(&layer) {
                SPAN.len()
            } else {
                0
            };
        assert_eq!(
            candidate_extent.per_layer_resident_rows[layer], expected,
            "layer {layer}"
        );
    }
    assert!(reference_extent.is_uniform());
    assert!(!candidate_extent.is_uniform());
}

/// FFN backend that records the layer sequence it was asked for.
struct RecordingFfn<'a> {
    inner: larql_compute::ffn::WeightFfn<'a>,
    calls: std::sync::Mutex<Vec<usize>>,
}

impl<'a> RecordingFfn<'a> {
    fn new(weights: &'a ModelWeights) -> Self {
        Self {
            inner: larql_compute::ffn::WeightFfn { weights },
            calls: std::sync::Mutex::new(Vec::new()),
        }
    }

    fn calls(&self) -> Vec<usize> {
        self.calls.lock().expect("recorder poisoned").clone()
    }
}

impl FfnBackend for RecordingFfn<'_> {
    fn forward(&self, layer: usize, x: &Array2<f32>) -> Array2<f32> {
        self.calls.lock().expect("recorder poisoned").push(layer);
        self.inner.forward(layer, x)
    }

    fn name(&self) -> &str {
        "recording"
    }
}
