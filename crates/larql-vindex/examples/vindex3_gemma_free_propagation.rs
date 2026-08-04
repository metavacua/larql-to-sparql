//! Free-propagation composition — each path carries its own residual.
//!
//! The locked-input sweep proved every MoE layer independently. It could not
//! prove that they *compose*, because it handed every layer the incumbent's own
//! activation. This one starts both paths from the same embedding and lets each
//! carry its own residual to the end:
//!
//! ```text
//! same token ids → same embedding
//!    ↓                              ↓
//! in-process MoE                 VINDEX3-bound MoE
//!    ↓ own residual                 ↓ own residual
//! …30 blocks…                    …30 blocks…
//!    ↓                              ↓
//!            compare, boundary by boundary
//! ```
//!
//! Attention, KV, the dense slab, the norms, PLE and the layer scalar are the
//! *same code* on both sides — both routes go through `predict_kquant_hidden`
//! and differ only in which `MoeExpertBackend` is installed. So a divergence is
//! the MoE path or it is nothing.
//!
//! # Two locations, never conflated
//!
//! ```text
//! first differing boundary   where a difference is first observable
//! earliest causal operation  the first MoE that differed on identical input
//! ```
//!
//! Those are not the same layer, and reporting only the first would be
//! actively misleading. Once layer 7's contribution differs, layer 8 reads a
//! contaminated residual and *everything* downstream differs — including its
//! post-attention state, which would then read as an attention defect. The
//! causal test is therefore conditional: a layer only indicts its own MoE if
//! its input residual was still identical when it ran.
//!
//! # Why the incumbent also runs through a backend
//!
//! `InProcessMoeBackend` is byte-identical to the block loop's default branch.
//! Putting the incumbent behind the same trait gives the harness a seam to
//! observe on both sides — and makes the seam itself falsifiable, which is what
//! the first check below does before any comparison is trusted.
//!
//! Usage:
//! ```text
//! cargo run --release -p larql-vindex --example vindex3_gemma_free_propagation -- \
//!   --vindex <path> [--prompt "The capital of France is"]
//! ```

use std::cell::RefCell;

use larql_inference::ffn::{
    BoundMoeBackend, InProcessMoeBackend, MoeBackendError, MoeExpertBackend,
};
use larql_inference::vindex::predict_kquant_hidden;
use larql_models::ModelWeights;
use ndarray::Array2;

/// Residual-scale values, judged relative to the reference's own magnitude —
/// the same policy as the single-layer harness, and for the same reason.
const RELATIVE_BAND: f32 = 0.05;
/// Below this the ratio stops meaning anything and the comparison is absolute.
const SCALE_FLOOR: f32 = 1e-6;

const DEFAULT_PROMPT: &str = "The capital of France is";
const ARG_VINDEX: &str = "--vindex";
const ARG_PROMPT: &str = "--prompt";
/// Enough to record a margin between the argmax and its nearest rival.
const TOP_K: usize = 5;
/// Unscaled: this is a parity comparison, not a sampling run.
const TEMPERATURE: f32 = 1.0;

fn arg(name: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn max_abs_diff(a: &Array2<f32>, b: &Array2<f32>) -> f32 {
    if a.shape() != b.shape() {
        return f32::INFINITY;
    }
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn magnitude(a: &Array2<f32>) -> f32 {
    a.iter().fold(0.0f32, |m, v| m.max(v.abs()))
}

/// What one layer's MoE call saw and produced.
#[derive(Clone)]
struct Observation {
    layer: usize,
    input: Array2<f32>,
    contribution: Array2<f32>,
}

/// Wraps a route and records every call, so both paths can be compared at the
/// same boundaries without either being reimplemented.
struct Recording<B: MoeExpertBackend> {
    inner: B,
    seen: RefCell<Vec<Observation>>,
}

impl<B: MoeExpertBackend> Recording<B> {
    fn new(inner: B) -> Self {
        Self {
            inner,
            seen: RefCell::new(Vec::new()),
        }
    }

    fn take(self) -> Vec<Observation> {
        self.seen.into_inner()
    }
}

impl<B: MoeExpertBackend> MoeExpertBackend for Recording<B> {
    fn forward_moe_seq(
        &self,
        weights: &ModelWeights,
        layer: usize,
        h: &Array2<f32>,
        norm_offset: f32,
        eps: f32,
    ) -> Result<Array2<f32>, MoeBackendError> {
        let contribution = self
            .inner
            .forward_moe_seq(weights, layer, h, norm_offset, eps)?;
        self.seen.borrow_mut().push(Observation {
            layer,
            input: h.clone(),
            contribution: contribution.clone(),
        });
        Ok(contribution)
    }

    fn name(&self) -> &'static str {
        self.inner.name()
    }
}

/// Where a divergence is first *visible*, and where it was first *caused*.
#[derive(Debug, Default)]
struct DivergenceReport {
    /// The first layer at which anything observable differs — input residual
    /// or contribution, whichever comes first.
    first_differing_boundary: Option<(usize, &'static str, f32)>,
    /// The first layer whose MoE contribution differed **while its input
    /// residual was still identical**. That layer's expert path is the defect;
    /// everything after it is contamination.
    earliest_causal_operation: Option<(usize, f32)>,
}

fn diverge(incumbent: &[Observation], bound: &[Observation]) -> DivergenceReport {
    let mut report = DivergenceReport::default();
    for (a, b) in incumbent.iter().zip(bound) {
        debug_assert_eq!(a.layer, b.layer);
        let input_diff = max_abs_diff(&a.input, &b.input);
        let contribution_diff = max_abs_diff(&a.contribution, &b.contribution);

        if report.first_differing_boundary.is_none() && input_diff != 0.0 {
            report.first_differing_boundary = Some((a.layer, "input residual", input_diff));
        }
        if report.first_differing_boundary.is_none() && contribution_diff != 0.0 {
            report.first_differing_boundary =
                Some((a.layer, "moe contribution", contribution_diff));
        }
        // The conditional that keeps contamination from reading as a defect:
        // a layer only indicts its own MoE if it ran on an identical input.
        if report.earliest_causal_operation.is_none()
            && input_diff == 0.0
            && contribution_diff != 0.0
        {
            report.earliest_causal_operation = Some((a.layer, contribution_diff));
        }
    }
    report
}

fn report_relative(label: &str, reference: &Array2<f32>, found: &Array2<f32>) -> bool {
    let diff = max_abs_diff(reference, found);
    if diff == 0.0 {
        println!("  {label:<26} BIT-IDENTICAL");
        return true;
    }
    let scale = magnitude(reference);
    if scale < SCALE_FLOOR {
        println!(
            "  {label:<26} max|Δ| = {diff:.3e}, reference magnitude {scale:.3e} \
             below the {SCALE_FLOOR:.0e} floor — absolute, not a ratio"
        );
        return diff < SCALE_FLOOR;
    }
    let relative = diff / scale;
    println!(
        "  {label:<26} max|Δ| = {diff:.3e} of {scale:.3e} = {:.3}%  {}",
        relative * 100.0,
        if relative <= RELATIVE_BAND {
            "within band"
        } else {
            "OUTSIDE"
        }
    );
    relative <= RELATIVE_BAND
}

fn main() -> Result<(), String> {
    let vindex = arg(ARG_VINDEX).ok_or(format!("set {ARG_VINDEX} <path>"))?;
    let prompt = arg(ARG_PROMPT).unwrap_or_else(|| DEFAULT_PROMPT.to_string());

    println!("Free-propagation composition — each path carries its own residual");
    println!("  vindex  {vindex}");
    println!("  prompt  {prompt:?}");

    let mut callbacks = larql_vindex::SilentLoadCallbacks;
    let path = std::path::Path::new(&vindex);
    let weights = larql_vindex::load_model_weights_kquant(path, &mut callbacks)
        .map_err(|e| format!("load weights: {e}"))?;
    let mut index = larql_vindex::VectorIndex::load_vindex(path, &mut callbacks)
        .map_err(|e| format!("load index: {e}"))?;
    // The same three region maps the CLI installs. Without them the forward
    // panics on the attention slices before ever reaching an MoE block.
    index
        .load_attn_kquant(path)
        .map_err(|e| format!("load attn Q4K: {e}"))?;
    index
        .load_interleaved_kquant(path)
        .map_err(|e| format!("load interleaved Q4K: {e}"))?;
    let _ = index.load_lm_head_kquant(path);
    let tokenizer =
        larql_vindex::load_vindex_tokenizer(path).map_err(|e| format!("tokenizer: {e}"))?;
    let encoding = tokenizer
        .encode(prompt.as_str(), true)
        .map_err(|e| format!("encode prompt: {e}"))?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();
    println!("  tokens  {}\n", token_ids.len());

    // ── The seam is falsifiable before anything is built on it ─────────────
    //
    // Running through `InProcessMoeBackend` must equal running with no backend
    // at all. If it does not, the trait changed the model rather than
    // relocating a call, and every comparison below would be measuring that
    // change instead of VINDEX3.
    println!("== seam ==");
    let default_route = predict_kquant_hidden(&weights, &token_ids, &index, None);
    let via_trait = predict_kquant_hidden(&weights, &token_ids, &index, Some(&InProcessMoeBackend));
    let seam_faithful = max_abs_diff(&default_route, &via_trait) == 0.0;
    println!(
        "  in-process via trait vs default branch: {}",
        if seam_faithful {
            "BIT-IDENTICAL — the seam relocates the call and nothing else"
        } else {
            "DIFFERS — the trait changed the model; stop here"
        }
    );
    if !seam_faithful {
        return Err("the seam is not faithful; no comparison below is meaningful".into());
    }

    // ── Both routes, each carrying its own residual ────────────────────────
    let incumbent = Recording::new(InProcessMoeBackend);
    let incumbent_h = predict_kquant_hidden(&weights, &token_ids, &index, Some(&incumbent));
    let incumbent_seen = incumbent.take();

    let bound = Recording::new(BoundMoeBackend::production());
    let bound_h = predict_kquant_hidden(&weights, &token_ids, &index, Some(&bound));
    let bound_seen = bound.take();

    println!("\n== boundaries ==");
    println!(
        "  {} MoE layers observed on each path",
        incumbent_seen.len()
    );
    if incumbent_seen.len() != bound_seen.len() {
        return Err(format!(
            "the two paths ran different numbers of MoE layers: {} vs {}",
            incumbent_seen.len(),
            bound_seen.len()
        ));
    }

    let report = diverge(&incumbent_seen, &bound_seen);
    match report.first_differing_boundary {
        None => println!("  first differing boundary   none — every boundary bit-identical"),
        Some((layer, what, diff)) => {
            println!("  first differing boundary   layer {layer}, {what}, max|Δ| = {diff:.3e}")
        }
    }
    match report.earliest_causal_operation {
        None => {
            println!("  earliest causal operation  none — no MoE differed on an identical input")
        }
        Some((layer, diff)) => println!(
            "  earliest causal operation  layer {layer} MoE, max|Δ| = {diff:.3e} \
             (its input was still identical, so this layer is the defect)"
        ),
    }
    if let (Some((first, _, _)), Some((causal, _))) = (
        report.first_differing_boundary,
        report.earliest_causal_operation,
    ) {
        if first != causal {
            println!(
                "  note: layers {first}..{causal} differ downstream of the cause — contamination, \
                 not independent defects"
            );
        }
    }

    // ── Composition ────────────────────────────────────────────────────────
    println!("\n== final hidden state ==");
    let composed = report_relative("pre-final-norm hidden", &incumbent_h, &bound_h);

    // ── The model's own endpoint ───────────────────────────────────────────
    //
    // Both boundaries go through production functions rather than a local
    // reimplementation: `apply_norm` is the one `logits_to_predictions` calls,
    // and the predictor is the one generation calls. A harness that rolled its
    // own lm_head here would be comparing the harness.
    println!("\n== endpoint ==");
    let norm_offset = weights.arch.norm_weight_offset();
    let final_key = weights.arch.final_norm_key();
    let incumbent_normed =
        larql_inference::forward::apply_norm(&weights, &incumbent_h, final_key, norm_offset);
    let bound_normed =
        larql_inference::forward::apply_norm(&weights, &bound_h, final_key, norm_offset);
    let normed = report_relative("post-final-norm hidden", &incumbent_normed, &bound_normed);

    let incumbent_pred = larql_inference::forward::predict::logits_to_predictions_pub(
        &weights,
        &incumbent_h,
        &tokenizer,
        TOP_K,
        TEMPERATURE,
    );
    let bound_pred = larql_inference::forward::predict::logits_to_predictions_pub(
        &weights,
        &bound_h,
        &tokenizer,
        TOP_K,
        TEMPERATURE,
    );

    let ids_match = incumbent_pred.token_ids == bound_pred.token_ids;
    let scores_match = incumbent_pred.predictions == bound_pred.predictions;
    println!(
        "  {:<26} {}",
        "top-k token ids",
        if ids_match { "IDENTICAL" } else { "DIFFER" }
    );
    println!(
        "  {:<26} {}",
        "top-k scores",
        if scores_match {
            "BIT-IDENTICAL"
        } else {
            "DIFFER"
        }
    );

    // The margin is recorded even though identical scores imply it. Later
    // approximate and partially-resident runs need this same report shape, and
    // a margin that only appears once something disagrees is a shape nobody can
    // compare against.
    let margin = |p: &larql_compute::forward::predict::PredictResult| -> f64 {
        match (p.predictions.first(), p.predictions.get(1)) {
            (Some((_, top)), Some((_, second))) => top - second,
            _ => f64::INFINITY,
        }
    };
    let argmax_match = incumbent_pred.token_ids.first() == bound_pred.token_ids.first();
    println!(
        "  {:<26} {:?} vs {:?}",
        "argmax token",
        incumbent_pred.predictions.first().map(|(t, _)| t),
        bound_pred.predictions.first().map(|(t, _)| t)
    );
    println!(
        "  {:<26} {:.6e} vs {:.6e}",
        "argmax margin",
        margin(&incumbent_pred),
        margin(&bound_pred)
    );

    let endpoint = normed && ids_match && scores_match && argmax_match;

    println!("\n== notes ==");
    println!("  Attention, KV, the dense slab, the norms, PLE and the layer scalar are the same");
    println!("  code on both paths. Only the MoE route differs, so a divergence is the MoE path.");
    if report.earliest_causal_operation.is_none() && composed && endpoint {
        println!("\nCOMPOSITION: every MoE ran on an identical input and produced an identical");
        println!("contribution; the free-propagated hidden states, the normed hidden states and");
        println!("the model's own predictions all agree.");
        Ok(())
    } else {
        Err("composition not established — see the causal layer above".into())
    }
}
