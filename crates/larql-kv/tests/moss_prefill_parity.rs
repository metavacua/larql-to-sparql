//! Funnel step 2 (`docs/tts-funnel.md` §3): MOSS-TTS-Realtime prefill
//! parity against the reference dump.
//!
//! Two comparisons, in causal order:
//!
//! 1. **The summed-embedding algebra** — `embed_tables_sum` over the
//!    `[T, 17]` prompt matrix vs the reference's per-position summed
//!    input embeddings. This isolates the new input primitive from the
//!    transformer entirely.
//! 2. **The backbone forward** — `StandardEngine::prefill_from_hidden`
//!    over those embeddings, final norm applied, vs the reference's
//!    `last_hidden_state` at the last position. This is the boundary
//!    crossing: LARQL's existing Qwen3 execution fed by a non-text input
//!    algebra it knows nothing about.
//!
//! NOT FOR CI: needs the ~4.3 GB checkpoint and the step-0 dump. Run:
//!
//! ```text
//! MOSS_TTS_REALTIME_DIR=<hf snapshot dir> \
//! MOSS_PARITY_BIN_DIR=<parity-dump/bin dir> \
//! cargo test -p larql-kv --test moss_prefill_parity -- --ignored --nocapture
//! ```
//!
//! The bin fixtures are flat little-endian arrays produced by
//! `moss_parity_dump.py --export-bin` (jarvis-voice), shapes in
//! `manifest.json` alongside them.

mod common;

use std::path::Path;

use ndarray::Array2;

use common::{max_abs_diff, model_dir_from_env, row_cosine, BinFixtures};
use larql_compute::forward::ops::apply_norm;
use larql_inference::ffn::WeightFfn;
use larql_inference::larql_models::loading::safetensors::load_model_dir;
use larql_inference::larql_models::speech::moss_tts_realtime::{
    load_moss_tts_aux_from_safetensors, MossTtsRealtimeConfig,
};
use larql_inference::KvEngine;
use larql_kv::engines::standard::StandardEngine;

/// Bit-exactness is not expected across engines (accumulation order in
/// fp32 GEMMs differs); these bounds are the "first gate" tolerances and
/// exist to be tightened, not relaxed.
const EMBED_MAX_ABS: f32 = 1e-5;
const HIDDEN_MAX_ABS: f32 = 1e-2;
const HIDDEN_MIN_COSINE: f64 = 0.999_99;

#[test]
#[ignore]
fn moss_prefill_matches_reference_dump() {
    let model_dir = model_dir_from_env();
    let fixtures = BinFixtures::open_from_env();

    // ── Load: backbone through the normal path, audio side through the
    // speech loader ──
    let weights = load_model_dir(&model_dir).expect("backbone load");
    assert_eq!(weights.arch.family(), "moss_tts_realtime");
    let config_json: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(Path::new(&model_dir).join("config.json")).unwrap(),
    )
    .unwrap();
    let aux_config = MossTtsRealtimeConfig::from_config_json(&config_json).unwrap();
    let aux = load_moss_tts_aux_from_safetensors(&model_dir, weights.arch.as_ref(), aux_config)
        .expect("aux load");

    // ── The prefill input: prompt matrix + the 12-token text lead the
    // reference appended (prefill_ids is the exact [T, 17] the reference
    // forwarded, so use it directly) ──
    let prefill_ids = fixtures.i64_matrix_as_u32("prefill_ids");
    let reference_embeds = fixtures.f32_matrix("prefill_embeds");
    let reference_hidden = fixtures.f32_matrix("prefill_hidden");
    let rows = prefill_ids.nrows();
    assert_eq!(reference_embeds.nrows(), rows);
    assert_eq!(reference_hidden.nrows(), rows);

    // ── Gate 1: the 17-way summed embedding ──
    let mut tables: Vec<ndarray::ArrayView2<f32>> =
        Vec::with_capacity(1 + aux.audio_embed_tables.len());
    tables.push(weights.embed.view());
    tables.extend(aux.audio_embed_tables.iter().map(|t| t.view()));
    let embeds = larql_compute::forward::embed::embed_tables_sum(&tables, &prefill_ids);

    let embed_diff = max_abs_diff(&embeds, &reference_embeds);
    println!("gate 1 — summed embeddings: max |Δ| = {embed_diff:e} over {rows} rows");
    assert!(
        embed_diff <= EMBED_MAX_ABS,
        "summed-embedding mismatch: max |Δ| {embed_diff:e} > {EMBED_MAX_ABS:e}"
    );

    // ── Gate 2: backbone forward from those embeddings ──
    let mut engine = StandardEngine::new(None);
    let ffn = WeightFfn { weights: &weights };
    let last_hidden = engine
        .prefill_from_hidden(&weights, &ffn, &embeds)
        .expect("prefill_from_hidden");
    let normed = apply_norm(
        &weights,
        &last_hidden,
        weights.arch.final_norm_key(),
        weights.arch.norm_weight_offset(),
    );

    let reference_last = {
        let mut last = Array2::<f32>::zeros((1, reference_hidden.ncols()));
        last.row_mut(0).assign(&reference_hidden.row(rows - 1));
        last
    };
    let hidden_diff = max_abs_diff(&normed, &reference_last);
    let cosine = row_cosine(&normed, &reference_last, 0);
    println!("gate 2 — final hidden (last row): max |Δ| = {hidden_diff:e}, cosine = {cosine:.8}");
    assert!(
        cosine >= HIDDEN_MIN_COSINE,
        "backbone hidden diverged: cosine {cosine} < {HIDDEN_MIN_COSINE}"
    );
    assert!(
        hidden_diff <= HIDDEN_MAX_ABS,
        "backbone hidden diverged: max |Δ| {hidden_diff:e} > {HIDDEN_MAX_ABS:e}"
    );
}
