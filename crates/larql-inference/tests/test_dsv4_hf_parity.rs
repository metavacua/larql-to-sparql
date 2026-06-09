//! HF-`transformers` parity gate for the DSv4-Flash GGUF forward
//! (`dsv4-hf-parity`; tasks #63/#69).
//!
//! ## Why
//!
//! Every other DSv4 check compares our own paths against each other
//! (CPU-vs-CUDA greedy equality; quant-vs-f32 tolerance). Those miss a
//! *shared* bug — a systematic forward error both paths reproduce. This
//! test pins the forward to an **external authority**: HuggingFace
//! `transformers` running the reference DeepSeek-V4-Flash on a fixed
//! prompt.
//!
//! ## How (decoupled)
//!
//! The HF reference can't run inside a Rust test, so it's generated
//! out-of-band by `scripts/dsv4_hf_reference.py` into a small JSON dump:
//!
//! ```json
//! { "model": "...", "prompt": "...", "token_ids": [...],
//!   "top_k": [{"token_id": 6342, "logit": 21.3}, ...] }
//! ```
//!
//! This test loads that dump (path via `LARQL_DSV4_HF_REF`, default
//! `tests/goldens/dsv4_flash_hf_reference.json`), runs the DSv4 GGUF
//! forward on the dump's `token_ids` (so the tokenizer is never a
//! variable), and compares the final-position next-token distribution.
//! It **skips** (not fails) when the dump or GGUF is absent, so default
//! CI stays green.
//!
//! ## What it asserts
//!
//! 1. **Greedy next-token == HF top-1** — the load-bearing assertion.
//! 2. **Top-K sets overlap** — robustness to rank swaps near the top.
//! 3. **Top-1 logit within a relative tolerance** — secondary; the GGUF
//!    is Q4_K-quantized vs the HF f16/bf16 reference, so values shift by
//!    quant error. The bound is generous + calibration-pending.
//!
//! ```sh
//! python scripts/dsv4_hf_reference.py --model <hf> --out <ref.json>
//! cargo test -p larql-inference --test test_dsv4_hf_parity -- --ignored --nocapture
//! ```

use std::collections::HashSet;
use std::path::PathBuf;

use larql_inference::attention::dsv4_head_storage::load_dsv4_head;
use larql_inference::attention::dsv4_storage_build::DsV4Hyperparams;
use larql_inference::attention::dsv4_streaming_model_forward::dsv4_streaming_model_forward_cached;
use larql_inference::attention::dsv4_topk_logits::dsv4_topk_logits;
use larql_models::loading::gguf::GgufFile;
use serde::Deserialize;

const GGUF_PATH: &str = "/tank/ai/deepseek-ai/DeepSeek-V4-Flash-GGUF/DeepSeek-V4-Flash-Q4_K_M.gguf";
const DEFAULT_REF: &str = "crates/larql-inference/tests/goldens/dsv4_flash_hf_reference.json";

/// DSv4-Flash is 43 transformer blocks (matches the full-model streaming
/// smoke test). Full-model parity must run them all.
const N_LAYER: usize = 43;

/// Q4_K (GGUF) vs HF f16/bf16 shifts logit magnitudes by quant error
/// (larger than the ~5e-2 CPU-vs-Metal noise the self-captured goldens
/// use). Generous + **calibration-pending** on the first real dump;
/// greedy-token match (assertion 1) is the load-bearing signal.
const LOGIT_REL_TOL: f32 = 0.15;

#[derive(Deserialize)]
struct RefTopK {
    token_id: u32,
    logit: f32,
}

#[derive(Deserialize)]
struct HfReference {
    #[allow(dead_code)]
    model: String,
    #[allow(dead_code)]
    prompt: String,
    token_ids: Vec<u32>,
    top_k: Vec<RefTopK>,
}

#[test]
#[ignore = "Requires the real DSv4-Flash GGUF + an HF reference dump (LARQL_DSV4_HF_REF); ~22 min full-model forward"]
fn dsv4_forward_matches_hf_reference() {
    let ref_path = PathBuf::from(
        std::env::var("LARQL_DSV4_HF_REF").unwrap_or_else(|_| DEFAULT_REF.to_string()),
    );
    if !ref_path.exists() {
        eprintln!(
            "skipping: HF reference dump {ref_path:?} not present \
             (generate via scripts/dsv4_hf_reference.py)"
        );
        return;
    }
    let gguf_path = std::path::Path::new(GGUF_PATH);
    if !gguf_path.exists() {
        eprintln!("skipping: {gguf_path:?} not present");
        return;
    }

    let reference: HfReference =
        serde_json::from_str(&std::fs::read_to_string(&ref_path).expect("read reference dump"))
            .expect("parse reference dump JSON");
    assert!(
        !reference.token_ids.is_empty(),
        "reference has no token_ids"
    );
    assert!(!reference.top_k.is_empty(), "reference has no top_k");

    let gguf = GgufFile::open(gguf_path).expect("open DSv4 GGUF");
    let hp = DsV4Hyperparams::from_gguf(&gguf).expect("hyperparams");
    let head = load_dsv4_head(&gguf, &hp).expect("head");

    let layer_indices: Vec<usize> = (0..N_LAYER).collect();
    let logits = dsv4_streaming_model_forward_cached(
        &gguf,
        &hp,
        &head,
        &reference.token_ids,
        &layer_indices,
        0,
        None,
        None,
    )
    .expect("DSv4 forward");

    // Final-position next-token top-K (last row = next-token prediction).
    let k = reference.top_k.len();
    let topk = dsv4_topk_logits(logits.view(), k);
    let ours = topk.last().expect("non-empty topk rows");

    let ref_top1 = reference.top_k[0].token_id;
    let our_top1 = ours[0].token_id;
    eprintln!("HF top-1 token = {ref_top1}, ours = {our_top1}");

    // 1. Greedy next-token must match the HF reference top-1.
    assert_eq!(
        our_top1, ref_top1,
        "greedy next-token differs from HF reference (ours {our_top1} vs HF {ref_top1})"
    );

    // 2. Top-K sets overlap (tolerate rank swaps near the top).
    let ref_set: HashSet<u32> = reference.top_k.iter().map(|e| e.token_id).collect();
    let our_set: HashSet<u32> = ours.iter().map(|e| e.token_id).collect();
    let overlap = ref_set.intersection(&our_set).count();
    assert!(
        overlap >= (k / 2).max(1),
        "top-{k} overlap only {overlap}/{k} between HF reference and DSv4 GGUF forward"
    );

    // 3. Top-1 logit within a (generous) relative tolerance.
    let ref_logit = reference.top_k[0].logit;
    let our_logit = ours[0].logit;
    let rel = (our_logit - ref_logit).abs() / ref_logit.abs().max(1.0);
    eprintln!("top-1 logit: HF {ref_logit:.3} vs ours {our_logit:.3} (rel diff {rel:.3})");
    assert!(
        rel < LOGIT_REL_TOL,
        "top-1 logit relative diff {rel:.3} exceeds tolerance {LOGIT_REL_TOL} \
         (HF {ref_logit:.3} vs ours {our_logit:.3})"
    );
}
