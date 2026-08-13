//! Real-model cross-backend prefill parity for the `standard` engine.
//!
//! The synthetic Gemma-3 fixture shows Metal's **batched** prefill
//! (`coarse_prefill` → `fused_prefill`) diverging 23-43% from both the
//! CPU and Metal's own iterative path, while an SWA-free fixture at
//! identical dims agrees to 1.6e-7 (see
//! `tests/gpu_engine_parity/gemma3_prefill_gap.rs`). Whether that is a
//! production defect or a fixture artefact cannot be answered on a
//! 2-layer synthetic model, so this drives the same comparison on a real
//! Gemma-3 Q4K vindex.
//!
//! It sweeps prompt length because the fixture's divergence is
//! position-dependent — absent at one token, large from two onward. It
//! also reports the argmax token at each length, because identical text
//! is what a `larql run` comparison sees and it can mask a hidden-state
//! difference that never flips a decision.
//!
//! ```sh
//! cargo run -p larql-kv --release --features gpu \
//!   --example gemma_prefill_parity -- <path-to.vindex> [max_len]
//! ```
//!
//! Exits 0 with a printed table; it is a diagnostic, not a gate.

use larql_inference::ffn::NullFfn;
use larql_kv::EngineKind;
use larql_vindex::SilentLoadCallbacks;
use ndarray::Array2;
use std::path::PathBuf;

/// Prompt lengths are taken as prefixes of a repeated filler so every
/// length is a strict prefix of the next — the only thing varying is how
/// many positions the prefill relates.
const FILLER: &str = "The capital of France is Paris and the history of Europe is long. ";

/// Default longest prefix swept.
const DEFAULT_MAX_LEN: usize = 24;

/// Above this relative L2 the two backends are not computing the same
/// thing; below it they differ by kernel rounding. The synthetic gap
/// sits at ~4e-1, ordinary Q4K reassociation at ~1e-3.
const DIVERGENCE_THRESHOLD: f32 = 1e-2;

fn relative_l2(a: &Array2<f32>, b: &Array2<f32>) -> f32 {
    let num: f32 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt();
    let den: f32 = b.iter().map(|y| y * y).sum::<f32>().sqrt().max(1e-6);
    num / den
}

fn argmax_slice(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: gemma_prefill_parity <path-to.vindex> [max_len]");
        return Ok(());
    };
    let vindex_path = PathBuf::from(path);
    let max_len: usize = args
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_MAX_LEN);

    if !vindex_path.exists() {
        eprintln!("vindex not found at {}; skipping", vindex_path.display());
        return Ok(());
    }

    let mut cb = SilentLoadCallbacks;
    let weights = larql_vindex::load_model_weights_kquant(&vindex_path, &mut cb)?;
    let mut index = larql_vindex::VectorIndex::load_vindex(&vindex_path, &mut cb)?;
    index.load_attn_kquant(&vindex_path)?;
    index.load_interleaved_kquant(&vindex_path)?;
    let tokenizer = larql_vindex::load_vindex_tokenizer(&vindex_path)?;

    let arch = &*weights.arch;
    println!(
        "model: {} layers, sliding_window={:?}, layer0_sliding={}",
        weights.num_layers,
        arch.sliding_window_size(),
        arch.is_sliding_window_layer(0),
    );

    let long_prompt = FILLER.repeat(max_len.div_ceil(10) + 2);
    let all_tokens = larql_inference::encode_prompt(&tokenizer, arch, long_prompt.as_str())
        .map_err(|e| format!("tokenize: {e}"))?;

    println!();
    println!(
        "  {:>7}  {:>12}  {:>10}  {:>10}",
        "tokens", "rel L2", "gpu tok", "cpu tok"
    );
    println!("  {}", "-".repeat(46));

    let mut worst = 0.0f32;
    let mut token_mismatches = 0usize;
    for n in 1..=max_len.min(all_tokens.len()) {
        let prompt = &all_tokens[..n];

        let mut gpu = EngineKind::Standard { window_size: None }
            .build(larql_inference::default_engine_backend());
        let mut cpu =
            EngineKind::Standard { window_size: None }.build(larql_inference::cpu_engine_backend());

        let h_gpu = gpu.prefill_quant(
            &weights,
            &NullFfn,
            &index,
            prompt,
            &*larql_inference::default_compute_backend(),
        )?;
        let h_cpu = cpu.prefill_quant(
            &weights,
            &NullFfn,
            &index,
            prompt,
            &*larql_inference::cpu_backend(),
        )?;

        let rel = relative_l2(&h_gpu, &h_cpu);
        worst = worst.max(rel);

        let t_gpu = argmax_slice(&larql_inference::forward::hidden_to_raw_logits(
            &weights, &h_gpu,
        ));
        let t_cpu = argmax_slice(&larql_inference::forward::hidden_to_raw_logits(
            &weights, &h_cpu,
        ));
        if t_gpu != t_cpu {
            token_mismatches += 1;
        }

        let flag = if rel > DIVERGENCE_THRESHOLD {
            "  <== DIVERGED"
        } else {
            ""
        };
        println!("  {n:>7}  {rel:>12.4e}  {t_gpu:>10}  {t_cpu:>10}{flag}");
    }

    println!();
    println!("worst relative L2: {worst:.4e}");
    println!("argmax mismatches: {token_mismatches}");
    if worst > DIVERGENCE_THRESHOLD {
        println!(
            "VERDICT: reproduces on this real model — the synthetic fixture was not the artefact."
        );
    } else {
        println!(
            "VERDICT: does NOT reproduce here; the divergence is specific to the \
             synthetic fixture, not this checkpoint."
        );
    }
    Ok(())
}
