use super::super::math::{gelu_tanh, silu};
use super::scratch::ExpertScratch;
use crate::cpu::ops::q4k_q8k_dot::{
    q4k_q8k_matvec_into, quantize_x_to_q8k, quantize_x_to_q8k_into, Q8KActivation,
};
use crate::options;
// `q4k_q8k_gate_up_into` exists for future kernel exploration but is not
// wired into the hot path — see comment in `run_single_expert_q4k_q8k_into`.

/// Pre-quantise `h_norm` to Q8_K once per layer (shared across the K
/// active experts).  Cost is amortised K-fold: at top_k=8 we save 7
/// quantisation passes per layer.
///
/// Returns `None` if `h_norm.len()` isn't a multiple of 256 (Q8_K block
/// size).  Caller falls back to the f32 path in that case.
pub fn quantize_h_norm_for_q4k(h_norm: &[f32]) -> Option<Q8KActivation> {
    if h_norm.is_empty() || !h_norm.len().is_multiple_of(256) {
        return None;
    }
    Some(quantize_x_to_q8k(h_norm))
}

/// Direct Q4_K-from-mmap expert kernel.  No f32 dequant cache; reads the
/// 144-byte Q4_K super-blocks straight from the per-layer mmap and accumulates
/// an integer dot product against the pre-quantised Q8_K activation.
///
/// On Apple Silicon the inner kernel uses `SDOT` (16 i8 × i8 → 4 i32 lanes
/// per instruction) via `crate::cpu::ops::q4k_q8k_dot::q4k_q8k_matvec_into`.
/// On other targets it falls through to the scalar Q8_K reference.
///
/// Why this is faster than the BLAS-on-cached-f32 path at Gemma 4 26B-A4B
/// sizes: the f32 cache is 24 MB per expert × 240 experts/token = 5.7 GB
/// of f32 weights walked per token, which exceeds L3 cache by ~30× on
/// M3 Max — DRAM bandwidth-bound at f32 reading.  Direct Q4_K reads are
/// ~12 MB Q4_K bytes per expert (4× smaller), so DRAM pressure drops 4×
/// and the kernel actually runs near the BW bound rather than way over it.
///
/// `h_norm_q8k` MUST be the Q8_K of the same `h_norm` that fed the f32
/// path — call `quantize_h_norm_for_q4k(&h_norm)` once outside the
/// per-expert loop and share it across the K active experts.
pub fn run_single_expert_q4k_q8k_into<'s>(
    scratch: &'s mut ExpertScratch,
    h_norm_q8k: &Q8KActivation,
    gate_up_bytes: &[u8],
    down_bytes: &[u8],
    inter: usize,
    activation: crate::Activation,
) -> &'s [f32] {
    // Per-stage timing for kernel diagnosis.  Enable with
    // `LARQL_KERNEL_TIMING=1`.  Cached in TLS to avoid syscall per call.
    thread_local! {
        static KERNEL_TIMING: bool = options::env_flag(options::ENV_KERNEL_TIMING);
    }
    let timing = KERNEL_TIMING.with(|t| *t);

    let hidden = h_norm_q8k.qs.len();
    if inter == 0 || hidden == 0 {
        for v in scratch.out.iter_mut() {
            *v = 0.0;
        }
        return &scratch.out;
    }

    // Q4_K weight stride (in bytes) per row: ceil(hidden / Q4_K_BLOCK_ELEMS) * Q4_K_BLOCK_BYTES.
    let block = larql_models::quant::ggml::Q4_K_BLOCK_ELEMS;
    let inter_padded = inter.div_ceil(block) * block;
    let row_block_bytes = (hidden / block) * larql_models::quant::ggml::Q4_K_BLOCK_BYTES;
    let half = inter * row_block_bytes;
    if gate_up_bytes.len() < 2 * half {
        for v in scratch.out.iter_mut() {
            *v = 0.0;
        }
        return &scratch.out;
    }
    let gate_bytes = &gate_up_bytes[..half];
    let up_bytes = &gate_up_bytes[half..2 * half];

    let mut t = std::time::Instant::now();
    // Back-to-back gate + up matvecs.  Tried fused-gate+up via
    // `q4k_q8k_gate_up_into` (2026-05-01): bench was within noise on the
    // single-layer floor and ~4% slower on the 30-layer sweep — the M3 Max
    // OoO engine already extracts plenty of ILP from these two independent
    // matvecs, and the manually-interleaved kernel adds register pressure
    // / hurts the L1 prefetcher.  Fused entry point is kept in
    // `q4k_q8k_dot.rs` (with bit-exact parity test) for future
    // CPU profiles where the trade-off may flip.
    q4k_q8k_matvec_into(&mut scratch.gate_out, h_norm_q8k, gate_bytes, inter, hidden);
    let t_gate = if timing { Some(t.elapsed()) } else { None };
    if timing {
        t = std::time::Instant::now();
    }

    q4k_q8k_matvec_into(&mut scratch.up_out, h_norm_q8k, up_bytes, inter, hidden);
    let t_up = if timing { Some(t.elapsed()) } else { None };
    if timing {
        t = std::time::Instant::now();
    }

    // GELU/SiLU(gate) ⊙ up.  Padding columns (`inter..inter_padded`) stay
    // at their zero-initialised value across reuses (we never write them),
    // matching the existing convention in `run_single_expert_into`.
    for j in 0..inter {
        let g = scratch.gate_out[j];
        let u = scratch.up_out[j];
        scratch.act[j] = match activation {
            crate::Activation::GeluTanh => gelu_tanh(g) * u,
            _ => silu(g) * u,
        };
    }
    let t_act = if timing { Some(t.elapsed()) } else { None };
    if timing {
        t = std::time::Instant::now();
    }

    // Within-expert feature routing (aim-validation probe). No-op (one relaxed
    // atomic load) unless a schedule is installed via
    // `super::within_expert::set_routing` — keeps this hot path byte-exact in
    // production. Prunes the `inter` post-activation features to the configured
    // subset before the `down` matvec; padding columns stay zero.
    super::super::within_expert::prune_act(&mut scratch.act, inter);

    // Quantise the per-expert activation to Q8_K in-place into the
    // caller-owned scratch buffer (no allocation on the hot path —
    // eliminates the 150 µs alloc spikes that drag par_iter wall up).
    quantize_x_to_q8k_into(&mut scratch.act_q8k, &scratch.act);
    let t_act_q8k = if timing { Some(t.elapsed()) } else { None };
    if timing {
        t = std::time::Instant::now();
    }

    // down matvec: out[hidden] = down_W[hidden, inter_padded] @ act
    q4k_q8k_matvec_into(
        &mut scratch.out,
        &scratch.act_q8k,
        down_bytes,
        hidden,
        inter_padded,
    );
    let t_down = if timing { Some(t.elapsed()) } else { None };

    if timing {
        eprintln!(
            "[expert_q4k_q8k] gate={:.0}us up={:.0}us act={:.0}us \
             act_q8k={:.0}us down={:.0}us",
            t_gate.unwrap().as_secs_f64() * 1e6,
            t_up.unwrap().as_secs_f64() * 1e6,
            t_act.unwrap().as_secs_f64() * 1e6,
            t_act_q8k.unwrap().as_secs_f64() * 1e6,
            t_down.unwrap().as_secs_f64() * 1e6,
        );
    }

    &scratch.out
}
