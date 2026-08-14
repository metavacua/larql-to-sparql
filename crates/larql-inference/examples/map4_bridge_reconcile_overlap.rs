//! One-off reconciliation: `map4_bridge_routing_lead_time.rs` reported
//! L20's true-vs-stale overlap as 7.50/8; `map4_bridge_topk_boundary_diagnostics.rs`
//! reported it as 7.25/8 (mean_swaps=0.75). Both read the exact same cached
//! residuals (`h_post_attn(L)` for true, `h_full(L-1)` for stale) — the only
//! difference is HOW the top-8 indices are derived from the router logits:
//! the routing script calls the library's `moe_route_from_router_input`
//! (which uses `math::matmul_vec`, ndarray/BLAS `dot`), the boundary script
//! reimplements the matmul+softmax by hand (naive left-to-right sum) since
//! `math` is crate-private. This checks per-prompt whether the two produce
//! the SAME top-8 set, to see if the discrepancy is a genuine bug or a
//! floating-point summation-order tie-flip at the rank-8/9 boundary.

use larql_compute::cpu::ops::moe::{
    moe_expert_input, moe_route_from_router_input, moe_router_input,
};
use larql_compute::forward::dump_config::{cpu_layer_h_post_attn_path, cpu_layer_path};
use larql_compute::pipeline_layer::build_moe_weights;
use larql_compute::MoeLayerWeights;

const NUM_PROMPTS: usize = 8;
const LAYER: usize = 20;

fn arg(name: &str) -> Option<String> {
    std::env::args()
        .collect::<Vec<_>>()
        .iter()
        .position(|a| a == name)
        .and_then(|i| std::env::args().nth(i + 1))
}

fn last_row(path: &str, width: usize) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let values: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let rows = values.len() / width;
    values[(rows - 1) * width..].to_vec()
}

fn router_full_probs(router_in: &[f32], moe: &MoeLayerWeights<'_>) -> Vec<f32> {
    let hidden = router_in.len();
    let mut logits: Vec<f32> = (0..moe.num_experts)
        .map(|e| {
            let row = &moe.router_proj[e * hidden..(e + 1) * hidden];
            row.iter().zip(router_in).map(|(w, x)| w * x).sum()
        })
        .collect();
    let max = logits.iter().copied().fold(f32::MIN, f32::max);
    let mut sum = 0.0f32;
    for v in &mut logits {
        *v = (*v - max).exp();
        sum += *v;
    }
    for v in &mut logits {
        *v /= sum;
    }
    logits
}

fn top_k_ids(probs: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..probs.len()).collect();
    idx.sort_unstable_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
    idx.truncate(k);
    idx.sort_unstable();
    idx
}

fn main() {
    let vindex = arg("--vindex").expect("--vindex");
    let dump_prefix = arg("--dump-prefix").expect("--dump-prefix");
    let mut cb = larql_vindex::SilentLoadCallbacks;
    let weights = larql_vindex::load_model_weights_kquant(std::path::Path::new(&vindex), &mut cb)
        .expect("load");
    let arch = &*weights.arch;
    let hidden = weights.hidden_size;
    let norm_offset = arch.norm_weight_offset();
    let eps = arch.norm_eps();
    let moe = build_moe_weights(&weights, arch, LAYER).expect("moe");

    for i in 0..NUM_PROMPTS {
        let dump_dir = format!("{dump_prefix}{i}");
        let h_true = last_row(&cpu_layer_h_post_attn_path(&dump_dir, LAYER), hidden);
        let h_stale = last_row(&cpu_layer_path(&dump_dir, LAYER - 1), hidden);

        for (label, h) in [("true", &h_true), ("stale", &h_stale)] {
            let expert_input = moe_expert_input(h, &moe, norm_offset, eps);
            let router_in = moe_router_input(h, &expert_input, &moe, norm_offset, eps);

            // library path (BLAS/ndarray matmul, used by map4_bridge_routing_lead_time.rs)
            let (mut lib_ids, _w) = moe_route_from_router_input(&router_in, &moe);
            lib_ids.sort_unstable();

            // hand-rolled path (naive summation, used by the boundary diagnostic)
            let probs = router_full_probs(&router_in, &moe);
            let hand_ids = top_k_ids(&probs, moe.top_k);

            let match_str = if lib_ids == hand_ids {
                "MATCH"
            } else {
                "DIFFER"
            };
            println!("prompt {i} [{label}]: lib={lib_ids:?} hand={hand_ids:?}  {match_str}");
            if lib_ids != hand_ids {
                // print raw probs near the boundary for both computations to see if it's a near-tie
                let mut sorted: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
                sorted.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                println!(
                    "  hand-computed ranks 6..10 (0-indexed): {:?}",
                    &sorted[6..10.min(sorted.len())]
                );
            }
        }
    }
}
