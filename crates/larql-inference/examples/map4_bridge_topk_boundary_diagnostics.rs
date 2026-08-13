//! MAP-4 bridge: top-8 boundary instability for L20, vs. the L26 control.
//!
//! `map4_bridge_router_relative_diagnostics.rs` found L20's global
//! router-probability response (cosine/KL between true and lead=1-stale
//! input) is ordinary within the late band, matching the non-dip control
//! L26 — ruling out "broad distribution rewrite" as L20's mechanism. But a
//! tiny probability shift concentrated exactly at the rank-8/rank-9 cutoff
//! could still flip which experts get selected while leaving the aggregate
//! distribution (and its cosine/KL to the true distribution) nearly
//! untouched. This tests that directly: for each layer, per prompt,
//! records which experts are lost/gained between the true and stale top-8,
//! how much TRUE probability mass sits on the lost experts (mass that
//! staleness would wrongly exclude), how much STALE probability mass sits
//! on the gained experts (mass staleness wrongly includes), and where
//! those specific experts rank in this layer's overall popularity —
//! testing whether L20 is "decision-boundary fragility" (many swaps,
//! ordinary distribution) as opposed to L27's "distribution rewrite" (both
//! the distribution AND the selection change).
//!
//! Usage: same dumps as the other map4_bridge examples.
//! ```text
//! cargo run --release -p larql-inference --example map4_bridge_topk_boundary_diagnostics -- \
//!   --vindex <path> --dump-prefix /tmp/dump_p
//! ```

use std::collections::HashMap;

use larql_compute::cpu::ops::moe::{moe_expert_input, moe_router_input};
use larql_compute::forward::dump_config::{cpu_layer_h_post_attn_path, cpu_layer_path};
use larql_compute::pipeline_layer::build_moe_weights;
use larql_compute::MoeLayerWeights;
use larql_models::ModelWeights;

const NUM_PROMPTS: usize = 8;
/// Layers this diagnostic exists to compare: L20 (the unexplained dip),
/// L26 (non-dip control, same whole-residual magnitude-jump extremity as
/// L20), L27 (the confirmed distribution-rewrite layer, for contrast),
/// L19 (band onset, for context).
const FOCUS_LAYERS: [usize; 4] = [19, 20, 26, 27];

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
    assert!(
        values.len().is_multiple_of(width) && !values.is_empty(),
        "{path}: {} values not a whole number of {width}-wide rows",
        values.len()
    );
    let rows = values.len() / width;
    values[(rows - 1) * width..].to_vec()
}

fn discover_moe_layers(
    weights: &ModelWeights,
    arch: &dyn larql_models::ModelArchitecture,
) -> Vec<usize> {
    (0..weights.num_layers)
        .filter(|&l| build_moe_weights(weights, arch, l).is_some())
        .collect()
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
    idx
}

fn main() {
    println!("=== MAP-4 bridge: top-8 boundary instability (L20 vs. L26 control) ===");
    let vindex = match arg("--vindex") {
        Some(v) => v,
        None => {
            println!("skipped — set --vindex <path> and --dump-prefix <dir prefix>");
            return;
        }
    };
    let dump_prefix = arg("--dump-prefix").expect("set --dump-prefix <dir prefix>");

    let mut cb = larql_vindex::SilentLoadCallbacks;
    let weights = larql_vindex::load_model_weights_kquant(std::path::Path::new(&vindex), &mut cb)
        .expect("load real vindex weights");
    let arch = &*weights.arch;
    let hidden = weights.hidden_size;
    let norm_offset = arch.norm_weight_offset();
    let eps = arch.norm_eps();

    let all_layers = discover_moe_layers(&weights, arch);
    let mut moe_at: HashMap<usize, MoeLayerWeights> = HashMap::new();
    for &l in &FOCUS_LAYERS {
        moe_at.insert(
            l,
            build_moe_weights(&weights, arch, l).expect("must be MoE"),
        );
    }

    for &l in &FOCUS_LAYERS {
        if !all_layers.contains(&l) {
            continue;
        }
        let moe = &moe_at[&l];
        let top_k = moe.top_k;

        // popularity ranking (1 = most frequent), from TRUE selections across all 8 prompts —
        // same construction as the routing-lead-time script's static-popularity baseline.
        let mut true_probs_by_prompt = Vec::with_capacity(NUM_PROMPTS);
        let mut stale_probs_by_prompt = Vec::with_capacity(NUM_PROMPTS);
        let mut counts: HashMap<usize, usize> = HashMap::new();
        for i in 0..NUM_PROMPTS {
            let dump_dir = format!("{dump_prefix}{i}");
            let h_true = last_row(&cpu_layer_h_post_attn_path(&dump_dir, l), hidden);
            let h_stale = last_row(&cpu_layer_path(&dump_dir, l - 1), hidden);

            let ei_true = moe_expert_input(&h_true, moe, norm_offset, eps);
            let router_in_true = moe_router_input(&h_true, &ei_true, moe, norm_offset, eps);
            let probs_true = router_full_probs(&router_in_true, moe);

            let ei_stale = moe_expert_input(&h_stale, moe, norm_offset, eps);
            let router_in_stale = moe_router_input(&h_stale, &ei_stale, moe, norm_offset, eps);
            let probs_stale = router_full_probs(&router_in_stale, moe);

            for &id in &top_k_ids(&probs_true, top_k) {
                *counts.entry(id).or_insert(0) += 1;
            }
            true_probs_by_prompt.push(probs_true);
            stale_probs_by_prompt.push(probs_stale);
        }
        let mut pop_order: Vec<usize> = counts.keys().copied().collect();
        pop_order.sort_unstable_by(|a, b| counts[b].cmp(&counts[a]).then(a.cmp(b)));
        let pop_rank: HashMap<usize, usize> = pop_order
            .iter()
            .enumerate()
            .map(|(i, &id)| (id, i + 1))
            .collect();
        let unranked = pop_order.len() + 1; // experts never in any prompt's true top-8

        println!("\n=== L{l} ===");
        let mut swaps_all = Vec::with_capacity(NUM_PROMPTS);
        let mut mass_lost_all = Vec::with_capacity(NUM_PROMPTS);
        let mut mass_gained_all = Vec::with_capacity(NUM_PROMPTS);
        let mut lost_pop_ranks = Vec::new();
        let mut gained_pop_ranks = Vec::new();

        for i in 0..NUM_PROMPTS {
            let probs_true = &true_probs_by_prompt[i];
            let probs_stale = &stale_probs_by_prompt[i];
            let true_ids = top_k_ids(probs_true, top_k);
            let stale_ids = top_k_ids(probs_stale, top_k);

            let lost: Vec<usize> = true_ids
                .iter()
                .copied()
                .filter(|id| !stale_ids.contains(id))
                .collect();
            let gained: Vec<usize> = stale_ids
                .iter()
                .copied()
                .filter(|id| !true_ids.contains(id))
                .collect();
            let mass_lost: f32 = lost.iter().map(|&id| probs_true[id]).sum();
            let mass_gained: f32 = gained.iter().map(|&id| probs_stale[id]).sum();

            swaps_all.push(lost.len());
            mass_lost_all.push(mass_lost);
            mass_gained_all.push(mass_gained);
            lost_pop_ranks.extend(lost.iter().map(|id| *pop_rank.get(id).unwrap_or(&unranked)));
            gained_pop_ranks.extend(
                gained
                    .iter()
                    .map(|id| *pop_rank.get(id).unwrap_or(&unranked)),
            );

            println!(
                "  prompt {i}: swaps={} lost={lost:?} gained={gained:?} mass_lost={mass_lost:.4} mass_gained={mass_gained:.4}",
                lost.len()
            );
        }

        let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
        let mean_usize = |v: &[usize]| v.iter().sum::<usize>() as f64 / v.len().max(1) as f64;
        println!(
            "  L{l} summary: mean_swaps={:.2}/{top_k}  mean_mass_lost={:.4}  mean_mass_gained={:.4}  mean_pop_rank_of_lost={:.1}  mean_pop_rank_of_gained={:.1}  (n_distinct_true_experts={})",
            swaps_all.iter().sum::<usize>() as f64 / swaps_all.len() as f64,
            mean(&mass_lost_all),
            mean(&mass_gained_all),
            mean_usize(&lost_pop_ranks),
            mean_usize(&gained_pop_ranks),
            pop_order.len(),
        );
    }

    println!(
        "\n== the comparison that matters ==\n\
         If L20 shows meaningfully MORE swaps / mass moved across the top-8 boundary than L26,\n\
         despite both having ordinary (matching) global router-probability cosine/KL (from the\n\
         router-relative diagnostic), that supports 'L20 = decision-boundary fragility' as a mechanism\n\
         genuinely distinct from L27's full-distribution rewrite (which should show both high swaps\n\
         AND high global KL). A low mean popularity-rank on lost/gained experts would mean the swaps\n\
         are just popular-expert churn, not a specific fragility."
    );
}
