//! MAP-4 bridge: router-relative geometry for the L20/L27 dips.
//!
//! `map4_bridge_depth_diagnostics.rs` measured whole-residual geometry
//! (write-delta magnitude/direction, independent of any router) and found
//! L20 and L27 each land at rank #2-of-29 on a DIFFERENT such signal — but
//! couldn't explain why L26's write, even larger in raw magnitude than
//! L20's, produces no routing dip at all. The missing piece: whether a
//! write is aligned with directions the layer's OWN router actually reads,
//! not just how big it is in the full residual space.
//!
//! For each layer L, this computes, using each layer's real router matrix:
//! - **router-probability cosine(true, stale)**: the continuous analogue of
//!   the discrete top-8 overlap already known — how similar the router's
//!   full response is to the true vs. the lead=1 stale input.
//! - **router-probability KL(true || stale)**: magnitude-sensitive
//!   complement — cosine can stay high even when the tails diverge a lot.
//! - **raw router-write projection ratio**: `||W_r,L @ delta|| / ||delta||`
//!   — how much this specific write, run through this specific layer's
//!   router matrix, moves the raw (pre-softmax) router scores per unit of
//!   its own residual magnitude. Tests whether L20's smaller write is
//!   nonetheless concentrated in router-sensitive directions, and whether
//!   L26's larger write is comparatively router-null.
//!
//! Usage: same dumps as the other map4_bridge examples.
//! ```text
//! cargo run --release -p larql-inference --example map4_bridge_router_relative_diagnostics -- \
//!   --vindex <path> --dump-prefix /tmp/dump_p
//! ```

use std::collections::HashMap;

use larql_compute::cpu::ops::moe::{moe_expert_input, moe_router_input};
use larql_compute::forward::dump_config::{cpu_layer_h_post_attn_path, cpu_layer_path};
use larql_compute::pipeline_layer::build_moe_weights;
use larql_compute::MoeLayerWeights;
use larql_models::ModelWeights;

const NUM_PROMPTS: usize = 8;
const KL_EPS: f64 = 1e-12;

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

fn sub(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| x - y).collect()
}

fn norm(a: &[f32]) -> f32 {
    a.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn cosine_f64(a: &[f32], b: &[f32]) -> f64 {
    let (na, nb) = (norm(a) as f64, norm(b) as f64);
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    a.iter()
        .zip(b)
        .map(|(x, y)| *x as f64 * *y as f64)
        .sum::<f64>()
        / (na * nb)
}

fn kl_bits(p: &[f32], q: &[f32]) -> f64 {
    p.iter()
        .zip(q)
        .map(|(&pi, &qi)| {
            let pi = (pi as f64).max(KL_EPS);
            let qi = (qi as f64).max(KL_EPS);
            pi * (pi.ln() - qi.ln())
        })
        .sum::<f64>()
        / std::f64::consts::LN_2
}

fn discover_moe_layers(
    weights: &ModelWeights,
    arch: &dyn larql_models::ModelArchitecture,
) -> Vec<usize> {
    (0..weights.num_layers)
        .filter(|&l| build_moe_weights(weights, arch, l).is_some())
        .collect()
}

/// Full per-expert router probability distribution — same math as
/// `moe_route_from_router_input` internally uses, but kept whole instead
/// of collapsed to just the top-k.
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

/// Raw (pre-norm, pre-softmax) router score displacement a residual delta
/// would cause if added directly — `W_r,L @ delta`. Deliberately bypasses
/// the router's RMSNorm pipeline (which is nonlinear in addition) to
/// isolate "how much of this direction does the router matrix itself
/// respond to," not "what routing decision would result."
fn raw_router_projection(delta: &[f32], moe: &MoeLayerWeights<'_>) -> Vec<f32> {
    let hidden = delta.len();
    (0..moe.num_experts)
        .map(|e| {
            let row = &moe.router_proj[e * hidden..(e + 1) * hidden];
            row.iter().zip(delta).map(|(w, x)| w * x).sum()
        })
        .collect()
}

fn main() {
    println!("=== MAP-4 bridge: router-relative geometry for the L20/L27 dips ===");
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

    let layers = discover_moe_layers(&weights, arch);
    let mut moe_at: HashMap<usize, MoeLayerWeights> = HashMap::new();
    for &l in &layers {
        moe_at.insert(
            l,
            build_moe_weights(&weights, arch, l).expect("must be MoE"),
        );
    }

    struct LayerStats {
        prob_cosine: f64,
        prob_kl: f64,
        raw_gain: f64, // ||W_r,L @ delta|| / ||delta||, this layer's own logit-sensitivity to its own real write
    }
    let mut stats: HashMap<usize, LayerStats> = HashMap::new();

    for &l in &layers {
        if l == 0 {
            continue; // no L-1 write to compare
        }
        let moe = &moe_at[&l];
        let mut cosines = Vec::with_capacity(NUM_PROMPTS);
        let mut kls = Vec::with_capacity(NUM_PROMPTS);
        let mut gains = Vec::with_capacity(NUM_PROMPTS);
        for i in 0..NUM_PROMPTS {
            let dump_dir = format!("{dump_prefix}{i}");
            let h_true = last_row(&cpu_layer_h_post_attn_path(&dump_dir, l), hidden);
            let h_stale = last_row(&cpu_layer_path(&dump_dir, l - 1), hidden);

            let expert_input_true = moe_expert_input(&h_true, moe, norm_offset, eps);
            let router_in_true =
                moe_router_input(&h_true, &expert_input_true, moe, norm_offset, eps);
            let probs_true = router_full_probs(&router_in_true, moe);

            let expert_input_stale = moe_expert_input(&h_stale, moe, norm_offset, eps);
            let router_in_stale =
                moe_router_input(&h_stale, &expert_input_stale, moe, norm_offset, eps);
            let probs_stale = router_full_probs(&router_in_stale, moe);

            cosines.push(cosine_f64(&probs_true, &probs_stale));
            kls.push(kl_bits(&probs_true, &probs_stale));

            // this layer's own write, run through this layer's own router.
            let h_full_l = last_row(&cpu_layer_path(&dump_dir, l), hidden);
            let h_full_l_minus_1 = last_row(&cpu_layer_path(&dump_dir, l - 1), hidden);
            let delta = sub(&h_full_l, &h_full_l_minus_1);
            let delta_norm = norm(&delta) as f64;
            if delta_norm > 0.0 {
                let projected = raw_router_projection(&delta, moe);
                gains.push(norm(&projected) as f64 / delta_norm);
            }
        }
        stats.insert(
            l,
            LayerStats {
                prob_cosine: cosines.iter().sum::<f64>() / cosines.len() as f64,
                prob_kl: kls.iter().sum::<f64>() / kls.len() as f64,
                raw_gain: gains.iter().sum::<f64>() / gains.len().max(1) as f64,
            },
        );
    }

    println!(
        "{:>4} {:>12} {:>10} {:>10}",
        "L", "prob_cos", "prob_KL", "raw_gain"
    );
    for &l in &layers {
        let Some(s) = stats.get(&l) else { continue };
        println!(
            "{l:>4} {:>12.4} {:>10.4} {:>10.2}",
            s.prob_cosine, s.prob_kl, s.raw_gain
        );
    }

    let mut by_low_cosine: Vec<usize> = stats.keys().copied().collect();
    by_low_cosine.sort_by(|a, b| {
        stats[a]
            .prob_cosine
            .partial_cmp(&stats[b].prob_cosine)
            .unwrap()
    });
    let mut by_high_kl: Vec<usize> = stats.keys().copied().collect();
    by_high_kl.sort_by(|a, b| stats[b].prob_kl.partial_cmp(&stats[a].prob_kl).unwrap());
    let mut by_high_gain: Vec<usize> = stats.keys().copied().collect();
    by_high_gain.sort_by(|a, b| stats[b].raw_gain.partial_cmp(&stats[a].raw_gain).unwrap());

    println!("\n== rankings (n={}) ==", stats.len());
    println!(
        "  lowest router-probability cosine(true,stale): {:?}",
        &by_low_cosine[..5.min(by_low_cosine.len())]
    );
    println!(
        "  highest router-probability KL(true||stale):    {:?}",
        &by_high_kl[..5.min(by_high_kl.len())]
    );
    println!(
        "  highest raw router-projection gain:            {:?}",
        &by_high_gain[..5.min(by_high_gain.len())]
    );

    println!("\n== the layers this whole diagnostic exists to compare ==");
    for l in [19usize, 20, 26, 27] {
        if let Some(s) = stats.get(&l) {
            let rank_cos = by_low_cosine.iter().position(|&x| x == l).map(|p| p + 1);
            let rank_kl = by_high_kl.iter().position(|&x| x == l).map(|p| p + 1);
            let rank_gain = by_high_gain.iter().position(|&x| x == l).map(|p| p + 1);
            println!(
                "  L{l}: prob_cos={:.4} (rank {:?}/{}), prob_KL={:.4} (rank {:?}/{}), raw_gain={:.2} (rank {:?}/{})",
                s.prob_cosine, rank_cos, stats.len(), s.prob_kl, rank_kl, stats.len(), s.raw_gain, rank_gain, stats.len()
            );
        }
    }
    println!(
        "\nThe hypothesis this tests: L26's write should show a LOW raw_gain rank (router-null: large residual write, \
        little effect on router scores) despite its extreme whole-residual magnitude; L20's write should show a HIGH \
        raw_gain rank (concentrated in router-sensitive directions) despite a SMALLER residual magnitude; L27 should \
        show a low prob_cos / high prob_KL (the router's actual response to true vs. stale input diverges sharply) \
        reflecting a directional rewrite that reorders experts, not just a scale change."
    );
}
