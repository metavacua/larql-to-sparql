//! MAP-4 bridge: does routing have usable lead time — and does it beat
//! the cheap alternatives?
//!
//! Registered as its own chuk-experiments experiment (`map-3-4-bridge`),
//! separate from `map-4` itself, per review: this answers "can future
//! operand demand be predicted early enough to prefetch," which is a
//! different, more immediately useful question than MAP-4e's "can an
//! approximation's danger be cheaply estimated."
//!
//! The core method (feed layer L's own router an EARLIER layer's
//! fully-computed residual instead of L's true current-time input, and
//! measure top-8 overlap against L's true selection) showed strong,
//! smoothly-decaying overlap with lead distance. But that alone doesn't
//! prove the mechanism is doing anything interesting — routing could
//! simply be "sticky" regardless of which residual drives it. Two cheap
//! competitors could in principle explain all of the observed overlap
//! without any real prediction happening:
//!
//! - **static popularity**: some experts are just selected often,
//!   regardless of the specific input: predicting "the usual top-8" could
//!   match the true set almost as well as a real prediction would.
//! - **previous-route reuse**: since MoE layers here are adjacent (L-1 is
//!   also MoE), just reusing L-1's OWN actual selection (no router
//!   substitution at all) might match L's true selection just as well as
//!   running L's router on L-1's residual does — i.e. routing might be
//!   "sticky across layers" for a reason that has nothing to do with this
//!   probe's specific mechanism.
//!
//! Both are computed here, at the lead=1 distance where they're most
//! directly comparable to the stale-residual method, with the popularity
//! baseline built LEAVE-ONE-OUT (excluding each prompt's own contribution)
//! to avoid trivially leaking the answer into its own baseline.
//!
//! Usage:
//! ```text
//! for i, prompt in prompts: LARQL_CPU_DUMP_LAYERS=/tmp/dump_p{i} larql run <vindex> "prompt" -n 1
//! cargo run --release -p larql-inference --example map4_bridge_routing_lead_time -- \
//!   --vindex <path> --dump-prefix /tmp/dump_p
//! ```

use std::collections::HashMap;

use larql_compute::cpu::ops::moe::{
    moe_expert_input, moe_route_from_router_input, moe_router_input,
};
use larql_compute::forward::dump_config::{cpu_layer_h_post_attn_path, cpu_layer_path};
use larql_compute::pipeline_layer::build_moe_weights;
use larql_compute::MoeLayerWeights;
use larql_models::ModelWeights;

const LEAD_TIMES: [usize; 5] = [1, 2, 4, 8, 16];
const NUM_PROMPTS: usize = 8;

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

fn route_at(moe: &MoeLayerWeights<'_>, h: &[f32], norm_offset: f32, eps: f32) -> Vec<usize> {
    let expert_input = moe_expert_input(h, moe, norm_offset, eps);
    let router_in = moe_router_input(h, &expert_input, moe, norm_offset, eps);
    let (ids, _weights) = moe_route_from_router_input(&router_in, moe);
    ids
}

fn overlap_count(a: &[usize], b: &[usize]) -> usize {
    a.iter().filter(|x| b.contains(x)).count()
}

/// Nearest earlier layer that's itself MoE (usually L-1 here, but not
/// assumed — searched explicitly in case of a hybrid gap).
fn nearest_earlier_moe_layer(weights: &ModelWeights, l: usize) -> Option<usize> {
    (0..l)
        .rev()
        .find(|&p| build_moe_weights(weights, &*weights.arch, p).is_some())
}

fn top_k_by_frequency(counts: &HashMap<usize, usize>, k: usize) -> Vec<usize> {
    let mut v: Vec<(usize, usize)> = counts.iter().map(|(&id, &c)| (id, c)).collect();
    v.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    v.into_iter().take(k).map(|(id, _)| id).collect()
}

/// Every layer the real architecture actually routes through MoE — not a
/// hand-picked sample. Cheap to test all of them: this probe only runs
/// router math on cached residuals, no forward pass.
fn discover_moe_layers(
    weights: &ModelWeights,
    arch: &dyn larql_models::ModelArchitecture,
) -> Vec<usize> {
    (0..weights.num_layers)
        .filter(|&l| build_moe_weights(weights, arch, l).is_some())
        .collect()
}

fn main() {
    println!("=== MAP-4 bridge: does routing have usable lead time, and does it beat cheap baselines? ===");
    let vindex = match arg("--vindex") {
        Some(v) => v,
        None => {
            println!("skipped — set --vindex <path> (a real MoE vindex, e.g. Gemma 4 26B-A4B) and --dump-prefix <dir prefix>");
            return;
        }
    };
    let dump_prefix = arg("--dump-prefix").expect("set --dump-prefix <dir prefix> (dirs <prefix>0 .. <prefix>{N-1}, one LARQL_CPU_DUMP_LAYERS dump per prompt)");

    println!("loading real vindex (weights only, no forward pass — activations come from real dumps): {vindex}");
    let mut cb = larql_vindex::SilentLoadCallbacks;
    let weights = larql_vindex::load_model_weights_kquant(std::path::Path::new(&vindex), &mut cb)
        .expect("load real vindex weights");
    let arch = &*weights.arch;
    let hidden = weights.hidden_size;
    let norm_offset = arch.norm_weight_offset();
    let eps = arch.norm_eps();

    let home_layers = discover_moe_layers(&weights, arch);
    println!(
        "discovered {} real MoE layers out of {}: {home_layers:?}\n",
        home_layers.len(),
        weights.num_layers
    );
    let sample_moe =
        build_moe_weights(&weights, arch, home_layers[0]).expect("discovered layer must be MoE");
    let top_k = sample_moe.top_k;
    let num_experts = sample_moe.num_experts;
    let chance_overlap = (top_k * top_k) as f64 / num_experts as f64;
    println!("top-{top_k} of {num_experts} experts; chance-level overlap ~= {chance_overlap:.2}\n");

    // moe weight structures, built once, reused across all prompts.
    let mut moe_at: HashMap<usize, MoeLayerWeights> = HashMap::new();
    let mut prev_of: HashMap<usize, usize> = HashMap::new();
    for &l in &home_layers {
        moe_at.insert(
            l,
            build_moe_weights(&weights, arch, l).expect("home layer must be MoE"),
        );
        if let Some(p) = nearest_earlier_moe_layer(&weights, l) {
            prev_of.insert(l, p);
            moe_at.entry(p).or_insert_with(|| {
                build_moe_weights(&weights, arch, p).expect("prev layer must be MoE")
            });
        }
    }

    // ── Pass 1: real true routing (and previous-layer routing) per prompt ──
    // true_ids[layer][prompt] = that layer's real selected experts for that prompt.
    let mut true_ids: HashMap<usize, Vec<Vec<usize>>> = HashMap::new();
    let mut layers_needed: Vec<usize> = home_layers.clone();
    layers_needed.extend(prev_of.values().copied());
    layers_needed.sort_unstable();
    layers_needed.dedup();

    for &l in &layers_needed {
        let moe = &moe_at[&l];
        let mut per_prompt = Vec::with_capacity(NUM_PROMPTS);
        for i in 0..NUM_PROMPTS {
            let dump_dir = format!("{dump_prefix}{i}");
            let h = last_row(&cpu_layer_h_post_attn_path(&dump_dir, l), hidden);
            per_prompt.push(route_at(moe, &h, norm_offset, eps));
        }
        true_ids.insert(l, per_prompt);
    }

    // ── Pass 2: stale-residual lead-time sweep (the original method) ──────
    println!("== stale-residual method (L's own router, fed L-k's residual) ==");
    struct LeadCell {
        home_layer: usize,
        lead: usize,
        prompt: usize,
        overlap: usize,
        pred_ids: Vec<usize>,
    }
    let mut lead_cells = Vec::new();
    for &l in &home_layers {
        let moe = &moe_at[&l];
        // `i` is the prompt ordinal, not merely an index into `true_ids`: it
        // also names the dump directory and is recorded on each cell.
        // Iterating `true_ids[&l]` instead would silently do fewer prompts if
        // that entry were short, where indexing panics — for a research sweep
        // a loud stop beats a quiet undercount.
        #[allow(clippy::needless_range_loop)]
        for i in 0..NUM_PROMPTS {
            let dump_dir = format!("{dump_prefix}{i}");
            for &k in &LEAD_TIMES {
                if k > l {
                    continue;
                }
                let h_early = last_row(&cpu_layer_path(&dump_dir, l - k), hidden);
                let pred_ids = route_at(moe, &h_early, norm_offset, eps);
                let ov = overlap_count(&true_ids[&l][i], &pred_ids);
                lead_cells.push(LeadCell {
                    home_layer: l,
                    lead: k,
                    prompt: i,
                    overlap: ov,
                    pred_ids,
                });
            }
        }
    }
    for &k in &LEAD_TIMES {
        let matching: Vec<&LeadCell> = lead_cells.iter().filter(|c| c.lead == k).collect();
        if matching.is_empty() {
            continue;
        }
        let mean = matching.iter().map(|c| c.overlap as f64).sum::<f64>() / matching.len() as f64;
        print!(
            "  lead {k:>2}: mean overlap {mean:.2}/{top_k}  (n={})  per-layer:",
            matching.len()
        );
        for &l in &home_layers {
            let per_layer: Vec<&LeadCell> = matching
                .iter()
                .filter(|c| c.home_layer == l)
                .copied()
                .collect();
            if per_layer.is_empty() {
                continue;
            }
            let m =
                per_layer.iter().map(|c| c.overlap as f64).sum::<f64>() / per_layer.len() as f64;
            print!(" L{l}={m:.2}");
        }
        println!();
    }

    // ── Pass 3: previous-route baseline (reuse L-1's OWN true selection) ──
    println!("\n== previous-route baseline (reuse the nearest earlier MoE layer's own true selection, no router substitution) ==");
    for &l in &home_layers {
        let Some(&prev) = prev_of.get(&l) else {
            continue;
        };
        let overlaps: Vec<usize> = (0..NUM_PROMPTS)
            .map(|i| overlap_count(&true_ids[&l][i], &true_ids[&prev][i]))
            .collect();
        let mean = overlaps.iter().sum::<usize>() as f64 / overlaps.len() as f64;
        println!(
            "  L{l:>2} <- L{prev}'s own selection: mean overlap {mean:.2}/{top_k}  ({overlaps:?})"
        );
    }

    // ── Pass 4: static popularity baseline (leave-one-out) ─────────────────
    println!("\n== static popularity baseline (top-{top_k} most frequent experts across the OTHER 7 prompts, leave-one-out) ==");
    // (layer, held_out_prompt) -> that prompt's leave-one-out popularity set, reused in passes 5/6.
    let mut pop_sets: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    for &l in &home_layers {
        let mut overlaps = Vec::with_capacity(NUM_PROMPTS);
        for held_out in 0..NUM_PROMPTS {
            let mut counts: HashMap<usize, usize> = HashMap::new();
            for (i, ids) in true_ids[&l].iter().enumerate() {
                if i == held_out {
                    continue;
                }
                for &id in ids {
                    *counts.entry(id).or_insert(0) += 1;
                }
            }
            let popular = top_k_by_frequency(&counts, top_k);
            overlaps.push(overlap_count(&true_ids[&l][held_out], &popular));
            pop_sets.insert((l, held_out), popular);
        }
        let mean = overlaps.iter().sum::<usize>() as f64 / overlaps.len() as f64;
        println!("  L{l:>2}: mean overlap {mean:.2}/{top_k}  ({overlaps:?})");
    }

    // ── Pass 5: excess of stale-residual over popularity, at EVERY lead ────
    // The single lead=1 comparison only checks whether the headline number
    // survives its cheapest competitor. This checks whether ANY lead time
    // beats popularity, and whether the gap (if any) itself decays with
    // lead the way a real predictive signal should but a lead-independent
    // popularity floor cannot.
    println!("\n== excess of stale-residual over popularity, per layer x lead ==");
    println!("  (positive = stale-residual beats the zero-cost popularity floor at that lead)");
    for &l in &home_layers {
        for &k in &LEAD_TIMES {
            let cells: Vec<&LeadCell> = lead_cells
                .iter()
                .filter(|c| c.home_layer == l && c.lead == k)
                .collect();
            if cells.is_empty() {
                continue;
            }
            let stale_mean =
                cells.iter().map(|c| c.overlap as f64).sum::<f64>() / cells.len() as f64;
            let pop_mean = cells
                .iter()
                .map(|c| overlap_count(&true_ids[&l][c.prompt], &pop_sets[&(l, c.prompt)]) as f64)
                .sum::<f64>()
                / cells.len() as f64;
            println!("  L{l:>2} lead={k:>2}: stale={stale_mean:.2}  popularity={pop_mean:.2}  excess={:+.2}", stale_mean - pop_mean);
        }
    }

    // ── Pass 6: union-under-budget (does stacking stale-residual ON TOP ──
    // of popularity buy real extra coverage, for a real prefetch budget) ──
    println!("\n== union(popularity, stale-residual) vs popularity alone, per layer x lead ==");
    println!("  (reports achieved overlap AND the union's actual candidate-set size, since the union isn't free)");
    for &l in &home_layers {
        for &k in &LEAD_TIMES {
            let cells: Vec<&LeadCell> = lead_cells
                .iter()
                .filter(|c| c.home_layer == l && c.lead == k)
                .collect();
            if cells.is_empty() {
                continue;
            }
            let mut union_overlaps = Vec::with_capacity(cells.len());
            let mut union_sizes = Vec::with_capacity(cells.len());
            let mut pop_overlaps = Vec::with_capacity(cells.len());
            for c in &cells {
                let pop = &pop_sets[&(l, c.prompt)];
                let mut union: Vec<usize> = pop.iter().chain(c.pred_ids.iter()).copied().collect();
                union.sort_unstable();
                union.dedup();
                union_sizes.push(union.len());
                union_overlaps.push(overlap_count(&true_ids[&l][c.prompt], &union));
                pop_overlaps.push(overlap_count(&true_ids[&l][c.prompt], pop));
            }
            let mean_union_ov =
                union_overlaps.iter().sum::<usize>() as f64 / union_overlaps.len() as f64;
            let mean_union_sz = union_sizes.iter().sum::<usize>() as f64 / union_sizes.len() as f64;
            let mean_pop_ov = pop_overlaps.iter().sum::<usize>() as f64 / pop_overlaps.len() as f64;
            println!(
                "  L{l:>2} lead={k:>2}: popularity-alone={mean_pop_ov:.2}/{top_k}  union={mean_union_ov:.2}/{mean_union_sz:.1} candidates  marginal-gain={:+.2}",
                mean_union_ov - mean_pop_ov
            );
        }
    }

    println!(
        "\n== the comparison that matters ==\n\
         If the stale-residual method's lead=1 overlap is not clearly ABOVE both the previous-route\n\
         and static-popularity baselines, the earlier finding is explained by routing being generically\n\
         'sticky' or popularity-skewed, not by this probe doing real prediction. Compare the lead=1 row\n\
         above against both baseline rows directly, per layer. Passes 5-6 extend this across every lead\n\
         time and check whether stacking the stale-residual method on top of popularity (rather than\n\
         picking one) buys real coverage for its added candidate-set cost."
    );
}
