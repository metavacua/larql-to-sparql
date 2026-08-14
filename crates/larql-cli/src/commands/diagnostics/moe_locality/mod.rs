//! `larql moe-locality` — expert-selection locality over a routing trace.
//!
//! Answers two questions from one trace, both of which gate whether a
//! disk-resident MoE can be served faster than its routed bytes suggest:
//!
//! 1. **Does speculative decoding amortise the expert bank?** Verifying `K`
//!    drafted tokens reads the *union* of their experts. If that union grows
//!    linearly in `K`, bytes-per-token is flat and speculation buys nothing on
//!    the routed path however good the draft is.
//! 2. **Can a hot cache work?** The LRU curve gives the hit rate a resident
//!    subset of experts would actually see.
//!
//! Both are reported against a uniform baseline *and* a skew-preserving
//! shuffle, because those two extrapolate differently to a load-balanced model.
//! See [`stats`] for the argument.

mod shuffle;
mod stats;
mod trace;

use std::collections::BTreeMap;
use std::path::PathBuf;

use clap::Args;

use shuffle::{shuffle_runs, Lcg};
use stats::Run;
use trace::Trace;

/// Cache capacities are stepped as multiples of top-k, doubling, up to this
/// many rows — enough to show the shape of the curve without burying it.
const MAX_CACHE_ROWS: usize = 6;

/// Block sizes for the union curve. Doubling from the smallest meaningful
/// block (a pair) to a draft length no speculative decoder exceeds in practice.
const DEFAULT_BLOCKS: &str = "2,4,8,16,32";

/// Block size for the per-layer breakdown — mid-range of [`DEFAULT_BLOCKS`],
/// where the union curve has bent but not yet saturated.
const DEFAULT_PER_LAYER_BLOCK: usize = 8;

/// Seed for the skew-preserving shuffle. Fixed so a reported ratio can be
/// re-derived from the same trace later.
const DEFAULT_SHUFFLE_SEED: u64 = 20_260_730;

/// Printed when a ratio has no informative baseline to divide by.
const MISSING_CELL: &str = "     —";

#[derive(Args, Debug)]
pub struct MoeLocalityArgs {
    /// JSONL routing trace, as written by `LARQL_MOE_ROUTE_TRACE=<path>`.
    trace: PathBuf,

    /// Expert count for the uniform baseline. Defaults to the largest expert id
    /// in the trace plus one, which under-counts if an expert never fired.
    #[arg(long)]
    num_experts: Option<usize>,

    /// Block sizes for the union curve.
    #[arg(long, value_delimiter = ',', default_value = DEFAULT_BLOCKS)]
    blocks: Vec<usize>,

    /// Block size for the per-layer breakdown.
    #[arg(long, default_value_t = DEFAULT_PER_LAYER_BLOCK)]
    per_layer_block: usize,

    /// Seed for the skew-preserving shuffle control.
    #[arg(long, default_value_t = DEFAULT_SHUFFLE_SEED)]
    seed: u64,
}

/// Layer id → runs of positions.
type Layers = BTreeMap<usize, Vec<Run>>;

/// Mean of a per-layer statistic across the layers that produced one.
///
/// Layers are averaged rather than pooled: routing behaviour differs by depth,
/// and pooling would let the longest layer dominate a number meant to describe
/// the stack.
fn mean_over_layers(layers: &Layers, stat: impl Fn(&[Run]) -> Option<f64>) -> Option<f64> {
    let values: Vec<f64> = layers.values().filter_map(|runs| stat(runs)).collect();
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}

/// Capacities for the LRU sweep: `top_k`, doubling, staying under `num_experts`.
fn cache_capacities(top_k: usize, num_experts: usize) -> Vec<usize> {
    let mut capacities = Vec::new();
    let mut capacity = top_k.max(1);
    while capacity < num_experts && capacities.len() < MAX_CACHE_ROWS {
        capacities.push(capacity);
        capacity *= 2;
    }
    capacities
}

/// `observed / baseline`, or `None` when the baseline carries no information.
fn ratio(observed: f64, baseline: f64) -> Option<f64> {
    (baseline > 0.0).then_some(observed / baseline)
}

/// Format an optional ratio for a table cell.
fn cell(value: Option<f64>) -> String {
    value.map_or_else(|| MISSING_CELL.to_string(), |v| format!("{v:6.3}"))
}

pub fn run(args: MoeLocalityArgs) -> Result<(), Box<dyn std::error::Error>> {
    let observed_trace = trace::load(&args.trace)?;
    let top_k = observed_trace.top_k();
    let observed_experts = observed_trace.observed_num_experts();
    let num_experts = args.num_experts.unwrap_or(observed_experts);

    if num_experts < observed_experts {
        return Err(format!(
            "--num-experts {num_experts} is below the largest id in the trace \
             ({}); those selections would be silently dropped",
            observed_experts - 1
        )
        .into());
    }

    let mut rng = Lcg::new(args.seed);
    let shuffled: Layers = observed_trace
        .layers
        .iter()
        .map(|(&layer, runs)| (layer, shuffle_runs(runs, &mut rng)))
        .collect();

    print_header(
        &observed_trace,
        num_experts,
        observed_experts,
        top_k,
        args.seed,
    );
    print_union_curve(
        &observed_trace.layers,
        &shuffled,
        &args.blocks,
        num_experts,
        top_k,
    );
    print_per_layer(
        &observed_trace.layers,
        &shuffled,
        args.per_layer_block,
        num_experts,
        top_k,
    );
    print_cache_curve(&observed_trace.layers, &shuffled, num_experts, top_k);
    Ok(())
}

fn print_header(
    trace: &Trace,
    num_experts: usize,
    observed_experts: usize,
    top_k: usize,
    seed: u64,
) {
    let activation = 100.0 * top_k as f64 / num_experts.max(1) as f64;
    println!(
        "trace: {} layers, {} runs, {} positions",
        trace.layers.len(),
        trace.num_runs(),
        trace.positions()
    );
    let source = if num_experts == observed_experts {
        "observed"
    } else {
        "declared"
    };
    println!("experts: {num_experts} ({source}), top-k: {top_k}, activation: {activation:.2}%");
    println!("shuffle control seed: {seed}");

    let adjacent = mean_over_layers(&trace.layers, stats::mean_adjacent_overlap);
    let uniform = stats::uniform_adjacent_overlap(num_experts, top_k);
    if let Some(adjacent) = adjacent {
        println!(
            "\nadjacent-position overlap: {adjacent:.3} experts \
             (uniform {uniform:.3}, ratio {})",
            cell(ratio(adjacent, uniform))
        );
    }
}

/// The union curve — the speculative-decoding answer.
fn print_union_curve(
    observed: &Layers,
    shuffled: &Layers,
    blocks: &[usize],
    num_experts: usize,
    top_k: usize,
) {
    println!("\n── distinct experts over K consecutive positions ──");
    println!(
        "{:>4}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>10}  {:>7}",
        "K", "observed", "shuffled", "uniform", "obs/unif", "obs/shuf", "exp/token", "saving"
    );

    for &block in blocks {
        let stat = |runs: &[Run]| stats::mean_union(runs, block, num_experts);
        let (Some(obs), Some(shuf)) = (
            mean_over_layers(observed, stat),
            mean_over_layers(shuffled, stat),
        ) else {
            println!("{block:>4}  (no run long enough)");
            continue;
        };
        let uniform = stats::uniform_union(block, num_experts, top_k);
        let per_token = obs / block as f64;
        let saving = 1.0 - per_token / top_k.max(1) as f64;
        println!(
            "{block:>4}  {obs:>8.2}  {shuf:>8.2}  {uniform:>8.2}  {}  {}  {per_token:>10.2}  {:>6.1}%",
            cell(ratio(obs, uniform)),
            cell(ratio(obs, shuf)),
            100.0 * saving
        );
    }
    println!(
        "\n  exp/token is bytes-per-token in units of one expert; saving is \
         against {top_k} (K=1)."
    );
}

/// Depth breakdown — locality is not expected to be uniform down the stack.
fn print_per_layer(
    observed: &Layers,
    shuffled: &Layers,
    block: usize,
    num_experts: usize,
    top_k: usize,
) {
    println!("\n── per layer, K={block} ──");
    println!(
        "{:>5}  {:>8}  {:>8}  {:>8}  {:>8}",
        "layer", "observed", "shuffled", "obs/unif", "obs/shuf"
    );
    let uniform = stats::uniform_union(block, num_experts, top_k);

    for (&layer, runs) in observed {
        let Some(obs) = stats::mean_union(runs, block, num_experts) else {
            continue;
        };
        let shuf = shuffled
            .get(&layer)
            .and_then(|runs| stats::mean_union(runs, block, num_experts));
        println!(
            "{layer:>5}  {obs:>8.2}  {}  {}  {}",
            shuf.map_or_else(|| "       —".to_string(), |v| format!("{v:8.2}")),
            cell(ratio(obs, uniform)),
            cell(shuf.and_then(|s| ratio(obs, s)))
        );
    }
}

/// The hot-cache answer.
fn print_cache_curve(observed: &Layers, shuffled: &Layers, num_experts: usize, top_k: usize) {
    println!("\n── LRU hit rate by resident experts per layer ──");
    println!(
        "{:>8}  {:>7}  {:>8}  {:>8}  {:>8}",
        "capacity", "of bank", "observed", "shuffled", "lift"
    );

    for capacity in cache_capacities(top_k, num_experts) {
        let stat = |runs: &[Run]| stats::lru_hit_rate(runs, capacity);
        let (Some(obs), Some(shuf)) = (
            mean_over_layers(observed, stat),
            mean_over_layers(shuffled, stat),
        ) else {
            continue;
        };
        let share = 100.0 * capacity as f64 / num_experts.max(1) as f64;
        println!(
            "{capacity:>8}  {share:>6.1}%  {:>7.1}%  {:>7.1}%  {}",
            100.0 * obs,
            100.0 * shuf,
            cell(ratio(obs, shuf))
        );
    }
    println!(
        "\n  lift is observed/shuffled: above 1.0 is temporal locality, \
         everything else is load skew."
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layers() -> Layers {
        BTreeMap::from([
            (0, vec![vec![vec![0, 1], vec![1, 2]]]),
            (1, vec![vec![vec![0, 1], vec![0, 1]]]),
        ])
    }

    #[test]
    fn mean_over_layers_averages_layers_not_windows() {
        // Layer 0 unions to 3, layer 1 to 2; the mean is 2.5 regardless of how
        // many windows each layer contributed.
        let mean = mean_over_layers(&layers(), |runs| stats::mean_union(runs, 2, 4));
        assert_eq!(mean, Some(2.5));
    }

    #[test]
    fn mean_over_layers_skips_layers_without_a_value() {
        let mean = mean_over_layers(&layers(), |runs| stats::mean_union(runs, 9, 4));
        assert_eq!(mean, None);
    }

    #[test]
    fn cache_capacities_double_from_top_k_and_stay_inside_the_bank() {
        assert_eq!(cache_capacities(4, 32), vec![4, 8, 16]);
        assert_eq!(cache_capacities(16, 896), vec![16, 32, 64, 128, 256, 512]);
    }

    #[test]
    fn cache_capacities_are_empty_when_the_bank_is_the_top_k() {
        assert!(cache_capacities(8, 8).is_empty());
    }

    #[test]
    fn ratio_guards_a_zero_baseline() {
        assert_eq!(ratio(1.0, 2.0), Some(0.5));
        assert_eq!(ratio(1.0, 0.0), None);
        assert_eq!(cell(None), MISSING_CELL);
        assert_eq!(cell(Some(0.5)), " 0.500");
    }
}
