//! `larql k3-ledger retention` — DEC-9A rendering.
//!
//! I/O, excluded from coverage. Reads a local routed capture pool; no
//! checkpoint, no network, no model.

use serde_json::json;

use super::args::RetentionArgs;
use super::retention::{self as ret, Policy, ReferenceStream, SimResult};
use super::selection_trace::SelectionTrace;
use crate::commands::primary::dec_bench::capture_format::CapturePool;

type R = Result<(), Box<dyn std::error::Error>>;

const ARMS: [Policy; 5] = [
    Policy::Min,
    Policy::StaticOracle,
    Policy::Lru,
    Policy::Lfu,
    Policy::Random,
];

pub fn run(a: &RetentionArgs, as_json: bool) -> R {
    let pool = CapturePool::open(&a.pool)?;
    let trace = SelectionTrace::from_routing_pool(&pool)?;
    let stream = ret::stream_from_trace(&trace);
    let expert_bytes = a.expert_bytes.unwrap_or(ret::K3_ROUTED_EXPERT_BYTES);

    let capacities: Vec<usize> = a
        .capacity_fracs
        .iter()
        .map(|f| ((stream.bank() as f64 * f).round() as usize).max(1))
        .collect();

    let mut results: Vec<SimResult> = Vec::new();
    for &cap in &capacities {
        for &p in &ARMS {
            results.push(ret::simulate(
                &stream,
                p,
                cap,
                !a.cold,
                a.seed,
                expert_bytes,
            ));
        }
    }
    let gates: Vec<_> = capacities
        .iter()
        .filter_map(|&c| ret::gate(&results, c))
        .collect();

    if as_json {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "pool": a.pool.display().to_string(),
                "model": pool.manifest.model,
                "warm": !a.cold,
                "bank": stream.bank(),
                "tokens": stream.tokens(),
                "requests": stream.requests(),
                "distinct_slots": stream.distinct_slots(),
                "activation_fraction": trace.activation_fraction(),
                "expert_bytes": expert_bytes,
                "results": results,
                "gates": gates,
            }))?
        );
        return Ok(());
    }

    render(a, &pool.manifest.model, &trace, &stream, &results, &gates);
    Ok(())
}

fn render(
    a: &RetentionArgs,
    model: &str,
    trace: &SelectionTrace,
    stream: &ReferenceStream,
    results: &[SimResult],
    gates: &[ret::RetentionGate],
) {
    println!();
    println!(
        "=== DEC-9A: retention oracle gate — {} ===",
        a.pool.display()
    );
    println!("model    {model}");
    println!(
        "trace    {} sessions x {} steps, {} strata, top-{} of {} ({:.2}% activation)",
        trace.sessions(),
        trace.steps(),
        trace.active_strata().len(),
        trace.width(),
        trace.alphabet(),
        trace.activation_fraction() * 100.0,
    );
    println!(
        "stream   {} requests over {} tokens; bank {} slots, {} ever touched",
        stream.requests(),
        stream.tokens(),
        stream.bank(),
        stream.distinct_slots(),
    );
    println!(
        "         one token needs {} distinct slots across all strata, {} per stratum",
        stream.peak_token_footprint(),
        stream.max_group(),
    );
    println!(
        "         cache is {} across sessions; the pool is a DOMAIN MIXTURE, which is",
        if a.cold {
            "COLD (reset)"
        } else {
            "WARM (kept)"
        }
    );
    println!(
        "         anti-conservative for warm reuse (R10) — a single-domain load would do better"
    );
    println!();

    println!("--- arms ---");
    for p in ARMS {
        println!(
            "  {:<14} {:<22} {}",
            short(p),
            p.label(),
            if p.is_oracle() {
                "NOT DEPLOYABLE — needs future knowledge; a ceiling, never a policy"
            } else {
                "deployable"
            }
        );
    }
    println!();

    println!("--- misses per token (lower is better) ---");
    print!("{:>9} {:>7}", "capacity", "%bank");
    for p in ARMS {
        print!(" {:>13}", short(p));
    }
    println!("   flags");
    let mut seen = Vec::new();
    for r in results {
        if seen.contains(&r.capacity) {
            continue;
        }
        seen.push(r.capacity);
        print!("{:>9} {:>6.2}%", r.capacity, 100.0 * r.capacity_frac);
        for p in ARMS {
            let v = results
                .iter()
                .find(|x| x.capacity == r.capacity && x.policy == p)
                .map(|x| x.misses_per_token)
                .unwrap_or(f64::NAN);
            print!(" {v:>13.2}");
        }
        println!(
            "   {}",
            if r.below_group_floor {
                "THRASH: below one stratum's simultaneous demand"
            } else if r.below_token_floor {
                "cyclic-thrash floor: below one token's 240-slot footprint"
            } else {
                ""
            }
        );
    }
    println!(
        "  compulsory floor {} misses total ({:.2}/token) — unavoidable at any capacity",
        results.first().map(|r| r.compulsory).unwrap_or(0),
        results.first().map(|r| r.compulsory).unwrap_or(0) as f64 / stream.tokens().max(1) as f64,
    );
    println!();

    println!("--- the two gaps, and which one prediction can target ---");
    println!(
        "{:>9} {:>7} {:>12} {:>12} {:>12}",
        "capacity", "%bank", "MIN/LRU", "MIN/static", "LRU/random"
    );
    for g in gates {
        println!(
            "{:>9} {:>6.2}% {:>11.1}% {:>11.1}% {:>11.1}%   {}",
            g.capacity,
            100.0 * g.capacity_frac,
            100.0 * g.oracle_over_lru,
            100.0 * g.temporal_prize,
            100.0 * g.recency_value,
            if g.below_token_floor {
                "<- LRU columns degenerate here"
            } else {
                ""
            }
        );
    }
    println!("  MIN/LRU     what a better POLICY could buy over recency.");
    println!("  MIN/static  the TEMPORAL prize — all a predictor, transition graph or");
    println!("              lookahead can ever target. A fixed set needs no prediction.");
    println!("  LRU/random  does recency carry information at all?");
    println!();

    println!(
        "--- external bytes/token at {:.2} MB per expert ---",
        a.expert_bytes.unwrap_or(ret::K3_ROUTED_EXPERT_BYTES) as f64 / 1e6
    );
    if a.expert_bytes.is_none() {
        println!("  NOTE: K3's expert size applied to another model's trace — a scaled");
        println!("  projection for readability, NOT a K3 measurement (R2).");
    }
    for g in gates {
        let per = |m: f64| m * a.expert_bytes.unwrap_or(ret::K3_ROUTED_EXPERT_BYTES) as f64 / 1e9;
        println!(
            "  {:>6.2}% bank   MIN {:>7.2} GB   LRU {:>7.2} GB   static {:>7.2} GB",
            100.0 * g.capacity_frac,
            per(g.min_misses_per_token),
            per(g.lru_misses_per_token),
            per(g.static_misses_per_token),
        );
    }
    println!();

    let best = gates
        .iter()
        .filter(|g| !g.below_token_floor)
        .max_by(|x, y| x.temporal_prize.total_cmp(&y.temporal_prize));
    match best {
        Some(g) => {
            println!("VERDICT: {}", ret::verdict(g.temporal_prize));
            println!(
                "  best temporal prize {:.1}% at {:.2}% residency, against bands {:.0}% / {:.0}%",
                100.0 * g.temporal_prize,
                100.0 * g.capacity_frac,
                100.0 * ret::PRIZE_CLOSE_PROGRAMME,
                100.0 * ret::PRIZE_INTERESTING,
            );
        }
        None => println!("VERDICT: no capacity produced a complete arm set"),
    }
}

fn short(p: Policy) -> &'static str {
    match p {
        Policy::Min => "MIN(oracle)",
        Policy::StaticOracle => "static(orcl)",
        Policy::Lru => "LRU",
        Policy::Lfu => "LFU",
        Policy::Random => "random",
    }
}
