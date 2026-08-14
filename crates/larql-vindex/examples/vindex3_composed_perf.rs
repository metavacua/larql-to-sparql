//! Where does a composed-run token actually go?
//!
//! The first composed run measured 9.2 s/token, which is a number nobody
//! should quote until it is decomposed. Two candidate stories:
//!
//! ```text
//! A  the loop        full-recompute (PLE, no KV cache) is simply this slow,
//!                    and VINDEX2 pays the same
//! B  the container   7680 region_bytes lookups per token cost real time
//! ```
//!
//! They are separable, because both routes run the *same* loop and the
//! decode-stage instrumentation already splits attention / dense / expert /
//! lm_head. Everything outside the expert stage is identical code on identical
//! bytes, so:
//!
//! ```text
//! container tax  =  expert_ms(container)  -  expert_ms(in-process)
//! ```
//!
//! Anything else that differs would be a bug, not a cost.
//!
//! # Reading the result honestly
//!
//! Run it on a cool machine on AC. This project has been bitten by 1.5–3×
//! phantom regressions measured on a hot box, and a composed-vs-control
//! comparison is *less* exposed to that than an absolute number is — both arms
//! see the same thermal state — but the absolute ms/token is not.
//!
//! Arms alternate rather than running one after the other for the same reason:
//! if the machine drifts during the run, drift lands on both arms instead of
//! whichever went second.
//!
//! ```text
//! LARQL_DECODE_STAGES=1 cargo run --release --example vindex3_composed_perf \
//!   -- <vindex2> <vindex3> [tokens] [rounds]
//! ```

use larql_inference::ffn::{ContainerRoutedBackend, InProcessMoeBackend, MoeExpertBackend};
use larql_inference::vindex::generate_kquant_cpu_routed;

const DEFAULT_TOKENS: usize = 4;
const DEFAULT_ROUNDS: usize = 2;
const PROMPT: &str = "The capital of France is";

struct Sample {
    wall_ms: f64,
    attn_ms: f64,
    dense_ms: f64,
    expert_ms: f64,
    lmhead_ms: f64,
    tokens: usize,
}

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let v2 = args
        .next()
        .ok_or("usage: vindex3_composed_perf <vindex2> <vindex3> [tokens] [rounds]")?;
    let v3 = args
        .next()
        .ok_or("usage: vindex3_composed_perf <vindex2> <vindex3> [tokens] [rounds]")?;
    let tokens: usize = args
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_TOKENS);
    let rounds: usize = args
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_ROUNDS);

    if !larql_inference::decode_stages::is_enabled() {
        eprintln!("note: set LARQL_DECODE_STAGES=1 for the per-stage split");
    }

    let v2_path = std::path::Path::new(&v2);
    let v3_path = std::path::Path::new(&v3);

    let mut cb = larql_vindex::SilentLoadCallbacks;
    let mut weights = larql_vindex::load_model_weights_kquant(v2_path, &mut cb)
        .map_err(|e| format!("load weights: {e}"))?;
    let tokenizer =
        larql_vindex::load_vindex_tokenizer(v2_path).map_err(|e| format!("load tokenizer: {e}"))?;
    let mut index = larql_vindex::VectorIndex::load_vindex(v2_path, &mut cb)
        .map_err(|e| format!("load vindex: {e}"))?;
    index
        .load_attn_kquant(v2_path)
        .map_err(|e| format!("load attn Q4K: {e}"))?;
    index
        .load_interleaved_kquant(v2_path)
        .map_err(|e| format!("load interleaved Q4K: {e}"))?;
    let _ = index.load_lm_head_kquant(v2_path);

    let wrapped = larql_inference::chat::render_user_prompt(v2_path, weights.arch.family(), PROMPT)
        .map_err(|e| format!("render prompt: {e}"))?;
    let ids = larql_inference::encode_prompt(&tokenizer, &*weights.arch, &wrapped)
        .map_err(|e| format!("tokenise: {e}"))?;

    let routed = ContainerRoutedBackend::open(v3_path, &weights, true)
        .map_err(|e| format!("compose: {e}"))?;
    let in_process = InProcessMoeBackend;

    println!("composed-run cost decomposition");
    println!("  prompt    {} token(s)", ids.len());
    println!("  generate  {tokens} token(s) x {rounds} round(s), arms alternating\n");

    let mut control: Vec<Sample> = Vec::new();
    let mut container: Vec<Sample> = Vec::new();

    // Round 0 is a discarded warm-up. The first pass faults in ~12 GB of
    // container and ~15 GB of model through the page cache, and its cost
    // decays for several rounds afterwards — measuring into that decay
    // attributes warm-up to whichever arm happened to go first.
    for round in 0..=rounds {
        // ABBA: swap which arm leads on alternate rounds. With a fixed order
        // the second arm is always the warmer one, which is enough on its own
        // to invent a difference in the direction of whoever runs second.
        let control_first = round % 2 == 0;
        for slot in 0..2 {
            let is_control = (slot == 0) == control_first;
            let backend: &dyn MoeExpertBackend = if is_control { &in_process } else { &routed };
            let label = if is_control { "vindex2" } else { "vindex3" };

            larql_inference::decode_stages::reset();
            let started = std::time::Instant::now();
            let out =
                generate_kquant_cpu_routed(&mut weights, &tokenizer, &ids, tokens, &index, backend)
                    .map_err(|e| format!("{label} arm: {e}"))?;
            let wall_ms = started.elapsed().as_secs_f64() * 1000.0;
            let (attn_ms, dense_ms, expert_ms, lmhead_ms) =
                larql_inference::decode_stages::snapshot_ms();

            let s = Sample {
                wall_ms,
                attn_ms,
                dense_ms,
                expert_ms,
                lmhead_ms,
                tokens: out.len().max(1),
            };
            let tag = if round == 0 {
                " (warm-up, discarded)"
            } else {
                ""
            };
            println!(
                "  round {round} {label:8}  {:8.0} ms  ({:7.0} ms/token){tag}",
                s.wall_ms,
                s.wall_ms / s.tokens as f64
            );
            if round == 0 {
                continue;
            }
            if is_control {
                control.push(s);
            } else {
                container.push(s);
            }
        }
    }

    fn best(v: &[Sample]) -> &Sample {
        // Fastest round: the minimum is the least contaminated by whatever
        // else the machine was doing, and the comparison is like-for-like.
        v.iter()
            .min_by(|a, b| {
                (a.wall_ms / a.tokens as f64)
                    .partial_cmp(&(b.wall_ms / b.tokens as f64))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("at least one round")
    }
    let c = best(&control);
    let k = best(&container);

    let per = |s: &Sample, v: f64| v / s.tokens as f64;
    println!("\n  best round, ms/token");
    println!(
        "  {:<10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "arm", "wall", "attn", "dense", "expert", "lm_head"
    );
    for (label, s) in [("vindex2", c), ("vindex3", k)] {
        println!(
            "  {:<10} {:>10.0} {:>10.0} {:>10.0} {:>10.0} {:>10.0}",
            label,
            per(s, s.wall_ms),
            per(s, s.attn_ms),
            per(s, s.dense_ms),
            per(s, s.expert_ms),
            per(s, s.lmhead_ms)
        );
    }

    let tax_ms = per(k, k.expert_ms) - per(c, c.expert_ms);
    let wall_delta = per(k, k.wall_ms) - per(c, c.wall_ms);
    println!("\n  container tax (expert stage)  {tax_ms:+.0} ms/token");
    println!("  wall delta                    {wall_delta:+.0} ms/token");
    if per(c, c.wall_ms) > 0.0 {
        println!(
            "  container / vindex2 wall      {:.3}x",
            per(k, k.wall_ms) / per(c, c.wall_ms)
        );
    }

    // What the instrumented stages do *not* explain. On the full-recompute
    // path most of a token is per-layer Q4_K dequantisation
    // (`insert_q4k_layer_tensors`), which no stage counter covers — so a
    // decomposition that only reported the four stages would silently account
    // for a minority of the time and invite the reader to divide the rest
    // among them.
    for (label, s) in [("vindex2", c), ("vindex3", k)] {
        let accounted =
            per(s, s.attn_ms) + per(s, s.dense_ms) + per(s, s.expert_ms) + per(s, s.lmhead_ms);
        let wall = per(s, s.wall_ms);
        println!(
            "  {label:<10} unaccounted {:>8.0} ms/token ({:.0}% of wall — not in any stage counter)",
            wall - accounted,
            100.0 * (wall - accounted) / wall.max(1.0)
        );
    }

    println!(
        "\n  Everything outside the expert stage is identical code on identical\n  \
         bytes; a large difference there would be a bug, not a cost."
    );
    Ok(())
}
