//! Where do the seconds in a V3 request actually go?
//!
//! The HTTP measurements said a warm 5-token / 1-token request costs
//! ~6.7 s of fixed work on a 3B container, and that server prefill runs
//! at roughly half the rate of `larql vindex3 exec` on the same
//! container and backend. Neither number tells you *which* call is
//! expensive, and HTTP is in the way.
//!
//! This profiles the serve path **below HTTP**, against a real
//! container, timing exactly the calls `generate_v3_resumable` makes:
//!
//! ```text
//! prefill_into(prompt, &mut kv)      <- batch prefill into the provider
//! session_with_kv(&mut kv)           <- open the decode session
//! session.step(token) x N            <- decode
//! ```
//!
//! It runs the **same requests twice** on one container in one process:
//! once through the load-per-call entry points (`Vindex3Runtime`) and
//! once over a prepared image (`PreparedVindex3`). Same tokens, same
//! backend, same warm page cache — the only difference is where the
//! operands come from.
//!
//! Usage:
//!   cargo run --release -p larql-server --example v3_request_phase_profile \
//!       -- <container> [prompt_tokens] [decode_tokens] [requests]

use std::path::PathBuf;
use std::time::Instant;

use larql_inference::vindex3::Vindex3Runtime;
use larql_kv::CanonicalKvState;
use larql_vindex::format::vindex3::opplan::exec::production::ProductionBackend;

const COMPONENT: &str = "target";

fn secs(t: Instant) -> f64 {
    t.elapsed().as_secs_f64()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let container = PathBuf::from(
        args.next()
            .expect("usage: <container> [prompt] [decode] [reqs]"),
    );
    let prompt_len: usize = args.next().map_or(5, |a| a.parse().unwrap());
    let decode_len: usize = args.next().map_or(1, |a| a.parse().unwrap());
    let requests: usize = args.next().map_or(3, |a| a.parse().unwrap());

    // Token ids are given, never tokenised: this profiles execution,
    // not text handling.
    let prompt: Vec<u32> = (0..prompt_len).map(|i| (i % 2000 + 100) as u32).collect();

    let t = Instant::now();
    let unprepared = Vindex3Runtime::open(&container, COMPONENT, ProductionBackend::new())?;
    println!("startup: open runtime           {:8.3} s", secs(t));

    // ── Arm A: load-per-call, as the serve path worked before ──────────
    println!("\nARM A — load per call (Vindex3Runtime):");
    println!(
        "{:>4}  {:>12}  {:>12}  {:>12}  {:>12}",
        "req", "prefill_into", "session_open", "decode", "total"
    );
    let mut arm_a = f64::MAX;
    for r in 0..requests {
        let mut kv = CanonicalKvState::new();
        let t = Instant::now();
        let _ = unprepared.prefill_into(&prompt, &mut kv)?;
        let t_prefill = secs(t);
        let t = Instant::now();
        let mut session = unprepared.session_with_kv(&mut kv)?;
        let t_session = secs(t);
        let t = Instant::now();
        {
            use larql_inference::vindex3::LogitsSession;
            let mut token = 1u32;
            for _ in 0..decode_len {
                token = argmax(&session.step(token)?);
            }
        }
        let t_decode = secs(t);
        let total = t_prefill + t_session + t_decode;
        arm_a = arm_a.min(total);
        println!(
            "{:>4}  {:>12.3}  {:>12.3}  {:>12.3}  {:>12.3}",
            r + 1,
            t_prefill,
            t_session,
            t_decode,
            total
        );
    }

    // ── Arm B: prepared once, as it works now ──────────────────────────
    let t = Instant::now();
    let runtime = unprepared.prepare()?;
    let prepare_cost = secs(t);
    println!("\nprepare operands (ONCE, at server boot)  {prepare_cost:8.3} s\n");

    println!("ARM B — over the prepared image (PreparedVindex3):");
    println!(
        "{:>4}  {:>12}  {:>12}  {:>12}  {:>12}",
        "req", "prefill_into", "session_open", "decode", "total"
    );

    let mut totals = [0.0f64; 3];
    for r in 0..requests {
        let mut kv = CanonicalKvState::new();

        let t = Instant::now();
        let _logits = runtime.prefill_into(&prompt, &mut kv)?;
        let t_prefill = secs(t);

        let t = Instant::now();
        let mut session = runtime.session_with_kv(&mut kv)?;
        let t_session = secs(t);

        let t = Instant::now();
        {
            use larql_inference::vindex3::LogitsSession;
            let mut token = 1u32;
            for _ in 0..decode_len {
                let logits = session.step(token)?;
                token = argmax(&logits);
            }
        }
        let t_decode = secs(t);

        totals[0] += t_prefill;
        totals[1] += t_session;
        totals[2] += t_decode;
        println!(
            "{:>4}  {:>12.3}  {:>12.3}  {:>12.3}  {:>12.3}",
            r + 1,
            t_prefill,
            t_session,
            t_decode,
            t_prefill + t_session + t_decode
        );
    }

    let n = requests as f64;
    let (p, s, d) = (totals[0] / n, totals[1] / n, totals[2] / n);
    let total = p + s + d;
    println!("\nmean per request ({prompt_len} prompt tokens, {decode_len} decoded):");
    println!("  prefill_into   {p:8.3} s   {:5.1}%", 100.0 * p / total);
    println!("  session_open   {s:8.3} s   {:5.1}%", 100.0 * s / total);
    println!("  decode         {d:8.3} s   {:5.1}%", 100.0 * d / total);
    println!("  total          {total:8.3} s");

    let arm_b = totals.iter().sum::<f64>() / n;
    println!("\n── gate 5 ───────────────────────────────────────────────");
    println!("  arm A, load per call (best)  {arm_a:8.3} s");
    println!("  arm B, prepared (mean)       {arm_b:8.3} s");
    println!("  speedup                      {:8.2}x", arm_a / arm_b);
    println!("  one-off preparation          {prepare_cost:8.3} s");

    Ok(())
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, v) in logits.iter().enumerate() {
        if *v > logits[best] {
            best = i;
        }
    }
    best as u32
}
