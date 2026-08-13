//! I/O-bound driver for `larql dec-bench drift` — the C6 wire-fidelity gate
//! (docs/dec-funnel.md §3 DEC-1A). Loads the model exactly like the bench /
//! capture remote-FFN runtimes, then scores the pinned corpus teacher-forced
//! through `score_forced_with_remote_ffn` once per wire arm, always starting
//! with the in-run f32/f32 baseline. Coverage-excluded (needs live model
//! weights + a running expert server); ALL decision logic — bits math, arm
//! plan, drift/gate arithmetic, records — lives in `drift.rs` and is gated.
//!
//! An end-to-end test would need a real Q4K vindex plus the fused Metal
//! decode path (`decode_token_with_moe` returns `None` on CPU), so there is
//! no cheap synthetic-fixture integration test for this driver — the same
//! constraint as `capture_runtime.rs` / `bench/remote_ffn_runtime.rs`.

use std::time::{Duration, Instant};

use super::args::DriftArgs;
use super::drift::{
    arm_connection, bits_for_target, drift_plan, format_drift_table, normalize_corpus,
    validate_gate_pct, DecDriftJsonResult, DriftArmResult, DETERMINISM_NOTE,
};
use super::pulse::to_jsonl;
use super::replay::WireSpec;
use larql_inference::{score_forced_with_remote_ffn, LayerShardedBackend};

pub(super) fn run_drift(args: &DriftArgs) -> Result<(), Box<dyn std::error::Error>> {
    validate_gate_pct(args.gate_pct)?;
    let requested = WireSpec::parse_list(&args.wire).map_err(|e| format!("--wire: {e}"))?;
    let plan = drift_plan(&requested);

    let vindex_path = crate::commands::primary::cache::resolve_model(&args.model)?;
    if !vindex_path.is_dir() {
        return Err(format!(
            "resolved model path is not a directory: {}",
            vindex_path.display(),
        )
        .into());
    }

    // An explicit `--metal` with no usable device is a loud error, not a CPU
    // fallback (see `backend_select`); without `--metal` the forced scorer
    // itself errors — the remote-FFN walk needs the fused GPU decode path.
    let backend: Box<dyn larql_compute::ComputeBackend> =
        crate::backend_select::backend_for_metal_flag(args.metal)?;

    let mut cb = larql_vindex::SilentLoadCallbacks;
    let weights = larql_vindex::load_model_weights_kquant(&vindex_path, &mut cb)
        .map_err(|e| format!("failed to load client weights: {e}"))?;
    let tokenizer = larql_vindex::load_vindex_tokenizer(&vindex_path)
        .map_err(|e| format!("failed to load tokenizer: {e}"))?;
    let mut index = larql_vindex::VectorIndex::load_vindex(&vindex_path, &mut cb)
        .map_err(|e| format!("failed to load vindex: {e}"))?;
    index.load_attn_kquant(&vindex_path)?;
    index.load_interleaved_kquant(&vindex_path)?;
    let _ = index.load_lm_head_kquant(&vindex_path);

    // Pinned corpus, normalized the way `larql shannon verify` normalizes
    // (CRLF → LF) so the bits/char here is comparable to the CI gate's.
    let raw_corpus = std::fs::read_to_string(&args.corpus)
        .map_err(|e| format!("read corpus {}: {e}", args.corpus.display()))?;
    let text = normalize_corpus(&raw_corpus, args.bytes);
    let corpus_chars = text.chars().count();
    let ids = larql_inference::encode_prompt(&tokenizer, &*weights.arch, &text)
        .map_err(|e| format!("tokenise corpus: {e}"))?;
    if ids.len() < 2 {
        return Err("corpus must tokenize to at least one scored token".into());
    }
    let n_targets = ids.len() - 1;
    let num_layers = weights.num_layers;
    let hidden = weights.hidden_size;
    let scaling = weights.arch.logits_scaling();
    let softcap = weights.arch.final_logit_softcapping();
    let timeout = Duration::from_secs(args.ffn_timeout_secs);

    eprintln!(
        "C6 drift gate: {} arms (baseline first) × {} scored tokens × {} layers vs {}",
        plan.len(),
        n_targets,
        num_layers,
        args.ffn,
    );

    let mut arms: Vec<DriftArmResult> = Vec::with_capacity(plan.len());
    let mut baseline_bpc: Option<f64> = None;
    for (i, spec) in plan.iter().enumerate() {
        let conn = arm_connection(*spec);
        eprintln!(
            "[{}/{}] wire {} (in {} / out {}) …",
            i + 1,
            plan.len(),
            spec.label(),
            spec.in_label(),
            spec.out_label(),
        );
        let remote = LayerShardedBackend::connect_with_wire_formats(
            &args.ffn,
            timeout,
            conn.wire_in,
            conn.wire_out,
        )
        .map_err(|e| format!("failed to connect to remote FFN: {e}"))?;
        remote.reset_wire_counters();

        let t0 = Instant::now();
        let mut total_bits = 0.0_f64;
        let scored = score_forced_with_remote_ffn(
            &weights,
            &ids,
            &index,
            &*backend,
            &remote,
            conn.use_q8k,
            |step, logits| {
                let bits = bits_for_target(logits, ids[step + 1] as usize, scaling, softcap)?;
                total_bits += bits;
                if args.verbose && (step + 1) % 64 == 0 {
                    eprintln!("  … {}/{} tokens", step + 1, n_targets);
                }
                Ok(())
            },
        )
        .map_err(|e| format!("arm {} failed: {e}", spec.label()))?;
        let elapsed = t0.elapsed().as_secs_f64();

        let calls = (scored * num_layers).max(1) as f64;
        let arm = DriftArmResult::new(
            *spec,
            scored,
            total_bits,
            corpus_chars,
            baseline_bpc,
            args.gate_pct,
            remote.wire_bytes_sent() as f64 / calls,
            remote.wire_bytes_recv() as f64 / calls,
            hidden,
            elapsed,
        );
        if baseline_bpc.is_none() {
            baseline_bpc = Some(arm.bits_per_char);
        }
        eprintln!(
            "  bits/char {:.4}  drift {:+.4}%  ({:.1}s)",
            arm.bits_per_char, arm.drift_pct, elapsed
        );
        arms.push(arm);
    }

    println!();
    print!("{}", format_drift_table(&arms, args.gate_pct));

    let pass = super::drift::overall_pass(&arms);
    let record = DecDriftJsonResult {
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs().to_string())
            .unwrap_or_default(),
        model: args.model.clone(),
        ffn_url: args.ffn.clone(),
        corpus: args.corpus.display().to_string(),
        corpus_bytes: text.len(),
        corpus_chars,
        tokens_scored: n_targets,
        gate_pct: args.gate_pct,
        baseline_wire: "f32".into(),
        determinism_note: DETERMINISM_NOTE.into(),
        pass,
        arms,
    };

    if let Some(path) = &args.output_file {
        std::fs::write(path, serde_json::to_string_pretty(&record)?)?;
        eprintln!("wrote run record: {}", path.display());
    }
    if let Some(path) = &args.pulse_file {
        let lines: Vec<serde_json::Value> = record
            .arms
            .iter()
            .enumerate()
            .map(|(i, a)| super::drift::drift_pulse_line(i, a, args.gate_pct))
            .collect();
        std::fs::write(path, to_jsonl(&lines))?;
        eprintln!("wrote pulse: {}", path.display());
    }

    println!();
    println!(
        "gate: {:.3}% |bits/char drift| vs in-run f32 baseline",
        args.gate_pct
    );
    if pass {
        println!("PASS");
        Ok(())
    } else {
        println!("FAIL");
        let failed: Vec<String> = record
            .arms
            .iter()
            .filter(|a| !a.pass)
            .map(|a| format!("{} ({:+.4}%)", a.wire_format, a.drift_pct))
            .collect();
        Err(format!(
            "C6 drift gate failed for {} of {} arms: {}",
            failed.len(),
            record.arms.len(),
            failed.join(", ")
        )
        .into())
    }
}
