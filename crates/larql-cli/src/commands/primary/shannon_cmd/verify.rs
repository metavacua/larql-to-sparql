//! Cross-engine verification: run the reference scorers and compare totals.

use super::*;

pub(super) struct TempFileGuard(pub(super) PathBuf);

#[derive(Debug)]
pub(super) struct VerifyResult {
    pub(super) engine: &'static str,
    pub(super) tokens_scored: usize,
    pub(super) total_bits: f64,
    pub(super) bits_per_token: f64,
    pub(super) bits_per_char: f64,
    pub(super) elapsed_secs: f64,
}

pub(super) fn run_verify(args: VerifyArgs) -> Result<(), Box<dyn std::error::Error>> {
    validate_window(args.context, args.stride)?;

    let (normalized, normalized_chars, tmp_corpus, _cleanup) =
        normalize_corpus_to_tempfile(&args.corpus, args.bytes)?;
    let engines = parse_engine_list(&args.engines)?;

    let mut results = vec![run_rust_scorer(&args, &normalized, normalized_chars)?];
    if engines.contains(&ENGINE_MLX) {
        results.push(spawn_reference_scorer(
            ENGINE_MLX,
            &args.python,
            &args.mlx_script,
            &args.model,
            &tmp_corpus,
            args.context,
            args.stride,
            None,
        )?);
    }
    if engines.contains(&ENGINE_HF) {
        results.push(spawn_reference_scorer(
            ENGINE_HF,
            &args.python,
            &args.hf_script,
            &args.model,
            &tmp_corpus,
            args.context,
            args.stride,
            Some(&args.hf_device),
        )?);
    }

    let reference_idx = pick_reference_index(&results);
    print_delta_table(&results, reference_idx);
    let (max_delta_pct, max_pair) = max_pairwise_delta_pct(&results);
    let pass = max_delta_pct <= args.threshold;
    if args.json {
        emit_verify_json(
            &results,
            reference_idx,
            max_delta_pct,
            max_pair,
            args.threshold,
            pass,
        );
    }
    print_verdict(max_delta_pct, max_pair, args.threshold)
}

/// Emit the `RESULT {...}` line consumed by `scripts/diagnose_models.py`.
/// Schema is intentionally flat so the Python side doesn't need to know
/// about Rust's `VerifyResult` struct layout.
pub(super) fn emit_verify_json(
    results: &[VerifyResult],
    reference_idx: usize,
    max_delta_pct: f64,
    max_pair: (&str, &str),
    threshold: f64,
    pass: bool,
) {
    let reference_engine = results.get(reference_idx).map(|r| r.engine).unwrap_or("");
    let engines_json: Vec<serde_json::Value> = results
        .iter()
        .map(|r| {
            serde_json::json!({
                "engine": r.engine,
                "tokens_scored": r.tokens_scored,
                "total_bits": r.total_bits,
                "bits_per_token": r.bits_per_token,
                "bits_per_char": r.bits_per_char,
                "elapsed_secs": r.elapsed_secs,
            })
        })
        .collect();
    let payload = serde_json::json!({
        "reference": reference_engine,
        "engines": engines_json,
        "max_delta_pct": max_delta_pct,
        "max_pair": [max_pair.0, max_pair.1],
        "threshold_pct": threshold,
        "pass": pass,
    });
    println!("{}{}", RESULT_PREFIX, payload);
}

/// Read the corpus, strip CRLF for cross-engine tokenization parity, and
/// write the result to a temp file the Python subprocesses can read.
/// Returns the in-memory string, its char count, the temp file path, and
/// a guard that deletes the file when dropped.
pub(super) fn normalize_corpus_to_tempfile(
    corpus: &Path,
    bytes: Option<usize>,
) -> Result<(String, usize, PathBuf, TempFileGuard), Box<dyn std::error::Error>> {
    let raw_text = read_text(&corpus.to_path_buf(), bytes)?;
    let normalized: String = raw_text.chars().filter(|c| *c != '\r').collect();
    let normalized_bytes = normalized.len();
    let normalized_chars = normalized.chars().count();
    let stripped = raw_text.len() - normalized_bytes;
    if stripped > 0 {
        eprintln!(
            "normalized corpus: stripped {} CR bytes ({} -> {} bytes)",
            stripped,
            raw_text.len(),
            normalized_bytes
        );
    }
    let tmp_corpus =
        std::env::temp_dir().join(format!("larql_shannon_verify_{}.txt", std::process::id()));
    fs::write(&tmp_corpus, normalized.as_bytes())?;
    let guard = TempFileGuard(tmp_corpus.clone());
    Ok((normalized, normalized_chars, tmp_corpus, guard))
}

/// Load the model and score the corpus through the LARQL Rust forward
/// path in-process. The model load + tokenize are isolated here so the
/// caller doesn't double-encode the corpus.
pub(super) fn run_rust_scorer(
    args: &VerifyArgs,
    normalized: &str,
    normalized_chars: usize,
) -> Result<VerifyResult, Box<dyn std::error::Error>> {
    eprintln!("[{ENGINE_RUST}] loading {}...", args.model);
    let start = Instant::now();
    let model = load_model(&args.model)?;
    let ids = encode_prompt(model.tokenizer(), &*model.weights().arch, normalized)?;
    if ids.len() < 2 {
        return Err("corpus must tokenize to at least one scored token".into());
    }
    eprintln!("[{ENGINE_RUST}] scoring {} target tokens...", ids.len() - 1);
    let summary = score_token_range(
        model.weights(),
        &ids,
        1..ids.len(),
        args.context,
        args.stride,
        Some(ENGINE_RUST),
    )?;
    Ok(VerifyResult {
        engine: ENGINE_RUST,
        tokens_scored: summary.token_bits.len(),
        total_bits: summary.total_bits,
        bits_per_token: summary.bits_per_token(),
        bits_per_char: summary.total_bits / normalized_chars.max(1) as f64,
        elapsed_secs: start.elapsed().as_secs_f64(),
    })
}

/// Spawn a Python reference scorer and time the run end-to-end. Wraps
/// `run_python_scorer` so the timing isn't measured per-fork-per-launch
/// and the `[engine] launching` line is uniform across MLX and HF.
#[allow(clippy::too_many_arguments)]
pub(super) fn spawn_reference_scorer(
    engine: &'static str,
    python: &Path,
    script: &Path,
    model: &str,
    corpus: &Path,
    context: usize,
    stride: usize,
    device: Option<&str>,
) -> Result<VerifyResult, Box<dyn std::error::Error>> {
    eprintln!("[{engine}] launching {} ...", script.display());
    let start = Instant::now();
    let mut r = run_python_scorer(
        engine, python, script, model, corpus, context, stride, device,
    )?;
    r.elapsed_secs = start.elapsed().as_secs_f64();
    Ok(r)
}

/// Reference = HF if present (canonical PyTorch-side number), else any
/// non-rust engine, else fall back to rust itself.
pub(super) fn pick_reference_index(results: &[VerifyResult]) -> usize {
    results
        .iter()
        .position(|r| r.engine == ENGINE_HF)
        .or_else(|| results.iter().position(|r| r.engine != ENGINE_RUST))
        .unwrap_or(0)
}

pub(super) fn print_delta_table(results: &[VerifyResult], reference_idx: usize) {
    let reference_bits = results[reference_idx].total_bits;
    println!();
    println!(
        "{:<8} {:>8} {:>12} {:>12} {:>14} {:>10} {:>10}",
        "engine", "tokens", "bits/token", "bits/char", "total bits", "Δ vs ref", "elapsed"
    );
    println!("{}", "-".repeat(80));
    for (i, r) in results.iter().enumerate() {
        let delta_pct = (r.total_bits - reference_bits) / reference_bits.max(1.0) * 100.0;
        let delta_str = if i == reference_idx {
            "—".to_string()
        } else if delta_pct >= 0.0 {
            format!("+{:.3}%", delta_pct)
        } else {
            format!("{:.3}%", delta_pct)
        };
        println!(
            "{:<8} {:>8} {:>12.4} {:>12.4} {:>14.3} {:>10} {:>9.1}s",
            r.engine,
            r.tokens_scored,
            r.bits_per_token,
            r.bits_per_char,
            r.total_bits,
            delta_str,
            r.elapsed_secs
        );
    }
}

/// Max pair-wise delta in percent (largest |Δ| relative to the larger
/// value in the pair). Returns 0.0 / ("", "") when fewer than two
/// engines ran.
pub(super) fn max_pairwise_delta_pct(
    results: &[VerifyResult],
) -> (f64, (&'static str, &'static str)) {
    let mut max_delta_pct = 0.0_f64;
    let mut max_pair: (&'static str, &'static str) = ("", "");
    for i in 0..results.len() {
        for j in (i + 1)..results.len() {
            let a = results[i].total_bits;
            let b = results[j].total_bits;
            let denom = a.max(b).max(1.0);
            let d = (a - b).abs() / denom * 100.0;
            if d > max_delta_pct {
                max_delta_pct = d;
                max_pair = (results[i].engine, results[j].engine);
            }
        }
    }
    (max_delta_pct, max_pair)
}

pub(super) fn print_verdict(
    max_delta_pct: f64,
    max_pair: (&'static str, &'static str),
    threshold: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    println!();
    println!(
        "max pair-wise delta: {:.3}% ({} vs {})",
        max_delta_pct, max_pair.0, max_pair.1
    );
    println!("threshold:           {:.3}%", threshold);
    if max_delta_pct > threshold {
        println!("FAIL");
        Err(format!(
            "engines disagree by {:.3}% (> {:.3}% threshold) between {} and {}",
            max_delta_pct, threshold, max_pair.0, max_pair.1
        )
        .into())
    } else {
        println!("PASS");
        Ok(())
    }
}

pub(super) fn parse_engine_list(
    spec: &str,
) -> Result<Vec<&'static str>, Box<dyn std::error::Error>> {
    let mut out = Vec::new();
    for token in spec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        match token {
            "mlx" => out.push(ENGINE_MLX),
            "hf" => out.push(ENGINE_HF),
            "rust" => {} // rust always runs; tolerate it in the list
            other => return Err(format!("unknown engine: {other} (expected mlx, hf, rust)").into()),
        }
    }
    Ok(out)
}

pub(super) fn run_python_scorer(
    engine: &'static str,
    python: &Path,
    script: &Path,
    model: &str,
    corpus: &Path,
    context: usize,
    stride: usize,
    device: Option<&str>,
) -> Result<VerifyResult, Box<dyn std::error::Error>> {
    use std::process::{Command, Stdio};

    let mut cmd = Command::new(python);
    cmd.arg(script)
        .arg(model)
        .arg("--corpus")
        .arg(corpus)
        .arg("--context")
        .arg(context.to_string())
        .arg("--stride")
        .arg(stride.to_string())
        .arg("--json");
    if let Some(dev) = device {
        cmd.arg("--device").arg(dev);
    }
    let output = cmd.stderr(Stdio::inherit()).output().map_err(|e| {
        format!(
            "failed to spawn {} scorer ({} {}): {e}",
            engine,
            python.display(),
            script.display()
        )
    })?;
    if !output.status.success() {
        return Err(format!("{} scorer exited with status {}", engine, output.status).into());
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let result_line = stdout
        .lines()
        .rev()
        .find(|l| l.starts_with(RESULT_PREFIX))
        .ok_or_else(|| {
            format!(
                "{engine} scorer did not emit a `{}` line; rerun without --json to inspect",
                RESULT_PREFIX.trim_end()
            )
        })?;
    let json_str = result_line.trim_start_matches(RESULT_PREFIX).trim();
    let parsed: serde_json::Value = serde_json::from_str(json_str).map_err(|e| {
        format!(
            "{engine} scorer emitted unparseable `{}` line: {e}",
            RESULT_PREFIX.trim_end()
        )
    })?;

    Ok(VerifyResult {
        engine,
        tokens_scored: parsed["tokens_scored"].as_u64().unwrap_or(0) as usize,
        total_bits: parsed["total_bits"].as_f64().unwrap_or(0.0),
        bits_per_token: parsed["bits_per_token"].as_f64().unwrap_or(0.0),
        bits_per_char: parsed["bits_per_char"].as_f64().unwrap_or(0.0),
        elapsed_secs: 0.0, // filled in by caller
    })
}
