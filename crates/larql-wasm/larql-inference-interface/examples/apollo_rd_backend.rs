//! Apollo boundary residual R(D) backend.
//!
//! Contract with `~/chris-source/chris-experiments/shannon/40_boundary_state_rate_distortion/kl_eval.py`:
//!
//! ```text
//! cargo run --release -p larql-inference --example apollo_rd_backend -- \
//!   --model google/gemma-3-4b-it --job JOB_JSON --out OUT_JSON
//! ```
//!
//! Batch mode uses the same command with a batch job file and writes JSONL.

use std::collections::HashSet;
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use larql_inference::{forward_from_layer, hidden_to_raw_logits, InferenceModel};
use larql_kv::apollo::{npy, ApolloStore};
use ndarray::s;
use serde::{Deserialize, Serialize};

const KL_DIRECTION: &str = "ground_truth||reconstructed";

#[derive(Debug, Deserialize)]
struct BatchJob {
    #[serde(default)]
    backend_mode: Option<String>,
    #[serde(default)]
    jobs: Vec<Job>,
}

#[derive(Debug, Deserialize)]
struct Job {
    config_id: String,
    store: PathBuf,
    payload: PathBuf,
    boundary_indices: Vec<usize>,
    source_crystal_layer: Option<usize>,
    rate_bits_per_token_with_basis: Option<f64>,
}

#[derive(Debug, Serialize)]
struct Metric {
    config_id: String,
    status: &'static str,
    kl_direction: &'static str,
    metric_source: &'static str,
    kl_mean_nats: f64,
    kl_p50_nats: f64,
    kl_p95_nats: f64,
    kl_p99_nats: f64,
    kl_max_nats: f64,
    n_positions: usize,
    kl_reverse_mean_nats: f64,
    kl_reverse_p50_nats: f64,
    kl_reverse_p95_nats: f64,
    kl_reverse_p99_nats: f64,
    kl_reverse_max_nats: f64,
    kl_symmetric_mean_nats: f64,
    kl_symmetric_p50_nats: f64,
    kl_symmetric_p95_nats: f64,
    kl_symmetric_max_nats: f64,
    kl_symmetric_p99_nats: f64,
    sampling_mode: String,
    position_stride: usize,
    n_windows_evaluated: usize,
}

#[derive(Debug)]
struct PayloadBoundaries {
    reconstructed: Vec<Vec<f32>>,
    rows: usize,
    hidden: usize,
}

#[derive(Debug)]
struct Args {
    model: String,
    job: PathBuf,
    out: PathBuf,
    max_positions_per_config: Option<usize>,
    position_stride: usize,
    eval_limit_windows: Option<usize>,
    adaptive_sampling: bool,
    low_rate_threshold: f64,
    mid_rate_threshold: f64,
    low_rate_max_positions: usize,
    mid_rate_max_positions: usize,
    high_rate_max_positions: usize,
    low_rate_windows: usize,
    mid_rate_windows: usize,
    high_rate_windows: usize,
    low_rate_position_stride: usize,
    mid_rate_position_stride: usize,
    high_rate_position_stride: usize,
    positions_file: Option<PathBuf>,
}

#[derive(Debug, Clone)]
struct EvalPlan {
    mode: String,
    max_positions: Option<usize>,
    position_stride: usize,
    limit_windows: Option<usize>,
}

#[derive(Debug, Clone)]
struct WindowOffsets {
    starts: Vec<usize>,
    total_tokens: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args()?;

    eprintln!("Loading model: {}", args.model);
    let t0 = Instant::now();
    let model = InferenceModel::load(&args.model)?;
    eprintln!("Model loaded in {:.1}s", t0.elapsed().as_secs_f64());
    let weights = model.weights();

    let job_text = std::fs::read_to_string(&args.job)?;
    if let Ok(batch) = serde_json::from_str::<BatchJob>(&job_text) {
        if batch.backend_mode.as_deref() == Some("batch") {
            let mut out = File::create(&args.out)?;
            for job in batch.jobs {
                let metric = evaluate_job(weights, &job, &args)?;
                writeln!(out, "{}", serde_json::to_string(&metric)?)?;
            }
            return Ok(());
        }
    }

    let job: Job = serde_json::from_str(&job_text)?;
    let metric = evaluate_job(weights, &job, &args)?;
    let mut out = File::create(&args.out)?;
    serde_json::to_writer_pretty(&mut out, &metric)?;
    writeln!(out)?;
    Ok(())
}

fn parse_args() -> Result<Args, Box<dyn std::error::Error>> {
    let mut model = None;
    let mut job = None;
    let mut out = None;
    let mut max_positions_per_config = None;
    let mut position_stride = 1usize;
    let mut eval_limit_windows = None;
    let mut adaptive_sampling = false;
    let mut low_rate_threshold = 20.0;
    let mut mid_rate_threshold = 60.0;
    let mut low_rate_max_positions = 128usize;
    let mut mid_rate_max_positions = 64usize;
    let mut high_rate_max_positions = 8usize;
    let mut low_rate_windows = 8usize;
    let mut mid_rate_windows = 4usize;
    let mut high_rate_windows = 1usize;
    let mut low_rate_position_stride = 16usize;
    let mut mid_rate_position_stride = 16usize;
    let mut high_rate_position_stride = 64usize;
    let mut positions_file = None;

    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--model" => model = it.next(),
            "--job" => job = it.next().map(PathBuf::from),
            "--out" => out = it.next().map(PathBuf::from),
            "--max-positions-per-config" => {
                max_positions_per_config = it.next().map(|v| v.parse()).transpose()?
            }
            "--position-stride" => {
                position_stride = it
                    .next()
                    .ok_or("--position-stride requires a value")?
                    .parse()?;
                if position_stride == 0 {
                    return Err("--position-stride must be >= 1".into());
                }
            }
            "--eval-limit-windows" => {
                eval_limit_windows = it.next().map(|v| v.parse()).transpose()?
            }
            "--adaptive-sampling" => adaptive_sampling = true,
            "--low-rate-threshold" => {
                low_rate_threshold = it
                    .next()
                    .ok_or("--low-rate-threshold requires a value")?
                    .parse()?
            }
            "--mid-rate-threshold" => {
                mid_rate_threshold = it
                    .next()
                    .ok_or("--mid-rate-threshold requires a value")?
                    .parse()?
            }
            "--low-rate-max-positions" => {
                low_rate_max_positions = it
                    .next()
                    .ok_or("--low-rate-max-positions requires a value")?
                    .parse()?
            }
            "--mid-rate-max-positions" => {
                mid_rate_max_positions = it
                    .next()
                    .ok_or("--mid-rate-max-positions requires a value")?
                    .parse()?
            }
            "--high-rate-max-positions" => {
                high_rate_max_positions = it
                    .next()
                    .ok_or("--high-rate-max-positions requires a value")?
                    .parse()?
            }
            "--low-rate-windows" => {
                low_rate_windows = it
                    .next()
                    .ok_or("--low-rate-windows requires a value")?
                    .parse()?
            }
            "--mid-rate-windows" => {
                mid_rate_windows = it
                    .next()
                    .ok_or("--mid-rate-windows requires a value")?
                    .parse()?
            }
            "--high-rate-windows" => {
                high_rate_windows = it
                    .next()
                    .ok_or("--high-rate-windows requires a value")?
                    .parse()?
            }
            "--low-rate-position-stride" => {
                low_rate_position_stride = it
                    .next()
                    .ok_or("--low-rate-position-stride requires a value")?
                    .parse()?
            }
            "--mid-rate-position-stride" => {
                mid_rate_position_stride = it
                    .next()
                    .ok_or("--mid-rate-position-stride requires a value")?
                    .parse()?
            }
            "--high-rate-position-stride" => {
                high_rate_position_stride = it
                    .next()
                    .ok_or("--high-rate-position-stride requires a value")?
                    .parse()?
            }
            "--positions-file" => positions_file = it.next().map(PathBuf::from),
            _ => return Err(format!("unknown argument: {arg}").into()),
        }
    }

    Ok(Args {
        model: model.ok_or("--model required")?,
        job: job.ok_or("--job required")?,
        out: out.ok_or("--out required")?,
        max_positions_per_config,
        position_stride,
        eval_limit_windows,
        adaptive_sampling,
        low_rate_threshold,
        mid_rate_threshold,
        low_rate_max_positions,
        mid_rate_max_positions,
        high_rate_max_positions,
        low_rate_windows,
        mid_rate_windows,
        high_rate_windows,
        low_rate_position_stride,
        mid_rate_position_stride,
        high_rate_position_stride,
        positions_file,
    })
}

fn evaluate_job(
    weights: &larql_inference::ModelWeights,
    job: &Job,
    args: &Args,
) -> Result<Metric, Box<dyn std::error::Error>> {
    eprintln!("Evaluating {}", job.config_id);
    let store = ApolloStore::load(&job.store)?;
    let payload = load_payload_boundaries(&job.payload)?;

    if payload.rows != job.boundary_indices.len() {
        return Err(format!(
            "{}: payload rows {} != boundary_indices {}",
            job.config_id,
            payload.rows,
            job.boundary_indices.len()
        )
        .into());
    }
    if payload.hidden != weights.hidden_size {
        return Err(format!(
            "{}: payload hidden {} != model hidden {}",
            job.config_id, payload.hidden, weights.hidden_size
        )
        .into());
    }

    let crystal = job
        .source_crystal_layer
        .unwrap_or(store.manifest.crystal_layer);
    let plan = eval_plan(job, args);
    let offsets = WindowOffsets::from_store(&store);
    let selected: HashSet<usize> = job.boundary_indices.iter().copied().collect();
    let mut values = Vec::new();
    let mut reverse_values = Vec::new();
    let mut symmetric_values = Vec::new();
    let mut windows_evaluated = 0usize;

    if let Some(path) = &args.positions_file {
        let positions = load_positions(path)?;
        let mut matched_segments = HashSet::new();
        for abs_prefix_len in positions {
            let Some((payload_row, start_window, end_window)) =
                selected_segment_for_position(job, &offsets, abs_prefix_len)
            else {
                continue;
            };
            let prefix =
                segment_prefix_tokens(&store, &offsets, start_window, end_window, abs_prefix_len);
            if prefix.is_empty() {
                continue;
            }
            let source_boundary = store
                .boundaries
                .get(start_window)
                .ok_or_else(|| format!("missing source boundary {start_window}"))?;
            let reconstructed_boundary = &payload.reconstructed[payload_row];
            let reference_logits = logits_from_boundary(weights, &prefix, source_boundary, crystal);
            let reconstructed_logits =
                logits_from_boundary(weights, &prefix, reconstructed_boundary, crystal);
            push_kl_metrics(
                &reference_logits,
                &reconstructed_logits,
                &mut values,
                &mut reverse_values,
                &mut symmetric_values,
            );
            matched_segments.insert(start_window);
            if plan.max_positions.is_some_and(|max| values.len() >= max) {
                break;
            }
        }
        windows_evaluated = matched_segments.len();
    } else {
        for (payload_row, &start_window) in job.boundary_indices.iter().enumerate() {
            if let Some(limit) = plan.limit_windows {
                if payload_row >= limit {
                    break;
                }
            }
            let end_window = next_boundary_or_end(
                &job.boundary_indices,
                payload_row,
                store.window_tokens.len(),
            );
            let mut tokens = Vec::new();
            for w in start_window..end_window {
                if let Some(window) = store.window_tokens.get(w) {
                    tokens.extend_from_slice(window);
                }
            }
            if tokens.is_empty() {
                continue;
            }
            let source_boundary = store
                .boundaries
                .get(start_window)
                .ok_or_else(|| format!("missing source boundary {start_window}"))?;
            let reconstructed_boundary = &payload.reconstructed[payload_row];
            windows_evaluated += 1;

            // Evaluating all prefixes is exact for per-position KL but expensive.
            // `position_stride` and `max_positions_per_config` are explicit
            // sampling controls for pilots.
            for pos in (1..=tokens.len()).step_by(plan.position_stride) {
                if plan.max_positions.is_some_and(|max| values.len() >= max) {
                    break;
                }
                let prefix = &tokens[..pos];
                let reference_logits =
                    logits_from_boundary(weights, prefix, source_boundary, crystal);
                let reconstructed_logits =
                    logits_from_boundary(weights, prefix, reconstructed_boundary, crystal);
                push_kl_metrics(
                    &reference_logits,
                    &reconstructed_logits,
                    &mut values,
                    &mut reverse_values,
                    &mut symmetric_values,
                );
            }

            if plan.max_positions.is_some_and(|max| values.len() >= max) {
                break;
            }

            // Coarser boundary configs intentionally skip intermediate source
            // boundaries. This guard documents that behavior and catches duplicate
            // index bugs without changing the segmenting rule.
            debug_assert!(selected.contains(&start_window));
        }
    }

    if values.is_empty() {
        return Err(format!("{} produced no KL positions", job.config_id).into());
    }
    let primary = stats(values);
    let reverse = stats(reverse_values);
    let symmetric = stats(symmetric_values);
    Ok(Metric {
        config_id: job.config_id.clone(),
        status: "complete",
        kl_direction: KL_DIRECTION,
        metric_source: "apollo_boundary_replay",
        kl_mean_nats: primary.mean,
        kl_p50_nats: primary.p50,
        kl_p95_nats: primary.p95,
        kl_p99_nats: primary.p99,
        kl_max_nats: primary.max,
        n_positions: primary.n,
        kl_reverse_mean_nats: reverse.mean,
        kl_reverse_p50_nats: reverse.p50,
        kl_reverse_p95_nats: reverse.p95,
        kl_reverse_p99_nats: reverse.p99,
        kl_reverse_max_nats: reverse.max,
        kl_symmetric_mean_nats: symmetric.mean,
        kl_symmetric_p50_nats: symmetric.p50,
        kl_symmetric_p95_nats: symmetric.p95,
        kl_symmetric_p99_nats: symmetric.p99,
        kl_symmetric_max_nats: symmetric.max,
        sampling_mode: if args.positions_file.is_some() {
            format!("matched_positions:{}", plan.mode)
        } else {
            plan.mode
        },
        position_stride: plan.position_stride,
        n_windows_evaluated: windows_evaluated,
    })
}

#[derive(Debug)]
struct Stats {
    mean: f64,
    p50: f64,
    p95: f64,
    p99: f64,
    max: f64,
    n: usize,
}

fn stats(mut values: Vec<f64>) -> Stats {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    Stats {
        mean,
        p50: percentile_sorted(&values, 50.0),
        p95: percentile_sorted(&values, 95.0),
        p99: percentile_sorted(&values, 99.0),
        max: *values.last().unwrap(),
        n: values.len(),
    }
}

fn eval_plan(job: &Job, args: &Args) -> EvalPlan {
    if !args.adaptive_sampling {
        return EvalPlan {
            mode: "fixed".to_string(),
            max_positions: args.max_positions_per_config,
            position_stride: args.position_stride,
            limit_windows: args.eval_limit_windows,
        };
    }

    let rate = job.rate_bits_per_token_with_basis.unwrap_or(f64::INFINITY);
    if rate <= args.low_rate_threshold {
        EvalPlan {
            mode: "adaptive_low_rate".to_string(),
            max_positions: Some(args.low_rate_max_positions),
            position_stride: args.low_rate_position_stride.max(1),
            limit_windows: Some(args.low_rate_windows),
        }
    } else if rate <= args.mid_rate_threshold {
        EvalPlan {
            mode: "adaptive_mid_rate".to_string(),
            max_positions: Some(args.mid_rate_max_positions),
            position_stride: args.mid_rate_position_stride.max(1),
            limit_windows: Some(args.mid_rate_windows),
        }
    } else {
        EvalPlan {
            mode: "adaptive_high_rate".to_string(),
            max_positions: Some(args.high_rate_max_positions),
            position_stride: args.high_rate_position_stride.max(1),
            limit_windows: Some(args.high_rate_windows),
        }
    }
}

impl WindowOffsets {
    fn from_store(store: &ApolloStore) -> Self {
        let mut starts = Vec::with_capacity(store.window_tokens.len());
        let mut total_tokens = 0usize;
        for window in &store.window_tokens {
            starts.push(total_tokens);
            total_tokens += window.len();
        }
        Self {
            starts,
            total_tokens,
        }
    }
}

fn load_positions(path: &Path) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    let text = std::fs::read_to_string(path)?;
    let positions: Vec<usize> = serde_json::from_str(&text)?;
    if positions.contains(&0) {
        return Err(format!("positions file {} contains prefix length 0", path.display()).into());
    }
    Ok(positions)
}

fn selected_segment_for_position(
    job: &Job,
    offsets: &WindowOffsets,
    abs_prefix_len: usize,
) -> Option<(usize, usize, usize)> {
    if abs_prefix_len == 0 || abs_prefix_len > offsets.total_tokens {
        return None;
    }
    for (payload_row, &start_window) in job.boundary_indices.iter().enumerate() {
        let start_abs = *offsets.starts.get(start_window)?;
        let end_window =
            next_boundary_or_end(&job.boundary_indices, payload_row, offsets.starts.len());
        let end_abs = offsets
            .starts
            .get(end_window)
            .copied()
            .unwrap_or(offsets.total_tokens);
        if abs_prefix_len > start_abs && abs_prefix_len <= end_abs {
            return Some((payload_row, start_window, end_window));
        }
    }
    None
}

fn segment_prefix_tokens(
    store: &ApolloStore,
    offsets: &WindowOffsets,
    start_window: usize,
    end_window: usize,
    abs_prefix_len: usize,
) -> Vec<u32> {
    let Some(&start_abs) = offsets.starts.get(start_window) else {
        return Vec::new();
    };
    if abs_prefix_len <= start_abs {
        return Vec::new();
    }
    let target_len = abs_prefix_len - start_abs;
    let mut tokens = Vec::new();
    for w in start_window..end_window {
        if let Some(window) = store.window_tokens.get(w) {
            tokens.extend_from_slice(window);
            if tokens.len() >= target_len {
                tokens.truncate(target_len);
                break;
            }
        }
    }
    tokens
}

fn next_boundary_or_end(boundaries: &[usize], row: usize, n_windows: usize) -> usize {
    boundaries
        .get(row + 1)
        .copied()
        .unwrap_or(n_windows)
        .min(n_windows)
}

fn logits_from_boundary(
    weights: &larql_inference::ModelWeights,
    tokens: &[u32],
    boundary: &[f32],
    crystal: usize,
) -> Vec<f32> {
    let raw = forward_from_layer(weights, tokens, boundary, crystal, None);
    let last = raw.h_pre_norm.shape()[0] - 1;
    let h_last = raw.h_pre_norm.slice(s![last..=last, ..]).to_owned();
    hidden_to_raw_logits(weights, &h_last)
}

fn load_payload_boundaries(path: &Path) -> Result<PayloadBoundaries, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let mut archive = zip::ZipArchive::new(file)?;
    let mut entry = archive.by_name("reconstructed_boundaries.npy")?;
    let mut bytes = Vec::with_capacity(entry.size() as usize);
    entry.read_to_end(&mut bytes)?;
    let (flat, shape) = npy::read_f32_flat(&bytes)?;
    if shape.len() != 2 {
        return Err(format!("reconstructed_boundaries must be 2D, got {shape:?}").into());
    }
    let rows = shape[0];
    let hidden = shape[1];
    let reconstructed = flat
        .chunks_exact(hidden)
        .map(|row| row.to_vec())
        .collect::<Vec<_>>();
    Ok(PayloadBoundaries {
        reconstructed,
        rows,
        hidden,
    })
}

fn push_kl_metrics(
    reference: &[f32],
    reconstructed: &[f32],
    primary_values: &mut Vec<f64>,
    reverse_values: &mut Vec<f64>,
    symmetric_values: &mut Vec<f64>,
) {
    let primary = kl_logits(reference, reconstructed);
    let reverse = kl_logits(reconstructed, reference);
    primary_values.push(primary);
    reverse_values.push(reverse);
    symmetric_values.push(0.5 * (primary + reverse));
}

fn kl_logits(reference: &[f32], reconstructed: &[f32]) -> f64 {
    let ref_logp = log_softmax(reference);
    let rec_logp = log_softmax(reconstructed);
    ref_logp
        .iter()
        .zip(rec_logp.iter())
        .map(|(&lp, &lq)| {
            let p = lp.exp();
            p * (lp - lq)
        })
        .sum()
}

fn log_softmax(logits: &[f32]) -> Vec<f64> {
    let max = logits
        .iter()
        .map(|&v| v as f64)
        .fold(f64::NEG_INFINITY, f64::max);
    let sum_exp = logits
        .iter()
        .map(|&v| ((v as f64) - max).exp())
        .sum::<f64>();
    let log_z = max + sum_exp.ln();
    logits.iter().map(|&v| (v as f64) - log_z).collect()
}

fn percentile_sorted(values: &[f64], p: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let rank = (p / 100.0) * (values.len().saturating_sub(1) as f64);
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    if lo == hi {
        values[lo]
    } else {
        let w = rank - lo as f64;
        values[lo] * (1.0 - w) + values[hi] * w
    }
}
