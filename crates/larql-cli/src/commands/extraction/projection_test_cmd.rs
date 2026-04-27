use std::path::PathBuf;
use std::time::Instant;

use clap::Args;
use larql_inference::{
    forward_to_layer, predict, predict_from_hidden, trace_forward, InferenceModel,
};

#[derive(Args)]
pub struct ProjectionTestArgs {
    /// Model path or HuggingFace model ID.
    model: String,

    /// Vectors binary file from trajectory-trace --dump-vectors.
    #[arg(long)]
    vectors: PathBuf,

    /// Vectors metadata JSON file.
    #[arg(long)]
    meta: PathBuf,

    /// Comma-separated prompts to test.
    #[arg(long)]
    prompts: Option<String>,

    /// File with one prompt per line.
    #[arg(long)]
    prompts_file: Option<PathBuf>,

    /// Projection rank.
    #[arg(long, default_value = "5")]
    rank: usize,

    /// Layer to inject projected residual (resume forward pass from here).
    #[arg(long, default_value = "15")]
    inject_layer: usize,

    /// Number of top predictions to show.
    #[arg(short = 'k', long, default_value = "5")]
    top_k: usize,
}

#[derive(serde::Deserialize)]
struct VectorMeta {
    shape: Vec<usize>,
    layers: Vec<usize>,
}

pub fn run(args: ProjectionTestArgs) -> Result<(), Box<dyn std::error::Error>> {
    // ── Load model ──
    eprintln!("Loading model: {}", args.model);
    let start = Instant::now();
    let model = InferenceModel::load(&args.model)?;
    let weights = model.weights();
    let hidden = weights.hidden_size;
    let num_layers = weights.num_layers;
    eprintln!(
        "  {} layers, hidden_size={} ({:.1}s)",
        num_layers, hidden,
        start.elapsed().as_secs_f64()
    );

    // ── Load trajectory vectors ──
    eprintln!("Loading trajectory vectors...");
    let meta: VectorMeta = serde_json::from_str(&std::fs::read_to_string(&args.meta)?)?;
    let n_train = meta.shape[0];
    let n_layers_file = meta.shape[1];
    let raw = std::fs::read(&args.vectors)?;
    let floats: Vec<f32> = raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // Source = L0, target = inject_layer
    let src_idx = meta.layers.iter().position(|&l| l == 0)
        .ok_or("L0 not in trajectory data")?;
    let tgt_idx = meta.layers.iter().position(|&l| l == args.inject_layer)
        .ok_or_else(|| format!("L{} not in trajectory data", args.inject_layer))?;

    eprintln!(
        "  {} training prompts, fitting L0→L{} rank-{} projection",
        n_train, args.inject_layer, args.rank
    );

    // ── Extract X (L0) and Y (inject_layer) from training data ──
    let stride = n_layers_file * hidden;
    let mut x_train: Vec<Vec<f32>> = Vec::with_capacity(n_train);
    let mut y_train: Vec<Vec<f32>> = Vec::with_capacity(n_train);
    for i in 0..n_train {
        let base = i * stride;
        x_train.push(floats[base + src_idx * hidden..base + src_idx * hidden + hidden].to_vec());
        y_train.push(floats[base + tgt_idx * hidden..base + tgt_idx * hidden + hidden].to_vec());
    }
    drop(floats); // free the 66MB

    // ── Compute means ──
    let mut x_mean = vec![0.0f32; hidden];
    let mut y_mean = vec![0.0f32; hidden];
    for i in 0..n_train {
        for j in 0..hidden {
            x_mean[j] += x_train[i][j];
            y_mean[j] += y_train[i][j];
        }
    }
    for j in 0..hidden {
        x_mean[j] /= n_train as f32;
        y_mean[j] /= n_train as f32;
    }

    // ── Center ──
    let mut xc: Vec<Vec<f32>> = Vec::with_capacity(n_train);
    let mut yc: Vec<Vec<f32>> = Vec::with_capacity(n_train);
    for i in 0..n_train {
        let mut xr = vec![0.0f32; hidden];
        let mut yr = vec![0.0f32; hidden];
        for j in 0..hidden {
            xr[j] = x_train[i][j] - x_mean[j];
            yr[j] = y_train[i][j] - y_mean[j];
        }
        xc.push(xr);
        yc.push(yr);
    }

    // ── Gram matrix + power iteration ──
    let fit_start = Instant::now();
    let mut gram = vec![0.0f32; n_train * n_train];
    for i in 0..n_train {
        for j in i..n_train {
            let d: f32 = xc[i].iter().zip(xc[j].iter()).map(|(a, b)| a * b).sum();
            gram[i * n_train + j] = d;
            gram[j * n_train + i] = d;
        }
    }

    let rank = args.rank.min(n_train);
    let mut eigenvalues: Vec<f32> = Vec::new();
    let mut eigenvectors: Vec<Vec<f32>> = Vec::new();

    for _ in 0..rank {
        let mut v = vec![1.0f32; n_train];
        let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in v.iter_mut() { *x /= n; }

        let mut ev = 0.0f32;
        for _ in 0..100 {
            let mut mv = vec![0.0f32; n_train];
            for i in 0..n_train {
                let mut s = 0.0f32;
                for j in 0..n_train {
                    s += gram[i * n_train + j] * v[j];
                }
                mv[i] = s;
            }
            ev = mv.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
            let n: f32 = mv.iter().map(|x| x * x).sum::<f32>().sqrt();
            if n < 1e-12 { break; }
            for (x, m) in v.iter_mut().zip(mv.iter()) { *x = m / n; }
        }
        if ev < 1e-8 { break; }

        eigenvalues.push(ev.sqrt());
        eigenvectors.push(v.clone());

        for i in 0..n_train {
            for j in 0..n_train {
                gram[i * n_train + j] -= ev * v[i] * v[j];
            }
        }
    }

    // ── Build projection: Vt rows (directions) and betas (coefficients) ──
    let mut vt_rows: Vec<Vec<f32>> = Vec::new();
    let mut betas: Vec<Vec<f32>> = Vec::new();

    for k in 0..eigenvalues.len() {
        // Vt[k] = normalized direction in hidden space
        let mut dir = vec![0.0f32; hidden];
        for i in 0..n_train {
            let c = eigenvectors[k][i] / eigenvalues[k];
            for j in 0..hidden { dir[j] += c * xc[i][j]; }
        }
        let n: f32 = dir.iter().map(|x| x * x).sum::<f32>().sqrt();
        if n > 1e-12 { for x in dir.iter_mut() { *x /= n; } }
        vt_rows.push(dir);

        // beta[k] = Y projected by same weights
        let mut beta = vec![0.0f32; hidden];
        for i in 0..n_train {
            let c = eigenvectors[k][i] / eigenvalues[k];
            for j in 0..hidden { beta[j] += c * yc[i][j]; }
        }
        betas.push(beta);
    }

    eprintln!("  Fitted in {:.0}ms", fit_start.elapsed().as_secs_f64() * 1000.0);

    // ── Project function: L0 last-token residual → predicted inject_layer residual ──
    let project = |x: &[f32]| -> Vec<f32> {
        let mut result = y_mean.clone();
        for k in 0..eigenvalues.len() {
            let score: f32 = (0..hidden).map(|j| (x[j] - x_mean[j]) * vt_rows[k][j]).sum();
            for j in 0..hidden { result[j] += score * betas[k][j]; }
        }
        result
    };

    // ── Load test prompts ──
    let test_prompts: Vec<String> = if let Some(ref file) = args.prompts_file {
        std::fs::read_to_string(file)?
            .lines().map(|l| l.trim().to_string()).filter(|l| !l.is_empty()).collect()
    } else if let Some(ref p) = args.prompts {
        p.split(',').map(|s| s.trim().to_string()).collect()
    } else {
        return Err("provide --prompts or --prompts-file".into());
    };

    // ── Run end-to-end tests ──
    eprintln!("\n── End-to-end: project L0→L{}, run L{}→L{} dense ──\n",
        args.inject_layer, args.inject_layer, num_layers - 1);

    println!(
        "{:<45} {:>12} {:>12} {:>8} {:>8} {:>8}",
        "Prompt", "Baseline", "Projected", "B_conf", "P_conf", "Cos@L"
    );
    println!("{}", "-".repeat(100));

    let inject_from = args.inject_layer; // resume from this layer
    let mut match_count = 0;
    let mut total = 0;
    let mut cosines = Vec::new();

    for prompt in &test_prompts {
        let encoding = model.tokenizer()
            .encode(prompt.as_str(), true)
            .map_err(|e| format!("tokenize error: {e}"))?;
        let token_ids: Vec<u32> = encoding.get_ids().to_vec();
        let seq_len = token_ids.len();
        if seq_len < 3 { continue; }

        // Baseline
        let baseline = predict(weights, model.tokenizer(), &token_ids, args.top_k);
        let (base_tok, base_conf) = baseline.predictions.first()
            .map(|(t, p)| (t.clone(), *p)).unwrap_or_default();

        // Get real hidden state at inject_layer (full sequence)
        // Run forward pass through layers 0..inject_layer-1
        let h_real = forward_to_layer(weights, &token_ids, inject_from - 1);

        // Get L0 last-token residual for projection
        let trace_l0 = trace_forward(weights, &token_ids, &[0], false, 0);
        let l0_last = &trace_l0.residuals[0].1;

        // Project L0 → inject_layer
        let projected = project(l0_last);

        // Cosine between projected and real at inject_layer
        let real_last_row = h_real.row(seq_len - 1);
        let cos: f32 = {
            let dot: f32 = projected.iter().zip(real_last_row.iter()).map(|(a, b)| a * b).sum();
            let na: f32 = projected.iter().map(|x| x * x).sum::<f32>().sqrt();
            let nb: f32 = real_last_row.iter().map(|x| x * x).sum::<f32>().sqrt();
            if na > 1e-12 && nb > 1e-12 { dot / (na * nb) } else { 0.0 }
        };
        cosines.push(cos);

        // Build hybrid hidden state: real for positions 0..seq_len-1, projected for last position
        let mut h_hybrid = h_real.clone();
        for j in 0..hidden {
            h_hybrid[[seq_len - 1, j]] = projected[j];
        }

        // Run from inject_layer to end
        let proj_result = predict_from_hidden(
            weights, model.tokenizer(), &h_hybrid, inject_from, args.top_k,
        );
        let (proj_tok, proj_conf) = proj_result.predictions.first()
            .map(|(t, p)| (t.clone(), *p)).unwrap_or_default();

        let matched = proj_tok == base_tok;
        if matched { match_count += 1; }
        total += 1;

        let match_str = if matched { "=" } else { "X" };
        println!(
            "{:<45} {:>12} {:>12} {:>7.2}% {:>7.2}% {:>7.4} {:>3}",
            &prompt[..prompt.len().min(44)],
            base_tok, proj_tok,
            base_conf * 100.0, proj_conf * 100.0,
            cos, match_str,
        );
    }

    // ── Summary ──
    eprintln!("\n── Summary ──");
    eprintln!("  Prompts: {}", total);
    eprintln!("  Token match: {}/{} ({:.1}%)", match_count, total, match_count as f64 / total as f64 * 100.0);
    let mean_cos: f32 = cosines.iter().sum::<f32>() / cosines.len() as f32;
    let min_cos: f32 = cosines.iter().copied().fold(f32::INFINITY, f32::min);
    eprintln!("  Cosine at L{}: mean={:.6}, min={:.6}", inject_from, mean_cos, min_cos);
    eprintln!("  Layers replaced: 0-{} ({} layers → rank-{} projection)", inject_from - 1, inject_from, args.rank);
    eprintln!("  Layers computed: {}-{} ({} layers dense)", inject_from, num_layers - 1, num_layers - inject_from);

    Ok(())
}
