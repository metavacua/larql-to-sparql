use std::path::PathBuf;
use std::time::Instant;

use clap::Args;
use larql_inference::{forward_to_layer, predict, predict_from_hidden, InferenceModel};

/// End-to-end proof: raw token embeddings → L13 → L14-33 dense → prediction.
/// Zero layers for L0-13. Just an embedding lookup + a learned projection.
#[derive(Args)]
pub struct EmbeddingJumpArgs {
    /// Model path or HuggingFace model ID.
    model: String,

    /// Training prompts file (one per line) — used to fit the projection.
    #[arg(long)]
    train_prompts: PathBuf,

    /// Test prompts (comma-separated).
    #[arg(long)]
    prompts: Option<String>,

    /// Test prompts file.
    #[arg(long)]
    prompts_file: Option<PathBuf>,

    /// Projection rank.
    #[arg(long, default_value = "5")]
    rank: usize,

    /// Target layer to jump to.
    #[arg(long, default_value = "13")]
    target_layer: usize,

    /// Top-k predictions to compare.
    #[arg(short = 'k', long, default_value = "5")]
    top_k: usize,

    /// Add layer-0 attention analytically before projecting.
    #[arg(long)]
    with_layer0_attention: bool,

    /// Use sum of ALL token embeddings (not just last token).
    #[arg(long)]
    sum_embeddings: bool,

    /// Source layer: run this many real layers, then project from there.
    /// 0 = raw embedding (default). 1 = after L0 attention+FFN. etc.
    #[arg(long, default_value = "0")]
    source_layers: usize,
}

pub fn run(args: EmbeddingJumpArgs) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("Loading model: {}", args.model);
    let start = Instant::now();
    let model = InferenceModel::load(&args.model)?;
    let weights = model.weights();
    let hidden = weights.hidden_size;
    let num_layers = weights.num_layers;
    let embed_scale = weights.arch.embed_scale();

    eprintln!(
        "  {} layers, hidden={}, embed_scale={:.1} ({:.1}s)",
        num_layers, hidden, embed_scale,
        start.elapsed().as_secs_f64()
    );

    // ── Load training prompts ──
    let train_prompts: Vec<String> = std::fs::read_to_string(&args.train_prompts)?
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect();

    eprintln!("Fitting projection from {} training prompts...", train_prompts.len());
    let fit_start = Instant::now();

    // ── For each training prompt: compute raw embedding AND real L_target ──
    let target = args.target_layer;
    let inject_at = target + 1;
    let rank = args.rank;

    let mut x_vecs: Vec<Vec<f32>> = Vec::new(); // raw embedding last-token
    let mut y_vecs: Vec<Vec<f32>> = Vec::new(); // real L_target last-token

    for (i, prompt) in train_prompts.iter().enumerate() {
        let encoding = model.tokenizer()
            .encode(prompt.as_str(), true)
            .map_err(|e| format!("tokenize: {e}"))?;
        let token_ids: Vec<u32> = encoding.get_ids().to_vec();
        let seq_len = token_ids.len();
        if seq_len < 3 { continue; }

        // Compute input vector
        let input_vec: Vec<f32> = if args.source_layers > 0 {
            // Run real layers 0..source_layers-1, take last-token residual
            let h = forward_to_layer(weights, &token_ids, args.source_layers - 1);
            h.row(seq_len - 1).to_vec()
        } else if args.sum_embeddings {
            let mut sum = vec![0.0f32; hidden];
            for &tid in &token_ids {
                let row = weights.embed.row(tid as usize);
                for j in 0..hidden { sum[j] += row[j] * embed_scale; }
            }
            sum
        } else {
            let last_tok = token_ids[seq_len - 1] as usize;
            let embed_row = weights.embed.row(last_tok);
            embed_row.iter().map(|&v| v * embed_scale).collect()
        };
        x_vecs.push(input_vec);

        // Real L_target via full forward pass
        let h_real = forward_to_layer(weights, &token_ids, target);
        let last_row: Vec<f32> = h_real.row(seq_len - 1).to_vec();
        y_vecs.push(last_row);

        if (i + 1) % 50 == 0 {
            eprintln!("  {}/{} training prompts...", i + 1, train_prompts.len());
        }
    }

    let n_train = x_vecs.len();
    eprintln!(
        "  {} prompts processed ({:.1}s)",
        n_train,
        fit_start.elapsed().as_secs_f64()
    );

    // ── Fit rank-k projection: Y ≈ (X - Xm) @ Vt[:k]^T @ B + Ym ──
    eprintln!("Fitting rank-{} projection...", rank);

    // Means
    let mut x_mean = vec![0.0f32; hidden];
    let mut y_mean = vec![0.0f32; hidden];
    for i in 0..n_train {
        for j in 0..hidden {
            x_mean[j] += x_vecs[i][j];
            y_mean[j] += y_vecs[i][j];
        }
    }
    for j in 0..hidden {
        x_mean[j] /= n_train as f32;
        y_mean[j] /= n_train as f32;
    }

    // Center X
    let xc: Vec<Vec<f32>> = x_vecs.iter()
        .map(|x| x.iter().zip(x_mean.iter()).map(|(a, m)| a - m).collect())
        .collect();
    let yc: Vec<Vec<f32>> = y_vecs.iter()
        .map(|y| y.iter().zip(y_mean.iter()).map(|(a, m)| a - m).collect())
        .collect();

    // Gram matrix for SVD
    let mut gram = vec![0.0f32; n_train * n_train];
    for i in 0..n_train {
        for j in i..n_train {
            let d: f32 = xc[i].iter().zip(xc[j].iter()).map(|(a, b)| a * b).sum();
            gram[i * n_train + j] = d;
            gram[j * n_train + i] = d;
        }
    }

    // Power iteration
    let r = rank.min(n_train);
    let mut eigenvalues: Vec<f32> = Vec::new();
    let mut eigenvectors: Vec<Vec<f32>> = Vec::new();

    for _ in 0..r {
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

    // Build projection directions and betas
    let mut vt_rows: Vec<Vec<f32>> = Vec::new();
    let mut betas: Vec<Vec<f32>> = Vec::new();

    for k in 0..eigenvalues.len() {
        // Direction in hidden space
        let mut dir = vec![0.0f32; hidden];
        for i in 0..n_train {
            let c = eigenvectors[k][i] / eigenvalues[k];
            for j in 0..hidden { dir[j] += c * xc[i][j]; }
        }
        let n: f32 = dir.iter().map(|x| x * x).sum::<f32>().sqrt();
        if n > 1e-12 { for x in dir.iter_mut() { *x /= n; } }
        vt_rows.push(dir);

        // Beta
        let mut beta = vec![0.0f32; hidden];
        for i in 0..n_train {
            let c = eigenvectors[k][i] / eigenvalues[k];
            for j in 0..hidden { beta[j] += c * yc[i][j]; }
        }
        betas.push(beta);
    }

    eprintln!("  Projection fitted: {} components", eigenvalues.len());

    // ── Load test prompts ──
    let test_prompts: Vec<String> = if let Some(ref file) = args.prompts_file {
        std::fs::read_to_string(file)?
            .lines().map(|l| l.trim().to_string()).filter(|l| !l.is_empty()).collect()
    } else if let Some(ref p) = args.prompts {
        p.split(',').map(|s| s.trim().to_string()).collect()
    } else {
        return Err("provide --prompts or --prompts-file".into());
    };

    // ── End-to-end test ──
    eprintln!(
        "\n── Embedding Jump: raw embed → rank-{} project → L{} → L{}-L{} dense ──\n",
        rank, target, inject_at, num_layers - 1
    );

    println!(
        "{:<45} {:>12} {:>12} {:>8} {:>8} {:>3}",
        "Prompt", "Baseline", "EmbJump", "B_conf", "E_conf", "="
    );
    println!("{}", "-".repeat(92));

    let mut match_count = 0;
    let mut total = 0;
    let mut cosines = Vec::new();

    for prompt in &test_prompts {
        let encoding = model.tokenizer()
            .encode(prompt.as_str(), true)
            .map_err(|e| format!("tokenize: {e}"))?;
        let token_ids: Vec<u32> = encoding.get_ids().to_vec();
        let seq_len = token_ids.len();
        if seq_len < 3 { continue; }

        // Baseline
        let baseline = predict(weights, model.tokenizer(), &token_ids, args.top_k);
        let (base_tok, base_conf) = baseline.predictions.first()
            .map(|(t, p)| (t.clone(), *p)).unwrap_or_default();

        // Compute input (same method as training)
        let input_vec: Vec<f32> = if args.source_layers > 0 {
            let h = forward_to_layer(weights, &token_ids, args.source_layers - 1);
            h.row(seq_len - 1).to_vec()
        } else if args.sum_embeddings {
            let mut sum = vec![0.0f32; hidden];
            for &tid in &token_ids {
                let row = weights.embed.row(tid as usize);
                for j in 0..hidden { sum[j] += row[j] * embed_scale; }
            }
            sum
        } else {
            let last_tok = token_ids[seq_len - 1] as usize;
            let embed_row = weights.embed.row(last_tok);
            embed_row.iter().map(|&v| v * embed_scale).collect()
        };

        // Project: input_vec → L_target
        let mut projected = y_mean.clone();
        for k in 0..eigenvalues.len() {
            let score: f32 = (0..hidden)
                .map(|j| (input_vec[j] - x_mean[j]) * vt_rows[k][j])
                .sum();
            for j in 0..hidden {
                projected[j] += score * betas[k][j];
            }
        }

        // Get real hidden state at L_target for context positions
        let h_real = forward_to_layer(weights, &token_ids, target);

        // Cosine between projected and real at target layer
        let real_last: Vec<f32> = h_real.row(seq_len - 1).to_vec();
        let cos: f32 = {
            let dot: f32 = projected.iter().zip(real_last.iter()).map(|(a, b)| a * b).sum();
            let na: f32 = projected.iter().map(|x| x * x).sum::<f32>().sqrt();
            let nb: f32 = real_last.iter().map(|x| x * x).sum::<f32>().sqrt();
            if na > 1e-12 && nb > 1e-12 { dot / (na * nb) } else { 0.0 }
        };
        cosines.push(cos);

        // Inject: replace last token with projection
        let mut h_hybrid = h_real;
        for j in 0..hidden {
            h_hybrid[[seq_len - 1, j]] = projected[j];
        }

        // Run decoder
        let jump_result = predict_from_hidden(
            weights, model.tokenizer(), &h_hybrid, inject_at, args.top_k,
        );
        let (jump_tok, jump_conf) = jump_result.predictions.first()
            .map(|(t, p)| (t.clone(), *p)).unwrap_or_default();

        let matched = jump_tok == base_tok;
        if matched { match_count += 1; }
        total += 1;

        let m = if matched { "=" } else { "X" };
        println!(
            "{:<45} {:>12} {:>12} {:>7.2}% {:>7.2}% {:>3}",
            &prompt[..prompt.len().min(44)],
            base_tok, jump_tok,
            base_conf * 100.0, jump_conf * 100.0, m,
        );
    }

    // Summary
    let mean_cos: f32 = cosines.iter().sum::<f32>() / cosines.len().max(1) as f32;
    let min_cos: f32 = cosines.iter().copied().fold(f32::INFINITY, f32::min);

    eprintln!("\n── Summary ──");
    eprintln!("  Prompts: {}", total);
    eprintln!(
        "  Token match: {}/{} ({:.1}%)",
        match_count, total,
        match_count as f64 / total.max(1) as f64 * 100.0
    );
    eprintln!("  Cosine at L{}: mean={:.6}, min={:.6}", target, mean_cos, min_cos);
    if args.source_layers > 0 {
        eprintln!("  Method: {} real layers → rank-{} projection → L{}-L{} dense",
            args.source_layers, rank, inject_at, num_layers - 1);
        eprintln!("  {} real layers + {} dot products → {} decoder layers.",
            args.source_layers, rank, num_layers - inject_at);
    } else {
        eprintln!("  Method: raw embedding → rank-{} projection → L{}-L{} dense",
            rank, inject_at, num_layers - 1);
        eprintln!("  Zero encoder layers. Just embedding lookup + {} dot products.", rank);
    }
    eprintln!("  Zero matmul layers. Just an embedding lookup + {} dot products.", rank);

    Ok(())
}
