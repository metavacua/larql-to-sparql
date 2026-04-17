use std::time::Instant;

use clap::Args;
use larql_inference::ndarray;
use larql_inference::InferenceModel;

#[derive(Args)]
pub struct QkRankArgs {
    /// Model path or HuggingFace model ID.
    model: String,

    /// Layers to analyze. Default: all.
    #[arg(short, long)]
    layers: Option<String>,

    /// Singular value threshold (fraction of largest). Values below this are noise.
    #[arg(long, default_value = "0.1")]
    threshold: f32,

    /// Show all heads (not just variable ones).
    #[arg(long)]
    all: bool,
}

pub fn run(args: QkRankArgs) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("Loading model: {}", args.model);
    let start = Instant::now();
    let model = InferenceModel::load(&args.model)?;
    let weights = model.weights();
    let num_layers = weights.num_layers;
    let num_q_heads = weights.num_q_heads;
    let num_kv_heads = weights.num_kv_heads;
    let head_dim = weights.head_dim;
    let reps = num_q_heads / num_kv_heads;
    let arch = &*weights.arch;

    eprintln!(
        "  {} layers, {} Q heads, {} KV heads, head_dim={} ({:.1}s)",
        num_layers, num_q_heads, num_kv_heads, head_dim,
        start.elapsed().as_secs_f64()
    );

    let layers: Vec<usize> = match &args.layers {
        Some(spec) => parse_layer_spec(spec)?,
        None => (0..num_layers).collect(),
    };

    eprintln!("\n── Computing QK rank per head ──\n");

    println!(
        "{:<6} {:<5} {:>6} {:>6} {:>8} {:>8} {:>8}  Spectrum (top 10 singular values)",
        "Layer", "Head", "Rank", "Dim", "S_max", "S_10", "S_50"
    );
    println!("{}", "-".repeat(100));

    let mut total_heads = 0;
    let mut rank_histogram: Vec<usize> = vec![0; head_dim + 1];
    let mut all_ranks: Vec<(usize, usize, usize)> = Vec::new(); // (layer, head, rank)

    for &layer in &layers {
        let w_q = match weights.tensors.get(&arch.attn_q_key(layer)) {
            Some(w) => w,
            None => continue,
        };
        let w_k = match weights.tensors.get(&arch.attn_k_key(layer)) {
            Some(w) => w,
            None => continue,
        };

        for q_head in 0..num_q_heads {
            let kv_head = q_head / reps;

            // Extract Q block: (head_dim, hidden_size)
            let q_start = q_head * head_dim;
            let q_block = w_q.slice(ndarray::s![q_start..q_start + head_dim, ..]);

            // Extract K block: (head_dim, hidden_size)
            let k_start = kv_head * head_dim;
            let k_block = w_k.slice(ndarray::s![k_start..k_start + head_dim, ..]);

            // QK = Q_block × K_block^T = (head_dim, hidden) × (hidden, head_dim) = (head_dim, head_dim)
            let qk = q_block.dot(&k_block.t());

            // SVD via eigendecomposition of QK^T × QK (symmetric positive semi-definite)
            // Singular values of QK = sqrt(eigenvalues of QK^T × QK)
            let qk_sq = qk.t().dot(&qk); // (head_dim, head_dim) symmetric

            // Power iteration to find singular values (simple, no external dependency)
            let singular_values = compute_singular_values(&qk_sq, head_dim);

            // Count significant singular values
            let s_max = singular_values[0];
            let threshold_val = s_max * args.threshold;
            let rank = singular_values.iter().filter(|&&s| s > threshold_val).count();

            rank_histogram[rank] += 1;
            all_ranks.push((layer, head_dim, rank));
            total_heads += 1;

            let s_10 = if singular_values.len() > 9 {
                singular_values[9]
            } else {
                0.0
            };
            let s_50 = if singular_values.len() > 49 {
                singular_values[49]
            } else {
                0.0
            };

            // Spectrum string: top 10 values normalized by max
            let spectrum: String = singular_values
                .iter()
                .take(10)
                .map(|&s| {
                    let normalized = if s_max > 0.0 { s / s_max } else { 0.0 };
                    if normalized > 0.5 {
                        "█"
                    } else if normalized > 0.2 {
                        "▓"
                    } else if normalized > 0.1 {
                        "▒"
                    } else if normalized > 0.05 {
                        "░"
                    } else {
                        "·"
                    }
                })
                .collect::<Vec<_>>()
                .join("");

            if args.all || rank <= head_dim / 2 {
                println!(
                    "L{:<4} H{:<4} {:>5} {:>5} {:>8.1} {:>8.1} {:>8.1}  {}",
                    layer, q_head, rank, head_dim, s_max, s_10, s_50, spectrum
                );
            }

            all_ranks.last_mut().unwrap().1 = q_head; // fix: store head not head_dim
        }
    }

    // ── Summary ──
    println!("\n═══ Summary ═══\n");
    println!("  Total heads analyzed: {}", total_heads);
    println!("  Head dimension: {}", head_dim);
    println!("  Threshold: {:.0}% of max singular value", args.threshold * 100.0);

    // Rank distribution
    println!("\n  Rank distribution:");
    let mut cumulative = 0;
    for (rank, &count) in rank_histogram.iter().enumerate() {
        if count > 0 {
            cumulative += count;
            println!(
                "    rank {:>3}: {:>4} heads ({:>5.1}% cumulative)",
                rank,
                count,
                cumulative as f64 / total_heads as f64 * 100.0
            );
        }
    }

    // Effective modes
    let ranks: Vec<usize> = all_ranks.iter().map(|&(_, _, r)| r).collect();
    let avg_rank: f64 = ranks.iter().sum::<usize>() as f64 / ranks.len() as f64;
    let min_rank = ranks.iter().min().copied().unwrap_or(0);
    let max_rank = ranks.iter().max().copied().unwrap_or(0);
    let median_rank = {
        let mut sorted = ranks.clone();
        sorted.sort();
        sorted[sorted.len() / 2]
    };

    println!("\n  Rank statistics:");
    println!("    min: {}", min_rank);
    println!("    median: {}", median_rank);
    println!("    mean: {:.1}", avg_rank);
    println!("    max: {}", max_rank);

    // Template capacity estimate
    // If median head has rank R, and heads are somewhat independent,
    // the number of distinct attention configurations ≈ R (not R^N, because
    // the modes are correlated across heads within a layer)
    println!("\n  Template capacity estimate:");
    println!(
        "    Per-head modes: {} (median rank at {:.0}% threshold)",
        median_rank,
        args.threshold * 100.0
    );
    println!(
        "    Estimated distinct templates: ~{}-{}",
        min_rank.max(2),
        max_rank.min(head_dim)
    );
    println!(
        "    (Actual number is constrained by correlations — likely closer to {})",
        median_rank * 2
    );

    Ok(())
}

/// Compute singular values of a symmetric PSD matrix via QR-like iteration.
/// Returns sorted descending.
fn compute_singular_values(ata: &ndarray::Array2<f32>, dim: usize) -> Vec<f32> {
    // For a (dim×dim) symmetric matrix, eigenvalues = singular values squared.
    // Use simple power iteration with deflation to get top eigenvalues.
    // This is O(dim² × iterations) per eigenvalue — fine for dim=256.

    let mut matrix = ata.clone();
    let mut eigenvalues: Vec<f32> = Vec::with_capacity(dim);

    let max_eigens = dim.min(100); // Don't need all 256, top 100 is enough
    let iterations = 50;

    for _ in 0..max_eigens {
        // Power iteration
        let mut v = ndarray::Array1::<f32>::ones(dim);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v /= norm;

        let mut eigenvalue = 0.0f32;

        for _ in 0..iterations {
            let mv = matrix.dot(&v);
            eigenvalue = mv.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
            let norm: f32 = mv.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm < 1e-10 {
                break;
            }
            v = mv / norm;
        }

        if eigenvalue < 1e-10 {
            break;
        }

        // Singular value = sqrt(eigenvalue of A^T A)
        eigenvalues.push(eigenvalue.sqrt());

        // Deflate: remove this eigenvalue's contribution
        let outer = {
            let mut o = ndarray::Array2::<f32>::zeros((dim, dim));
            for i in 0..dim {
                for j in 0..dim {
                    o[[i, j]] = eigenvalue * v[i] * v[j];
                }
            }
            o
        };
        matrix = matrix - outer;
    }

    eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    eigenvalues
}

fn parse_layer_spec(spec: &str) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    let mut layers = Vec::new();
    for part in spec.split(',') {
        let part = part.trim();
        if part.contains('-') {
            let (a, b) = part
                .split_once('-')
                .ok_or_else(|| format!("invalid range: {part}"))?;
            let start: usize = a.parse()?;
            let end: usize = b.parse()?;
            layers.extend(start..=end);
        } else {
            layers.push(part.parse()?);
        }
    }
    Ok(layers)
}
