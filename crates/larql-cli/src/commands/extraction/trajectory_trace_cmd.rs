use std::path::PathBuf;
use std::time::Instant;

use clap::Args;
use larql_inference::{trace_forward, InferenceModel};
use serde::Serialize;

#[derive(Args)]
pub struct TrajectoryTraceArgs {
    /// Model path or HuggingFace model ID.
    model: String,

    /// Comma-separated prompts to trace.
    #[arg(long, required_unless_present = "prompts_file")]
    prompts: Option<String>,

    /// File with one prompt per line.
    #[arg(long)]
    prompts_file: Option<PathBuf>,

    /// Output JSONL file.
    #[arg(short, long)]
    output: PathBuf,

    /// Layers to capture (default: all).
    #[arg(long)]
    layers: Option<String>,

    /// Dump raw residual vectors to a numpy-compatible binary file (.npz-like).
    /// Creates <output>.vectors.bin (f32, row-major: [prompt][layer][hidden_size])
    /// and <output>.vectors.json (metadata: shapes, prompts, layers).
    #[arg(long)]
    dump_vectors: bool,
}

// ── Output structs ──

#[derive(Serialize)]
struct TrajectoryHeader {
    _header: bool,
    model: String,
    hidden_size: usize,
    num_layers: usize,
    num_prompts: usize,
    extraction_date: String,
}

#[derive(Serialize)]
struct LayerPoint {
    layer: usize,
    norm: f32,
    cosine_to_prev: Option<f32>,
    angular_displacement: Option<f32>,
    delta_norm: Option<f32>,
    // Force decomposition
    radial_force: Option<f32>,
    tangential_force: Option<f32>,
    radial_fraction: Option<f32>,
}

#[derive(Serialize)]
struct PcaInfo {
    singular_values: Vec<f32>,
    cumulative_variance: Vec<f32>,
    dims_for_90: usize,
    dims_for_95: usize,
    dims_for_99: usize,
}

#[derive(Serialize)]
struct TrajectoryRecord {
    prompt: String,
    prompt_index: usize,
    num_tokens: usize,
    points: Vec<LayerPoint>,
    total_arc_length: f32,
    mean_curvature: f32,
    norm_drift: f32,
    prediction: String,
    confidence: f64,
    pca: PcaInfo,
}

#[derive(Serialize)]
struct CrossPromptPca {
    _type: String,
    shared_singular_values: Vec<f32>,
    shared_cumulative_variance: Vec<f32>,
    shared_dims_for_95: usize,
    // Each prompt's trajectory projected into shared PCA space: [prompt][layer][component]
    projections: Vec<PromptProjection>,
    // Pairwise cosine similarity of projected trajectories
    pairwise_trajectory_cosine: Vec<PairwiseSim>,
    // Layer-by-layer divergence between prompt pairs
    divergence_profiles: Vec<DivergenceProfile>,
}

#[derive(Serialize)]
struct PromptProjection {
    prompt: String,
    prompt_index: usize,
    // coords[layer_idx] = [pc0, pc1, pc2, ...] (top 10 components)
    coords: Vec<Vec<f32>>,
}

#[derive(Serialize)]
struct PairwiseSim {
    prompt_a: String,
    prompt_b: String,
    cosine: f32,
}

#[derive(Serialize)]
struct DivergenceProfile {
    prompt_a: String,
    prompt_b: String,
    // Per-layer cosine similarity between the two trajectories' residuals
    layer_cosines: Vec<LayerCosine>,
    diverge_layer: Option<usize>,
}

#[derive(Serialize)]
struct LayerCosine {
    layer: usize,
    cosine: f32,
}

// ── Math helpers ──

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn vec_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let na = vec_norm(a);
    let nb = vec_norm(b);
    if na < 1e-12 || nb < 1e-12 {
        return 0.0;
    }
    (dot(a, b) / (na * nb)).clamp(-1.0, 1.0)
}

/// SVD of an n_rows × n_cols matrix (n_rows << n_cols).
/// Computes via eigendecomposition of the n_rows × n_rows Gram matrix.
/// Returns singular values sorted descending.
fn svd_singular_values(rows: &[Vec<f32>], n_rows: usize) -> Vec<f32> {
    // Build Gram matrix G = X × X^T  (n_rows × n_rows)
    let mut gram = vec![0.0f32; n_rows * n_rows];
    for i in 0..n_rows {
        for j in i..n_rows {
            let d = dot(&rows[i], &rows[j]);
            gram[i * n_rows + j] = d;
            gram[j * n_rows + i] = d;
        }
    }

    // Power iteration with deflation on the Gram matrix
    let mut eigenvalues = Vec::with_capacity(n_rows);
    let iterations = 80;

    for _ in 0..n_rows {
        let mut v = vec![1.0f32; n_rows];
        let n = vec_norm(&v);
        for x in v.iter_mut() {
            *x /= n;
        }

        let mut eigenvalue = 0.0f32;
        for _ in 0..iterations {
            // mv = gram × v
            let mut mv = vec![0.0f32; n_rows];
            for i in 0..n_rows {
                let mut s = 0.0f32;
                for j in 0..n_rows {
                    s += gram[i * n_rows + j] * v[j];
                }
                mv[i] = s;
            }
            eigenvalue = dot(&mv, &v);
            let n = vec_norm(&mv);
            if n < 1e-12 {
                break;
            }
            for (x, m) in v.iter_mut().zip(mv.iter()) {
                *x = m / n;
            }
        }

        if eigenvalue < 1e-8 {
            break;
        }

        eigenvalues.push(eigenvalue.sqrt());

        // Deflate
        for i in 0..n_rows {
            for j in 0..n_rows {
                gram[i * n_rows + j] -= eigenvalue * v[i] * v[j];
            }
        }
    }

    eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap());
    eigenvalues
}

/// Compute top-k eigenvectors of an n×n Gram matrix.
/// Returns (eigenvalues, eigenvectors) where eigenvectors[k] has length n.
fn svd_eigenvectors(rows: &[Vec<f32>], n_rows: usize, top_k: usize) -> (Vec<f32>, Vec<Vec<f32>>) {
    let mut gram = vec![0.0f32; n_rows * n_rows];
    for i in 0..n_rows {
        for j in i..n_rows {
            let d = dot(&rows[i], &rows[j]);
            gram[i * n_rows + j] = d;
            gram[j * n_rows + i] = d;
        }
    }

    let mut eigenvalues = Vec::with_capacity(top_k);
    let mut eigenvectors: Vec<Vec<f32>> = Vec::with_capacity(top_k);
    let iterations = 80;

    for _ in 0..top_k.min(n_rows) {
        let mut v = vec![1.0f32; n_rows];
        let n = vec_norm(&v);
        for x in v.iter_mut() {
            *x /= n;
        }

        let mut eigenvalue = 0.0f32;
        for _ in 0..iterations {
            let mut mv = vec![0.0f32; n_rows];
            for i in 0..n_rows {
                let mut s = 0.0f32;
                for j in 0..n_rows {
                    s += gram[i * n_rows + j] * v[j];
                }
                mv[i] = s;
            }
            eigenvalue = dot(&mv, &v);
            let n = vec_norm(&mv);
            if n < 1e-12 {
                break;
            }
            for (x, m) in v.iter_mut().zip(mv.iter()) {
                *x = m / n;
            }
        }

        if eigenvalue < 1e-8 {
            break;
        }

        eigenvalues.push(eigenvalue.sqrt());
        eigenvectors.push(v.clone());

        for i in 0..n_rows {
            for j in 0..n_rows {
                gram[i * n_rows + j] -= eigenvalue * v[i] * v[j];
            }
        }
    }

    (eigenvalues, eigenvectors)
}

fn cumulative_variance(singular_values: &[f32]) -> Vec<f32> {
    let total: f32 = singular_values.iter().map(|s| s * s).sum();
    if total < 1e-12 {
        return vec![0.0; singular_values.len()];
    }
    let mut cum = 0.0f32;
    singular_values
        .iter()
        .map(|s| {
            cum += s * s;
            round4(cum / total)
        })
        .collect()
}

fn dims_for_threshold(cumvar: &[f32], threshold: f32) -> usize {
    cumvar
        .iter()
        .position(|&v| v >= threshold)
        .map(|i| i + 1)
        .unwrap_or(cumvar.len())
}

fn make_pca_info(singular_values: Vec<f32>) -> PcaInfo {
    let cumvar = cumulative_variance(&singular_values);
    let dims_90 = dims_for_threshold(&cumvar, 0.90);
    let dims_95 = dims_for_threshold(&cumvar, 0.95);
    let dims_99 = dims_for_threshold(&cumvar, 0.99);
    PcaInfo {
        singular_values: singular_values.iter().map(|v| round4(*v)).collect(),
        cumulative_variance: cumvar,
        dims_for_90: dims_90,
        dims_for_95: dims_95,
        dims_for_99: dims_99,
    }
}

// ── Main ──

pub fn run(args: TrajectoryTraceArgs) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("Loading model: {}", args.model);
    let start = Instant::now();
    let model = InferenceModel::load(&args.model)?;
    let weights = model.weights();
    let num_layers = weights.num_layers;
    let hidden_size = weights.hidden_size;

    eprintln!(
        "  {} layers, hidden_size={} ({:.1}s)",
        num_layers, hidden_size,
        start.elapsed().as_secs_f64()
    );

    let capture_layers: Vec<usize> = match &args.layers {
        Some(spec) => parse_layer_spec(spec, num_layers)?,
        None => (0..num_layers).collect(),
    };

    let prompt_strings: Vec<String> = if let Some(ref file) = args.prompts_file {
        std::fs::read_to_string(file)?
            .lines()
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty())
            .collect()
    } else if let Some(ref p) = args.prompts {
        p.split(',').map(|s| s.trim().to_string()).collect()
    } else {
        return Err("provide --prompts or --prompts-file".into());
    };
    let prompts: Vec<&str> = prompt_strings.iter().map(|s| s.as_str()).collect();
    eprintln!(
        "  {} prompts, capturing {} layers\n",
        prompts.len(),
        capture_layers.len()
    );

    // ── Phase 1: Collect all trajectories ──

    struct RawTrajectory {
        prompt: String,
        num_tokens: usize,
        layers: Vec<usize>,
        residuals: Vec<Vec<f32>>,  // residuals[layer_idx] = hidden_size vector
        prediction: String,
        confidence: f64,
    }

    let mut trajectories: Vec<RawTrajectory> = Vec::with_capacity(prompts.len());

    for (idx, prompt) in prompts.iter().enumerate() {
        let prompt = prompt.trim();
        let pass_start = Instant::now();

        let encoding = model
            .tokenizer()
            .encode(prompt, true)
            .map_err(|e| format!("tokenize error: {e}"))?;
        let token_ids: Vec<u32> = encoding.get_ids().to_vec();

        let trace = trace_forward(weights, &token_ids, &capture_layers, false, 0);
        let pred = larql_inference::predict(weights, model.tokenizer(), &token_ids, 1);
        let (prediction, confidence) = pred
            .predictions
            .first()
            .map(|(t, p)| (t.clone(), *p))
            .unwrap_or_default();

        let layers: Vec<usize> = trace.residuals.iter().map(|(l, _)| *l).collect();
        let residuals: Vec<Vec<f32>> = trace.residuals.into_iter().map(|(_, v)| v).collect();

        let elapsed = pass_start.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "  [{}/{}] {:40} → {:12} ({:.2})  {:.0}ms",
            idx + 1, prompts.len(), prompt, prediction, confidence, elapsed
        );

        trajectories.push(RawTrajectory {
            prompt: prompt.to_string(),
            num_tokens: token_ids.len(),
            layers,
            residuals,
            prediction,
            confidence,
        });
    }

    eprintln!("\n── Per-prompt PCA + Force Decomposition ──\n");

    // ── Phase 2: Per-prompt analysis ──

    let mut records: Vec<TrajectoryRecord> = Vec::with_capacity(trajectories.len());

    for (idx, traj) in trajectories.iter().enumerate() {
        let n_layers = traj.residuals.len();

        // -- Trajectory points with force decomposition --
        let mut points = Vec::with_capacity(n_layers);
        let mut total_arc = 0.0f32;
        let mut curvatures = Vec::new();

        for i in 0..n_layers {
            let n = vec_norm(&traj.residuals[i]);

            let (cos, angle, delta_n) = if i > 0 {
                let cos = cosine_sim(&traj.residuals[i], &traj.residuals[i - 1]);
                let angle = cos.acos();
                let dn = n - vec_norm(&traj.residuals[i - 1]);
                total_arc += angle;
                (Some(cos), Some(angle), Some(dn))
            } else {
                (None, None, None)
            };

            if i >= 2 {
                let cos_prev = cosine_sim(&traj.residuals[i - 1], &traj.residuals[i - 2]);
                let cos_curr = cosine_sim(&traj.residuals[i], &traj.residuals[i - 1]);
                curvatures.push((cos_curr.acos() - cos_prev.acos()).abs());
            }

            // Force decomposition: delta = residual[i] - residual[i-1]
            let (radial, tangential, rad_frac) = if i > 0 {
                let prev = &traj.residuals[i - 1];
                let curr = &traj.residuals[i];
                let prev_norm = vec_norm(prev);

                // delta = curr - prev
                let delta: Vec<f32> = curr.iter().zip(prev.iter()).map(|(c, p)| c - p).collect();
                let delta_mag = vec_norm(&delta);

                if prev_norm < 1e-12 || delta_mag < 1e-12 {
                    (Some(0.0f32), Some(0.0f32), Some(0.0f32))
                } else {
                    // Unit vector along previous residual direction
                    // Radial component = projection of delta onto prev direction
                    let radial_component = dot(&delta, prev) / prev_norm;
                    // Tangential = what's left (Pythagorean)
                    let tang_sq = (delta_mag * delta_mag - radial_component * radial_component).max(0.0);
                    let tangential_component = tang_sq.sqrt();
                    let frac = if delta_mag > 0.0 {
                        (radial_component.abs() / delta_mag).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    (
                        Some(radial_component),
                        Some(tangential_component),
                        Some(frac),
                    )
                }
            } else {
                (None, None, None)
            };

            points.push(LayerPoint {
                layer: traj.layers[i],
                norm: round4(n),
                cosine_to_prev: cos.map(round4),
                angular_displacement: angle.map(round4),
                delta_norm: delta_n.map(round4),
                radial_force: radial.map(round4),
                tangential_force: tangential.map(round4),
                radial_fraction: rad_frac.map(round4),
            });
        }

        let norm_drift = if !traj.residuals.is_empty() {
            vec_norm(traj.residuals.last().unwrap()) - vec_norm(traj.residuals.first().unwrap())
        } else {
            0.0
        };
        let mean_curvature = if curvatures.is_empty() {
            0.0
        } else {
            curvatures.iter().sum::<f32>() / curvatures.len() as f32
        };

        // -- Per-prompt PCA: 34×2560 matrix, SVD via Gram --
        let svs = svd_singular_values(&traj.residuals, n_layers);
        let pca = make_pca_info(svs);

        eprintln!(
            "  {:40} PCA: {} dims for 90%, {} for 95%, {} for 99%  (top SV: {:.0})",
            traj.prompt,
            pca.dims_for_90,
            pca.dims_for_95,
            pca.dims_for_99,
            pca.singular_values.first().copied().unwrap_or(0.0),
        );

        records.push(TrajectoryRecord {
            prompt: traj.prompt.clone(),
            prompt_index: idx,
            num_tokens: traj.num_tokens,
            points,
            total_arc_length: round4(total_arc),
            mean_curvature: round4(mean_curvature),
            norm_drift: round4(norm_drift),
            prediction: traj.prediction.clone(),
            confidence: traj.confidence,
            pca,
        });
    }

    // ── Phase 3: Cross-prompt PCA ──

    eprintln!("\n── Cross-prompt PCA ──\n");

    // Stack all trajectories into one big matrix: (num_prompts * n_layers) × hidden_size
    // Mean-center each row (remove per-layer mean across prompts)
    let n_layers = trajectories[0].residuals.len();
    let n_prompts = trajectories.len();

    // Compute per-layer mean
    let mut layer_means = vec![vec![0.0f32; hidden_size]; n_layers];
    for traj in &trajectories {
        for (l, res) in traj.residuals.iter().enumerate() {
            for (j, v) in res.iter().enumerate() {
                layer_means[l][j] += v;
            }
        }
    }
    for mean in &mut layer_means {
        for v in mean.iter_mut() {
            *v /= n_prompts as f32;
        }
    }

    // Build centered rows
    let mut all_rows: Vec<Vec<f32>> = Vec::with_capacity(n_prompts * n_layers);
    for traj in &trajectories {
        for (l, res) in traj.residuals.iter().enumerate() {
            let centered: Vec<f32> = res
                .iter()
                .zip(layer_means[l].iter())
                .map(|(v, m)| v - m)
                .collect();
            all_rows.push(centered);
        }
    }

    let total_rows = all_rows.len();
    let top_k = 10.min(total_rows);
    let (shared_svs, shared_evecs) = svd_eigenvectors(&all_rows, total_rows, top_k);
    let shared_cumvar = cumulative_variance(&shared_svs);
    let shared_dims_95 = dims_for_threshold(&shared_cumvar, 0.95);

    eprintln!(
        "  Shared PCA: {} total rows, top-{} components",
        total_rows, top_k
    );
    for (i, (sv, cv)) in shared_svs.iter().zip(shared_cumvar.iter()).enumerate() {
        eprintln!("    PC{i}: SV={sv:.1}  cumvar={cv:.4}");
    }

    // Project each prompt's trajectory into shared PCA space
    // The eigenvectors are in row-space (coefficients over the stacked rows).
    // To get the actual principal components in hidden_size space:
    //   pc_k = sum_i evec_k[i] * all_rows[i]  (then normalize)
    // Then project each residual onto these PCs.
    let mut pc_directions: Vec<Vec<f32>> = Vec::with_capacity(top_k);
    for evec in shared_evecs.iter().take(shared_svs.len()) {
        let mut direction = vec![0.0f32; hidden_size];
        for (i, row) in all_rows.iter().enumerate() {
            let coeff = evec[i];
            for (j, v) in row.iter().enumerate() {
                direction[j] += coeff * v;
            }
        }
        // Normalize
        let n = vec_norm(&direction);
        if n > 1e-12 {
            for v in direction.iter_mut() {
                *v /= n;
            }
        }
        pc_directions.push(direction);
    }

    let mut projections = Vec::with_capacity(n_prompts);
    for traj in &trajectories {
        let mut coords = Vec::with_capacity(n_layers);
        for (l, res) in traj.residuals.iter().enumerate() {
            // Center
            let centered: Vec<f32> = res
                .iter()
                .zip(layer_means[l].iter())
                .map(|(v, m)| v - m)
                .collect();
            // Project onto each PC
            let pc_coords: Vec<f32> = pc_directions
                .iter()
                .map(|pc| round4(dot(&centered, pc)))
                .collect();
            coords.push(pc_coords);
        }
        projections.push(PromptProjection {
            prompt: traj.prompt.clone(),
            prompt_index: traj.residuals.len(), // will fix below
            coords,
        });
    }
    // Fix prompt_index
    for (i, p) in projections.iter_mut().enumerate() {
        p.prompt_index = i;
    }

    // Pairwise trajectory cosine: flatten each prompt's full trajectory, cosine between them
    let mut pairwise = Vec::new();
    let flat_trajs: Vec<Vec<f32>> = trajectories
        .iter()
        .map(|t| t.residuals.iter().flatten().copied().collect())
        .collect();
    for i in 0..n_prompts {
        for j in (i + 1)..n_prompts {
            pairwise.push(PairwiseSim {
                prompt_a: trajectories[i].prompt.clone(),
                prompt_b: trajectories[j].prompt.clone(),
                cosine: round4(cosine_sim(&flat_trajs[i], &flat_trajs[j])),
            });
        }
    }

    // Layer-by-layer divergence profiles between selected pairs
    let mut divergence_profiles = Vec::new();
    for i in 0..n_prompts {
        for j in (i + 1)..n_prompts {
            let mut layer_cosines = Vec::with_capacity(n_layers);
            let mut diverge_layer = None;
            for l in 0..n_layers {
                let cos = cosine_sim(&trajectories[i].residuals[l], &trajectories[j].residuals[l]);
                if diverge_layer.is_none() && cos < 0.95 {
                    diverge_layer = Some(trajectories[i].layers[l]);
                }
                layer_cosines.push(LayerCosine {
                    layer: trajectories[i].layers[l],
                    cosine: round4(cos),
                });
            }
            divergence_profiles.push(DivergenceProfile {
                prompt_a: trajectories[i].prompt.clone(),
                prompt_b: trajectories[j].prompt.clone(),
                layer_cosines,
                diverge_layer,
            });
        }
    }

    // Print divergence summary
    eprintln!("\n── Divergence Summary ──\n");
    eprintln!(
        "  {:<25} {:<25} {:>8} {:>10}",
        "Prompt A", "Prompt B", "TrajCos", "DivergeL"
    );
    eprintln!("  {}", "-".repeat(72));
    for (pw, div) in pairwise.iter().zip(divergence_profiles.iter()) {
        let dl = div
            .diverge_layer
            .map(|l| format!("L{l}"))
            .unwrap_or_else(|| "never".to_string());
        eprintln!(
            "  {:<25} {:<25} {:>8.4} {:>10}",
            pw.prompt_a, pw.prompt_b, pw.cosine, dl
        );
    }

    // ── Phase 3b: Dump raw vectors if requested ──

    if args.dump_vectors {
        let bin_path = args.output.with_extension("vectors.bin");
        let meta_path = args.output.with_extension("vectors.json");

        let n_layers_captured = trajectories[0].residuals.len();
        let total_floats = n_prompts * n_layers_captured * hidden_size;

        eprintln!(
            "\n── Dumping raw vectors: {} prompts × {} layers × {} dims = {} floats ({:.1} MB) ──",
            n_prompts, n_layers_captured, hidden_size, total_floats,
            total_floats as f64 * 4.0 / 1_048_576.0
        );

        // Write binary: f32 little-endian, row-major [prompt][layer][hidden]
        {
            use std::io::Write;
            let mut f = std::io::BufWriter::new(std::fs::File::create(&bin_path)?);
            for traj in &trajectories {
                for res in &traj.residuals {
                    let bytes: &[u8] = unsafe {
                        std::slice::from_raw_parts(
                            res.as_ptr() as *const u8,
                            res.len() * 4,
                        )
                    };
                    f.write_all(bytes)?;
                }
            }
            f.flush()?;
        }

        // Write metadata JSON
        let meta = serde_json::json!({
            "shape": [n_prompts, n_layers_captured, hidden_size],
            "dtype": "float32",
            "byte_order": "little_endian",
            "prompts": trajectories.iter().map(|t| &t.prompt).collect::<Vec<_>>(),
            "layers": &trajectories[0].layers,
            "model": &args.model,
        });
        std::fs::write(&meta_path, serde_json::to_string_pretty(&meta)?)?;

        eprintln!("  Vectors: {}", bin_path.display());
        eprintln!("  Metadata: {}", meta_path.display());
    }

    // ── Phase 4: Write output ──

    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut out = std::io::BufWriter::new(std::fs::File::create(&args.output)?);
    use std::io::Write;

    let header = TrajectoryHeader {
        _header: true,
        model: args.model.clone(),
        hidden_size,
        num_layers,
        num_prompts: prompts.len(),
        extraction_date: current_date(),
    };
    serde_json::to_writer(&mut out, &header)?;
    writeln!(out)?;

    for record in &records {
        serde_json::to_writer(&mut out, record)?;
        writeln!(out)?;
    }

    let cross = CrossPromptPca {
        _type: "cross_prompt_pca".to_string(),
        shared_singular_values: shared_svs.iter().map(|v| round4(*v)).collect(),
        shared_cumulative_variance: shared_cumvar,
        shared_dims_for_95: shared_dims_95,
        projections,
        pairwise_trajectory_cosine: pairwise,
        divergence_profiles,
    };
    serde_json::to_writer(&mut out, &cross)?;
    writeln!(out)?;

    eprintln!("\nTrajectories saved: {}", args.output.display());

    // Print force decomposition summary
    eprintln!("\n── Force Decomposition Summary ──\n");
    eprintln!(
        "  {:<5} {:>10} {:>10} {:>10}  (averaged across all prompts)",
        "Layer", "Radial", "Tangent", "RadFrac%"
    );
    eprintln!("  {}", "-".repeat(45));
    let n_pts = records[0].points.len();
    for l in 1..n_pts {
        let mut rad_sum = 0.0f32;
        let mut tan_sum = 0.0f32;
        let mut frac_sum = 0.0f32;
        let mut count = 0;
        for rec in &records {
            if let (Some(r), Some(t), Some(f)) = (
                rec.points[l].radial_force,
                rec.points[l].tangential_force,
                rec.points[l].radial_fraction,
            ) {
                rad_sum += r;
                tan_sum += t;
                frac_sum += f;
                count += 1;
            }
        }
        if count > 0 {
            let c = count as f32;
            eprintln!(
                "  L{:<3} {:>10.1} {:>10.1} {:>9.1}%",
                records[0].points[l].layer,
                rad_sum / c,
                tan_sum / c,
                frac_sum / c * 100.0,
            );
        }
    }

    Ok(())
}

fn round4(v: f32) -> f32 {
    (v * 10000.0).round() / 10000.0
}

fn current_date() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let days = now / 86400;
    let year = 1970 + (days / 365);
    let remaining = days % 365;
    let month = remaining / 30 + 1;
    let day = remaining % 30 + 1;
    format!("{year}-{month:02}-{day:02}")
}

fn parse_layer_spec(
    spec: &str,
    num_layers: usize,
) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    let mut layers = Vec::new();
    for part in spec.split(',') {
        let part = part.trim();
        if let Some((start, end)) = part.split_once('-') {
            let s: usize = start.parse()?;
            let e: usize = end.parse()?;
            layers.extend(s..=e.min(num_layers - 1));
        } else {
            let l: usize = part.parse()?;
            if l < num_layers {
                layers.push(l);
            }
        }
    }
    Ok(layers)
}
