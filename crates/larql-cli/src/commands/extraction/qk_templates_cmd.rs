use std::time::Instant;

use clap::Args;
use larql_inference::{trace_forward_full, InferenceModel, WeightFfn};

#[derive(Args)]
pub struct QkTemplatesArgs {
    /// Model path or HuggingFace model ID.
    model: String,

    /// Template prompts to compare (one per relation type).
    /// Format: "relation:prompt" pairs, comma-separated.
    /// e.g., "capital-of:The capital of France is,language-of:The language of France is"
    /// If not provided, uses built-in templates.
    #[arg(short, long, value_delimiter = ',')]
    templates: Option<Vec<String>>,

    /// Layers to analyze. Default: all.
    #[arg(short, long)]
    layers: Option<String>,

    /// Correlation threshold below which a head is considered "variable".
    #[arg(long, default_value = "0.95")]
    threshold: f32,

    /// Number of top SVD components to show per head.
    #[arg(long, default_value = "5")]
    top_components: usize,
}

/// Default templates covering the 12 discovered relation types.
fn default_templates() -> Vec<(String, String)> {
    vec![
        ("capital-of".into(), "The capital of France is".into()),
        ("language-of".into(), "The language of France is".into()),
        ("located-in".into(), "France is located in".into()),
        ("currency".into(), "The currency of France is".into()),
        ("continent".into(), "The continent of France is".into()),
        ("nationality".into(), "The nationality of someone from France is".into()),
        ("birthplace".into(), "The birthplace of Napoleon is".into()),
        ("known-for".into(), "France is known for".into()),
        ("spoken-in".into(), "The language spoken in France is".into()),
        ("author-of".into(), "The author of Les Misérables is".into()),
        ("birth-year".into(), "Napoleon was born in the year".into()),
        ("death-year".into(), "Napoleon died in the year".into()),
    ]
}

pub fn run(args: QkTemplatesArgs) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("Loading model: {}", args.model);
    let start = Instant::now();
    let model = InferenceModel::load(&args.model)?;
    let weights = model.weights();
    let num_layers = weights.num_layers;
    let num_heads = weights.num_q_heads;
    let head_dim = weights.head_dim;
    eprintln!(
        "  {} layers, {} heads, head_dim={} ({:.1}s)",
        num_layers, num_heads, head_dim,
        start.elapsed().as_secs_f64()
    );

    // Parse templates
    let templates: Vec<(String, String)> = match &args.templates {
        Some(specs) => specs
            .iter()
            .map(|s| {
                let (rel, prompt) = s.split_once(':').unwrap_or(("unknown", s.as_str()));
                (rel.to_string(), prompt.to_string())
            })
            .collect(),
        None => default_templates(),
    };

    eprintln!("\n{} templates:", templates.len());
    for (rel, prompt) in &templates {
        eprintln!("  {rel}: {prompt:?}");
    }

    let layers: Vec<usize> = match &args.layers {
        Some(spec) => parse_layer_spec(spec)?,
        None => (0..num_layers).collect(),
    };

    // ── Step 1: Run forward passes and capture attention patterns ──
    eprintln!("\n── Capturing attention patterns ──");
    let ffn = WeightFfn { weights };

    // captures[template_idx][layer_idx] = per-head attention weights
    let mut all_captures: Vec<Vec<Vec<Vec<f32>>>> = Vec::new();
    let mut all_token_labels: Vec<Vec<String>> = Vec::new();

    for (rel, prompt) in templates.iter() {
        let encoding = model
            .tokenizer()
            .encode(prompt.as_str(), true)
            .map_err(|e| format!("tokenize error: {e}"))?;
        let token_ids: Vec<u32> = encoding.get_ids().to_vec();
        let labels: Vec<String> = token_ids
            .iter()
            .map(|&id| {
                model.tokenizer().decode(&[id], true)
                    .unwrap_or_else(|_| format!("T{id}"))
                    .trim().to_string()
            })
            .collect();

        eprint!("  {rel}...");
        let trace = trace_forward_full(
            weights, &token_ids, &layers, false, 0, true, &ffn,
        );

        let mut prompt_captures: Vec<Vec<Vec<f32>>> = Vec::new();
        for capture in &trace.attention {
            prompt_captures.push(capture.weights.heads.clone());
        }
        all_captures.push(prompt_captures);
        all_token_labels.push(labels);
        eprintln!(" done ({} tokens)", token_ids.len());
    }

    // ── Step 2: Find variable heads ──
    eprintln!("\n── Identifying variable heads ──");

    // For each head, compute average pairwise correlation across all template pairs
    struct HeadInfo {
        layer: usize,
        head: usize,
        avg_corr: f32,
        min_corr: f32,
        max_attn: f32,
    }

    let mut variable_heads: Vec<HeadInfo> = Vec::new();
    let mut fixed_count = 0;

    for (li, &layer) in layers.iter().enumerate() {
        for head in 0..num_heads {
            let mut corrs: Vec<f32> = Vec::new();
            let mut max_attn: f32 = 0.0;

            for i in 0..templates.len() {
                for j in (i + 1)..templates.len() {
                    let w_i = match all_captures.get(i).and_then(|c| c.get(li)).and_then(|h| h.get(head)) {
                        Some(w) => w,
                        None => continue,
                    };
                    let w_j = match all_captures.get(j).and_then(|c| c.get(li)).and_then(|h| h.get(head)) {
                        Some(w) => w,
                        None => continue,
                    };

                    // Use min length for comparison (different prompt lengths)
                    let len = w_i.len().min(w_j.len());
                    let corr = cosine_similarity(&w_i[..len], &w_j[..len]);
                    corrs.push(corr);
                    max_attn = max_attn
                        .max(w_i.iter().copied().fold(0.0f32, f32::max))
                        .max(w_j.iter().copied().fold(0.0f32, f32::max));
                }
            }

            if corrs.is_empty() || max_attn < 0.1 {
                continue;
            }

            let avg_corr: f32 = corrs.iter().sum::<f32>() / corrs.len() as f32;
            let min_corr = corrs.iter().copied().fold(f32::INFINITY, f32::min);

            if avg_corr < args.threshold {
                variable_heads.push(HeadInfo {
                    layer, head, avg_corr, min_corr, max_attn,
                });
            } else {
                fixed_count += 1;
            }
        }
    }

    variable_heads.sort_by(|a, b| a.avg_corr.partial_cmp(&b.avg_corr).unwrap());

    println!("\n═══ Variable Heads ({} variable, {} fixed) ═══\n", variable_heads.len(), fixed_count);
    println!(
        "{:<8} {:<6} {:>8} {:>8} {:>8}",
        "Layer", "Head", "AvgCorr", "MinCorr", "MaxAttn"
    );
    println!("{}", "-".repeat(45));
    for h in &variable_heads {
        println!(
            "L{:<6} H{:<5} {:>7.4} {:>7.4} {:>7.3}",
            h.layer, h.head, h.avg_corr, h.min_corr, h.max_attn
        );
    }

    // ── Step 3: For variable heads, compute per-template attention fingerprint ──
    // Group variable heads by layer for analysis
    println!("\n═══ Template Fingerprints (variable heads) ═══\n");

    // For each template, build a fingerprint vector from variable head patterns
    // Fingerprint = concatenation of attention patterns at variable heads
    let num_templates = templates.len();

    // Build fingerprint matrix: templates × variable_heads
    // For each variable head, which position gets the most attention?
    // This creates a compact signature.

    println!(
        "{:<20} {}",
        "Template",
        variable_heads.iter()
            .take(15)
            .map(|h| format!("L{}H{}", h.layer, h.head))
            .collect::<Vec<_>>()
            .join("  ")
    );
    println!("{}", "-".repeat(20 + variable_heads.len().min(15) * 7));

    for (ti, (rel, _)) in templates.iter().enumerate() {
        let mut cells: Vec<String> = Vec::new();
        for vh in variable_heads.iter().take(15) {
            let li = layers.iter().position(|&l| l == vh.layer).unwrap_or(0);
            let pattern = all_captures.get(ti)
                .and_then(|c| c.get(li))
                .and_then(|h| h.get(vh.head));

            if let Some(weights) = pattern {
                // Find the position with max attention
                let (max_pos, max_val) = weights.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap_or((0, &0.0));
                let label = all_token_labels.get(ti)
                    .and_then(|l| l.get(max_pos))
                    .map(|s| s.as_str())
                    .unwrap_or("?");
                cells.push(format!("{:.0}%{}", max_val * 100.0, &label[..label.len().min(3)]));
            } else {
                cells.push("---".into());
            }
        }
        println!("{:<20} {}", rel, cells.join("  "));
    }

    // ── Step 4: Template clustering via pairwise correlation ──
    println!("\n═══ Template Similarity Matrix ═══\n");

    // Build full fingerprint vectors for correlation
    let mut fingerprints: Vec<Vec<f32>> = Vec::new();
    for (ti, _) in templates.iter().enumerate() {
        let mut fp: Vec<f32> = Vec::new();
        for vh in &variable_heads {
            let li = layers.iter().position(|&l| l == vh.layer).unwrap_or(0);
            if let Some(weights) = all_captures.get(ti)
                .and_then(|c| c.get(li))
                .and_then(|h| h.get(vh.head))
            {
                fp.extend_from_slice(weights);
            }
        }
        fingerprints.push(fp);
    }

    // Print correlation matrix header
    let short_names: Vec<String> = templates.iter()
        .map(|(r, _)| r.chars().take(10).collect())
        .collect();

    print!("{:<14}", "");
    for name in &short_names {
        print!("{:>12}", name);
    }
    println!();

    for i in 0..num_templates {
        print!("{:<14}", short_names[i]);
        for j in 0..num_templates {
            if fingerprints[i].is_empty() || fingerprints[j].is_empty() {
                print!("{:>12}", "---");
            } else {
                let len = fingerprints[i].len().min(fingerprints[j].len());
                let corr = cosine_similarity(&fingerprints[i][..len], &fingerprints[j][..len]);
                print!("{:>11.4} ", corr);
            }
        }
        println!();
    }

    // ── Step 5: Cluster templates ──
    println!("\n═══ Template Clusters ═══\n");

    // Simple single-linkage clustering at threshold 0.9
    let cluster_threshold = 0.9;
    let mut clusters: Vec<Vec<usize>> = Vec::new();
    let mut assigned: Vec<bool> = vec![false; num_templates];

    for i in 0..num_templates {
        if assigned[i] {
            continue;
        }
        let mut cluster = vec![i];
        assigned[i] = true;

        for j in (i + 1)..num_templates {
            if assigned[j] {
                continue;
            }
            let len = fingerprints[i].len().min(fingerprints[j].len());
            if len > 0 {
                let corr = cosine_similarity(&fingerprints[i][..len], &fingerprints[j][..len]);
                if corr > cluster_threshold {
                    cluster.push(j);
                    assigned[j] = true;
                }
            }
        }
        clusters.push(cluster);
    }

    for (ci, cluster) in clusters.iter().enumerate() {
        let members: Vec<String> = cluster.iter()
            .map(|&i| templates[i].0.clone())
            .collect();
        println!("  Cluster {}: {}", ci + 1, members.join(", "));
    }

    println!("\n  {} distinct attention circuits for {} relation types",
        clusters.len(), num_templates);

    if clusters.len() < num_templates {
        println!("  → Some relations share attention circuits (can reuse cached patterns)");
    }
    if clusters.len() <= 5 {
        println!("  → Very few distinct circuits. Attention is highly reusable.");
    }

    Ok(())
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a < 1e-10 || norm_b < 1e-10 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

fn parse_layer_spec(spec: &str) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    let mut layers = Vec::new();
    for part in spec.split(',') {
        let part = part.trim();
        if part.contains('-') {
            let (a, b) = part.split_once('-')
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
