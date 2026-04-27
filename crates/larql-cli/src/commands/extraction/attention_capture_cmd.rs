use std::time::Instant;

use clap::Args;
use larql_inference::{trace_forward_full, InferenceModel, WeightFfn};

#[derive(Args)]
pub struct AttentionCaptureArgs {
    /// Model path or HuggingFace model ID.
    model: String,

    /// Prompts to compare (comma-separated).
    /// e.g., "The capital of France is,The capital of Germany is,The capital of Japan is"
    #[arg(short, long, value_delimiter = ',')]
    prompts: Vec<String>,

    /// Layers to capture. Comma-separated or range. Default: all.
    #[arg(short, long)]
    layers: Option<String>,

    /// Attention threshold — only show heads with max attention > this value.
    #[arg(long, default_value = "0.1")]
    threshold: f32,

    /// Show verbose per-head details.
    #[arg(short, long)]
    verbose: bool,
}

pub fn run(args: AttentionCaptureArgs) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("Loading model: {}", args.model);
    let start = Instant::now();
    let model = InferenceModel::load(&args.model)?;
    let weights = model.weights();
    let num_layers = weights.num_layers;
    let num_heads = weights.num_q_heads;
    eprintln!(
        "  {} layers, {} heads, hidden_size={} ({:.1}s)",
        num_layers,
        num_heads,
        weights.hidden_size,
        start.elapsed().as_secs_f64()
    );

    let layers: Vec<usize> = match &args.layers {
        Some(spec) => parse_layer_spec(spec)?,
        None => (0..num_layers).collect(),
    };

    // Tokenize all prompts and get token labels
    let mut all_token_ids: Vec<Vec<u32>> = Vec::new();
    let mut all_token_labels: Vec<Vec<String>> = Vec::new();

    for prompt in &args.prompts {
        let encoding = model
            .tokenizer()
            .encode(prompt.as_str(), true)
            .map_err(|e| format!("tokenize error: {e}"))?;
        let ids: Vec<u32> = encoding.get_ids().to_vec();
        let labels: Vec<String> = ids
            .iter()
            .map(|&id| {
                model
                    .tokenizer()
                    .decode(&[id], true)
                    .unwrap_or_else(|_| format!("T{id}"))
                    .trim()
                    .to_string()
            })
            .collect();
        eprintln!("  {:?} → {} tokens: {:?}", prompt, ids.len(), labels);
        all_token_ids.push(ids);
        all_token_labels.push(labels);
    }

    // Run forward pass with attention capture for each prompt
    let ffn = WeightFfn { weights };

    // Store: captures[prompt_idx][layer_idx] = per-head attention weights
    let mut all_captures: Vec<Vec<Vec<Vec<f32>>>> = Vec::new();

    for (i, token_ids) in all_token_ids.iter().enumerate() {
        eprintln!("\nRunning forward pass for prompt {}...", i + 1);
        let start = Instant::now();
        let trace = trace_forward_full(
            weights,
            token_ids,
            &layers,
            false, // no activation capture
            0,
            true, // capture attention
            &ffn,
        );
        eprintln!("  {:.1}s", start.elapsed().as_secs_f64());

        // Extract per-head weights: [layer][head] = [seq_len]
        let mut prompt_captures: Vec<Vec<Vec<f32>>> = Vec::new();
        for capture in &trace.attention {
            prompt_captures.push(capture.weights.heads.clone());
        }
        all_captures.push(prompt_captures);
    }

    let num_prompts = args.prompts.len();
    let _seq_len = all_token_ids[0].len();

    // ── Analysis ──

    // 1. Per-layer, per-head: where does the last token attend?
    println!("\n═══ Attention Patterns (last token → positions) ═══\n");

    for (li, &layer) in layers.iter().enumerate() {
        let mut any_active = false;

        for head in 0..num_heads {
            // Check if this head is active (above threshold) for any prompt
            let max_attn: f32 = (0..num_prompts)
                .filter_map(|pi| {
                    all_captures.get(pi)
                        .and_then(|c| c.get(li))
                        .and_then(|h| h.get(head))
                        .map(|w| w.iter().copied().fold(0.0f32, f32::max))
                })
                .fold(0.0f32, f32::max);

            if max_attn < args.threshold {
                continue;
            }
            any_active = true;

            if args.verbose || num_prompts <= 3 {
                println!("L{layer} H{head} (max={max_attn:.3}):");
                for (pi, prompt) in args.prompts.iter().enumerate() {
                    if let Some(weights) = all_captures.get(pi)
                        .and_then(|c| c.get(li))
                        .and_then(|h| h.get(head))
                    {
                        let pattern: String = weights
                            .iter()
                            .enumerate()
                            .filter(|(_, &w)| w > 0.01)
                            .map(|(j, &w)| {
                                let label = all_token_labels.get(pi)
                                    .and_then(|l| l.get(j))
                                    .map(|s| s.as_str())
                                    .unwrap_or("?");
                                format!("{}={:.2}", label, w)
                            })
                            .collect::<Vec<_>>()
                            .join("  ");
                        let short_prompt: String = prompt.chars().take(40).collect();
                        println!("  {short_prompt:<40} [{pattern}]");
                    }
                }
            }
        }

        if !any_active && args.verbose {
            println!("L{layer}: no heads above threshold {:.2}", args.threshold);
        }
    }

    // 2. Cross-prompt correlation: are patterns the same across entities?
    if num_prompts >= 2 {
        println!("\n═══ Cross-Prompt Correlation ═══\n");
        println!(
            "{:<8} {:<6} {:>8} {:>10}  Classification",
            "Layer", "Head", "MaxAttn", "Corr(0,1)"
        );
        println!("{}", "-".repeat(65));

        for (li, &layer) in layers.iter().enumerate() {
            for head in 0..num_heads {
                // Get attention patterns for first two prompts
                let w0 = match all_captures.first().and_then(|c| c.get(li)).and_then(|h| h.get(head)) {
                    Some(w) => w,
                    None => continue,
                };
                let w1 = match all_captures.get(1).and_then(|c| c.get(li)).and_then(|h| h.get(head)) {
                    Some(w) => w,
                    None => continue,
                };

                let max_attn = w0.iter().copied().fold(0.0f32, f32::max)
                    .max(w1.iter().copied().fold(0.0f32, f32::max));

                if max_attn < args.threshold {
                    continue;
                }

                let corr = cosine_similarity(w0, w1);
                let classification = if corr > 0.95 {
                    "FIXED (template-invariant)"
                } else if corr > 0.8 {
                    "SIMILAR"
                } else if corr > 0.5 {
                    "PARTIAL"
                } else {
                    "DIFFERENT (entity-sensitive)"
                };

                println!(
                    "L{:<6} H{:<5} {:>7.3} {:>9.4}  {}",
                    layer, head, max_attn, corr, classification
                );
            }
        }

        // Summary stats
        let mut fixed = 0;
        let mut similar = 0;
        let mut partial = 0;
        let mut different = 0;
        let mut total_active = 0;

        for (li, _) in layers.iter().enumerate() {
            for head in 0..num_heads {
                let w0 = match all_captures.first().and_then(|c| c.get(li)).and_then(|h| h.get(head)) {
                    Some(w) => w,
                    None => continue,
                };
                let w1 = match all_captures.get(1).and_then(|c| c.get(li)).and_then(|h| h.get(head)) {
                    Some(w) => w,
                    None => continue,
                };

                let max_attn = w0.iter().copied().fold(0.0f32, f32::max)
                    .max(w1.iter().copied().fold(0.0f32, f32::max));
                if max_attn < args.threshold {
                    continue;
                }

                total_active += 1;
                let corr = cosine_similarity(w0, w1);
                if corr > 0.95 {
                    fixed += 1;
                } else if corr > 0.8 {
                    similar += 1;
                } else if corr > 0.5 {
                    partial += 1;
                } else {
                    different += 1;
                }
            }
        }

        println!("\n═══ Summary ═══");
        println!("  Active heads (above threshold): {total_active}");
        println!("  FIXED (corr > 0.95):    {fixed} ({:.0}%)", fixed as f64 / total_active as f64 * 100.0);
        println!("  SIMILAR (corr > 0.8):   {similar} ({:.0}%)", similar as f64 / total_active as f64 * 100.0);
        println!("  PARTIAL (corr > 0.5):   {partial} ({:.0}%)", partial as f64 / total_active as f64 * 100.0);
        println!("  DIFFERENT (corr < 0.5): {different} ({:.0}%)", different as f64 / total_active as f64 * 100.0);

        if fixed + similar > total_active * 80 / 100 {
            println!("\n  → Attention is largely TEMPLATE-FIXED. Circuit caching viable.");
        } else if different > total_active * 50 / 100 {
            println!("\n  → Attention is largely ENTITY-SENSITIVE. Circuit caching not viable.");
        } else {
            println!("\n  → Mixed. Early layers fixed, late layers may vary.");
        }
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
