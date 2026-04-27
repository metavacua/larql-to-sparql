use std::path::PathBuf;
use std::time::Instant;

use clap::Args;
use larql_inference::{forward_to_layer, predict, predict_from_hidden, InferenceModel};

#[derive(Args)]
pub struct BottleneckTestArgs {
    /// Model path or HuggingFace model ID.
    model: String,

    /// L13 bottleneck vectors JSON (from bottleneck analysis).
    #[arg(long)]
    bottleneck: PathBuf,

    /// Comma-separated prompts to test.
    #[arg(long)]
    prompts: Option<String>,

    /// File with one prompt per line.
    #[arg(long)]
    prompts_file: Option<PathBuf>,

    /// Number of top predictions to show.
    #[arg(short = 'k', long, default_value = "5")]
    top_k: usize,
}

#[derive(serde::Deserialize)]
struct BottleneckData {
    hidden_size: usize,
    layer: usize,
    base_vector: Vec<f32>,
    pc1_direction: Vec<f32>,
}

/// 9 rules that replace 13 layers of computation.
fn rule_score(prompt: &str) -> f32 {
    let p = prompt.to_lowercase();

    // Non-ASCII fraction (multilingual detection)
    let ascii_frac = prompt.chars().filter(|c| c.is_ascii()).count() as f32
        / prompt.len().max(1) as f32;
    if ascii_frac < 0.7 {
        return 6000.0;
    }

    // Instructions
    if p.starts_with("describe ")
        || p.starts_with("explain ")
        || p.starts_with("name ")
        || p.starts_with("list ")
        || p.starts_with("summarize ")
    {
        return -5500.0;
    }
    if p.starts_with("write ")
        || p.starts_with("translate ")
        || p.starts_with("compare ")
        || p.starts_with("define ")
        || p.starts_with("calculate ")
        || p.starts_with("convert ")
    {
        return -4500.0;
    }

    // Questions
    if p.starts_with("how ") {
        return -5500.0;
    }
    if p.starts_with("what ")
        || p.starts_with("who ")
        || p.starts_with("where ")
        || p.starts_with("when ")
        || p.starts_with("why ")
    {
        return -4500.0;
    }

    // Capital pattern
    if p.contains("capital of") {
        return 800.0;
    }

    // Code
    if p.starts_with("def ")
        || p.starts_with("class ")
        || p.starts_with("import ")
        || p.starts_with("select ")
        || p.starts_with("for ")
        || p.starts_with("return ")
        || p.starts_with("async ")
    {
        return 2000.0;
    }

    // Factual "The X is" pattern
    if p.starts_with("the ") && p.trim_end().ends_with(" is") {
        return 500.0;
    }

    // Default: narrative
    1500.0
}

pub fn run(args: BottleneckTestArgs) -> Result<(), Box<dyn std::error::Error>> {
    // Load model
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

    // Load bottleneck vectors
    eprintln!("Loading bottleneck vectors...");
    let bn: BottleneckData = serde_json::from_str(&std::fs::read_to_string(&args.bottleneck)?)?;
    let inject_layer = bn.layer + 1; // Reconstruct L13, inject at L14 (resume from L14)
    eprintln!(
        "  Bottleneck at L{}, inject at L{}, hidden={}",
        bn.layer, inject_layer, bn.hidden_size
    );

    // Load prompts
    let test_prompts: Vec<String> = if let Some(ref file) = args.prompts_file {
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

    eprintln!(
        "\n── End-to-end: 9 rules → L{} state → L{}-L{} dense ──\n",
        bn.layer, inject_layer, num_layers - 1
    );

    println!(
        "{:<45} {:>12} {:>12} {:>8} {:>8} {:>8} {:>3}",
        "Prompt", "Baseline", "Rules", "B_conf", "R_conf", "Score", "="
    );
    println!("{}", "-".repeat(100));

    let mut match_count = 0;
    let mut total = 0;

    for prompt in &test_prompts {
        let encoding = model
            .tokenizer()
            .encode(prompt.as_str(), true)
            .map_err(|e| format!("tokenize error: {e}"))?;
        let token_ids: Vec<u32> = encoding.get_ids().to_vec();
        let seq_len = token_ids.len();
        if seq_len < 3 {
            continue;
        }

        // Baseline: full forward pass
        let baseline = predict(weights, model.tokenizer(), &token_ids, args.top_k);
        let (base_tok, base_conf) = baseline
            .predictions
            .first()
            .map(|(t, p)| (t.clone(), *p))
            .unwrap_or_default();

        // Rule-based score
        let score = rule_score(prompt);

        // Reconstruct L13 last-token state: base + score * pc1
        let mut l13_last_token = vec![0.0f32; hidden];
        for (j, slot) in l13_last_token.iter_mut().enumerate() {
            *slot = bn.base_vector[j] + score * bn.pc1_direction[j];
        }

        // Get real hidden state at L13 for all positions except last
        // (the real forward pass provides context for positions 0..seq_len-1,
        //  we only replace the last token's state with our reconstruction)
        let h_real = forward_to_layer(weights, &token_ids, bn.layer);

        // Build hybrid: real for all positions, rule-reconstructed for last
        let mut h_hybrid = h_real.clone();
        for j in 0..hidden {
            h_hybrid[[seq_len - 1, j]] = l13_last_token[j];
        }

        // Run L14-33
        let rule_result =
            predict_from_hidden(weights, model.tokenizer(), &h_hybrid, inject_layer, args.top_k);
        let (rule_tok, rule_conf) = rule_result
            .predictions
            .first()
            .map(|(t, p)| (t.clone(), *p))
            .unwrap_or_default();

        let matched = rule_tok == base_tok;
        if matched {
            match_count += 1;
        }
        total += 1;

        let m = if matched { "=" } else { "X" };
        println!(
            "{:<45} {:>12} {:>12} {:>7.2}% {:>7.2}% {:>7.0} {:>3}",
            &prompt[..prompt.len().min(44)],
            base_tok,
            rule_tok,
            base_conf * 100.0,
            rule_conf * 100.0,
            score,
            m,
        );
    }

    eprintln!("\n── Summary ──");
    eprintln!("  Prompts: {}", total);
    eprintln!(
        "  Token match: {}/{} ({:.1}%)",
        match_count,
        total,
        match_count as f64 / total as f64 * 100.0
    );
    eprintln!(
        "  Layers replaced: 0-{} ({} layers → 9 if-else rules)",
        bn.layer,
        bn.layer + 1
    );
    eprintln!(
        "  Layers computed: {}-{} ({} layers dense)",
        inject_layer,
        num_layers - 1,
        num_layers - inject_layer
    );

    Ok(())
}
