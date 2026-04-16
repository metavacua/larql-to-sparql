use std::path::PathBuf;
use std::time::Instant;

use clap::Args;
#[allow(deprecated)]
use larql_inference::{
    predict, predict_with_ffn, FeatureListFfn, InferenceModel,
};

#[derive(Args)]
pub struct GraphWalkArgs {
    /// Model path or HuggingFace model ID.
    #[arg(short, long)]
    model: String,

    /// Comma-separated prompts to evaluate.
    #[arg(long)]
    prompts: String,

    /// Top-K features per layer for feature list calibration.
    #[arg(short = 'k', long, default_value = "50")]
    top_k: usize,

    /// Number of top predictions to show.
    #[arg(long, default_value = "5")]
    predict_top_k: usize,

    /// Also run dense ground truth for comparison.
    #[arg(long)]
    compare: bool,

    /// Save feature lists to this directory.
    #[arg(long)]
    save: Option<PathBuf>,

    /// Load feature lists from file instead of calibrating.
    #[arg(long)]
    load: Option<PathBuf>,
}

pub fn run(args: GraphWalkArgs) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("Loading model: {}", args.model);
    let model = InferenceModel::load(&args.model)?;
    let weights = model.weights();
    eprintln!("  {} layers, hidden_size={}", weights.num_layers, weights.hidden_size);

    let prompts: Vec<&str> = args.prompts.split(',').map(|s| s.trim()).collect();

    if args.compare {
        println!(
            "{:40} {:>8} {:>10} {:>7}  {:>8} {:>10} {:>7}  {:>5}",
            "Prompt", "FL ms", "Top-1", "Prob", "Dense", "Top-1", "Prob", "Match",
        );
        println!("{}", "-".repeat(100));
    }

    let mut total_fl_ms = 0.0;
    let mut total_dense_ms = 0.0;
    let mut matches = 0;
    let mut total = 0;

    for (i, prompt) in prompts.iter().enumerate() {
        let encoding = model.tokenizer().encode(*prompt, true)
            .map_err(|e| format!("tokenize error: {e}"))?;
        let token_ids: Vec<u32> = encoding.get_ids().to_vec();

        // Get or build feature lists
        let fl_ffn = if let Some(ref path) = args.load {
            FeatureListFfn::load(weights, path)?
        } else {
            let cal_start = Instant::now();
            let fl = FeatureListFfn::calibrate(weights, &token_ids, args.top_k);
            let cal_ms = cal_start.elapsed().as_secs_f64() * 1000.0;
            eprint!("\r  Cal {:?}: {:.0}ms, {:.0} feats/layer   ",
                prompt, cal_ms, fl.avg_features_per_layer());

            if let Some(ref dir) = args.save {
                std::fs::create_dir_all(dir)?;
                let path = dir.join(format!("prompt_{}.features", i));
                fl.save(&path)?;
                eprint!("→ {}", path.display());
            }
            eprintln!();
            fl
        };

        // Inference: attention live + sparse FFN on preselected features
        let start = Instant::now();
        let fl_result = predict_with_ffn(
            weights, model.tokenizer(), &token_ids, args.predict_top_k, &fl_ffn,
        );
        let fl_ms = start.elapsed().as_secs_f64() * 1000.0;
        total_fl_ms += fl_ms;

        let (f_tok, f_prob) = fl_result.predictions.first()
            .map(|(t, p)| (t.as_str(), *p)).unwrap_or(("?", 0.0));

        if args.compare {
            let start = Instant::now();
            let dense_result = predict(weights, model.tokenizer(), &token_ids, args.predict_top_k);
            let dense_ms = start.elapsed().as_secs_f64() * 1000.0;
            total_dense_ms += dense_ms;

            let (d_tok, d_prob) = dense_result.predictions.first()
                .map(|(t, p)| (t.as_str(), *p)).unwrap_or(("?", 0.0));

            let is_match = f_tok == d_tok;
            if is_match { matches += 1; }
            total += 1;

            println!(
                "{:40} {:>6.0}ms {:>10} {:>6.1}%  {:>6.0}ms {:>10} {:>6.1}%  {:>5}",
                prompt, fl_ms, f_tok, f_prob * 100.0,
                dense_ms, d_tok, d_prob * 100.0,
                if is_match { "yes" } else { "NO" },
            );
        } else {
            total += 1;
            println!("{:40} {:>6.0}ms {:>10} {:>6.1}%", prompt, fl_ms, f_tok, f_prob * 100.0);
        }
    }

    if args.compare {
        println!("{}", "-".repeat(100));
        println!(
            "  Match: {}/{} ({:.0}%)  |  FL avg: {:.0}ms  Dense avg: {:.0}ms  Speedup: {:.2}x  (K={})",
            matches, total, matches as f64 / total as f64 * 100.0,
            total_fl_ms / total as f64, total_dense_ms / total as f64,
            total_dense_ms / total_fl_ms, args.top_k,
        );
    }

    Ok(())
}
