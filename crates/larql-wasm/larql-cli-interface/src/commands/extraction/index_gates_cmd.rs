use std::path::PathBuf;
use std::time::Instant;

use clap::Args;
use larql_inference::{GateIndex, IndexBuildCallbacks, InferenceModel};

#[derive(Args)]
pub struct IndexGatesArgs {
    /// Model path or HuggingFace model ID.
    model: String,

    /// Output index file (.gate-index.jsonl).
    #[arg(short, long)]
    output: PathBuf,

    /// Features to index per token per layer.
    #[arg(long, default_value = "100")]
    features_per_token: usize,

    /// Top tokens to match at runtime (stored in header).
    #[arg(long, default_value = "10")]
    top_tokens: usize,

    /// Layers to index (e.g. "0-33" or "26,27,28"). Default: all.
    #[arg(long)]
    layers: Option<String>,
}

struct ProgressCallbacks {
    total: usize,
}

impl IndexBuildCallbacks for ProgressCallbacks {
    fn on_layer_start(&mut self, layer: usize, _total: usize) {
        eprint!("  Layer {layer}/{} ...", self.total);
    }

    fn on_layer_done(&mut self, _layer: usize, elapsed_ms: f64) {
        eprintln!(" {:.1}s", elapsed_ms / 1000.0);
    }
}

pub fn run(args: IndexGatesArgs) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("Loading model: {}", args.model);
    let start = Instant::now();
    let model = InferenceModel::load(&args.model)?;
    eprintln!(
        "  {} layers, hidden_size={}, vocab_size={} ({:.1}s)",
        model.num_layers(),
        model.hidden_size(),
        model.weights().vocab_size,
        start.elapsed().as_secs_f64()
    );

    // Parse layers
    let layers: Vec<usize> = if let Some(ref spec) = args.layers {
        parse_layers(spec)
    } else {
        (0..model.num_layers()).collect()
    };

    eprintln!(
        "Building gate index: {} layers, {} features/token, {} top_tokens",
        layers.len(),
        args.features_per_token,
        args.top_tokens,
    );

    let mut callbacks = ProgressCallbacks {
        total: layers.len(),
    };

    let build_start = Instant::now();

    // Stream directly to disk — never holds more than one layer in memory.
    eprintln!("Streaming to {}...", args.output.display());
    GateIndex::build_streaming(
        model.weights(),
        &layers,
        args.features_per_token,
        args.top_tokens,
        &args.output,
        &mut callbacks,
    )?;
    let build_elapsed = build_start.elapsed();

    let size = std::fs::metadata(&args.output)?.len();
    eprintln!(
        "\nIndex built in {:.1}s ({} layers, {:.1} MB)",
        build_elapsed.as_secs_f64(),
        layers.len(),
        size as f64 / 1024.0 / 1024.0,
    );

    eprintln!("\nDone. Total: {:.1}s", start.elapsed().as_secs_f64());
    eprintln!(
        "Use with: larql predict {} --ffn graph --gate-index {}",
        args.model,
        args.output.display()
    );

    Ok(())
}

fn parse_layers(s: &str) -> Vec<usize> {
    let mut layers = Vec::new();
    for part in s.split(',') {
        let part = part.trim();
        if let Some((a, b)) = part.split_once('-') {
            let start: usize = a.parse().unwrap_or(0);
            let end: usize = b.parse().unwrap_or(0);
            layers.extend(start..=end);
        } else if let Ok(l) = part.parse::<usize>() {
            layers.push(l);
        }
    }
    layers
}
