use std::path::PathBuf;

use clap::Args;
use larql_inference::{
    trace_forward, GateIndex, InferenceModel,
};

#[derive(Args)]
pub struct FfnOverlapArgs {
    /// Model path or HuggingFace model ID.
    #[arg(short, long)]
    model: String,

    /// Path to gate index file.
    #[arg(long)]
    gate_index: PathBuf,

    /// Prompt to analyze.
    #[arg(short, long, default_value = "The capital of France is")]
    prompt: String,

    /// Layers to check.
    #[arg(short, long, default_value = "0,4,8,12,16,20,24,28,32")]
    layers: String,
}

pub fn run(args: FfnOverlapArgs) -> Result<(), Box<dyn std::error::Error>> {
    let model = InferenceModel::load(&args.model)?;
    let weights = model.weights();

    let gi = GateIndex::load(&args.gate_index, 10)?;

    let encoding = model.tokenizer().encode(args.prompt.as_str(), true)
        .map_err(|e| format!("tokenize error: {e}"))?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();

    let layers: Vec<usize> = args.layers.split(',')
        .map(|s| s.trim().parse().unwrap())
        .collect();

    // Capture residuals at each layer
    let trace = trace_forward(weights, &token_ids, &layers, false, 0);

    // Entity tokens for gate index lookup
    let entity_tokens: Vec<(usize, f32)> = token_ids.iter().map(|&t| (t as usize, 1.0)).collect();

    println!("{:>5} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "Layer", "Entity", "Gate64", "Gate256", "Overlap64", "Overlap256");
    println!("{}", "-".repeat(55));

    for (layer, residual_vec) in &trace.residuals {
        let arch = &*weights.arch;
        let w_gate = weights.tensors.get(&arch.ffn_gate_key(*layer)).unwrap();
        let _hidden = weights.hidden_size;

        // Ground truth: actual gate matmul on the residual
        let residual = larql_inference::ndarray::Array1::from_vec(residual_vec.clone());
        let gate_scores = w_gate.dot(&residual);

        // Top-64 and top-256 from actual gate matmul
        let mut indexed: Vec<(usize, f32)> = gate_scores.iter().copied().enumerate()
            .map(|(i, v)| (i, v * larql_inference::ffn::sigmoid(v)))
            .collect();
        indexed.sort_unstable_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
        let gate_top64: std::collections::HashSet<usize> = indexed.iter().take(64).map(|x| x.0).collect();
        let gate_top256: std::collections::HashSet<usize> = indexed.iter().take(256).map(|x| x.0).collect();

        // Entity-routed features from gate index
        let entity_feats64 = gi.lookup_from_tokens(&entity_tokens, *layer, 64);
        let entity_feats256 = gi.lookup_from_tokens(&entity_tokens, *layer, 256);

        let entity_set64: std::collections::HashSet<usize> = entity_feats64.iter().copied().collect();
        let entity_set256: std::collections::HashSet<usize> = entity_feats256.iter().copied().collect();

        let overlap64 = entity_set64.intersection(&gate_top64).count();
        let overlap256 = entity_set256.intersection(&gate_top256).count();

        println!("{:>5} {:>8} {:>8} {:>8} {:>7}/{:<3} {:>7}/{:<3}",
            layer, entity_feats64.len(), gate_top64.len(), gate_top256.len(),
            overlap64, 64, overlap256, 256);
    }

    Ok(())
}
