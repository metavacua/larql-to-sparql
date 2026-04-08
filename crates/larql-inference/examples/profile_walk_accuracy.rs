//! Walk FFN accuracy vs K: what top-K gives the same output as dense?
//!
//! Runs dense FFN and walk FFN at various K values on the same input,
//! measures cosine similarity of the outputs.
//!
//! Run:
//!   cargo run --release -p larql-inference --example profile_walk_accuracy -- \
//!     --model google/gemma-3-4b-it --vindex output/gemma3-4b-v2.vindex

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_path = args.iter().position(|a| a == "--model")
        .and_then(|i| args.get(i + 1)).map(|s| s.as_str())
        .unwrap_or("google/gemma-3-4b-it");
    let vindex_path = args.iter().position(|a| a == "--vindex")
        .and_then(|i| args.get(i + 1)).map(|s| s.as_str())
        .unwrap_or("output/gemma3-4b-v2.vindex");

    println!("=== Walk FFN Accuracy vs K ===\n");

    let model = larql_inference::InferenceModel::load(model_path).unwrap();
    let weights = model.weights();
    let vindex_dir = std::path::PathBuf::from(vindex_path);
    let mut index = larql_vindex::VectorIndex::load_vindex(
        &vindex_dir, &mut larql_vindex::SilentLoadCallbacks,
    ).unwrap();
    let _ = index.load_down_features(&vindex_dir);
    let _ = index.load_up_features(&vindex_dir);

    let hidden = weights.hidden_size;
    let intermediate = index.num_features(14);
    println!("  hidden={hidden}, intermediate={intermediate}\n");

    // Run forward to get realistic hidden state at L14
    let prompt = "The capital of France is";
    let encoding = model.tokenizer().encode(prompt, true).unwrap();
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();
    let mut h = larql_inference::forward::embed_tokens_pub(weights, &token_ids);
    for layer in 0..14 {
        let (h_pa, _, _) = larql_inference::attention::run_attention_block_gpu(
            weights, &h, layer, false, None,
        ).unwrap();
        let dense_ffn = larql_inference::WeightFfn { weights };
        let (h_out, _) = larql_inference::forward::run_ffn(weights, &h_pa, layer, &dense_ffn, false);
        h = h_out;
    }

    // Get the post-attention state at L14
    let (h_post_attn, _, _) = larql_inference::attention::run_attention_block_gpu(
        weights, &h, 14, false, None,
    ).unwrap();

    // Dense FFN output (ground truth)
    let dense_ffn = larql_inference::WeightFfn { weights };
    let (dense_out, _) = larql_inference::forward::run_ffn(weights, &h_post_attn, 14, &dense_ffn, false);
    let dense_row = dense_out.row(dense_out.shape()[0] - 1);
    let dense_norm = larql_compute::norm(&dense_row);

    // Count non-zero activations in dense path
    let norm_offset = weights.arch.norm_weight_offset();
    let h_ffn = larql_inference::forward::apply_norm(
        weights, &h_post_attn,
        &weights.arch.post_attention_layernorm_key(14), norm_offset,
    );
    let gate_w = weights.tensors.get(&weights.arch.ffn_gate_key(14)).unwrap();
    let up_w = weights.tensors.get(&weights.arch.ffn_up_key(14)).unwrap();
    let gate_scores = h_ffn.row(h_ffn.shape()[0]-1).dot(&gate_w.t());
    let up_scores = h_ffn.row(h_ffn.shape()[0]-1).dot(&up_w.t());
    let mut activations: Vec<f32> = gate_scores.iter().zip(up_scores.iter())
        .map(|(&g, &u)| {
            let act_g = larql_inference::ffn::gelu_tanh(g);
            act_g * u
        })
        .collect();
    activations.sort_by(|a, b| b.abs().partial_cmp(&a.abs()).unwrap());

    let nonzero = activations.iter().filter(|a| a.abs() > 1e-6).count();
    let top10_energy: f32 = activations[..10].iter().map(|a| a * a).sum();
    let total_energy: f32 = activations.iter().map(|a| a * a).sum();

    println!("  Dense FFN activation profile (L14):");
    println!("    Non-zero activations:  {nonzero}/{intermediate}");
    println!("    Top-10 energy:         {:.1}%", top10_energy / total_energy * 100.0);
    println!("    Top-50 energy:         {:.1}%", activations[..50].iter().map(|a| a*a).sum::<f32>() / total_energy * 100.0);
    println!("    Top-200 energy:        {:.1}%", activations[..200].iter().map(|a| a*a).sum::<f32>() / total_energy * 100.0);
    println!("    Top-500 energy:        {:.1}%", activations[..500].iter().map(|a| a*a).sum::<f32>() / total_energy * 100.0);
    println!("    Top-2000 energy:       {:.1}%\n", activations[..2000].iter().map(|a| a*a).sum::<f32>() / total_energy * 100.0);

    // Walk FFN at various K
    println!("  K       cosine    max_diff   energy%   time/layer");
    println!("  ───── ──────── ────────── ──────── ──────────");
    let gate_index: &dyn larql_vindex::GateIndex = &index;
    for k in [10, 50, 100, 200, 500, 1000, 2000, 4000, 8192, intermediate] {
        let walk_ffn = larql_inference::vindex::WalkFfn::new(weights, &index, k);
        let t = std::time::Instant::now();
        let (walk_out, _) = larql_inference::forward::run_ffn(weights, &h_post_attn, 14, &walk_ffn, false);
        let walk_ms = t.elapsed().as_secs_f64() * 1000.0;

        let walk_row = walk_out.row(walk_out.shape()[0] - 1);
        let walk_norm = larql_compute::norm(&walk_row);

        let cosine = larql_compute::dot(&dense_row, &walk_row) / (dense_norm * walk_norm + 1e-10);
        let max_diff: f32 = dense_row.iter().zip(walk_row.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        // Energy captured
        let energy_pct = if k < intermediate {
            activations[..k].iter().map(|a| a*a).sum::<f32>() / total_energy * 100.0
        } else { 100.0 };

        println!("  {k:>5}  {cosine:>8.6}  {max_diff:>9.4}  {energy_pct:>6.1}%  {walk_ms:>8.1}ms");
    }

    // Run full inference at different K values
    println!("\n=== Full Inference Accuracy vs K ===\n");
    println!("  K       prediction  prob     match_dense");
    println!("  ───── ──────────── ─────── ─────────────");

    // Dense baseline
    let dense_result = larql_inference::predict(weights, model.tokenizer(), &token_ids, 5);
    let (dense_tok, dense_prob) = dense_result.predictions.first()
        .map(|(t, p)| (t.clone(), *p)).unwrap_or(("?".into(), 0.0));
    println!("  dense  {dense_tok:>12}  {:.1}%    (baseline)", dense_prob * 100.0);

    for k in [50, 200, 500, 2000, 8192] {
        let walk_ffn = larql_inference::vindex::WalkFfn::new(weights, &index, k);
        let walk_graph = larql_inference::WalkLayerGraph { ffn: &walk_ffn, backend: None };
        let result = larql_inference::predict_with_graph(
            weights, model.tokenizer(), &token_ids, 5, &walk_graph,
        );
        let (tok, prob) = result.predictions.first()
            .map(|(t, p)| (t.clone(), *p)).unwrap_or(("?".into(), 0.0));
        let matches = tok == dense_tok;
        println!("  {k:>5}  {tok:>12}  {:.1}%    {}", prob * 100.0, if matches { "YES" } else { "NO" });
    }

    println!("\n=== Done ===");
}
