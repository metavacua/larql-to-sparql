// Quick latency benchmark: forward_to_layer vs generate_cached timing
// Run as: cargo test --test bench_probe_latency --release -- --nocapture
use std::time::Instant;
use larql_inference::{encode_prompt, forward::forward_to_layer, InferenceModel, WeightFfn};
use larql_inference::forward::generate_cached_constrained;

#[test]
fn bench_probe_vs_generate() {
    let mid = std::env::var("LARQL_MODEL").unwrap_or_else(|_| "google/gemma-3-4b-it".to_string());
    let model = match InferenceModel::load(&mid) {
        Ok(m) => m,
        Err(e) => { eprintln!("skip: {e}"); return; }
    };
    let prompt = "What is the GCD of 144 and 60?";
    let ids = encode_prompt(model.tokenizer(), &*model.weights().arch, prompt).unwrap();
    let probe_layer = 5usize;

    // Warm up
    let _ = forward_to_layer(model.weights(), &ids, probe_layer);

    // Benchmark probe (forward_to_layer at L5)
    const N: usize = 5;
    let t0 = Instant::now();
    for _ in 0..N {
        let _ = forward_to_layer(model.weights(), &ids, probe_layer);
    }
    let probe_ms = t0.elapsed().as_millis() as f64 / N as f64;

    // Benchmark full generate (ids_gen, chat-wrapped)
    let wrapped = format!("<start_of_turn>user\nRespond with ONLY a JSON object.\n\nQuestion: {prompt}\n<end_of_turn>\n<start_of_turn>model\n");
    let ids_gen = encode_prompt(model.tokenizer(), &*model.weights().arch, &wrapped).unwrap();
    let ffn = WeightFfn { weights: model.weights() };

    let t1 = Instant::now();
    let mut out = String::new();
    generate_cached_constrained(
        model.weights(), model.tokenizer(), &ffn, &ids_gen, 64,
        |_, _| {}, |_, tok| out.push_str(tok),
    );
    let gen_ms = t1.elapsed().as_millis() as f64;

    eprintln!("model:       {mid}");
    eprintln!("probe L{probe_layer}:   {probe_ms:.0} ms  ({} tokens)", ids.len());
    eprintln!("generate:    {gen_ms:.0} ms  ({} prompt tokens, 64 max new)", ids_gen.len());
    eprintln!("ratio:       {:.1}×", gen_ms / probe_ms);
    eprintln!("probe share: {:.1}%", 100.0 * probe_ms / gen_ms);
}
