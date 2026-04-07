use larql_inference::forward::{forward_to_layer, embed_tokens_pub};
use larql_models::{load_model_dir, resolve_model_path};

fn main() {
    let path = resolve_model_path("google/gemma-4-E2B-it").expect("resolve failed");
    let weights = load_model_dir(&path).expect("load failed");

    let token_ids: Vec<u32> = vec![818, 5279, 529, 7001, 563];

    // Compare norms layer by layer vs HF
    // HF: L0=39.58, L1=35.35, L2=22.63, L3=27.22, L4=52.55, L5=81.40,
    //     L10=59.95, L15=68.49, L20=65.15, L25=65.46, L30=90.94, L34=62.45
    let hf_norms = [
        39.58, 35.35, 22.63, 27.22, 52.55, 81.40,
        69.49, 63.98, 62.36, 68.07, 59.95,
        46.83, 34.30, 52.89, 37.28, 68.49,
        72.90, 68.79, 64.46, 71.57, 65.15,
        57.99, 58.17, 60.10, 55.17, 65.46,
        56.64, 58.42, 70.98, 92.74, 90.94,
        88.84, 82.76, 77.28, 62.45,
    ];

    for stop in 0..35 {
        let h = forward_to_layer(&weights, &token_ids, stop);
        let norm: f32 = h.row(4).iter().map(|x| x*x).sum::<f32>().sqrt();
        let hf = hf_norms.get(stop + 1).copied().unwrap_or(0.0);
        let pct = if hf > 0.0 { ((norm as f64 - hf) / hf * 100.0) } else { 0.0 };
        let marker = if pct.abs() > 5.0 { " <<<" } else { "" };
        println!("L{stop:2}: ours={norm:7.2}  HF={hf:7.2}  diff={pct:+6.1}%{marker}");
    }
}
