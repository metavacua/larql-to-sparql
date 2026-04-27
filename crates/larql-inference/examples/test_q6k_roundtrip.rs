//! Quick Q6_K roundtrip test

fn main() {
    let data: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.1).collect();
    println!("Input: {} values, max={:.2}", data.len(), data.iter().fold(0.0f32, |a, &b| a.max(b.abs())));

    let q6k = larql_compute::cpu::ops::q4_common::quantize_q6_k(&data);
    println!("Quantized: {} bytes (expected 210)", q6k.len());

    // First few bytes
    println!("First 10 bytes: {:?}", &q6k[..10]);
    println!("Scale bytes [208..210]: {:?}", &q6k[208..210]);

    // Dequantize via ggml
    let deq = larql_models::quant::ggml::dequantize(&q6k, larql_models::quant::ggml::TYPE_Q6_K, 256);
    match deq {
        Ok(ref d) => {
            let nz = d.iter().filter(|v| v.abs() > 1e-6).count();
            let max_err: f32 = data.iter().zip(d.iter()).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
            println!("Dequantized: {} values, nonzero={}, max_err={:.4}", d.len(), nz, max_err);
            println!("First 5: {:?}", &d[..5]);
        }
        Err(e) => println!("Dequantize FAILED: {}", e),
    }

    // Also try dequantize_q6_k directly
    let deq2 = larql_models::quant::ggml::dequantize_q6_k(&q6k, 256);
    match deq2 {
        Ok(ref d) => {
            let nz = d.iter().filter(|v| v.abs() > 1e-6).count();
            println!("dequantize_q6_k: {} values, nonzero={}", d.len(), nz);
            println!("First 5: {:?}", &d[..5]);
        }
        Err(e) => println!("dequantize_q6_k FAILED: {}", e),
    }
}
