fn main() {
    use larql_compute::{ComputeBackend, cpu_backend, default_backend};
    use larql_compute::cpu::q4::{quantize_q4_0, quantize_to_q8};

    let hidden = 256;
    let rows = 32;
    let matrix: Vec<f32> = (0..rows * hidden).map(|i| (i as f32 * 0.001).cos()).collect();
    let q4_data = quantize_q4_0(&matrix);
    let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.01).sin()).collect();
    let (q8_x, q8_scales) = quantize_to_q8(&x);

    let cpu = cpu_backend();
    let gpu = default_backend();

    let cpu_result = cpu.q4_matvec(&q4_data, &q8_x, &q8_scales, rows, hidden).unwrap();
    let gpu_result = gpu.q4_matvec(&q4_data, &q8_x, &q8_scales, rows, hidden).unwrap();

    let max_diff: f32 = cpu_result.iter().zip(gpu_result.iter())
        .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);

    println!("Small matrix [32, 256]:");
    println!("  CPU[0..4]: {:?}", &cpu_result[..4]);
    println!("  GPU[0..4]: {:?}", &gpu_result[..4]);
    println!("  Max diff: {max_diff:.2e}");

    // Now test at bench_full dimensions
    let hidden = 2560;
    let rows = 10240;
    let matrix: Vec<f32> = (0..rows * hidden).map(|i| (i as f32 * 0.0001).cos()).collect();
    let q4_data = quantize_q4_0(&matrix);
    let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.001).sin()).collect();
    let (q8_x, q8_scales) = quantize_to_q8(&x);

    let cpu_result = cpu.q4_matvec(&q4_data, &q8_x, &q8_scales, rows, hidden).unwrap();
    let gpu_result = gpu.q4_matvec(&q4_data, &q8_x, &q8_scales, rows, hidden).unwrap();

    let max_diff: f32 = cpu_result.iter().zip(gpu_result.iter())
        .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);

    println!("\nLarge matrix [10240, 2560]:");
    println!("  CPU[0..4]: {:?}", &cpu_result[..4]);
    println!("  GPU[0..4]: {:?}", &gpu_result[..4]);
    println!("  Max diff: {max_diff:.2e}");
    println!("  OK: {}", if max_diff < 1.0 { "yes" } else { "NO" });
}
