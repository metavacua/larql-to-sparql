//! Debug: quantize actual V weights and check roundtrip

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let vd = std::path::PathBuf::from("output/gemma3-4b-v2.vindex");

    // Read raw V weights from attn_weights.bin
    let file = std::fs::File::open(vd.join("attn_weights.bin"))?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };

    // V proj L0: offset=31457280, length=10485760, shape=[1024, 2560]
    let offset = 31457280;
    let n_floats = 1024 * 2560;
    let f32_data = unsafe {
        std::slice::from_raw_parts(
            mmap[offset..offset + n_floats * 4].as_ptr() as *const f32,
            n_floats,
        )
    };

    println!("V proj L0 f32 data:");
    let amax = f32_data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let nz = f32_data.iter().filter(|v| v.abs() > 1e-10).count();
    println!("  n={}, nonzero={}, amax={:.6}", n_floats, nz, amax);
    println!("  first 5: {:?}", &f32_data[..5]);

    // Quantize to Q6_K
    let padded_len = n_floats.div_ceil(256) * 256;
    let mut padded = f32_data.to_vec();
    padded.resize(padded_len, 0.0);

    let q6k = larql_compute::cpu::ops::q4_common::quantize_q6_k(&padded);
    println!("\nQ6_K quantized: {} bytes", q6k.len());

    // Check the d scale of first superblock
    let d_bytes = &q6k[208..210];
    let d = larql_models::quant::half::f16_to_f32(u16::from_le_bytes([d_bytes[0], d_bytes[1]]));
    println!("First superblock d: {:.8} (bytes: {:02x} {:02x})", d, d_bytes[0], d_bytes[1]);

    // First 256 floats amax
    let first_256_amax = f32_data[..256].iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    println!("First 256 values amax: {:.6}", first_256_amax);
    println!("Expected d = amax/32 = {:.8}", first_256_amax / 32.0);

    // Dequantize
    let deq = larql_models::quant::ggml::dequantize_q6_k(&q6k, padded_len)?;
    let deq_nz = deq[..n_floats].iter().filter(|v| v.abs() > 1e-10).count();
    let max_err: f32 = f32_data.iter().zip(deq.iter()).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
    println!("\nRoundtrip: nonzero={}/{}, max_err={:.6}", deq_nz, n_floats, max_err);
    println!("Dequantized first 5: {:?}", &deq[..5]);

    // NOW compare with what's in the q4k file
    let q4k_file = std::fs::File::open(vd.join("attn_weights_q4k.bin"))?;
    let q4k_mmap = unsafe { memmap2::Mmap::map(&q4k_file)? };
    // V proj in q4k file: offset=4546560, length=2150400
    let q4k_v = &q4k_mmap[4546560..4546560 + 2150400];
    let q4k_d_bytes = &q4k_v[208..210];
    let q4k_d = larql_models::quant::half::f16_to_f32(u16::from_le_bytes([q4k_d_bytes[0], q4k_d_bytes[1]]));
    println!("\nOn-disk Q4K file V scale: {:.8} (bytes: {:02x} {:02x})", q4k_d, q4k_d_bytes[0], q4k_d_bytes[1]);
    println!("Fresh quantize scale:    {:.8}", d);
    println!("Match: {}", (d - q4k_d).abs() < 1e-10);

    // Check if the stored data matches fresh quantization
    let bytes_match = q6k[..2150400] == q4k_v[..2150400];
    println!("Byte-for-byte match (fresh vs disk): {}", bytes_match);
    if !bytes_match {
        // Find first difference
        for i in 0..2150400 {
            if q6k[i] != q4k_v[i] {
                println!("First diff at byte {}: fresh={:02x}, disk={:02x}", i, q6k[i], q4k_v[i]);
                break;
            }
        }
    }

    Ok(())
}
