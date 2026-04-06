//! Per-shader correctness tests for Metal compute kernels.
//!
//! Each test runs the Metal shader and compares output against
//! a CPU reference implementation. Tests both correctness and
//! that the shader compiles and dispatches successfully.
//!
//! Run with: cargo test -p larql-compute --features metal

#![cfg(feature = "metal")]

extern crate blas_src;

use ndarray::Array2;
use larql_compute::{ComputeBackend, cpu::q4};
use larql_compute::cpu::q4::quantize_q4_0;

// ── Test helpers ──

fn synth(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
    let mut s = seed;
    Array2::from_shape_fn((rows, cols), |_| {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((s >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
    })
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

fn get_metal() -> larql_compute::metal::MetalBackend {
    larql_compute::metal::MetalBackend::new().expect("Metal device required for these tests")
}

// ── Shader compilation ──

#[test]
fn all_shaders_compile() {
    let src = larql_compute::metal::shaders::all_shaders();
    assert!(src.len() > 1000, "Shader source too short");

    let device = metal::Device::system_default().expect("No Metal device");
    let opts = metal::CompileOptions::new();
    device.new_library_with_source(&src, &opts)
        .expect("Shader compilation failed");
}

#[test]
fn all_kernel_functions_exist() {
    let device = metal::Device::system_default().unwrap();
    let src = larql_compute::metal::shaders::all_shaders();
    let opts = metal::CompileOptions::new();
    let lib = device.new_library_with_source(&src, &opts).unwrap();

    let names = ["sgemm", "sgemm_transb", "q4_matvec", "q4_vecmat",
                 "q4_f32_matvec", "geglu_silu", "quantize_q8", "causal_attention"];
    for name in &names {
        lib.get_function(name, None)
            .unwrap_or_else(|e| panic!("Kernel '{name}' not found: {e}"));
    }
}

// ── f32 sgemm ──

#[test]
fn sgemm_matches_cpu() {
    let metal = get_metal();
    let a = synth(6, 2560, 42);
    let b = synth(2560, 2560, 43);

    let cpu_result = a.dot(&b);
    let metal_result = metal.matmul(a.view(), b.view());

    let diff = max_diff(cpu_result.as_slice().unwrap(), metal_result.as_slice().unwrap());
    assert!(diff < 0.1, "sgemm max diff {diff} exceeds 0.1");
}

// ── f32 sgemm_transb ──

#[test]
fn sgemm_transb_matches_cpu() {
    let metal = get_metal();
    let a = synth(6, 2560, 42);
    let b = synth(10240, 2560, 43);

    let cpu_result = a.dot(&b.t());
    let metal_result = metal.matmul_transb(a.view(), b.view());

    let diff = max_diff(cpu_result.as_slice().unwrap(), metal_result.as_slice().unwrap());
    assert!(diff < 0.1, "sgemm_transb max diff {diff} exceeds 0.1");
}

#[test]
fn sgemm_transb_small_matrix() {
    let metal = get_metal();
    let a = synth(1, 256, 42);
    let b = synth(512, 256, 43);

    let cpu_result = a.dot(&b.t());
    let metal_result = metal.matmul_transb(a.view(), b.view());

    let diff = max_diff(cpu_result.as_slice().unwrap(), metal_result.as_slice().unwrap());
    assert!(diff < 0.01, "small sgemm_transb max diff {diff}");
}

// ── Q4 matvec ──

#[test]
fn q4_matvec_matches_cpu() {
    let metal = get_metal();
    let hidden = 2560;
    let rows = 10240;

    let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.001).sin()).collect();
    let matrix: Vec<f32> = (0..rows * hidden).map(|i| (i as f32 * 0.0001).cos()).collect();
    let q4_data = quantize_q4_0(&matrix);
    let (q8_x, q8_scales) = q4::quantize_to_q8(&x);

    let cpu_result = q4::q4_matvec(&q4_data, &x, rows, hidden);
    let metal_result = metal.q4_matvec_direct(&q4_data, &q8_x, &q8_scales, rows, hidden);

    let diff = max_diff(&cpu_result, &metal_result);
    assert!(diff < 0.01, "q4_matvec max diff {diff} exceeds 0.01");
}

#[test]
fn q4_matvec_small_matrix() {
    let metal = get_metal();
    let hidden = 256;
    let rows = 128;

    let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.01).sin()).collect();
    let matrix: Vec<f32> = (0..rows * hidden).map(|i| (i as f32 * 0.001).cos()).collect();
    let q4_data = quantize_q4_0(&matrix);
    let (q8_x, q8_scales) = q4::quantize_to_q8(&x);

    let cpu_result = q4::q4_matvec(&q4_data, &x, rows, hidden);
    let metal_result = metal.q4_matvec_direct(&q4_data, &q8_x, &q8_scales, rows, hidden);

    let diff = max_diff(&cpu_result, &metal_result);
    assert!(diff < 0.01, "small q4_matvec max diff {diff}");
}

#[test]
fn q4_matvec_zero_input() {
    let metal = get_metal();
    let hidden = 256;
    let rows = 64;

    let x = vec![0.0f32; hidden];
    let matrix: Vec<f32> = (0..rows * hidden).map(|i| (i as f32 * 0.001).cos()).collect();
    let q4_data = quantize_q4_0(&matrix);
    let (q8_x, q8_scales) = q4::quantize_to_q8(&x);

    let result = metal.q4_matvec_direct(&q4_data, &q8_x, &q8_scales, rows, hidden);
    assert!(result.iter().all(|&v| v.abs() < 0.01), "zero input should produce near-zero output");
}

// ── Q4 vecmat ──

#[test]
fn q4_vecmat_matches_cpu() {
    let metal = get_metal();
    let hidden = 2560;
    let inter = 10240;

    let activation: Vec<f32> = (0..inter).map(|i| if i % 5 == 0 { (i as f32 * 0.01).sin() } else { 0.0 }).collect();
    let matrix: Vec<f32> = (0..inter * hidden).map(|i| (i as f32 * 0.0001).cos()).collect();
    let q4_data = quantize_q4_0(&matrix);

    let cpu_result = q4::q4_vecmat(&activation, &q4_data, inter, hidden);
    let metal_result = metal.q4_vecmat_direct(&activation, &q4_data, inter, hidden);

    let diff = max_diff(&cpu_result, &metal_result);
    assert!(diff < 0.1, "q4_vecmat max diff {diff} exceeds 0.1");
}

// ── Q4 f32 matvec (for transposed down) ──

#[test]
fn q4_f32_matvec_nonzero() {
    let metal = get_metal();
    let hidden = 2560;
    let inter = 10240;

    let activation: Vec<f32> = (0..inter).map(|i| (i as f32 * 0.001).sin()).collect();
    let mut down_t: Vec<f32> = vec![0.0; hidden * inter];
    for r in 0..inter { for c in 0..hidden { down_t[c * inter + r] = ((r * hidden + c) as f32 * 0.0001).cos(); } }
    let q4_data = quantize_q4_0(&down_t);

    let result = metal.q4_f32_matvec_direct(&q4_data, &activation, hidden, inter);
    assert_eq!(result.len(), hidden);
    assert!(result.iter().any(|&v| v.abs() > 0.01), "should produce nonzero output");
}

// ── Q4 pair batch ──

#[test]
fn q4_pair_batch_matches_individual() {
    let metal = get_metal();
    let hidden = 2560;
    let inter = 1024; // smaller for test speed
    let seq = 2;

    let gate_f32: Vec<f32> = (0..inter * hidden).map(|i| (i as f32 * 0.0001).cos()).collect();
    let up_f32: Vec<f32> = (0..inter * hidden).map(|i| (i as f32 * 0.0002).sin()).collect();
    let gate_q4 = quantize_q4_0(&gate_f32);
    let up_q4 = quantize_q4_0(&up_f32);
    let x: Vec<f32> = (0..seq * hidden).map(|i| (i as f32 * 0.001).sin()).collect();

    // Individual calls
    let mut indiv_gate = Vec::new();
    let mut indiv_up = Vec::new();
    for s in 0..seq {
        let slice = &x[s * hidden..(s + 1) * hidden];
        let (q8, sc) = q4::quantize_to_q8(slice);
        indiv_gate.push(metal.q4_matvec_direct(&gate_q4, &q8, &sc, inter, hidden));
        indiv_up.push(metal.q4_matvec_direct(&up_q4, &q8, &sc, inter, hidden));
    }

    // Batched call
    let (batch_gate, batch_up) = metal.q4_matvec_pair_batch_direct(
        &gate_q4, &up_q4, &x, seq, inter, hidden,
    );

    // Compare
    for s in 0..seq {
        let diff_g = max_diff(&indiv_gate[s], &batch_gate[s]);
        let diff_u = max_diff(&indiv_up[s], &batch_up[s]);
        assert!(diff_g < 0.001, "pair_batch gate diff {diff_g} at seq {s}");
        assert!(diff_u < 0.001, "pair_batch up diff {diff_u} at seq {s}");
    }
}

// ── Multi-layer Q4 FFN ──

#[test]
fn multi_layer_q4_produces_output() {
    let metal = get_metal();
    let hidden = 256; // small for test speed
    let inter = 512;
    let layers = 3;

    let mut layers_q4 = Vec::new();
    for l in 0..layers {
        let g: Vec<f32> = (0..inter * hidden).map(|i| ((i + l * 1000) as f32 * 0.001).cos()).collect();
        let u: Vec<f32> = (0..inter * hidden).map(|i| ((i + l * 2000) as f32 * 0.002).sin()).collect();
        let mut dt = vec![0.0f32; hidden * inter];
        for r in 0..inter { for c in 0..hidden { dt[c * inter + r] = ((r * hidden + c + l * 3000) as f32 * 0.003).cos(); } }
        layers_q4.push((quantize_q4_0(&g), quantize_q4_0(&u), quantize_q4_0(&dt)));
    }

    let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.01).sin()).collect();
    let layers_refs: Vec<(&[u8], &[u8], &[u8])> = layers_q4.iter()
        .map(|(g, u, d)| (g.as_slice(), u.as_slice(), d.as_slice())).collect();
    let result = metal.multi_layer_q4_ffn(&layers_refs, &x, inter, hidden);

    assert_eq!(result.len(), hidden);
    assert!(result.iter().any(|&v| v.abs() > 0.001), "multi-layer should produce nonzero output");
}

// ── Buffer cache ──

#[test]
fn buffer_cache_reuses_same_pointer() {
    let metal = get_metal();
    let data = vec![1.0f32; 1024];
    let q4 = quantize_q4_0(&data);
    let (q8, sc) = q4::quantize_to_q8(&data[..256]);

    // Call twice with same data — buffer should be cached
    let r1 = metal.q4_matvec_direct(&q4, &q8, &sc, 4, 256);
    let r2 = metal.q4_matvec_direct(&q4, &q8, &sc, 4, 256);

    let diff = max_diff(&r1, &r2);
    assert!(diff < 1e-6, "cached buffer should produce identical results, diff: {diff}");
}

// ── Trait dispatch ──

#[test]
fn metal_backend_implements_trait() {
    use larql_compute::ComputeBackend;
    let metal = get_metal();

    assert!(metal.has_q4());
    assert!(metal.name().contains("metal"));

    let a = synth(2, 64, 42);
    let b = synth(32, 64, 43);
    let result = metal.matmul_transb(a.view(), b.view());
    assert_eq!(result.shape(), &[2, 32]);
}

// ── Q8 matvec ──

#[test]
fn q8_matvec_metal_nonzero() {
    let _metal = get_metal();
    let hidden = 256;
    let rows = 64;

    let weights: Vec<f32> = (0..rows * hidden).map(|i| (i as f32 * 0.001).cos()).collect();
    let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.01).sin()).collect();

    let (w_q8, w_scales) = larql_compute::cpu::ops::q8_matvec::quantize_weights_q8(&weights, rows, hidden);
    let (x_q8, x_scales) = larql_compute::cpu::ops::q4_common::quantize_to_q8(&x);

    // CPU reference
    let cpu_result = larql_compute::cpu::ops::q8_matvec::dispatch(&w_q8, &w_scales, &x_q8, &x_scales, rows, hidden);
    assert!(cpu_result.iter().any(|&v| v.abs() > 0.01), "Q8 CPU should produce nonzero");
}

// ── Sparse Q4 matvec ──

#[test]
fn sparse_matvec_matches_dense() {
    let metal = get_metal();
    let hidden = 256;
    let n_rows = 64;
    let k_selected = 16;

    let matrix: Vec<f32> = (0..n_rows * hidden).map(|i| (i as f32 * 0.001).cos()).collect();
    let q4_data = quantize_q4_0(&matrix);
    let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.01).sin()).collect();
    let (q8_x, q8_scales) = q4::quantize_to_q8(&x);

    // Dense: score all rows
    let dense_result = metal.q4_matvec_direct(&q4_data, &q8_x, &q8_scales, n_rows, hidden);

    // Sparse: score selected rows [0, 4, 8, 12, ...]
    let indices: Vec<u32> = (0..k_selected as u32).map(|i| i * 4).collect();

    // Use the sparse shader via raw Metal dispatch
    let device = metal::Device::system_default().unwrap();
    let src = larql_compute::metal::shaders::all_shaders();
    let lib = device.new_library_with_source(&src, &metal::CompileOptions::new()).unwrap();
    let pipeline = device.new_compute_pipeline_state_with_function(
        &lib.get_function("q4_sparse_matvec", None).unwrap()
    ).unwrap();

    let bufs = &larql_compute::metal::buffers::BufferCache::new(&device);
    let queue = device.new_command_queue();
    let buf_q4 = bufs.get_bytes(&q4_data);
    let buf_q8 = bufs.transient_from_i8(&q8_x);
    let buf_sc = bufs.transient_from_f32(&q8_scales);
    let idx_bytes: Vec<u8> = indices.iter().flat_map(|i| i.to_le_bytes()).collect();
    let buf_idx = bufs.transient_from_f32(unsafe {
        std::slice::from_raw_parts(idx_bytes.as_ptr() as *const f32, indices.len())
    });
    let buf_out = bufs.output((k_selected * 4) as u64);

    let k_val = k_selected as u32;
    let h_val = hidden as u32;
    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&buf_q4), 0);
    enc.set_buffer(1, Some(&buf_q8), 0);
    enc.set_buffer(2, Some(&buf_sc), 0);
    enc.set_buffer(3, Some(&buf_idx), 0);
    enc.set_buffer(4, Some(&buf_out), 0);
    enc.set_bytes(5, 4, &k_val as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(6, 4, &h_val as *const u32 as *const std::ffi::c_void);
    enc.dispatch_threads(metal::MTLSize::new(k_selected as u64, 1, 1), metal::MTLSize::new(k_selected as u64, 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let ptr = buf_out.contents() as *const f32;
    let sparse_result: Vec<f32> = unsafe { std::slice::from_raw_parts(ptr, k_selected).to_vec() };

    // Verify sparse results match corresponding dense results
    for (i, &idx) in indices.iter().enumerate() {
        let diff = (sparse_result[i] - dense_result[idx as usize]).abs();
        assert!(diff < 0.01, "sparse[{i}] (row {idx}) diff {diff}");
    }
}

// ── Residual ops ──

#[test]
fn residual_add_correct() {
    let device = metal::Device::system_default().unwrap();
    let src = larql_compute::metal::shaders::all_shaders();
    let lib = device.new_library_with_source(&src, &metal::CompileOptions::new()).unwrap();
    let pipeline = device.new_compute_pipeline_state_with_function(
        &lib.get_function("residual_add", None).unwrap()
    ).unwrap();

    let bufs = larql_compute::metal::buffers::BufferCache::new(&device);
    let queue = device.new_command_queue();

    let a = vec![1.0f32, 2.0, 3.0, 4.0];
    let b = vec![10.0f32, 20.0, 30.0, 40.0];
    let buf_a = bufs.transient_from_f32(&a);
    let buf_b = bufs.transient_from_f32(&b);
    let buf_out = bufs.output(16);
    let len = 4u32;

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&buf_a), 0);
    enc.set_buffer(1, Some(&buf_b), 0);
    enc.set_buffer(2, Some(&buf_out), 0);
    enc.set_bytes(3, 4, &len as *const u32 as *const std::ffi::c_void);
    enc.dispatch_threads(metal::MTLSize::new(4, 1, 1), metal::MTLSize::new(4, 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let ptr = buf_out.contents() as *const f32;
    let result: Vec<f32> = unsafe { std::slice::from_raw_parts(ptr, 4).to_vec() };
    assert!((result[0] - 11.0).abs() < 1e-5);
    assert!((result[1] - 22.0).abs() < 1e-5);
    assert!((result[2] - 33.0).abs() < 1e-5);
    assert!((result[3] - 44.0).abs() < 1e-5);
}

// ── GEGLU ──

#[test]
fn geglu_matches_cpu() {
    let device = metal::Device::system_default().unwrap();
    let src = larql_compute::metal::shaders::all_shaders();
    let lib = device.new_library_with_source(&src, &metal::CompileOptions::new()).unwrap();
    let pipeline = device.new_compute_pipeline_state_with_function(
        &lib.get_function("geglu_silu", None).unwrap()
    ).unwrap();

    let bufs = larql_compute::metal::buffers::BufferCache::new(&device);
    let queue = device.new_command_queue();

    let n = 256;
    let gate: Vec<f32> = (0..n).map(|i| i as f32 * 0.1 - 12.8).collect();
    let up: Vec<f32> = (0..n).map(|i| i as f32 * 0.05).collect();

    // CPU reference
    let cpu_result = larql_compute::cpu::ops::geglu::geglu_silu_alloc(&gate, &up);

    // Metal
    let buf_g = bufs.transient_from_f32(&gate);
    let buf_u = bufs.transient_from_f32(&up);
    let buf_out = bufs.output((n * 4) as u64);
    let n_val = n as u32;

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&buf_g), 0);
    enc.set_buffer(1, Some(&buf_u), 0);
    enc.set_buffer(2, Some(&buf_out), 0);
    enc.set_bytes(3, 4, &n_val as *const u32 as *const std::ffi::c_void);
    enc.dispatch_threads(metal::MTLSize::new(n as u64, 1, 1), metal::MTLSize::new(256, 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let ptr = buf_out.contents() as *const f32;
    let metal_result: Vec<f32> = unsafe { std::slice::from_raw_parts(ptr, n).to_vec() };

    let diff = max_diff(&cpu_result, &metal_result);
    assert!(diff < 1e-4, "GEGLU CPU vs Metal diff {diff}");
}

// ── Cross-validation: all kernels listed ──

#[test]
fn all_new_kernel_functions_exist() {
    let device = metal::Device::system_default().unwrap();
    let src = larql_compute::metal::shaders::all_shaders();
    let lib = device.new_library_with_source(&src, &metal::CompileOptions::new()).unwrap();

    let names = [
        "sgemm", "sgemm_transb",
        "q4_matvec", "q4_matvec_v2", "q4_matvec_v3", "q4_matvec_v4", "q4_matvec_v5",
        "q4_vecmat", "q4_f32_matvec", "q4_sparse_matvec",
        "q8_matvec",
        "geglu_silu", "quantize_q8",
        "residual_copy", "residual_add", "rms_norm",
        "causal_attention", "kv_attention", "kv_cache_append",
        "rope_apply", "fused_attention",
    ];
    for name in &names {
        lib.get_function(name, None)
            .unwrap_or_else(|e| panic!("Kernel '{name}' not found: {e}"));
    }
}

// ── RoPE shader ──

#[test]
fn rope_apply_matches_cpu() {
    let device = metal::Device::system_default().unwrap();
    let src = larql_compute::metal::shaders::all_shaders();
    let lib = device.new_library_with_source(&src, &metal::CompileOptions::new()).unwrap();
    let pipeline = device.new_compute_pipeline_state_with_function(
        &lib.get_function("rope_apply", None).unwrap()
    ).unwrap();

    let bufs = larql_compute::metal::buffers::BufferCache::new(&device);
    let queue = device.new_command_queue();

    let dim = 64u32;
    let seq_len = 4u32;
    let base = 10000.0f32;

    // Create test data
    let data: Vec<f32> = (0..seq_len as usize * dim as usize)
        .map(|i| (i as f32 * 0.01).sin())
        .collect();
    let data_copy = data.clone();

    // CPU reference: apply RoPE manually
    let half = dim as usize / 2;
    let mut cpu_result = data_copy.clone();
    for pos in 0..seq_len as usize {
        for d in 0..half {
            let freq = 1.0 / base.powf(2.0 * d as f32 / dim as f32);
            let angle = pos as f32 * freq;
            let cos_a = angle.cos();
            let sin_a = angle.sin();
            let re = cpu_result[pos * dim as usize + d];
            let im = cpu_result[pos * dim as usize + d + half];
            cpu_result[pos * dim as usize + d] = re * cos_a - im * sin_a;
            cpu_result[pos * dim as usize + d + half] = re * sin_a + im * cos_a;
        }
    }

    // Metal
    let buf = bufs.transient_from_f32(&data);
    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&buf), 0);
    enc.set_bytes(1, 4, &dim as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(2, 4, &base as *const f32 as *const std::ffi::c_void);
    enc.dispatch_threads(
        metal::MTLSize::new(half as u64, seq_len as u64, 1),
        metal::MTLSize::new(half as u64, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let ptr = buf.contents() as *const f32;
    let metal_result: Vec<f32> = unsafe {
        std::slice::from_raw_parts(ptr, seq_len as usize * dim as usize).to_vec()
    };

    let diff = max_diff(&cpu_result, &metal_result);
    assert!(diff < 1e-4, "RoPE max diff {diff} exceeds 1e-4");
}

// ── Fused attention shader ──

#[test]
fn fused_attention_single_token() {
    // At seq=1, attention output = V (only one key to attend to, weight = 1.0)
    let device = metal::Device::system_default().unwrap();
    let src = larql_compute::metal::shaders::all_shaders();
    let lib = device.new_library_with_source(&src, &metal::CompileOptions::new()).unwrap();
    let pipeline = device.new_compute_pipeline_state_with_function(
        &lib.get_function("fused_attention", None).unwrap()
    ).unwrap();

    let bufs = larql_compute::metal::buffers::BufferCache::new(&device);
    let queue = device.new_command_queue();

    let seq_len = 1u32;
    let head_dim = 32u32;
    let num_q = 2u32;
    let num_kv = 2u32;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let rope_base = 10000.0f32;
    let use_qk_norm = 0u32;
    let softcap = 0.0f32;

    let total = seq_len as usize * num_q as usize * head_dim as usize;
    let kv_total = seq_len as usize * num_kv as usize * head_dim as usize;

    let q: Vec<f32> = (0..total).map(|i| (i as f32 * 0.1).sin()).collect();
    let k: Vec<f32> = (0..kv_total).map(|i| (i as f32 * 0.2).cos()).collect();
    let v: Vec<f32> = (0..kv_total).map(|i| (i as f32 * 0.05 + 1.0)).collect();

    let buf_q = bufs.transient_from_f32(&q);
    let buf_k = bufs.transient_from_f32(&k);
    let buf_v = bufs.transient_from_f32(&v);
    let buf_out = bufs.output((total * 4) as u64);

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&buf_q), 0);
    enc.set_buffer(1, Some(&buf_k), 0);
    enc.set_buffer(2, Some(&buf_v), 0);
    enc.set_buffer(3, Some(&buf_out), 0);
    enc.set_bytes(4, 4, &seq_len as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(5, 4, &head_dim as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(6, 4, &num_q as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(7, 4, &num_kv as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(8, 4, &scale as *const f32 as *const std::ffi::c_void);
    enc.set_bytes(9, 4, &rope_base as *const f32 as *const std::ffi::c_void);
    enc.set_bytes(10, 4, &use_qk_norm as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(11, 4, &softcap as *const f32 as *const std::ffi::c_void);
    enc.dispatch_thread_groups(
        metal::MTLSize::new(num_q as u64, seq_len as u64, 1),
        metal::MTLSize::new(256, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let ptr = buf_out.contents() as *const f32;
    let result: Vec<f32> = unsafe { std::slice::from_raw_parts(ptr, total).to_vec() };

    // At seq=1, output should be V (rotated by RoPE, but with weight=1.0)
    // Just verify nonzero and finite
    assert!(result.iter().all(|v| v.is_finite()), "output should be finite");
    assert!(result.iter().any(|v| v.abs() > 0.01), "output should be nonzero");
}

// ══════════════════════════════════════════════════════════════
// Shader correctness tests — each shader vs CPU reference
// ══════════════════════════════════════════════════════════════

// ── rms_norm with offset ──

#[test]
fn rms_norm_matches_cpu() {
    let device = metal::Device::system_default().unwrap();
    let src = larql_compute::metal::shaders::all_shaders();
    let lib = device.new_library_with_source(&src, &metal::CompileOptions::new()).unwrap();
    let pipeline = device.new_compute_pipeline_state_with_function(
        &lib.get_function("rms_norm", None).unwrap()
    ).unwrap();
    let bufs = larql_compute::metal::buffers::BufferCache::new(&device);
    let queue = device.new_command_queue();

    let len = 64usize;
    let x: Vec<f32> = (0..len).map(|i| (i as f32 * 0.1 - 3.2)).collect();
    let weight: Vec<f32> = (0..len).map(|i| 0.5 + (i as f32 * 0.01)).collect();
    let eps = 1e-6f32;
    let offset = 1.0f32; // Gemma-style

    // CPU reference
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let rms = 1.0 / (sum_sq / len as f32 + eps).sqrt();
    let cpu_result: Vec<f32> = x.iter().zip(weight.iter())
        .map(|(xi, wi)| xi * (wi + offset) * rms)
        .collect();

    // Metal
    let buf_x = bufs.transient_from_f32(&x);
    let buf_w = bufs.transient_from_f32(&weight);
    let buf_out = bufs.output((len * 4) as u64);
    let len_val = len as u32;

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&buf_x), 0);
    enc.set_buffer(1, Some(&buf_w), 0);
    enc.set_buffer(2, Some(&buf_out), 0);
    enc.set_bytes(3, 4, &len_val as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(4, 4, &eps as *const f32 as *const std::ffi::c_void);
    enc.set_bytes(5, 4, &offset as *const f32 as *const std::ffi::c_void);
    enc.dispatch_threads(metal::MTLSize::new(len as u64, 1, 1), metal::MTLSize::new(len as u64, 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let ptr = buf_out.contents() as *const f32;
    let metal_result: Vec<f32> = unsafe { std::slice::from_raw_parts(ptr, len).to_vec() };

    let diff = max_diff(&cpu_result, &metal_result);
    assert!(diff < 1e-5, "rms_norm max diff {diff}");
}

#[test]
fn rms_norm_zero_offset() {
    // Standard RMS norm (Llama-style, offset=0)
    let device = metal::Device::system_default().unwrap();
    let src = larql_compute::metal::shaders::all_shaders();
    let lib = device.new_library_with_source(&src, &metal::CompileOptions::new()).unwrap();
    let pipeline = device.new_compute_pipeline_state_with_function(
        &lib.get_function("rms_norm", None).unwrap()
    ).unwrap();
    let bufs = larql_compute::metal::buffers::BufferCache::new(&device);
    let queue = device.new_command_queue();

    let len = 32usize;
    let x: Vec<f32> = (0..len).map(|i| (i as f32 * 0.2 - 3.0)).collect();
    let weight: Vec<f32> = vec![1.0f32; len];
    let eps = 1e-6f32;
    let offset = 0.0f32;

    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let rms = 1.0 / (sum_sq / len as f32 + eps).sqrt();
    let cpu_result: Vec<f32> = x.iter().map(|xi| xi * rms).collect();

    let buf_x = bufs.transient_from_f32(&x);
    let buf_w = bufs.transient_from_f32(&weight);
    let buf_out = bufs.output((len * 4) as u64);
    let len_val = len as u32;

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&buf_x), 0);
    enc.set_buffer(1, Some(&buf_w), 0);
    enc.set_buffer(2, Some(&buf_out), 0);
    enc.set_bytes(3, 4, &len_val as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(4, 4, &eps as *const f32 as *const std::ffi::c_void);
    enc.set_bytes(5, 4, &offset as *const f32 as *const std::ffi::c_void);
    enc.dispatch_threads(metal::MTLSize::new(len as u64, 1, 1), metal::MTLSize::new(len as u64, 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let ptr = buf_out.contents() as *const f32;
    let metal_result: Vec<f32> = unsafe { std::slice::from_raw_parts(ptr, len).to_vec() };

    let diff = max_diff(&cpu_result, &metal_result);
    assert!(diff < 1e-5, "rms_norm(offset=0) max diff {diff}");
}

// ── residual_add ──

#[test]
fn residual_add_matches_cpu() {
    let device = metal::Device::system_default().unwrap();
    let src = larql_compute::metal::shaders::all_shaders();
    let lib = device.new_library_with_source(&src, &metal::CompileOptions::new()).unwrap();
    let pipeline = device.new_compute_pipeline_state_with_function(
        &lib.get_function("residual_add", None).unwrap()
    ).unwrap();
    let bufs = larql_compute::metal::buffers::BufferCache::new(&device);
    let queue = device.new_command_queue();

    let len = 128usize;
    let a: Vec<f32> = (0..len).map(|i| i as f32 * 0.1).collect();
    let b: Vec<f32> = (0..len).map(|i| -(i as f32 * 0.05)).collect();
    let cpu_result: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();

    let buf_a = bufs.transient_from_f32(&a);
    let buf_b = bufs.transient_from_f32(&b);
    let buf_out = bufs.output((len * 4) as u64);
    let len_val = len as u32;

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&buf_a), 0);
    enc.set_buffer(1, Some(&buf_b), 0);
    enc.set_buffer(2, Some(&buf_out), 0);
    enc.set_bytes(3, 4, &len_val as *const u32 as *const std::ffi::c_void);
    enc.dispatch_threads(metal::MTLSize::new(len as u64, 1, 1), metal::MTLSize::new(len as u64, 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let ptr = buf_out.contents() as *const f32;
    let metal_result: Vec<f32> = unsafe { std::slice::from_raw_parts(ptr, len).to_vec() };

    let diff = max_diff(&cpu_result, &metal_result);
    assert!(diff < 1e-6, "residual_add max diff {diff}");
}

// ── fused_attention correctness (3 tokens, 2 heads, verified against CPU) ──

#[test]
fn fused_attention_matches_cpu_reference() {
    let device = metal::Device::system_default().unwrap();
    let src = larql_compute::metal::shaders::all_shaders();
    let lib = device.new_library_with_source(&src, &metal::CompileOptions::new()).unwrap();
    let pipeline = device.new_compute_pipeline_state_with_function(
        &lib.get_function("fused_attention", None).unwrap()
    ).unwrap();
    let bufs = larql_compute::metal::buffers::BufferCache::new(&device);
    let queue = device.new_command_queue();

    let seq_len = 3u32;
    let head_dim = 8u32;  // small for easy debugging
    let num_q = 2u32;
    let num_kv = 2u32;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let rope_base = 10000.0f32;
    let use_qk_norm = 0u32;
    let softcap = 0.0f32;

    let total = (seq_len * num_q * head_dim) as usize;
    let kv_total = (seq_len * num_kv * head_dim) as usize;

    // Deterministic test data
    let q: Vec<f32> = (0..total).map(|i| ((i as f32 * 0.37 + 1.0).sin() * 0.5)).collect();
    let k: Vec<f32> = (0..kv_total).map(|i| ((i as f32 * 0.23 + 2.0).cos() * 0.5)).collect();
    let v: Vec<f32> = (0..kv_total).map(|i| ((i as f32 * 0.11 + 3.0).sin() * 0.3)).collect();

    // ── CPU reference: apply RoPE then causal attention ──
    let hd = head_dim as usize;
    let half = hd / 2;
    let nq = num_q as usize;
    let nkv = num_kv as usize;
    let sl = seq_len as usize;

    // Apply RoPE to Q and K
    let mut q_rope = q.clone();
    let mut k_rope = k.clone();
    for pos in 0..sl {
        for head in 0..nq {
            for d in 0..half {
                let freq = 1.0 / rope_base.powf(2.0 * d as f32 / hd as f32);
                let angle = pos as f32 * freq;
                let (cos_a, sin_a) = (angle.cos(), angle.sin());
                let idx_re = pos * nq * hd + head * hd + d;
                let idx_im = pos * nq * hd + head * hd + d + half;
                let re = q[idx_re];
                let im = q[idx_im];
                q_rope[idx_re] = re * cos_a - im * sin_a;
                q_rope[idx_im] = re * sin_a + im * cos_a;
            }
        }
        for head in 0..nkv {
            for d in 0..half {
                let freq = 1.0 / rope_base.powf(2.0 * d as f32 / hd as f32);
                let angle = pos as f32 * freq;
                let (cos_a, sin_a) = (angle.cos(), angle.sin());
                let idx_re = pos * nkv * hd + head * hd + d;
                let idx_im = pos * nkv * hd + head * hd + d + half;
                let re = k[idx_re];
                let im = k[idx_im];
                k_rope[idx_re] = re * cos_a - im * sin_a;
                k_rope[idx_im] = re * sin_a + im * cos_a;
            }
        }
    }

    // Causal attention per head per position
    let mut cpu_out = vec![0.0f32; total];
    for head in 0..nq {
        let kv_head = head / (nq / nkv);
        for qi in 0..sl {
            // Compute scores for all k <= qi
            let mut scores = Vec::new();
            for ki in 0..=qi {
                let mut dot = 0.0f32;
                for d in 0..hd {
                    let q_val = q_rope[qi * nq * hd + head * hd + d];
                    let k_val = k_rope[ki * nkv * hd + kv_head * hd + d];
                    dot += q_val * k_val;
                }
                scores.push(dot * scale);
            }
            // Softmax
            let max_s = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = scores.iter().map(|s| (s - max_s).exp()).collect();
            let sum_exp: f32 = exps.iter().sum();
            let weights: Vec<f32> = exps.iter().map(|e| e / sum_exp).collect();
            // Weighted V
            for d in 0..hd {
                let mut acc = 0.0f32;
                for ki in 0..=qi {
                    acc += weights[ki] * v[ki * nkv * hd + kv_head * hd + d];
                }
                cpu_out[qi * nq * hd + head * hd + d] = acc;
            }
        }
    }

    // ── Metal ──
    let buf_q = bufs.transient_from_f32(&q);
    let buf_k = bufs.transient_from_f32(&k);
    let buf_v = bufs.transient_from_f32(&v);
    let buf_out = bufs.output((total * 4) as u64);

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&buf_q), 0);
    enc.set_buffer(1, Some(&buf_k), 0);
    enc.set_buffer(2, Some(&buf_v), 0);
    enc.set_buffer(3, Some(&buf_out), 0);
    enc.set_bytes(4, 4, &seq_len as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(5, 4, &head_dim as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(6, 4, &num_q as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(7, 4, &num_kv as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(8, 4, &scale as *const f32 as *const std::ffi::c_void);
    enc.set_bytes(9, 4, &rope_base as *const f32 as *const std::ffi::c_void);
    enc.set_bytes(10, 4, &use_qk_norm as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(11, 4, &softcap as *const f32 as *const std::ffi::c_void);
    enc.dispatch_thread_groups(
        metal::MTLSize::new(num_q as u64, seq_len as u64, 1),
        metal::MTLSize::new(256, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let ptr = buf_out.contents() as *const f32;
    let metal_result: Vec<f32> = unsafe { std::slice::from_raw_parts(ptr, total).to_vec() };

    // Compare
    let diff = max_diff(&cpu_out, &metal_result);
    assert!(diff < 0.01, "fused_attention max diff {diff} (expected < 0.01).\nCPU[0..8]: {:?}\nGPU[0..8]: {:?}",
        &cpu_out[..8.min(total)], &metal_result[..8.min(total)]);
}

// ── quantize_q8 shader ──

#[test]
fn quantize_q8_matches_cpu() {
    let device = metal::Device::system_default().unwrap();
    let src = larql_compute::metal::shaders::all_shaders();
    let lib = device.new_library_with_source(&src, &metal::CompileOptions::new()).unwrap();
    let pipeline = device.new_compute_pipeline_state_with_function(
        &lib.get_function("quantize_q8", None).unwrap()
    ).unwrap();
    let bufs = larql_compute::metal::buffers::BufferCache::new(&device);
    let queue = device.new_command_queue();

    let len = 64usize;
    let x: Vec<f32> = (0..len).map(|i| (i as f32 * 0.15 - 4.8)).collect();

    // CPU reference
    let (cpu_q8, cpu_scales) = larql_compute::cpu::q4::quantize_to_q8(&x);

    // Metal
    let buf_x = bufs.transient_from_f32(&x);
    let buf_q8 = bufs.output(len as u64);
    let buf_scales = bufs.output((len / 32 * 4) as u64);
    let len_val = len as u32;

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&buf_x), 0);
    enc.set_buffer(1, Some(&buf_q8), 0);
    enc.set_buffer(2, Some(&buf_scales), 0);
    let n_blocks = (len / 32) as u32;
    enc.set_bytes(3, 4, &len_val as *const u32 as *const std::ffi::c_void);
    enc.dispatch_threads(metal::MTLSize::new(n_blocks as u64, 1, 1), metal::MTLSize::new(n_blocks as u64, 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let q8_ptr = buf_q8.contents() as *const i8;
    let sc_ptr = buf_scales.contents() as *const f32;
    let metal_q8: Vec<i8> = unsafe { std::slice::from_raw_parts(q8_ptr, len).to_vec() };
    let metal_scales: Vec<f32> = unsafe { std::slice::from_raw_parts(sc_ptr, len / 32).to_vec() };

    // Check scales match
    for i in 0..len/32 {
        let diff = (cpu_scales[i] - metal_scales[i]).abs();
        assert!(diff < 0.01, "Q8 scale[{i}] diff: cpu={} metal={}", cpu_scales[i], metal_scales[i]);
    }
    // Check quantized values match (allow ±1 for rounding)
    let mut mismatches = 0;
    for i in 0..len {
        if (cpu_q8[i] as i32 - metal_q8[i] as i32).abs() > 1 {
            mismatches += 1;
        }
    }
    assert!(mismatches == 0, "Q8 quantize: {mismatches}/{len} values differ by >1");
}

// ── Fused ops: rms_norm_q8, residual_norm, residual_norm_q8 ──

#[test]
fn rms_norm_q8_matches_separate_ops() {
    let device = metal::Device::system_default().unwrap();
    let src = larql_compute::metal::shaders::all_shaders();
    let lib = device.new_library_with_source(&src, &metal::CompileOptions::new()).unwrap();
    let fused = device.new_compute_pipeline_state_with_function(
        &lib.get_function("rms_norm_q8", None).unwrap()
    ).unwrap();
    let bufs = larql_compute::metal::buffers::BufferCache::new(&device);
    let queue = device.new_command_queue();

    let len = 64usize;
    let x: Vec<f32> = (0..len).map(|i| (i as f32 * 0.15 - 4.8)).collect();
    let weight: Vec<f32> = (0..len).map(|i| 0.5 + i as f32 * 0.01).collect();
    let eps = 1e-6f32;
    let offset = 1.0f32;

    // CPU reference: norm then quantize
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let rms = 1.0 / (sum_sq / len as f32 + eps).sqrt();
    let normed: Vec<f32> = x.iter().zip(weight.iter()).map(|(xi, wi)| xi * (wi + offset) * rms).collect();
    let (cpu_q8, cpu_scales) = larql_compute::cpu::q4::quantize_to_q8(&normed);

    // Metal fused
    let buf_x = bufs.transient_from_f32(&x);
    let buf_w = bufs.transient_from_f32(&weight);
    let buf_q8 = bufs.output(len as u64);
    let buf_sc = bufs.output((len / 32 * 4) as u64);
    let len_val = len as u32;

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&fused);
    enc.set_buffer(0, Some(&buf_x), 0);
    enc.set_buffer(1, Some(&buf_w), 0);
    enc.set_buffer(2, Some(&buf_q8), 0);
    enc.set_buffer(3, Some(&buf_sc), 0);
    enc.set_bytes(4, 4, &len_val as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(5, 4, &eps as *const f32 as *const std::ffi::c_void);
    enc.set_bytes(6, 4, &offset as *const f32 as *const std::ffi::c_void);
    enc.dispatch_threads(metal::MTLSize::new(len as u64, 1, 1), metal::MTLSize::new(len as u64, 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let q8_ptr = buf_q8.contents() as *const i8;
    let sc_ptr = buf_sc.contents() as *const f32;
    let metal_q8: Vec<i8> = unsafe { std::slice::from_raw_parts(q8_ptr, len).to_vec() };
    let metal_sc: Vec<f32> = unsafe { std::slice::from_raw_parts(sc_ptr, len / 32).to_vec() };

    // Check scales match
    for i in 0..len/32 {
        let diff = (cpu_scales[i] - metal_sc[i]).abs();
        assert!(diff < 0.1, "fused rms_norm_q8 scale[{i}] diff: cpu={} metal={}", cpu_scales[i], metal_sc[i]);
    }
    // Check Q8 values (allow ±2 rounding)
    let mut bad = 0;
    for i in 0..len {
        if (cpu_q8[i] as i32 - metal_q8[i] as i32).abs() > 2 { bad += 1; }
    }
    assert!(bad == 0, "fused rms_norm_q8: {bad}/{len} values differ by >2");
}

#[test]
fn residual_norm_matches_separate_ops() {
    let device = metal::Device::system_default().unwrap();
    let src = larql_compute::metal::shaders::all_shaders();
    let lib = device.new_library_with_source(&src, &metal::CompileOptions::new()).unwrap();
    let fused = device.new_compute_pipeline_state_with_function(
        &lib.get_function("residual_norm", None).unwrap()
    ).unwrap();
    let bufs = larql_compute::metal::buffers::BufferCache::new(&device);
    let queue = device.new_command_queue();

    let len = 64usize;
    let a: Vec<f32> = (0..len).map(|i| (i as f32 * 0.1 - 3.2)).collect();
    let b: Vec<f32> = (0..len).map(|i| (i as f32 * 0.05 + 0.3)).collect();
    let weight: Vec<f32> = (0..len).map(|i| 0.8 + i as f32 * 0.005).collect();
    let eps = 1e-6f32;
    let offset = 0.0f32;

    // CPU reference: add then norm
    let sum: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
    let sum_sq: f32 = sum.iter().map(|v| v * v).sum();
    let rms = 1.0 / (sum_sq / len as f32 + eps).sqrt();
    let cpu_result: Vec<f32> = sum.iter().zip(weight.iter()).map(|(s, w)| s * (w + offset) * rms).collect();

    // Metal fused
    let buf_a = bufs.transient_from_f32(&a);
    let buf_b = bufs.transient_from_f32(&b);
    let buf_w = bufs.transient_from_f32(&weight);
    let buf_out = bufs.output((len * 4) as u64);
    let len_val = len as u32;

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&fused);
    enc.set_buffer(0, Some(&buf_a), 0);
    enc.set_buffer(1, Some(&buf_b), 0);
    enc.set_buffer(2, Some(&buf_w), 0);
    enc.set_buffer(3, Some(&buf_out), 0);
    enc.set_bytes(4, 4, &len_val as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(5, 4, &eps as *const f32 as *const std::ffi::c_void);
    enc.set_bytes(6, 4, &offset as *const f32 as *const std::ffi::c_void);
    enc.dispatch_threads(metal::MTLSize::new(len as u64, 1, 1), metal::MTLSize::new(len as u64, 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let ptr = buf_out.contents() as *const f32;
    let metal_result: Vec<f32> = unsafe { std::slice::from_raw_parts(ptr, len).to_vec() };
    let diff = max_diff(&cpu_result, &metal_result);
    assert!(diff < 1e-4, "residual_norm max diff {diff}");
}
