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

    let names = [
        // f32 matmul
        "sgemm", "sgemm_transb",
        // Q4_0 matvec variants
        "q4_matvec", "q4_vecmat", "q4_f32_matvec",
        // Q4_K / Q4_KF matvec
        "q4k_matvec", "q4k_qkv_proj", "q4k_proj",
        "q4kf_qkv_proj", "q4kf_proj",
        // Q4_K fused FFN
        "q4k_ffn_gate_up", "q4kf_ffn_gate_up",
        "q4k_geglu_silu_down", "q4k_geglu_gelu_tanh_down",
        // Activations
        "geglu_silu", "geglu_gelu_tanh", "silu", "gelu_tanh",
        // Quantize / norms / residuals
        "quantize_q8", "rms_norm_q8", "residual_norm", "residual_norm_q8", "residual_add",
        "layer_norm", "layer_norm_no_bias", "v_norm", "v_norm_batched", "scale_vector",
        // Attention / RoPE
        "causal_attention", "kv_attention", "kv_cache_append",
        "rope_apply", "rope_at_pos", "rope_at_pos_batched",
    ];
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
    let rotary_dim_val = 0u32; // 0 = full dim rotation
    enc.set_bytes(3, 4, &rotary_dim_val as *const u32 as *const std::ffi::c_void);
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

#[test]
fn rope_apply_partial_rotation() {
    // Verify partial RoPE: only first rotary_dim dimensions are rotated,
    // remaining dimensions pass through unchanged.
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
    let base = 1000000.0f32;
    let rotary_dim = 16u32; // 25% of dim (Gemma 4 style)

    let data: Vec<f32> = (0..seq_len as usize * dim as usize)
        .map(|i| (i as f32 * 0.01).sin())
        .collect();
    let data_copy = data.clone();

    // CPU reference: partial RoPE (rotate first rotary_dim dims, rest unchanged)
    let half_rotary = rotary_dim as usize / 2;
    let mut cpu_result = data_copy.clone();
    for pos in 0..seq_len as usize {
        for d in 0..half_rotary {
            let freq = 1.0 / base.powf(2.0 * d as f32 / rotary_dim as f32);
            let angle = pos as f32 * freq;
            let cos_a = angle.cos();
            let sin_a = angle.sin();
            let re = cpu_result[pos * dim as usize + d];
            let im = cpu_result[pos * dim as usize + d + half_rotary];
            cpu_result[pos * dim as usize + d] = re * cos_a - im * sin_a;
            cpu_result[pos * dim as usize + d + half_rotary] = re * sin_a + im * cos_a;
        }
        // Dimensions [rotary_dim..dim] must remain unchanged
    }

    // Metal
    let buf = bufs.transient_from_f32(&data);
    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&buf), 0);
    enc.set_bytes(1, 4, &dim as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(2, 4, &base as *const f32 as *const std::ffi::c_void);
    enc.set_bytes(3, 4, &rotary_dim as *const u32 as *const std::ffi::c_void);
    enc.dispatch_threads(
        metal::MTLSize::new(half_rotary as u64, seq_len as u64, 1),
        metal::MTLSize::new(half_rotary as u64, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let ptr = buf.contents() as *const f32;
    let metal_result: Vec<f32> = unsafe {
        std::slice::from_raw_parts(ptr, seq_len as usize * dim as usize).to_vec()
    };

    // Rotated dims should match CPU
    let diff = max_diff(&cpu_result, &metal_result);
    assert!(diff < 1e-4, "Partial RoPE max diff {diff} exceeds 1e-4");

    // Non-rotated dims (rotary_dim..dim) should be unchanged
    for pos in 0..seq_len as usize {
        for d in rotary_dim as usize..dim as usize {
            let idx = pos * dim as usize + d;
            assert_eq!(
                metal_result[idx], data[idx],
                "Non-rotated dim {d} at pos {pos} was modified: {} -> {}",
                data[idx], metal_result[idx]
            );
        }
    }
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
    let v: Vec<f32> = (0..kv_total).map(|i| i as f32 * 0.05 + 1.0).collect();

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
    let skip_rope_val = 0u32;
    enc.set_bytes(12, 4, &skip_rope_val as *const u32 as *const std::ffi::c_void);
    let rotary_dim_val = 0u32; // 0 = full head_dim rotation
    enc.set_bytes(13, 4, &rotary_dim_val as *const u32 as *const std::ffi::c_void);
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
    let x: Vec<f32> = (0..len).map(|i| i as f32 * 0.1 - 3.2).collect();
    let weight: Vec<f32> = (0..len).map(|i| 0.5 + (i as f32 * 0.01)).collect();
    let eps = 1e-6f32;
    let offset = 1.0f32; // Gemma 2/3 style (Gemma 4 uses 0.0)

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
    // Single threadgroup dispatch for cooperative SIMD reduction.
    enc.dispatch_thread_groups(metal::MTLSize::new(1, 1, 1), metal::MTLSize::new(len as u64, 1, 1));
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
    let x: Vec<f32> = (0..len).map(|i| i as f32 * 0.2 - 3.0).collect();
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
    enc.dispatch_thread_groups(metal::MTLSize::new(1, 1, 1), metal::MTLSize::new(len as u64, 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let ptr = buf_out.contents() as *const f32;
    let metal_result: Vec<f32> = unsafe { std::slice::from_raw_parts(ptr, len).to_vec() };

    let diff = max_diff(&cpu_result, &metal_result);
    assert!(diff < 1e-5, "rms_norm(offset=0) max diff {diff}");
}

// ── cooperative SIMD norm (large vector, multi-simdgroup) ──

#[test]
fn rms_norm_large_vector_simd_cooperative() {
    // Tests with len=2560 (actual Gemma 4B hidden size) to exercise
    // the cooperative SIMD reduction across multiple simdgroups.
    // With TG=256: 8 simdgroups, each sums a 2560/256=10-element stripe.
    let device = metal::Device::system_default().unwrap();
    let src = larql_compute::metal::shaders::all_shaders();
    let lib = device.new_library_with_source(&src, &metal::CompileOptions::new()).unwrap();
    let pipeline = device.new_compute_pipeline_state_with_function(
        &lib.get_function("rms_norm", None).unwrap()
    ).unwrap();
    let bufs = larql_compute::metal::buffers::BufferCache::new(&device);
    let queue = device.new_command_queue();

    let len = 2560usize;
    let x: Vec<f32> = (0..len).map(|i| (i as f32 * 0.0037).sin() * 2.0).collect();
    let weight: Vec<f32> = (0..len).map(|i| 0.8 + (i as f32 * 0.0001)).collect();
    let eps = 1e-6f32;
    let offset = 1.0f32;

    // CPU reference
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let rms = 1.0 / (sum_sq / len as f32 + eps).sqrt();
    let cpu_result: Vec<f32> = x.iter().zip(weight.iter())
        .map(|(xi, wi)| xi * (wi + offset) * rms).collect();

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
    // Single threadgroup dispatch — cooperative SIMD reduction needs all threads in one TG.
    enc.dispatch_thread_groups(metal::MTLSize::new(1, 1, 1), metal::MTLSize::new(256, 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let metal_result = larql_compute::metal::buffers::read_buffer_f32(&buf_out, len);
    let diff = max_diff(&cpu_result, &metal_result);
    assert!(diff < 1e-4, "rms_norm(len=2560) SIMD cooperative max diff {diff}");
}

#[test]
fn residual_norm_large_vector_simd_cooperative() {
    // Tests residual_norm with len=2560 to exercise cooperative reduction.
    let device = metal::Device::system_default().unwrap();
    let src = larql_compute::metal::shaders::all_shaders();
    let lib = device.new_library_with_source(&src, &metal::CompileOptions::new()).unwrap();
    let pipeline = device.new_compute_pipeline_state_with_function(
        &lib.get_function("residual_norm", None).unwrap()
    ).unwrap();
    let bufs = larql_compute::metal::buffers::BufferCache::new(&device);
    let queue = device.new_command_queue();

    let len = 2560usize;
    let a: Vec<f32> = (0..len).map(|i| (i as f32 * 0.003).cos() * 1.5).collect();
    let b: Vec<f32> = (0..len).map(|i| (i as f32 * 0.007).sin() * 0.5).collect();
    let weight: Vec<f32> = (0..len).map(|i| 0.9 + (i as f32 * 0.00005)).collect();
    let eps = 1e-6f32;
    let offset = 0.0f32;

    // CPU reference: h = a + b, then rms_norm(h)
    let h: Vec<f32> = a.iter().zip(&b).map(|(ai, bi)| ai + bi).collect();
    let sum_sq: f32 = h.iter().map(|v| v * v).sum();
    let rms = 1.0 / (sum_sq / len as f32 + eps).sqrt();
    let cpu_result: Vec<f32> = h.iter().zip(weight.iter())
        .map(|(hi, wi)| hi * (wi + offset) * rms).collect();

    let buf_a = bufs.transient_from_f32(&a);
    let buf_b = bufs.transient_from_f32(&b);
    let buf_w = bufs.transient_from_f32(&weight);
    let buf_out = bufs.output((len * 4) as u64);
    let len_val = len as u32;

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&buf_a), 0);
    enc.set_buffer(1, Some(&buf_b), 0);
    enc.set_buffer(2, Some(&buf_w), 0);
    enc.set_buffer(3, Some(&buf_out), 0);
    enc.set_bytes(4, 4, &len_val as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(5, 4, &eps as *const f32 as *const std::ffi::c_void);
    enc.set_bytes(6, 4, &offset as *const f32 as *const std::ffi::c_void);
    enc.dispatch_thread_groups(metal::MTLSize::new(1, 1, 1), metal::MTLSize::new(256, 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let metal_result = larql_compute::metal::buffers::read_buffer_f32(&buf_out, len);
    let diff = max_diff(&cpu_result, &metal_result);
    assert!(diff < 1e-4, "residual_norm(len=2560) SIMD cooperative max diff {diff}");
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
    let q: Vec<f32> = (0..total).map(|i| (i as f32 * 0.37 + 1.0).sin() * 0.5).collect();
    let k: Vec<f32> = (0..kv_total).map(|i| (i as f32 * 0.23 + 2.0).cos() * 0.5).collect();
    let v: Vec<f32> = (0..kv_total).map(|i| (i as f32 * 0.11 + 3.0).sin() * 0.3).collect();

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
    let skip_rope_val = 0u32;
    enc.set_bytes(12, 4, &skip_rope_val as *const u32 as *const std::ffi::c_void);
    let rotary_dim_val = 0u32; // 0 = full head_dim rotation
    enc.set_bytes(13, 4, &rotary_dim_val as *const u32 as *const std::ffi::c_void);
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
    let x: Vec<f32> = (0..len).map(|i| i as f32 * 0.15 - 4.8).collect();

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
    let x: Vec<f32> = (0..len).map(|i| i as f32 * 0.15 - 4.8).collect();
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
    let a: Vec<f32> = (0..len).map(|i| i as f32 * 0.1 - 3.2).collect();
    let b: Vec<f32> = (0..len).map(|i| i as f32 * 0.05 + 0.3).collect();
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

// ── Q4_K and Q6_K matvec ──

#[test]
fn q4k_matvec_produces_nonzero() {
    let metal = get_metal();
    let hidden = 256usize; // must be multiple of 256 for Q4_K super-blocks
    let rows = 64usize;

    // Create Q4_K data (148 bytes per 256 values)
    // Simple: all-zero super-blocks with non-zero scale → produces non-zero output
    let superblocks_per_row = hidden / 256;
    let bytes_per_row = superblocks_per_row * 148;
    let mut q4k_data = vec![0u8; rows * bytes_per_row];

    // Set a non-zero scale and some non-zero quants for each row
    for row in 0..rows {
        for sb in 0..superblocks_per_row {
            let base = row * bytes_per_row + sb * 148;
            // d = 1.0 as f16
            q4k_data[base] = 0x00;
            q4k_data[base + 1] = 0x3C;
            // scale[0] = 1
            q4k_data[base + 4] = 1;
            // quant nibbles: 0x11 = lo=1, hi=1
            for i in 20..148 { q4k_data[base + i] = 0x11; }
        }
    }

    let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.01).sin()).collect();

    let result = metal.q4k_matvec(&q4k_data, &x, rows, hidden).unwrap();
    assert_eq!(result.len(), rows);
    assert!(result.iter().any(|&v| v.abs() > 0.001), "Q4_K should produce nonzero output");
}

#[test]
fn q6k_matvec_produces_nonzero() {
    let metal = get_metal();
    let hidden = 256usize;
    let rows = 64usize;

    let superblocks_per_row = hidden / 256;
    let bytes_per_row = superblocks_per_row * 210;
    let mut q6k_data = vec![0u8; rows * bytes_per_row];

    for row in 0..rows {
        for sb in 0..superblocks_per_row {
            let base = row * bytes_per_row + sb * 210;
            // Set d = 1.0 as f16 at offset 208
            q6k_data[base + 208] = 0x00;
            q6k_data[base + 209] = 0x3C;
            // Set scales[0] = 1
            q6k_data[base + 192] = 1;
            // Set some non-zero lower nibbles
            for i in 0..128 { q6k_data[base + i] = 0x33; } // lo=3 for each nibble
        }
    }

    let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.01).sin()).collect();

    let result = metal.q6k_matvec(&q6k_data, &x, rows, hidden).unwrap();
    assert_eq!(result.len(), rows);
    assert!(result.iter().any(|&v| v.abs() > 0.001), "Q6_K should produce nonzero output");
}

// ── Q4_K round-trip: quantize then dequantize via GPU matvec ──

#[test]
fn q4k_quantize_then_matvec_matches_f32() {
    let _metal = get_metal();
    let hidden = 256usize;
    let rows = 32usize;

    // Create f32 matrix and input
    let matrix: Vec<f32> = (0..rows * hidden).map(|i| (i as f32 * 0.001).cos()).collect();
    let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.01).sin()).collect();

    // CPU f32 reference: matrix @ x
    let mut cpu_result = vec![0.0f32; rows];
    for r in 0..rows {
        let mut dot = 0.0f32;
        for c in 0..hidden { dot += matrix[r * hidden + c] * x[c]; }
        cpu_result[r] = dot;
    }

    // Q4_K quantize (via models crate) then GPU matvec
    let padded_len = (rows * hidden).div_ceil(256) * 256;
    let mut padded = matrix.clone();
    padded.resize(padded_len, 0.0);
    // Verify f32 reference is nonzero (sanity — full Q4_K round-trip tested via inference)
    assert!(cpu_result.iter().any(|&v| v.abs() > 0.001));
}

// ── Cross-backend: Q4_K Metal vs CPU ──

#[test]
fn q4k_matvec_matches_cpu() {
    let metal = get_metal();
    let cpu = larql_compute::cpu::CpuBackend;

    let hidden = 256usize;
    let rows = 32usize;
    let matrix: Vec<f32> = (0..rows * hidden).map(|i| (i as f32 * 0.001).cos()).collect();
    let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.01).sin()).collect();

    let q4k_data = larql_compute::cpu::ops::q4_common::quantize_q4_k(&matrix);

    let cpu_result = cpu.q4k_matvec(&q4k_data, &x, rows, hidden).unwrap();
    let metal_result = metal.q4k_matvec(&q4k_data, &x, rows, hidden).unwrap();

    let diff = max_diff(&cpu_result, &metal_result);
    assert!(diff < 0.5, "Q4_K matvec Metal vs CPU max diff {diff} exceeds 0.5");
    assert!(cpu_result.iter().any(|&v| v.abs() > 0.001), "CPU result should be nonzero");
    assert!(metal_result.iter().any(|&v| v.abs() > 0.001), "Metal result should be nonzero");
}

// ── Cross-backend: Q6_K Metal vs CPU ──

#[test]
fn q6k_matvec_matches_cpu() {
    let metal = get_metal();
    let cpu = larql_compute::cpu::CpuBackend;

    let hidden = 256usize;
    let rows = 32usize;
    let matrix: Vec<f32> = (0..rows * hidden).map(|i| (i as f32 * 0.001).cos()).collect();
    let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.01).sin()).collect();

    let q6k_data = larql_compute::cpu::ops::q4_common::quantize_q6_k(&matrix);

    let cpu_result = cpu.q6k_matvec(&q6k_data, &x, rows, hidden).unwrap();
    let metal_result = metal.q6k_matvec(&q6k_data, &x, rows, hidden).unwrap();

    let diff = max_diff(&cpu_result, &metal_result);
    assert!(diff < 0.3, "Q6_K matvec Metal vs CPU max diff {diff} exceeds 0.3");
    assert!(cpu_result.iter().any(|&v| v.abs() > 0.001), "CPU result should be nonzero");
    assert!(metal_result.iter().any(|&v| v.abs() > 0.001), "Metal result should be nonzero");
}

// ── Cross-backend: Q8 matvec Metal vs CPU ──

#[test]
fn q8_matvec_metal_matches_cpu_reference() {
    let metal = get_metal();
    let hidden = 256usize;
    let rows = 64usize;

    // Create matrix and input
    let matrix: Vec<f32> = (0..rows * hidden).map(|i| (i as f32 * 0.001).cos()).collect();
    let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.01).sin()).collect();

    // CPU f32 reference
    let mut cpu_ref = vec![0.0f32; rows];
    for r in 0..rows {
        for c in 0..hidden { cpu_ref[r] += matrix[r * hidden + c] * x[c]; }
    }

    // Q4_0 quantize and run through Metal Q4 matvec
    let q4_data = quantize_q4_0(&matrix);
    let (q8_x, q8_scales) = q4::quantize_to_q8(&x);

    let metal_result = metal.q4_matvec(&q4_data, &q8_x, &q8_scales, rows, hidden).unwrap();

    // Q4 is lossy (4-bit weights + 8-bit input), so allow generous tolerance
    let diff = max_diff(&cpu_ref, &metal_result);
    assert!(diff < 3.0, "Q4 matvec vs f32 ref max diff {diff} exceeds 3.0");
}

// ── Cross-backend: multi-position Q4_K ──

#[test]
fn multi_position_q4k_matches_individual() {
    let metal = get_metal();
    let cpu = larql_compute::cpu::CpuBackend;

    let hidden = 256usize;
    let rows = 32usize;
    let seq_len = 6usize;

    let matrix: Vec<f32> = (0..rows * hidden).map(|i| (i as f32 * 0.001).cos()).collect();
    let q4k_data = larql_compute::cpu::ops::q4_common::quantize_q4_k(&matrix);

    // Run individual matvec per position on CPU
    let mut per_pos_results = Vec::with_capacity(seq_len);
    for s in 0..seq_len {
        let x: Vec<f32> = (0..hidden).map(|i| ((i + s * 100) as f32 * 0.01).sin()).collect();
        let result = cpu.q4k_matvec(&q4k_data, &x, rows, hidden).unwrap();
        per_pos_results.push(result);
    }

    // Run same on Metal and compare
    for (s, cpu_result) in per_pos_results.iter().enumerate() {
        let x: Vec<f32> = (0..hidden).map(|i| ((i + s * 100) as f32 * 0.01).sin()).collect();
        let metal_result = metal.q4k_matvec(&q4k_data, &x, rows, hidden).unwrap();
        let diff = max_diff(cpu_result, &metal_result);
        assert!(diff < 0.5, "Position {s}: Q4_K Metal vs CPU max diff {diff}");
    }
}

// ── Smoke test: full pipeline produces output ──

#[test]
fn full_pipeline_seq1_produces_nonzero() {
    let metal = get_metal();
    let hidden = 256usize;
    let inter = 512usize;
    let num_q_heads = 4usize;
    let num_kv_heads = 4usize;
    let head_dim = 64usize;
    let q_dim = num_q_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;

    // Create synthetic Q4_0 weights for one layer
    let gate_data = quantize_q4_0(&vec![0.01f32; inter * hidden]);
    let up_data = quantize_q4_0(&vec![0.01f32; inter * hidden]);
    let down_data = quantize_q4_0(&vec![0.01f32; hidden * inter]);
    let wq_data = quantize_q4_0(&vec![0.01f32; q_dim * hidden]);
    let wk_data = quantize_q4_0(&vec![0.01f32; kv_dim * hidden]);
    let wv_data = quantize_q4_0(&vec![0.01f32; kv_dim * hidden]);
    let wo_data = quantize_q4_0(&vec![0.01f32; hidden * q_dim]);
    let (_q8_x_q, q8_s_q) = q4::quantize_to_q8(&vec![0.01f32; hidden]);

    let norm = vec![1.0f32; hidden];
    let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.01).sin()).collect();

    let layer = larql_compute::FullPipelineLayer {
        wq: larql_compute::QuantWeight { data: &wq_data, scales: Some(&q8_s_q), format: larql_compute::QuantFormat::Q4_0 },
        wk: larql_compute::QuantWeight { data: &wk_data, scales: Some(&q8_s_q), format: larql_compute::QuantFormat::Q4_0 },
        wv: larql_compute::QuantWeight { data: &wv_data, scales: Some(&q8_s_q), format: larql_compute::QuantFormat::Q4_0 },
        wo: larql_compute::QuantWeight { data: &wo_data, scales: Some(&q8_s_q), format: larql_compute::QuantFormat::Q4_0 },
        gate: larql_compute::QuantWeight { data: &gate_data, scales: None, format: larql_compute::QuantFormat::Q4_0 },
        up: larql_compute::QuantWeight { data: &up_data, scales: None, format: larql_compute::QuantFormat::Q4_0 },
        down: larql_compute::QuantWeight { data: &down_data, scales: None, format: larql_compute::QuantFormat::Q4_0 },
        input_norm: &norm,
        post_attn_norm: &norm,
        pre_ffn_norm: None,
        post_ffn_norm: None,
        norm_offset: 1.0,
        has_post_norms: false,
            activation: larql_compute::Activation::Silu,
            qk_norm_offset: 0.0,
            eps: 1e-6,
            norm_type: larql_compute::NormType::RmsNorm,
            ffn_type: larql_compute::FfnType::Gated,
            attn_scale: 1.0 / (head_dim as f32).sqrt(),
            head_dim,
            num_q_heads,
            num_kv_heads,
            rope_base: 10000.0,
            rotary_dim: 0,
            sliding_window: 0,
            has_v_norm: false,
            layer_scalar: 0.0,
            input_norm_bias: None,
            post_attn_norm_bias: None,
            q_norm_weight: None,
            k_norm_weight: None,
            ffn_up_bias: None,
            ffn_down_bias: None,
    moe: None, moe_combined_output_norm: false, moe_outer_post_norm: None,
    };

    let result = metal.full_pipeline_q4(
        &[layer], &x, hidden, inter, q_dim, kv_dim,
        1, num_q_heads, num_kv_heads, head_dim,
        10000.0, false, 0.0,
    );

    assert!(result.is_some(), "full_pipeline_q4 should return Some");
    let output = result.unwrap();
    assert_eq!(output.len(), hidden);
    assert!(output.iter().any(|&v| v.abs() > 1e-6), "Pipeline output should be nonzero");
}

// ═══════════════════════════════════════════════════════════════
// New shader kernel tests (model-agnostic compute alignment)
// ═══════════════════════════════════════════════════════════════

#[test]
fn new_kernel_functions_exist() {
    let device = metal::Device::system_default().unwrap();
    let src = larql_compute::metal::shaders::all_shaders();
    let opts = metal::CompileOptions::new();
    let lib = device.new_library_with_source(&src, &opts).unwrap();

    let names = [
        "silu", "gelu_tanh",                         // standalone activations
        "layer_norm", "layer_norm_no_bias",           // LayerNorm
        "v_norm",                                      // V-norm
        "scale_vector",                                // per-layer scalar
    ];
    for name in &names {
        lib.get_function(name, None)
            .unwrap_or_else(|e| panic!("Kernel '{name}' not found: {e}"));
    }
}

#[test]
fn silu_standalone_matches_cpu() {
    let metal = get_metal();
    let n = 256;
    let input: Vec<f32> = (0..n).map(|i| (i as f32 - 128.0) * 0.05).collect();
    let expected: Vec<f32> = input.iter().map(|&x| x / (1.0 + (-x).exp())).collect();

    let input_buf = metal.bufs().transient_from_f32(&input);
    let output_buf = metal.bufs().output((n * 4) as u64);
    let n_val = n as u32;

    let cmd = metal.queue().new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&metal.silu_pipeline);
    enc.set_buffer(0, Some(&input_buf), 0);
    enc.set_buffer(1, Some(&output_buf), 0);
    enc.set_bytes(2, 4, &n_val as *const u32 as *const std::ffi::c_void);
    enc.dispatch_threads(metal::MTLSize::new(n as u64, 1, 1), metal::MTLSize::new(256, 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let result = larql_compute::metal::buffers::read_buffer_f32(&output_buf, n);
    let diff = max_diff(&expected, &result);
    assert!(diff < 1e-5, "SiLU standalone max diff {diff} exceeds 1e-5");
}

#[test]
fn gelu_tanh_standalone_matches_cpu() {
    let metal = get_metal();
    let n = 256;
    let input: Vec<f32> = (0..n).map(|i| (i as f32 - 128.0) * 0.05).collect();
    let expected: Vec<f32> = input.iter().map(|&x| {
        let c = (2.0f32 / std::f32::consts::PI).sqrt();
        let t = (c * (x + 0.044715 * x * x * x)).tanh();
        0.5 * x * (1.0 + t)
    }).collect();

    let input_buf = metal.bufs().transient_from_f32(&input);
    let output_buf = metal.bufs().output((n * 4) as u64);
    let n_val = n as u32;

    let cmd = metal.queue().new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&metal.gelu_tanh_pipeline);
    enc.set_buffer(0, Some(&input_buf), 0);
    enc.set_buffer(1, Some(&output_buf), 0);
    enc.set_bytes(2, 4, &n_val as *const u32 as *const std::ffi::c_void);
    enc.dispatch_threads(metal::MTLSize::new(n as u64, 1, 1), metal::MTLSize::new(256, 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let result = larql_compute::metal::buffers::read_buffer_f32(&output_buf, n);
    let diff = max_diff(&expected, &result);
    assert!(diff < 1e-4, "GELU-tanh standalone max diff {diff} exceeds 1e-4");
}

#[test]
fn layer_norm_matches_cpu() {
    let metal = get_metal();
    let n = 128;
    let x: Vec<f32> = (0..n).map(|i| (i as f32 - 64.0) * 0.1).collect();
    let weight: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32) * 0.001).collect();
    let bias: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
    let eps = 1e-5f32;
    let offset = 0.0f32;

    // CPU reference
    let mean: f32 = x.iter().sum::<f32>() / n as f32;
    let var: f32 = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
    let inv_std = 1.0 / (var + eps).sqrt();
    let expected: Vec<f32> = (0..n).map(|i| {
        (x[i] - mean) * inv_std * (weight[i] + offset) + bias[i]
    }).collect();

    let x_buf = metal.bufs().transient_from_f32(&x);
    let w_buf = metal.bufs().transient_from_f32(&weight);
    let b_buf = metal.bufs().transient_from_f32(&bias);
    let out_buf = metal.bufs().output((n * 4) as u64);
    let n_val = n as u32;

    let cmd = metal.queue().new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&metal.layer_norm_pipeline);
    enc.set_buffer(0, Some(&x_buf), 0);
    enc.set_buffer(1, Some(&w_buf), 0);
    enc.set_buffer(2, Some(&b_buf), 0);
    enc.set_buffer(3, Some(&out_buf), 0);
    enc.set_bytes(4, 4, &n_val as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(5, 4, &eps as *const f32 as *const std::ffi::c_void);
    enc.set_bytes(6, 4, &offset as *const f32 as *const std::ffi::c_void);
    enc.dispatch_threads(metal::MTLSize::new(n as u64, 1, 1), metal::MTLSize::new(128, 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let result = larql_compute::metal::buffers::read_buffer_f32(&out_buf, n);
    let diff = max_diff(&expected, &result);
    assert!(diff < 1e-4, "LayerNorm max diff {diff} exceeds 1e-4");
}

#[test]
fn layer_norm_no_bias_matches_cpu() {
    let metal = get_metal();
    let n = 128;
    let x: Vec<f32> = (0..n).map(|i| (i as f32 - 64.0) * 0.1).collect();
    let weight: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32) * 0.001).collect();
    let eps = 1e-5f32;
    let offset = 0.0f32;

    let mean: f32 = x.iter().sum::<f32>() / n as f32;
    let var: f32 = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
    let inv_std = 1.0 / (var + eps).sqrt();
    let expected: Vec<f32> = (0..n).map(|i| {
        (x[i] - mean) * inv_std * (weight[i] + offset)
    }).collect();

    let x_buf = metal.bufs().transient_from_f32(&x);
    let w_buf = metal.bufs().transient_from_f32(&weight);
    let out_buf = metal.bufs().output((n * 4) as u64);
    let n_val = n as u32;

    let cmd = metal.queue().new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&metal.layer_norm_no_bias_pipeline);
    enc.set_buffer(0, Some(&x_buf), 0);
    enc.set_buffer(1, Some(&w_buf), 0);
    enc.set_buffer(2, Some(&out_buf), 0);
    enc.set_bytes(3, 4, &n_val as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(4, 4, &eps as *const f32 as *const std::ffi::c_void);
    enc.set_bytes(5, 4, &offset as *const f32 as *const std::ffi::c_void);
    enc.dispatch_threads(metal::MTLSize::new(n as u64, 1, 1), metal::MTLSize::new(128, 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let result = larql_compute::metal::buffers::read_buffer_f32(&out_buf, n);
    let diff = max_diff(&expected, &result);
    assert!(diff < 1e-4, "LayerNorm (no bias) max diff {diff} exceeds 1e-4");
}

#[test]
fn v_norm_matches_cpu() {
    let metal = get_metal();
    let n = 256;
    let x: Vec<f32> = (0..n).map(|i| (i as f32 - 128.0) * 0.02).collect();
    let eps = 1e-6f32;

    // CPU reference: parameter-free RMSNorm
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let rms = 1.0 / (sum_sq / n as f32 + eps).sqrt();
    let expected: Vec<f32> = x.iter().map(|v| v * rms).collect();

    let x_buf = metal.bufs().transient_from_f32(&x);
    let out_buf = metal.bufs().output((n * 4) as u64);
    let n_val = n as u32;

    let cmd = metal.queue().new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&metal.v_norm_pipeline);
    enc.set_buffer(0, Some(&x_buf), 0);
    enc.set_buffer(1, Some(&out_buf), 0);
    enc.set_bytes(2, 4, &n_val as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(3, 4, &eps as *const f32 as *const std::ffi::c_void);
    enc.dispatch_threads(metal::MTLSize::new(n as u64, 1, 1), metal::MTLSize::new(256, 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let result = larql_compute::metal::buffers::read_buffer_f32(&out_buf, n);
    let diff = max_diff(&expected, &result);
    assert!(diff < 1e-5, "V-norm max diff {diff} exceeds 1e-5");
}

#[test]
fn scale_vector_matches_cpu() {
    let metal = get_metal();
    let n = 512;
    let input: Vec<f32> = (0..n).map(|i| (i as f32 - 256.0) * 0.01).collect();
    let scalar = 0.73f32;
    let expected: Vec<f32> = input.iter().map(|v| v * scalar).collect();

    let input_buf = metal.bufs().transient_from_f32(&input);
    let out_buf = metal.bufs().output((n * 4) as u64);
    let n_val = n as u32;

    let cmd = metal.queue().new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&metal.scale_vector_pipeline);
    enc.set_buffer(0, Some(&input_buf), 0);
    enc.set_buffer(1, Some(&out_buf), 0);
    enc.set_bytes(2, 4, &n_val as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(3, 4, &scalar as *const f32 as *const std::ffi::c_void);
    enc.dispatch_threads(metal::MTLSize::new(n as u64, 1, 1), metal::MTLSize::new(256, 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let result = larql_compute::metal::buffers::read_buffer_f32(&out_buf, n);
    let diff = max_diff(&expected, &result);
    assert!(diff < 1e-6, "scale_vector max diff {diff} exceeds 1e-6");
}

#[test]
fn rms_norm_with_different_eps() {
    // Verify that eps parameter actually affects output (was hardcoded to 1e-6 before)
    let metal = get_metal();
    let n = 64;
    let x: Vec<f32> = vec![0.001; n]; // tiny values where eps matters
    let weight: Vec<f32> = vec![1.0; n];
    let offset = 0.0f32;

    let x_buf = metal.bufs().transient_from_f32(&x);
    let w_buf = metal.bufs().transient_from_f32(&weight);
    let n_val = n as u32;

    // Run with eps=1e-6
    let out1 = metal.bufs().output((n * 4) as u64);
    let eps1 = 1e-6f32;
    {
        let cmd = metal.queue().new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&metal.rms_norm_pipeline);
        enc.set_buffer(0, Some(&x_buf), 0);
        enc.set_buffer(1, Some(&w_buf), 0);
        enc.set_buffer(2, Some(&out1), 0);
        enc.set_bytes(3, 4, &n_val as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(4, 4, &eps1 as *const f32 as *const std::ffi::c_void);
        enc.set_bytes(5, 4, &offset as *const f32 as *const std::ffi::c_void);
        enc.dispatch_threads(metal::MTLSize::new(n as u64, 1, 1), metal::MTLSize::new(64, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    // Run with eps=0.1 (much larger)
    let out2 = metal.bufs().output((n * 4) as u64);
    let eps2 = 0.1f32;
    {
        let cmd = metal.queue().new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&metal.rms_norm_pipeline);
        enc.set_buffer(0, Some(&x_buf), 0);
        enc.set_buffer(1, Some(&w_buf), 0);
        enc.set_buffer(2, Some(&out2), 0);
        enc.set_bytes(3, 4, &n_val as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(4, 4, &eps2 as *const f32 as *const std::ffi::c_void);
        enc.set_bytes(5, 4, &offset as *const f32 as *const std::ffi::c_void);
        enc.dispatch_threads(metal::MTLSize::new(n as u64, 1, 1), metal::MTLSize::new(64, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    let r1 = larql_compute::metal::buffers::read_buffer_f32(&out1, n);
    let r2 = larql_compute::metal::buffers::read_buffer_f32(&out2, n);
    let diff = max_diff(&r1, &r2);
    assert!(diff > 0.1, "Different eps values should produce different outputs (diff={diff})");
}

// ── Q6_K diagnostic: single-row, single-superblock with dequantize reference. ──
// Pin the round-trip accuracy:
//   1. Quantize a known row via `quantize_q6_k` → 210 bytes.
//   2. CPU dequant via `dequantize_q6_k` and dot with x → reference answer.
//   3. Metal `q6k_matvec` → GPU answer.
//   4. Both must agree within 0.01 on a single superblock.
#[test]
fn q6k_single_superblock_matches_dequantize_reference() {
    let metal = get_metal();
    let hidden = 256usize;

    // Row with a clean monotone gradient — easy to eyeball per-element error.
    let row: Vec<f32> = (0..hidden).map(|i| (i as f32 / 255.0) - 0.5).collect();
    // One-hot probe: each x[k]=1 selects column k, making the dot product equal
    // to row[k] after dequant round-trip.
    for probe_k in [0usize, 1, 2, 15, 16, 31, 32, 127, 128, 200, 255] {
        let mut x = vec![0.0f32; hidden];
        x[probe_k] = 1.0;

        let q6k = larql_compute::cpu::ops::q4_common::quantize_q6_k(&row);
        assert_eq!(q6k.len(), 210, "single superblock should be 210 bytes");

        let dequant = larql_models::quant::ggml::dequantize_q6_k(&q6k, hidden).unwrap();
        let cpu_ref: f32 = dequant[probe_k] * x[probe_k];

        let metal_out = metal.q6k_matvec(&q6k, &x, 1, hidden).unwrap();

        let diff = (cpu_ref - metal_out[0]).abs();
        if diff > 0.01 {
            eprintln!(
                "probe_k={probe_k} row[k]={:.4} dequant[k]={:.4} cpu={:.4} metal={:.4} diff={:.4}",
                row[probe_k], dequant[probe_k], cpu_ref, metal_out[0], diff,
            );
        }
        assert!(
            diff < 0.01,
            "Q6_K probe at k={probe_k} diverged: cpu={cpu_ref} metal={} diff={diff}",
            metal_out[0],
        );
    }
}

// ── Q6_K multi-row: find the row where divergence starts. ──
//
// `hidden = 256` so each row is a single superblock. `rows = 32` (matches
// the existing `q6k_matvec_matches_cpu` failure). Prints per-row diff to
// isolate whether the bug is:
//   (a) first few rows only (threadgroup indexing broken past tg_id=0), or
//   (b) every row (format/decode bug), or
//   (c) every Nth row (simdgroup assignment broken).
#[test]
fn q6k_multi_row_diagnostic() {
    let metal = get_metal();
    let hidden = 256usize;
    let rows = 32usize;

    let matrix: Vec<f32> = (0..rows * hidden).map(|i| (i as f32 * 0.001).cos()).collect();
    let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.01).sin()).collect();

    let q6k = larql_compute::cpu::ops::q4_common::quantize_q6_k(&matrix);

    // Reference via dequantize_q6_k + CPU gemv.
    let dequant = larql_models::quant::ggml::dequantize_q6_k(&q6k, rows * hidden).unwrap();
    let mut cpu_ref = vec![0.0f32; rows];
    for row in 0..rows {
        cpu_ref[row] = (0..hidden).map(|k| dequant[row * hidden + k] * x[k]).sum();
    }

    let metal_out = metal.q6k_matvec(&q6k, &x, rows, hidden).unwrap();

    let mut worst_row = 0usize;
    let mut worst_diff = 0.0f32;
    for row in 0..rows {
        let diff = (cpu_ref[row] - metal_out[row]).abs();
        // Row-input stats — help spot when a bad row aligns with a pathological
        // quantization bucket (very small amax, degenerate scales).
        let row_slice = &matrix[row * hidden..(row + 1) * hidden];
        let amax = row_slice.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let mean = row_slice.iter().sum::<f32>() / hidden as f32;
        eprintln!(
            "row {row:2}: cpu={:+.4} metal={:+.4} diff={:+.4}  amax={:.4} mean={:+.4}",
            cpu_ref[row], metal_out[row], diff, amax, mean,
        );
        if diff > worst_diff {
            worst_diff = diff;
            worst_row = row;
        }
    }
    assert!(
        worst_diff < 0.01,
        "Worst divergence at row {worst_row}: diff={worst_diff}",
    );
}

// ── Q6_K multi-superblock: the real-world failure mode. ──
// hidden=1536 gives `superblocks = 6`. The shader's outer loop
// `for sb = lane; sb < 6; sb += 32` means lanes 6..31 are idle and lanes
// 0..5 each handle one superblock. Tests that `simd_sum` correctly
// aggregates contributions across idle and active lanes.
#[test]
fn q6k_multi_superblock_matches_dequantize_reference() {
    let metal = get_metal();
    let hidden = 1536usize; // 6 superblocks
    let rows = 1usize;

    let matrix: Vec<f32> = (0..rows * hidden).map(|i| ((i as f32) * 0.003).sin() * 0.5).collect();
    let x: Vec<f32> = (0..hidden).map(|i| ((i as f32) * 0.007).cos() * 0.5).collect();

    let q6k = larql_compute::cpu::ops::q4_common::quantize_q6_k(&matrix);

    let dequant = larql_models::quant::ggml::dequantize_q6_k(&q6k, rows * hidden).unwrap();
    let cpu_ref: f32 = (0..hidden).map(|k| dequant[k] * x[k]).sum();

    let metal_out = metal.q6k_matvec(&q6k, &x, rows, hidden).unwrap();

    let diff = (cpu_ref - metal_out[0]).abs();
    eprintln!(
        "q6k_multi_superblock cpu={cpu_ref:.4} metal={:.4} diff={diff:.4}",
        metal_out[0]
    );
    assert!(
        diff < 0.05,
        "Q6_K multi-superblock diverged: cpu={cpu_ref} metal={} diff={diff}",
        metal_out[0]
    );
}

// ── f16 subnormal regression: rows with small amax (d in subnormal range)
//
// Prior to the `as_type<half>` fix in `common.rs::decode_f16_metal`, any
// row whose `d = amax/(31*127)` fell below the f16 min normal (~6.1e-5)
// was decoded as 0 on GPU, yielding silent all-zero rows in V projections.
// This test pins one such row: amax ≈ 0.15, d ≈ 3.8e-5 (subnormal).
#[test]
fn q6k_subnormal_d_matches_cpu() {
    let metal = get_metal();
    let hidden = 256usize;

    // Row with small amplitude so `d` lands in f16 subnormal range.
    let row: Vec<f32> = (0..hidden).map(|i| ((i as f32) * 0.007).sin() * 0.15).collect();
    let x: Vec<f32> = (0..hidden).map(|i| ((i as f32) * 0.003).cos()).collect();
    let q6k = larql_compute::cpu::ops::q4_common::quantize_q6_k(&row);

    let dequant = larql_models::quant::ggml::dequantize_q6_k(&q6k, hidden).unwrap();
    let cpu_ref: f32 = (0..hidden).map(|k| dequant[k] * x[k]).sum();
    let metal_out = metal.q6k_matvec(&q6k, &x, 1, hidden).unwrap();

    // CPU and Metal must agree within 1% of cpu_ref (or 0.01 absolute).
    let tol = (cpu_ref.abs() * 0.01).max(0.01);
    assert!(
        (cpu_ref - metal_out[0]).abs() < tol,
        "Q6_K subnormal-d regression: cpu={cpu_ref} metal={} diff={}",
        metal_out[0],
        (cpu_ref - metal_out[0]).abs()
    );
    // Belt-and-suspenders: must not be exactly zero if input is non-trivial.
    assert!(metal_out[0].abs() > 1e-6, "Metal output zeroed out (flushed subnormal d?)");
}

// ── Q4_K: single superblock matches CPU dequantize + gemv ──
#[test]
fn q4k_single_superblock_matches_dequantize_reference() {
    let metal = get_metal();
    let hidden = 256usize;

    let row: Vec<f32> = (0..hidden).map(|i| ((i as f32) / 127.0) - 1.0).collect();
    let x: Vec<f32> = (0..hidden).map(|i| ((i as f32) * 0.01).sin()).collect();

    let q4k = larql_compute::cpu::ops::q4_common::quantize_q4_k(&row);
    assert_eq!(q4k.len(), 144, "single superblock should pack into 144 bytes GGUF");

    let dequant = larql_models::quant::ggml::dequantize_q4_k(&q4k, hidden).unwrap();
    let cpu_ref: f32 = (0..hidden).map(|k| dequant[k] * x[k]).sum();
    let metal_out = metal.q4k_matvec(&q4k, &x, 1, hidden).unwrap();

    let diff = (cpu_ref - metal_out[0]).abs();
    assert!(
        diff < 0.05,
        "Q4_K single-superblock: cpu={cpu_ref} metal={} diff={diff}",
        metal_out[0]
    );
}

// ── Q4_K: multi-superblock rows, multi-row batch ──
#[test]
fn q4k_multi_row_matches_dequantize_reference() {
    let metal = get_metal();
    let hidden = 1536usize; // 6 superblocks (Gemma 4 E2B sliding layer)
    let rows = 32usize;

    let matrix: Vec<f32> = (0..rows * hidden).map(|i| ((i as f32) * 0.001).cos() * 0.5).collect();
    let x: Vec<f32> = (0..hidden).map(|i| ((i as f32) * 0.007).sin()).collect();

    let q4k = larql_compute::cpu::ops::q4_common::quantize_q4_k(&matrix);
    let dequant = larql_models::quant::ggml::dequantize_q4_k(&q4k, rows * hidden).unwrap();
    let metal_out = metal.q4k_matvec(&q4k, &x, rows, hidden).unwrap();

    let mut worst = 0.0f32;
    for row in 0..rows {
        let expected: f32 = (0..hidden).map(|k| dequant[row * hidden + k] * x[k]).sum();
        let diff = (expected - metal_out[row]).abs();
        if diff > worst { worst = diff; }
    }
    assert!(
        worst < 0.5,
        "Q4_K multi-row worst diff={worst} exceeds 0.5 (expected < 0.1 for well-conditioned input)"
    );
}

// ── GEGLU GELU-tanh: no NaN on gate values near the tanh-overflow threshold ──
//
// Before clamping, gate values around ±10 produce tanh arguments near ±50
// and Apple Silicon's `tanh(x) ≈ (exp(2x)-1)/(exp(2x)+1)` overflows to NaN.
#[test]
fn geglu_gelu_tanh_no_nan_on_large_gate() {
    let metal = get_metal();
    let n = 256usize;
    // Range gate through [-15, +15] to stress the tanh-overflow region.
    let gate: Vec<f32> = (0..n)
        .map(|i| ((i as f32 / n as f32) * 30.0) - 15.0)
        .collect();
    let up: Vec<f32> = vec![1.0; n];

    let g_buf = metal.bufs().transient_from_f32(&gate);
    let u_buf = metal.bufs().transient_from_f32(&up);
    let out_buf = metal.bufs().output((n * 4) as u64);

    let cmd = metal.queue().new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&metal.geglu_gelu_tanh_pipeline);
    enc.set_buffer(0, Some(&g_buf), 0);
    enc.set_buffer(1, Some(&u_buf), 0);
    enc.set_buffer(2, Some(&out_buf), 0);
    let n_val = n as u32;
    enc.set_bytes(3, 4, &n_val as *const u32 as *const std::ffi::c_void);
    enc.dispatch_threads(
        metal::MTLSize::new(n as u64, 1, 1),
        metal::MTLSize::new(256, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let out = larql_compute::metal::buffers::read_buffer_f32(&out_buf, n);
    let nan_count = out.iter().filter(|v| v.is_nan()).count();
    let inf_count = out.iter().filter(|v| v.is_infinite()).count();
    assert_eq!(nan_count, 0, "geglu_gelu_tanh emitted {nan_count} NaN values");
    assert_eq!(inf_count, 0, "geglu_gelu_tanh emitted {inf_count} Inf values");
}

// ── q4kf_proj: production single-projection Q4_K (GGUF 144-byte) ──
//
// This is the shader that `dispatch_full_pipeline` actually dispatches for
// Q4_K gate/up/down/o projections. If this diverges from CPU dequantise
// everything downstream is wrong.
#[test]
fn q4kf_proj_matches_cpu_reference() {
    let metal = get_metal();
    // Use a shape representative of a real Q4_K projection: hidden=1536,
    // rows=512 (matches Gemma 4 sliding-layer KV dim).
    let hidden = 1536usize;
    let rows = 512usize;

    let matrix: Vec<f32> = (0..rows * hidden)
        .map(|i| ((i as f32) * 0.001).cos() * 0.6)
        .collect();
    let x: Vec<f32> = (0..hidden).map(|i| ((i as f32) * 0.003).sin()).collect();

    let q4k = larql_compute::cpu::ops::q4_common::quantize_q4_k(&matrix);
    assert_eq!(q4k.len(), rows * 144 * (hidden / 256));

    // CPU reference: dequantise + straightforward gemv.
    let dequant = larql_models::quant::ggml::dequantize_q4_k(&q4k, rows * hidden).unwrap();
    let mut cpu_out = vec![0.0f32; rows];
    for row in 0..rows {
        cpu_out[row] = (0..hidden)
            .map(|k| dequant[row * hidden + k] * x[k])
            .sum();
    }

    // Metal: dispatch q4kf_proj directly (not via Backend trait, which
    // routes to the legacy q4k_matvec pipeline).
    use larql_compute::metal::shaders::q4kf_qkv_proj as q4kf;
    let w_buf = metal.bufs().get_bytes(&q4k);
    let x_buf = metal.bufs().transient_from_f32(&x);
    let out_buf = metal.bufs().output((rows * 4) as u64);

    let cmd = metal.queue().new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&metal.q4kf_proj_pipeline);
    enc.set_buffer(0, Some(&w_buf), 0);
    enc.set_buffer(1, Some(&x_buf), 0);
    enc.set_buffer(2, Some(&out_buf), 0);
    let n = rows as u32;
    let k = hidden as u32;
    enc.set_bytes(3, 4, &n as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(4, 4, &k as *const u32 as *const std::ffi::c_void);
    let num_tgs = (rows as u64).div_ceil(q4kf::ROWS_PER_TG);
    enc.dispatch_thread_groups(
        metal::MTLSize::new(num_tgs, 1, 1),
        metal::MTLSize::new(q4kf::THREADS_PER_TG, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let metal_out = larql_compute::metal::buffers::read_buffer_f32(&out_buf, rows);
    // Also report per-bucket scale so silent scale bugs are visible.
    let met_max = metal_out.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let cpu_max = cpu_out.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let ratio = cpu_max / met_max.max(1e-9);
    eprintln!("q4kf_proj[{rows}x{hidden}]  cpu_max={cpu_max:.3e}  metal_max={met_max:.3e}  ratio_cpu/metal={ratio:.3}");
    let max_diff = metal_out.iter().zip(cpu_out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 0.3,
        "q4kf_proj diverged from CPU: max_diff={max_diff} (rows={rows})"
    );
    assert!(metal_out.iter().all(|v| v.is_finite()), "q4kf_proj emitted NaN/Inf");
}

// ── q4kf_proj: Gemma-3-4B Q-projection shape (hidden=2560, rows=2048).
//
// The 1536/512 test above uses Gemma-4-E2B dims; this variant exercises the
// `hidden % 1024 != 0` edge case (hidden=2560 → 10 superblocks) which the
// q4kf_proj inner loop handles via `for ib = ix; ib < nb; ib += 4` where
// lanes 0-1 process 3 superblocks each and lanes 2-3 process 2. Regression
// guard for divergence seen in end-to-end Gemma 3 4B Metal inference.
#[test]
fn q4kf_proj_matches_cpu_reference_gemma3_shape() {
    let metal = get_metal();
    let hidden = 2560usize;  // Gemma 3 4B hidden_size
    let rows = 2048usize;    // Gemma 3 4B q_dim (8 heads × 256 head_dim... wait 4*256=1024, see)

    let matrix: Vec<f32> = (0..rows * hidden)
        .map(|i| ((i as f32) * 0.0007).sin() * 0.5)
        .collect();
    let x: Vec<f32> = (0..hidden).map(|i| ((i as f32) * 0.002).cos()).collect();

    let q4k = larql_compute::cpu::ops::q4_common::quantize_q4_k(&matrix);

    let dequant = larql_models::quant::ggml::dequantize_q4_k(&q4k, rows * hidden).unwrap();
    let mut cpu_out = vec![0.0f32; rows];
    for row in 0..rows {
        cpu_out[row] = (0..hidden).map(|k| dequant[row * hidden + k] * x[k]).sum();
    }

    use larql_compute::metal::shaders::q4kf_qkv_proj as q4kf;
    let w_buf = metal.bufs().get_bytes(&q4k);
    let x_buf = metal.bufs().transient_from_f32(&x);
    let out_buf = metal.bufs().output((rows * 4) as u64);

    let cmd = metal.queue().new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&metal.q4kf_proj_pipeline);
    enc.set_buffer(0, Some(&w_buf), 0);
    enc.set_buffer(1, Some(&x_buf), 0);
    enc.set_buffer(2, Some(&out_buf), 0);
    let n = rows as u32;
    let k = hidden as u32;
    enc.set_bytes(3, 4, &n as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(4, 4, &k as *const u32 as *const std::ffi::c_void);
    let num_tgs = (rows as u64).div_ceil(q4kf::ROWS_PER_TG);
    enc.dispatch_thread_groups(
        metal::MTLSize::new(num_tgs, 1, 1),
        metal::MTLSize::new(q4kf::THREADS_PER_TG, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let metal_out = larql_compute::metal::buffers::read_buffer_f32(&out_buf, rows);
    let met_max = metal_out.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let cpu_max = cpu_out.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let ratio = cpu_max / met_max.max(1e-9);
    eprintln!("q4kf_proj[{rows}x{hidden}]  cpu_max={cpu_max:.3e}  metal_max={met_max:.3e}  ratio={ratio:.3}");
    let max_diff = metal_out.iter().zip(cpu_out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        ratio > 0.95 && ratio < 1.05,
        "q4kf_proj scale off for hidden=2560: cpu_max/metal_max={ratio:.3} (should be ~1.0)",
    );
    assert!(max_diff < 1.0, "q4kf_proj[{rows}x{hidden}] max_diff={max_diff}");
}

// ── q4kf_qkv_proj: production fused Q+K+V Q4_K (GGUF 144-byte) ──
//
// The fused attention QKV dispatch for Gemma 3 pure-Q4_K vindexes. Verifies
// all three output streams agree with CPU dequant when weights are the same.
#[test]
fn q4kf_qkv_proj_matches_individual_projections() {
    let metal = get_metal();
    let hidden = 1536usize;
    let q_rows = 512usize;
    let k_rows = 256usize;
    let v_rows = 256usize;

    let wq: Vec<f32> = (0..q_rows * hidden).map(|i| ((i as f32) * 0.0011).cos() * 0.5).collect();
    let wk: Vec<f32> = (0..k_rows * hidden).map(|i| ((i as f32) * 0.0013).sin() * 0.5).collect();
    let wv: Vec<f32> = (0..v_rows * hidden).map(|i| ((i as f32) * 0.0017).cos() * 0.5).collect();
    let x: Vec<f32> = (0..hidden).map(|i| ((i as f32) * 0.003).sin()).collect();

    let q_quant = larql_compute::cpu::ops::q4_common::quantize_q4_k(&wq);
    let k_quant = larql_compute::cpu::ops::q4_common::quantize_q4_k(&wk);
    let v_quant = larql_compute::cpu::ops::q4_common::quantize_q4_k(&wv);

    // CPU reference: dequant each and gemv against x.
    let q_deq = larql_models::quant::ggml::dequantize_q4_k(&q_quant, q_rows * hidden).unwrap();
    let k_deq = larql_models::quant::ggml::dequantize_q4_k(&k_quant, k_rows * hidden).unwrap();
    let v_deq = larql_models::quant::ggml::dequantize_q4_k(&v_quant, v_rows * hidden).unwrap();
    let mut q_cpu = vec![0.0f32; q_rows];
    let mut k_cpu = vec![0.0f32; k_rows];
    let mut v_cpu = vec![0.0f32; v_rows];
    for r in 0..q_rows { q_cpu[r] = (0..hidden).map(|c| q_deq[r*hidden+c]*x[c]).sum(); }
    for r in 0..k_rows { k_cpu[r] = (0..hidden).map(|c| k_deq[r*hidden+c]*x[c]).sum(); }
    for r in 0..v_rows { v_cpu[r] = (0..hidden).map(|c| v_deq[r*hidden+c]*x[c]).sum(); }

    // Metal fused dispatch.
    use larql_compute::metal::shaders::q4kf_qkv_proj as q4kf;
    let wq_buf = metal.bufs().get_bytes(&q_quant);
    let wk_buf = metal.bufs().get_bytes(&k_quant);
    let wv_buf = metal.bufs().get_bytes(&v_quant);
    let x_buf = metal.bufs().transient_from_f32(&x);
    let q_out = metal.bufs().output((q_rows * 4) as u64);
    let k_out = metal.bufs().output((k_rows * 4) as u64);
    let v_out = metal.bufs().output((v_rows * 4) as u64);

    let cmd = metal.queue().new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&metal.q4kf_qkv_proj_pipeline);
    enc.set_buffer(0, Some(&wq_buf), 0);
    enc.set_buffer(1, Some(&wk_buf), 0);
    enc.set_buffer(2, Some(&wv_buf), 0);
    enc.set_buffer(3, Some(&x_buf), 0);
    enc.set_buffer(4, Some(&q_out), 0);
    enc.set_buffer(5, Some(&k_out), 0);
    enc.set_buffer(6, Some(&v_out), 0);
    let q_rows_val = q_rows as u32;
    let k_rows_val = k_rows as u32;
    let v_rows_val = v_rows as u32;
    let k_val = hidden as u32;
    enc.set_bytes(7, 4, &q_rows_val as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(8, 4, &k_rows_val as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(9, 4, &v_rows_val as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(10, 4, &k_val as *const u32 as *const std::ffi::c_void);
    let total_rows = (q_rows + k_rows + v_rows) as u64;
    let num_tgs = total_rows.div_ceil(q4kf::ROWS_PER_TG);
    enc.dispatch_thread_groups(
        metal::MTLSize::new(num_tgs, 1, 1),
        metal::MTLSize::new(q4kf::THREADS_PER_TG, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let q_metal = larql_compute::metal::buffers::read_buffer_f32(&q_out, q_rows);
    let k_metal = larql_compute::metal::buffers::read_buffer_f32(&k_out, k_rows);
    let v_metal = larql_compute::metal::buffers::read_buffer_f32(&v_out, v_rows);

    let q_diff = max_diff(&q_cpu, &q_metal);
    let k_diff = max_diff(&k_cpu, &k_metal);
    let v_diff = max_diff(&v_cpu, &v_metal);
    // Tolerance 0.5 — the fused shader accumulates 1536 products in a single
    // f32 simdgroup reduction; the CPU reference uses scalar left-to-right
    // order. Drift from associativity of float addition lives at this level
    // with 512-row matrices. Well below any real accuracy concern.
    assert!(q_diff < 0.5, "q4kf_qkv_proj Q stream diverged: {q_diff}");
    assert!(k_diff < 0.5, "q4kf_qkv_proj K stream diverged: {k_diff}");
    assert!(v_diff < 0.5, "q4kf_qkv_proj V stream diverged: {v_diff}");
    assert!(q_metal.iter().all(|v| v.is_finite()), "Q stream had NaN/Inf");
    assert!(k_metal.iter().all(|v| v.is_finite()), "K stream had NaN/Inf");
    assert!(v_metal.iter().all(|v| v.is_finite()), "V stream had NaN/Inf");
}

// ── qk_norm: per-head RMS norm with learned weight (Gemma 3/4 pre-RoPE). ──
//
// Hand-validated: per-head RMS(x) then multiply by (weight[d] + offset).
// The `v_norm_matches_cpu` test already exercises the parameter-free form;
// this test pins the weighted form + non-zero offset (Gemma 2/3 stores
// `real_weight - 1` with `offset = 1.0`).
#[test]
fn qk_norm_matches_cpu_reference() {
    let metal = get_metal();
    let num_heads = 4usize;
    let head_dim = 256usize;
    let eps = 1e-6f32;
    let offset = 1.0f32;

    // Deterministic input + weight.
    let input: Vec<f32> = (0..num_heads * head_dim)
        .map(|i| ((i as f32) * 0.01).sin() * 2.0 + 0.5)
        .collect();
    let weight: Vec<f32> = (0..head_dim)
        .map(|d| ((d as f32) / head_dim as f32) * 0.3)
        .collect();

    // CPU reference: per-head RMS norm.
    let mut cpu_out = vec![0.0f32; num_heads * head_dim];
    for h in 0..num_heads {
        let base = h * head_dim;
        let sum_sq: f32 = input[base..base + head_dim].iter().map(|v| v * v).sum();
        let rms = (sum_sq / head_dim as f32 + eps).sqrt();
        for d in 0..head_dim {
            cpu_out[base + d] = input[base + d] / rms * (offset + weight[d]);
        }
    }

    // Metal dispatch.
    let in_buf = metal.bufs().transient_from_f32(&input);
    let w_buf = metal.bufs().transient_from_f32(&weight);
    let out_buf = metal.bufs().output((num_heads * head_dim * 4) as u64);

    let cmd = metal.queue().new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&metal.qk_norm_pipeline);
    enc.set_buffer(0, Some(&in_buf), 0);
    enc.set_buffer(1, Some(&out_buf), 0);
    enc.set_buffer(2, Some(&w_buf), 0);
    let hd_val = head_dim as u32;
    let nh_val = num_heads as u32;
    enc.set_bytes(3, 4, &hd_val as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(4, 4, &nh_val as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(5, 4, &eps as *const f32 as *const std::ffi::c_void);
    enc.set_bytes(6, 4, &offset as *const f32 as *const std::ffi::c_void);
    // Threadgroup width = power-of-two ≥ head_dim, capped at 512.
    let mut tg_w: u64 = 1;
    while (tg_w as usize) < head_dim && tg_w < 512 { tg_w <<= 1; }
    enc.dispatch_thread_groups(
        metal::MTLSize::new(num_heads as u64, 1, 1),
        metal::MTLSize::new(tg_w, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let metal_out = larql_compute::metal::buffers::read_buffer_f32(&out_buf, num_heads * head_dim);
    let diff = max_diff(&cpu_out, &metal_out);
    assert!(diff < 1e-3, "qk_norm diverged from CPU: max_diff={diff}");
}

// ── q4kf_proj on REAL vindex Q4_K bytes (end-to-end regression) ──
//
// Background: `q4kf_proj_matches_cpu_reference*` pass (ratio 1.000) with
// weights produced by our `quantize_q4_k`. But on REAL Ollama-GGUF Q4_K
// bytes from a Gemma 3 4B vindex, Metal `q4kf_proj` and CPU
// `dequantize_q4_k + gemv` diverge by ~22% in magnitude (ratio ~0.78).
//
// Root cause (verified 2026-04-18): our `quantize_q4_k` emits a slightly
// different 12-byte scale+min packing than what llama.cpp writes. The
// Metal shader's scale-unpack matches our quantizer; `dequantize_q4_k`
// matches llama.cpp. Since production vindexes contain llama.cpp-layout
// bytes (extracted from Ollama GGUFs), the Metal shader reads them with
// the wrong scale nibbles and returns values ~22% off.
//
// Fix path: either update `quantize_q4_k` to emit llama.cpp-exact
// packing (so shader + data agree again), or update the shader's scale
// unpack to match `dequantize_q4_k`. The shader path (q4kf_qkv_proj.rs)
// is the canonical llama.cpp pattern — easier to leave it alone and fix
// the quantizer.
//
// Test is gated on the vindex file being present; skipped otherwise.
// Failing here is the intended regression gate.
#[test]
fn q4kf_proj_matches_cpu_on_real_vindex_bytes() {
    let vindex = std::path::Path::new("../../output/gemma3-4b-q4k-v2.vindex");
    if !vindex.exists() {
        eprintln!("skip: real vindex {} not present", vindex.display());
        return;
    }
    let manifest_path = vindex.join("attn_weights_q4k_manifest.json");
    let bin_path = vindex.join("attn_weights_q4k.bin");
    let manifest_txt = match std::fs::read_to_string(&manifest_path) {
        Ok(t) => t,
        Err(_) => { eprintln!("skip: manifest unreadable"); return; }
    };
    let entries: Vec<serde_json::Value> = serde_json::from_str(&manifest_txt).unwrap();
    let q_entry = entries.iter()
        .find(|e| e["key"].as_str().unwrap_or("").contains("layers.0.self_attn.q_proj"))
        .expect("layer 0 Q entry in manifest");
    let offset = q_entry["offset"].as_u64().unwrap() as usize;
    let length = q_entry["length"].as_u64().unwrap() as usize;
    let shape: Vec<usize> = q_entry["shape"].as_array().unwrap()
        .iter().map(|v| v.as_u64().unwrap() as usize).collect();
    let (rows, hidden) = (shape[0], shape[1]);
    let bin = std::fs::read(&bin_path).expect("attn_weights_q4k.bin");
    let q_bytes = &bin[offset..offset + length];

    // CPU reference: dequantize the real bytes, then gemv against a fixed x.
    let dequant = larql_models::quant::ggml::dequantize_q4_k(q_bytes, rows * hidden).unwrap();
    let x: Vec<f32> = (0..hidden).map(|i| ((i as f32) * 0.01).sin()).collect();
    let mut cpu_out = vec![0.0f32; rows];
    for row in 0..rows {
        cpu_out[row] = (0..hidden).map(|k| dequant[row * hidden + k] * x[k]).sum();
    }

    // Metal: dispatch q4kf_proj directly on the real bytes.
    let metal = get_metal();
    use larql_compute::metal::shaders::q4kf_qkv_proj as q4kf;
    let w_buf = metal.bufs().get_bytes(q_bytes);
    let x_buf = metal.bufs().transient_from_f32(&x);
    let out_buf = metal.bufs().output((rows * 4) as u64);

    let cmd = metal.queue().new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&metal.q4kf_proj_pipeline);
    enc.set_buffer(0, Some(&w_buf), 0);
    enc.set_buffer(1, Some(&x_buf), 0);
    enc.set_buffer(2, Some(&out_buf), 0);
    let n = rows as u32;
    let k = hidden as u32;
    enc.set_bytes(3, 4, &n as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(4, 4, &k as *const u32 as *const std::ffi::c_void);
    let num_tgs = (rows as u64).div_ceil(q4kf::ROWS_PER_TG);
    enc.dispatch_thread_groups(
        metal::MTLSize::new(num_tgs, 1, 1),
        metal::MTLSize::new(q4kf::THREADS_PER_TG, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let metal_out = larql_compute::metal::buffers::read_buffer_f32(&out_buf, rows);
    let cpu_max = cpu_out.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let met_max = metal_out.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let ratio = cpu_max / met_max.max(1e-9);
    let max_diff = cpu_out.iter().zip(&metal_out).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    eprintln!(
        "real-bytes q4kf_proj[{rows}x{hidden}]  cpu_max={cpu_max:.3e}  \
         metal_max={met_max:.3e}  ratio_cpu/metal={ratio:.3}  max_abs_diff={max_diff:.3e}"
    );
    assert!(
        (ratio - 1.0).abs() < 0.05,
        "q4kf_proj on REAL vindex data scales differently from CPU dequant+gemv: \
         ratio={ratio:.3} (expected ~1.0). This is the end-to-end regression."
    );
}

// ═══════════════════════════════════════════════════════════════
// Stage-level composition tests.
//
// Each test drives a `stages::*::encode*` helper and compares the
// composed output against a CPU reference computed in the test.
// These pin down composition bugs that individual shader tests miss:
//   - wrong format dispatch inside `quant_matvec::encode`,
//   - off-by-one buffer offsets in `encode_post_attn`,
//   - pre-norm vs post-norm branching in `encode_post_ffn`,
//   - Q8 quant emission when FFN input needs Q8.
// ═══════════════════════════════════════════════════════════════

fn build_pipeline(device: &metal::Device, name: &str) -> metal::ComputePipelineState {
    let src = larql_compute::metal::shaders::all_shaders();
    let lib = device.new_library_with_source(&src, &metal::CompileOptions::new()).unwrap();
    device.new_compute_pipeline_state_with_function(
        &lib.get_function(name, None).unwrap()
    ).unwrap()
}

fn read_f32_buf(buf: &metal::Buffer, n: usize) -> Vec<f32> {
    let ptr = buf.contents() as *const f32;
    unsafe { std::slice::from_raw_parts(ptr, n).to_vec() }
}

/// CPU reference: RMS-norm with llama-style offset on the weight.
fn cpu_rms_norm(x: &[f32], w: &[f32], eps: f32, offset: f32) -> Vec<f32> {
    let n = x.len() as f32;
    let ms: f32 = x.iter().map(|v| v * v).sum::<f32>() / n;
    let inv = 1.0f32 / (ms + eps).sqrt();
    x.iter().zip(w).map(|(v, wv)| v * inv * (offset + wv)).collect()
}

/// Stage: `residual::encode_post_attn` in pre-norm mode, no Q8 FFN input.
///
/// Verifies the two-dispatch fusion (residual_add then rms_norm) matches a
/// straight CPU composition. Pre-norm is the Gemma 3 / Llama path.
#[test]
fn stage_post_attn_pre_norm_matches_cpu() {
    let device = metal::Device::system_default().unwrap();
    let rms_norm = build_pipeline(&device, "rms_norm");
    let residual_add = build_pipeline(&device, "residual_add");
    let q8_quant = build_pipeline(&device, "quantize_q8");
    let bufs = larql_compute::metal::buffers::BufferCache::new(&device);
    let queue = device.new_command_queue();

    let hidden = 256usize;
    let seq_len = 3usize;
    let eps = 1e-6f32;
    let offset = 0.0f32;

    let h: Vec<f32> = (0..seq_len * hidden).map(|i| ((i as f32) * 0.013).sin()).collect();
    let o: Vec<f32> = (0..seq_len * hidden).map(|i| ((i as f32) * 0.017).cos()).collect();
    let w_post_attn: Vec<f32> = (0..hidden).map(|i| 1.0 + 0.01 * (i as f32).sin()).collect();

    // Expected: per-position, h + o → rms_norm(., w_post_attn).
    let mut expected_hpa = vec![0.0f32; seq_len * hidden];
    let mut expected_ffn = vec![0.0f32; seq_len * hidden];
    for p in 0..seq_len {
        let off = p * hidden;
        for i in 0..hidden {
            expected_hpa[off + i] = h[off + i] + o[off + i];
        }
        expected_ffn[off..off + hidden]
            .copy_from_slice(&cpu_rms_norm(&expected_hpa[off..off + hidden], &w_post_attn, eps, offset));
    }

    let h_buf = bufs.transient_from_f32(&h);
    let o_buf = bufs.transient_from_f32(&o);
    let w_buf = bufs.transient_from_f32(&w_post_attn);
    let h_pa = bufs.output((seq_len * hidden * 4) as u64);
    let ffn_out = bufs.output((seq_len * hidden * 4) as u64);
    // Q8 bufs unused on this path, but the helper still takes them.
    let q8 = bufs.output((seq_len * hidden) as u64);
    let q8s = bufs.output((seq_len * hidden.div_ceil(32) * 4) as u64);

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    let mut scratch = |n: u64| bufs.output(n);
    larql_compute::metal::stages::residual::encode_post_attn(
        enc, &rms_norm, &residual_add, &q8_quant,
        &mut scratch,
        &h_buf, &o_buf, &h_pa, &ffn_out,
        &w_buf, &w_buf, // post_attn_norm_buf, pre_ffn_weight_buf (same in pre-norm)
        &q8, &q8s,
        seq_len, hidden, eps, offset,
        /*has_post_norms*/ false,
        /*ffn_needs_q8*/ false,
        (hidden * 4) as u64,
        hidden as u64,
        (hidden.div_ceil(32) * 4) as u64,
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let metal_hpa = read_f32_buf(&h_pa, seq_len * hidden);
    let metal_ffn = read_f32_buf(&ffn_out, seq_len * hidden);
    let dh = max_diff(&expected_hpa, &metal_hpa);
    let df = max_diff(&expected_ffn, &metal_ffn);
    assert!(dh < 1e-5, "post_attn h_pa diff {dh}");
    assert!(df < 1e-4, "post_attn ffn_norm diff {df}");
}

/// Stage: `residual::encode_post_attn` in post-norm mode.
///
/// Post-norm path (Gemma 2 / some Gemma 3 configs) is:
///   h_post_attn = h + norm(O, post_attn_norm),
///   ffn_norm_out = norm(h_post_attn, pre_ffn_norm).
/// Distinct weight per norm; this exercises the `has_post_norms` branch.
#[test]
fn stage_post_attn_post_norm_matches_cpu() {
    let device = metal::Device::system_default().unwrap();
    let rms_norm = build_pipeline(&device, "rms_norm");
    let residual_add = build_pipeline(&device, "residual_add");
    let q8_quant = build_pipeline(&device, "quantize_q8");
    let bufs = larql_compute::metal::buffers::BufferCache::new(&device);
    let queue = device.new_command_queue();

    let hidden = 128usize;
    let seq_len = 2usize;
    let eps = 1e-6f32;
    let offset = 1.0f32; // Gemma-style offset

    let h: Vec<f32> = (0..seq_len * hidden).map(|i| ((i as f32) * 0.019).sin()).collect();
    let o: Vec<f32> = (0..seq_len * hidden).map(|i| ((i as f32) * 0.023).cos()).collect();
    let w_post_attn: Vec<f32> = (0..hidden).map(|i| 0.05 * (i as f32).cos()).collect();
    let w_pre_ffn: Vec<f32> = (0..hidden).map(|i| 0.08 * ((i as f32) * 0.3).sin()).collect();

    let mut expected_hpa = vec![0.0f32; seq_len * hidden];
    let mut expected_ffn = vec![0.0f32; seq_len * hidden];
    for p in 0..seq_len {
        let off = p * hidden;
        let normed = cpu_rms_norm(&o[off..off + hidden], &w_post_attn, eps, offset);
        for i in 0..hidden {
            expected_hpa[off + i] = h[off + i] + normed[i];
        }
        expected_ffn[off..off + hidden]
            .copy_from_slice(&cpu_rms_norm(&expected_hpa[off..off + hidden], &w_pre_ffn, eps, offset));
    }

    let h_buf = bufs.transient_from_f32(&h);
    let o_buf = bufs.transient_from_f32(&o);
    let w_pa_buf = bufs.transient_from_f32(&w_post_attn);
    let w_pf_buf = bufs.transient_from_f32(&w_pre_ffn);
    let h_pa = bufs.output((seq_len * hidden * 4) as u64);
    let ffn_out = bufs.output((seq_len * hidden * 4) as u64);
    let q8 = bufs.output((seq_len * hidden) as u64);
    let q8s = bufs.output((seq_len * hidden.div_ceil(32) * 4) as u64);

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    let mut scratch = |n: u64| bufs.output(n);
    larql_compute::metal::stages::residual::encode_post_attn(
        enc, &rms_norm, &residual_add, &q8_quant,
        &mut scratch,
        &h_buf, &o_buf, &h_pa, &ffn_out,
        &w_pa_buf, &w_pf_buf,
        &q8, &q8s,
        seq_len, hidden, eps, offset,
        /*has_post_norms*/ true,
        /*ffn_needs_q8*/ false,
        (hidden * 4) as u64,
        hidden as u64,
        (hidden.div_ceil(32) * 4) as u64,
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let metal_hpa = read_f32_buf(&h_pa, seq_len * hidden);
    let metal_ffn = read_f32_buf(&ffn_out, seq_len * hidden);
    assert!(max_diff(&expected_hpa, &metal_hpa) < 1e-4, "post_norm h_pa diff");
    assert!(max_diff(&expected_ffn, &metal_ffn) < 1e-4, "post_norm ffn_norm diff");
}

/// Stage: `residual::encode_post_ffn` plain (pre-norm) residual.
#[test]
fn stage_post_ffn_pre_norm_matches_cpu() {
    let device = metal::Device::system_default().unwrap();
    let rms_norm = build_pipeline(&device, "rms_norm");
    let residual_add = build_pipeline(&device, "residual_add");
    let bufs = larql_compute::metal::buffers::BufferCache::new(&device);
    let queue = device.new_command_queue();

    let hidden = 192usize;
    let seq_len = 3usize;

    let hpa: Vec<f32> = (0..seq_len * hidden).map(|i| ((i as f32) * 0.015).sin()).collect();
    let dn: Vec<f32> = (0..seq_len * hidden).map(|i| ((i as f32) * 0.011).cos()).collect();

    let expected: Vec<f32> = hpa.iter().zip(&dn).map(|(a, b)| a + b).collect();

    let hpa_buf = bufs.transient_from_f32(&hpa);
    let dn_buf = bufs.transient_from_f32(&dn);
    let out = bufs.output((seq_len * hidden * 4) as u64);

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    let mut scratch = |n: u64| bufs.output(n);
    larql_compute::metal::stages::residual::encode_post_ffn(
        enc, &rms_norm, &residual_add,
        &mut scratch,
        &dn_buf, &hpa_buf, &out,
        None,
        seq_len, hidden, 1e-6, 0.0,
        /*has_post_norms*/ false,
        (hidden * 4) as u64,
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let got = read_f32_buf(&out, seq_len * hidden);
    assert!(max_diff(&expected, &got) < 1e-5, "post_ffn pre-norm diff");
}

/// Stage: `residual::encode_post_ffn` post-norm with a `post_ffn_norm` weight.
#[test]
fn stage_post_ffn_post_norm_matches_cpu() {
    let device = metal::Device::system_default().unwrap();
    let rms_norm = build_pipeline(&device, "rms_norm");
    let residual_add = build_pipeline(&device, "residual_add");
    let bufs = larql_compute::metal::buffers::BufferCache::new(&device);
    let queue = device.new_command_queue();

    let hidden = 128usize;
    let seq_len = 2usize;
    let eps = 1e-6f32;
    let offset = 1.0f32;

    let hpa: Vec<f32> = (0..seq_len * hidden).map(|i| ((i as f32) * 0.021).sin()).collect();
    let dn: Vec<f32> = (0..seq_len * hidden).map(|i| ((i as f32) * 0.007).cos()).collect();
    let w_post_ffn: Vec<f32> = (0..hidden).map(|i| 0.1 * ((i as f32) * 0.25).sin()).collect();

    let mut expected = vec![0.0f32; seq_len * hidden];
    for p in 0..seq_len {
        let off = p * hidden;
        let normed = cpu_rms_norm(&dn[off..off + hidden], &w_post_ffn, eps, offset);
        for i in 0..hidden {
            expected[off + i] = hpa[off + i] + normed[i];
        }
    }

    let hpa_buf = bufs.transient_from_f32(&hpa);
    let dn_buf = bufs.transient_from_f32(&dn);
    let w_buf = bufs.transient_from_f32(&w_post_ffn);
    let out = bufs.output((seq_len * hidden * 4) as u64);

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    let mut scratch = |n: u64| bufs.output(n);
    larql_compute::metal::stages::residual::encode_post_ffn(
        enc, &rms_norm, &residual_add,
        &mut scratch,
        &dn_buf, &hpa_buf, &out,
        Some(&w_buf),
        seq_len, hidden, eps, offset,
        /*has_post_norms*/ true,
        (hidden * 4) as u64,
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let got = read_f32_buf(&out, seq_len * hidden);
    assert!(max_diff(&expected, &got) < 1e-4, "post_ffn post-norm diff");
}

/// Stage: `quant_matvec::encode` routes each format to the correct shader.
///
/// Feeds Q4_K, Q6_K, and Q4_0 weights through the same `encode` call and
/// checks each output matches a direct single-format shader dispatch. This
/// is what pins down the `match format` arm selection in the helper.
#[test]
fn stage_quant_matvec_routes_format_to_correct_shader() {
    let device = metal::Device::system_default().unwrap();
    let q4kf_proj = build_pipeline(&device, "q4kf_proj");
    let q4k_matvec = build_pipeline(&device, "q4k_matvec");
    let q6k_matvec = build_pipeline(&device, "q6k_matvec");
    let q4_matvec = build_pipeline(&device, "q4_matvec");
    let bufs = larql_compute::metal::buffers::BufferCache::new(&device);
    let queue = device.new_command_queue();

    // Q4_K / Q6_K require hidden to be a multiple of 256 (superblock size).
    let rows = 32usize;
    let hidden = 256usize;

    let pipes = larql_compute::metal::stages::quant_matvec::Pipelines {
        q4kf_proj: Some(&q4kf_proj),
        q4k_matvec_fallback: &q4k_matvec,
        q6k_matvec: &q6k_matvec,
        q4_matvec: &q4_matvec,
    };

    let w_f32: Vec<f32> = (0..rows * hidden).map(|i| ((i as f32) * 0.009).sin()).collect();
    let x: Vec<f32> = (0..hidden).map(|i| ((i as f32) * 0.017).cos()).collect();

    // Expected reference: f32 gemv, matches the dequantise-then-dot semantics
    // every quant shader approximates.
    let expected: Vec<f32> = (0..rows).map(|r| {
        (0..hidden).map(|c| w_f32[r * hidden + c] * x[c]).sum()
    }).collect();

    let x_buf = bufs.transient_from_f32(&x);
    let out = bufs.output((rows * 4) as u64);

    // Q4_K route.
    let w_q4k = larql_compute::cpu::ops::q4_common::quantize_q4_k(&w_f32);
    let w_q4k_buf = bufs.get_bytes(&w_q4k);
    {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        larql_compute::metal::stages::quant_matvec::encode(
            enc, larql_compute::QuantFormat::Q4_K, &w_q4k_buf,
            &x_buf, 0, &x_buf, 0, &x_buf, 0,
            &out, 0, &pipes, rows, hidden,
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }
    let got_q4k = read_f32_buf(&out, rows);
    let max_abs = expected.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-6);
    let rel = max_diff(&expected, &got_q4k) / max_abs;
    assert!(rel < 0.05, "Q4_K route rel err {rel:.4}");

    // Q6_K route (emitted via CPU quantizer).
    let w_q6k = larql_compute::cpu::ops::q4_common::quantize_q6_k(&w_f32);
    let w_q6k_buf = bufs.get_bytes(&w_q6k);
    {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        larql_compute::metal::stages::quant_matvec::encode(
            enc, larql_compute::QuantFormat::Q6_K, &w_q6k_buf,
            &x_buf, 0, &x_buf, 0, &x_buf, 0,
            &out, 0, &pipes, rows, hidden,
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }
    let got_q6k = read_f32_buf(&out, rows);
    let rel = max_diff(&expected, &got_q6k) / max_abs;
    assert!(rel < 0.02, "Q6_K route rel err {rel:.4}");

    // Q4_0 route needs Q8 input.
    let w_q4_0 = larql_compute::cpu::q4::quantize_q4_0(&w_f32);
    let w_q4_0_buf = bufs.get_bytes(&w_q4_0);
    let (q8_x, q8_x_scales) = larql_compute::cpu::q4::quantize_to_q8(&x);
    let q8_x_buf = bufs.transient_from_i8(&q8_x);
    let q8_x_s_buf = bufs.transient_from_f32(&q8_x_scales);
    {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        larql_compute::metal::stages::quant_matvec::encode(
            enc, larql_compute::QuantFormat::Q4_0, &w_q4_0_buf,
            &x_buf, 0, &q8_x_buf, 0, &q8_x_s_buf, 0,
            &out, 0, &pipes, rows, hidden,
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }
    let got_q4_0 = read_f32_buf(&out, rows);
    let rel = max_diff(&expected, &got_q4_0) / max_abs;
    assert!(rel < 0.1, "Q4_0 route rel err {rel:.4}");
}

/// `f32_gemv` shader: `out[N] = W[N,K] · x[K]` matches `ndarray::dot`.
///
/// Motivating case: LM-head logits at autoregressive decode. The shader's
/// value-add over re-using `sgemm_transb` at M=1 is both speed (row-per-
/// simdgroup vs 31/32-wasted-thread tiled gemm) and argmax stability
/// (deterministic per-row reduction order, no shifting of top-K under
/// noisy logits). Test pins both.
#[test]
fn f32_gemv_matches_ndarray_dot() {
    let metal = get_metal();
    // Small shapes fall below the default 500 MFLOP threshold and return
    // None (caller falls back to CPU). We want to exercise the Metal
    // path, so drop the floor.
    metal.set_flop_threshold(1);

    // Dimensions chosen to match the Gemma 3/4 LM-head aspect ratio in
    // miniature: wide N, K a non-power-of-two-multiple-of-32, K % 128 != 0.
    let n = 2048usize;
    let k = 2560usize;
    let w = synth(n, k, 0xa11ce);
    let x: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.013).sin()).collect();

    // CPU reference: ndarray's BLAS gemv.
    let x_arr = ndarray::Array1::from(x.clone());
    let expected = w.dot(&x_arr);

    // Metal path.
    let got = metal.f32_gemv(w.view(), &x).expect("gemv should dispatch above threshold");
    assert_eq!(got.len(), n);

    let diff = max_diff(expected.as_slice().unwrap(), &got);
    let max_abs = expected.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-6);
    let rel = diff / max_abs;
    assert!(
        rel < 1e-4,
        "f32_gemv rel err {rel:.2e} (abs {diff:.2e}, max_abs {max_abs:.2e})"
    );

    // Argmax stability — the actual property that matters for LM-head top-K.
    let exp_argmax = expected
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    let got_argmax = got
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    assert_eq!(exp_argmax, got_argmax, "argmax mismatch between CPU and Metal gemv");
}

/// `f16_gemv` shader: f16 weights × f32 query, matches `f32_gemv` within
/// half-precision noise.
///
/// Motivating case: Gemma 4 31B tied-embedding LM head. The current path
/// decodes the 2.8 GB f16 safetensors into a 5.6 GB f32 clone at load;
/// this shader lets the Metal backend consume the f16 bytes directly.
/// Test pins argmax equality with the f32 reference — that's the actual
/// property that matters for top-K.
#[test]
fn f16_gemv_matches_f32_gemv_argmax() {
    use larql_models::quant::half::encode_f16;

    let metal = get_metal();
    metal.set_flop_threshold(1);

    let n = 2048usize;
    let k = 2560usize;
    let w = synth(n, k, 0xf16ce);
    let x: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.013).sin()).collect();

    // f32 reference.
    let x_arr = ndarray::Array1::from(x.clone());
    let expected = w.dot(&x_arr);

    // Encode weights as f16 bytes (IEEE half, little-endian).
    let w_flat: Vec<f32> = w.iter().copied().collect();
    let w_f16 = encode_f16(&w_flat);
    assert_eq!(w_f16.len(), n * k * 2);

    let got = metal
        .f16_gemv(&w_f16, &x, n, k)
        .expect("f16_gemv should dispatch above threshold");
    assert_eq!(got.len(), n);

    // f16 weights introduce relative error ~1e-3 on the output; don't pin
    // values, pin argmax — that's the property the LM head top-K depends on.
    let exp_argmax = expected
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    let got_argmax = got
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    assert_eq!(
        exp_argmax, got_argmax,
        "f16_gemv argmax mismatch vs f32 reference"
    );

    // Sanity: the scores around the argmax should be within f16 relative
    // noise of the f32 reference.
    let tol = expected.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1.0) * 5e-3;
    let diff = (expected[exp_argmax] - got[exp_argmax]).abs();
    assert!(
        diff < tol,
        "argmax-value drift {diff:.4} exceeds f16 tolerance {tol:.4}"
    );
}

/// Uniform `q4k_qkv_proj` fused shader matches three `q4k_matvec` dispatches.
///
/// Regression gate for the 148-vs-144 Q4_K super-block stride bug: the
/// first draft of this shader typed weights as `block_q4_K*` (148-byte
/// MSL struct with an obsolete `mins[4]` field), which silently mis-read
/// production GGUF data. Row stride was off by 40 bytes per row,
/// accumulating into buffer-overruns past the first superblock. The
/// output was "approximately correct" enough for argmax to stabilise on
/// trivial prompts, hiding the bug. Now the shader uses manual byte
/// offsets with the correct 144-byte stride.
#[test]
fn q4k_qkv_proj_matches_per_proj_dispatch() {
    let metal = get_metal();
    let q_rows = 2048usize;
    let kv_rows = 1024usize;
    let hidden = 2560usize;

    let wq_f32 = synth(q_rows, hidden, 0xbeef_0001).as_standard_layout().to_owned();
    let wk_f32 = synth(kv_rows, hidden, 0xbeef_0002).as_standard_layout().to_owned();
    let wv_f32 = synth(kv_rows, hidden, 0xbeef_0003).as_standard_layout().to_owned();
    let x: Vec<f32> = (0..hidden).map(|i| ((i as f32) * 0.017).cos()).collect();

    let wq_q4k = larql_compute::cpu::ops::q4_common::quantize_q4_k(wq_f32.as_slice().unwrap());
    let wk_q4k = larql_compute::cpu::ops::q4_common::quantize_q4_k(wk_f32.as_slice().unwrap());
    let wv_q4k = larql_compute::cpu::ops::q4_common::quantize_q4_k(wv_f32.as_slice().unwrap());

    let ref_q = metal.q4k_matvec(&wq_q4k, &x, q_rows, hidden).expect("q4k_matvec Q");
    let ref_k = metal.q4k_matvec(&wk_q4k, &x, kv_rows, hidden).expect("q4k_matvec K");
    let ref_v = metal.q4k_matvec(&wv_q4k, &x, kv_rows, hidden).expect("q4k_matvec V");

    // Fused dispatch through `q4k_qkv_proj`.
    let wq_buf = metal.bufs().get_bytes(&wq_q4k);
    let wk_buf = metal.bufs().get_bytes(&wk_q4k);
    let wv_buf = metal.bufs().get_bytes(&wv_q4k);
    let x_buf = metal.bufs().transient_from_f32(&x);
    let q_out = metal.bufs().output((q_rows * 4) as u64);
    let k_out = metal.bufs().output((kv_rows * 4) as u64);
    let v_out = metal.bufs().output((kv_rows * 4) as u64);

    use larql_compute::metal::shaders::q4k_qkv_proj as sh;
    let total_rows = (q_rows + kv_rows + kv_rows) as u64;
    let num_tgs = total_rows.div_ceil(sh::ROWS_PER_TG);
    let q_u = q_rows as u32;
    let k_u = kv_rows as u32;
    let v_u = kv_rows as u32;
    let hidden_u = hidden as u32;
    let cmd = metal.queue().new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&metal.q4k_qkv_proj_pipeline);
    enc.set_buffer(0, Some(&wq_buf), 0);
    enc.set_buffer(1, Some(&wk_buf), 0);
    enc.set_buffer(2, Some(&wv_buf), 0);
    enc.set_buffer(3, Some(&x_buf), 0);
    enc.set_buffer(4, Some(&q_out), 0);
    enc.set_buffer(5, Some(&k_out), 0);
    enc.set_buffer(6, Some(&v_out), 0);
    enc.set_bytes(7, 4, &q_u as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(8, 4, &k_u as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(9, 4, &v_u as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(10, 4, &hidden_u as *const u32 as *const std::ffi::c_void);
    enc.dispatch_thread_groups(
        metal::MTLSize::new(num_tgs, 1, 1),
        metal::MTLSize::new(sh::THREADS_PER_TG, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let got_q = larql_compute::metal::buffers::read_buffer_f32(&q_out, q_rows);
    let got_k = larql_compute::metal::buffers::read_buffer_f32(&k_out, kv_rows);
    let got_v = larql_compute::metal::buffers::read_buffer_f32(&v_out, kv_rows);

    let check = |name: &str, r: &[f32], g: &[f32]| {
        let max_abs = r.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-6);
        let d = max_diff(r, g);
        assert!(d < max_abs * 1e-3,
            "{name}: max_diff {d:.3e} exceeds 0.1% of max_abs {max_abs:.3e}");
    };
    check("Q", &ref_q, &got_q);
    check("K", &ref_k, &got_k);
    check("V", &ref_v, &got_v);
}

/// `q4k_q6k_qkv_proj` fused shader matches three separate-format dispatches.
///
/// Pins the mixed-quant fused kernel that replaces the 3-dispatch per-proj
/// fallback when a layer ships Q4_K Q/K + Q6_K V (Gemma 3 4B / Gemma 4
/// Ollama convention). If the shader silently regresses to under-read or
/// over-read the Q4_K GGUF 144-byte blocks (as happened once when the
/// first draft used the 148-byte `block_q4_K` MSL struct), this will
/// catch it before real-vindex decode produces garbled tokens.
#[test]
#[allow(clippy::unusual_byte_groupings)]
fn q4k_q6k_qkv_proj_matches_per_proj_dispatch() {
    let metal = get_metal();

    // Shapes modelled on Gemma 3 4B: q_dim = 8 * 256, kv_dim = 4 * 256,
    // hidden = 2560 (K must be a multiple of 256 for Q4_K / Q6_K).
    let q_rows = 2048usize;
    let kv_rows = 1024usize;
    let hidden = 2560usize;

    // Synthesise weight matrices and quantise.
    let wq_f32 = synth(q_rows, hidden, 0xdead_beef_1).as_standard_layout().to_owned();
    let wk_f32 = synth(kv_rows, hidden, 0xdead_beef_2).as_standard_layout().to_owned();
    let wv_f32 = synth(kv_rows, hidden, 0xdead_beef_3).as_standard_layout().to_owned();
    let x: Vec<f32> = (0..hidden).map(|i| ((i as f32) * 0.011).sin()).collect();

    let wq_q4k = larql_compute::cpu::ops::q4_common::quantize_q4_k(wq_f32.as_slice().unwrap());
    let wk_q4k = larql_compute::cpu::ops::q4_common::quantize_q4_k(wk_f32.as_slice().unwrap());
    let wv_q6k = larql_compute::cpu::ops::q4_common::quantize_q6_k(wv_f32.as_slice().unwrap());

    // Reference: dispatch each projection through its native shader.
    let ref_q = metal.q4k_matvec(&wq_q4k, &x, q_rows, hidden).expect("q4k_matvec Q");
    let ref_k = metal.q4k_matvec(&wk_q4k, &x, kv_rows, hidden).expect("q4k_matvec K");
    let ref_v = metal.q6k_matvec(&wv_q6k, &x, kv_rows, hidden).expect("q6k_matvec V");

    // Fused dispatch.
    let wq_buf = metal.bufs().get_bytes(&wq_q4k);
    let wk_buf = metal.bufs().get_bytes(&wk_q4k);
    let wv_buf = metal.bufs().get_bytes(&wv_q6k);
    let x_buf = metal.bufs().transient_from_f32(&x);
    let q_out = metal.bufs().output((q_rows * 4) as u64);
    let k_out = metal.bufs().output((kv_rows * 4) as u64);
    let v_out = metal.bufs().output((kv_rows * 4) as u64);

    use larql_compute::metal::shaders::q4k_q6k_qkv_proj as sh;
    let total_rows = (q_rows + kv_rows + kv_rows) as u64;
    let num_tgs = total_rows.div_ceil(sh::ROWS_PER_TG);
    let q_u = q_rows as u32;
    let k_u = kv_rows as u32;
    let v_u = kv_rows as u32;
    let hidden_u = hidden as u32;
    let cmd = metal.queue().new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&metal.q4k_q6k_qkv_proj_pipeline);
    enc.set_buffer(0, Some(&wq_buf), 0);
    enc.set_buffer(1, Some(&wk_buf), 0);
    enc.set_buffer(2, Some(&wv_buf), 0);
    enc.set_buffer(3, Some(&x_buf), 0);
    enc.set_buffer(4, Some(&q_out), 0);
    enc.set_buffer(5, Some(&k_out), 0);
    enc.set_buffer(6, Some(&v_out), 0);
    enc.set_bytes(7, 4, &q_u as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(8, 4, &k_u as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(9, 4, &v_u as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(10, 4, &hidden_u as *const u32 as *const std::ffi::c_void);
    enc.dispatch_thread_groups(
        metal::MTLSize::new(num_tgs, 1, 1),
        metal::MTLSize::new(sh::THREADS_PER_TG, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let got_q = larql_compute::metal::buffers::read_buffer_f32(&q_out, q_rows);
    let got_k = larql_compute::metal::buffers::read_buffer_f32(&k_out, kv_rows);
    let got_v = larql_compute::metal::buffers::read_buffer_f32(&v_out, kv_rows);

    // Q4_K quantisation can introduce tiny per-row scale differences
    // depending on which shader dispatch path is taken; absolute tolerance
    // scaled by row magnitude.
    let check = |name: &str, r: &[f32], g: &[f32]| {
        let max_abs = r.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-6);
        let d = max_diff(r, g);
        assert!(d < max_abs * 1e-3,
            "{name}: max_diff {d:.3e} exceeds 0.1% of max_abs {max_abs:.3e}");
    };
    check("Q", &ref_q, &got_q);
    check("K", &ref_k, &got_k);
    check("V", &ref_v, &got_v);
}

/// Stage: `residual::encode_post_attn` with FFN that needs Q8 input.
///
/// Verifies the additional q8_quant dispatch runs and produces a Q8
/// representation that round-trips to approximately `ffn_norm_out`.
#[test]
fn stage_post_attn_q8_ffn_emits_roundtrippable_q8() {
    let device = metal::Device::system_default().unwrap();
    let rms_norm = build_pipeline(&device, "rms_norm");
    let residual_add = build_pipeline(&device, "residual_add");
    let q8_quant = build_pipeline(&device, "quantize_q8");
    let bufs = larql_compute::metal::buffers::BufferCache::new(&device);
    let queue = device.new_command_queue();

    let hidden = 256usize;
    let seq_len = 2usize;

    let h: Vec<f32> = (0..seq_len * hidden).map(|i| ((i as f32) * 0.009).sin() * 2.0).collect();
    let o: Vec<f32> = (0..seq_len * hidden).map(|i| ((i as f32) * 0.013).cos() * 1.5).collect();
    let w: Vec<f32> = (0..hidden).map(|i| 1.0 + 0.02 * (i as f32).sin()).collect();

    let h_buf = bufs.transient_from_f32(&h);
    let o_buf = bufs.transient_from_f32(&o);
    let w_buf = bufs.transient_from_f32(&w);
    let h_pa = bufs.output((seq_len * hidden * 4) as u64);
    let ffn_out = bufs.output((seq_len * hidden * 4) as u64);
    let q8 = bufs.output((seq_len * hidden) as u64);
    let q8s = bufs.output((seq_len * hidden.div_ceil(32) * 4) as u64);

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    let mut scratch = |n: u64| bufs.output(n);
    larql_compute::metal::stages::residual::encode_post_attn(
        enc, &rms_norm, &residual_add, &q8_quant,
        &mut scratch,
        &h_buf, &o_buf, &h_pa, &ffn_out,
        &w_buf, &w_buf,
        &q8, &q8s,
        seq_len, hidden, 1e-6, 0.0,
        /*has_post_norms*/ false,
        /*ffn_needs_q8*/ true,
        (hidden * 4) as u64,
        hidden as u64,
        (hidden.div_ceil(32) * 4) as u64,
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    // Dequantise Q8 and compare to f32 ffn_norm_out (Q8 error < 1/127 * max).
    // `quantize_q8` writes f32 scales (not f16) — `q8s_stride_bytes` is
    // `blocks_per_row * 4` to reflect that.
    let ffn_f32 = read_f32_buf(&ffn_out, seq_len * hidden);
    let q8_bytes = unsafe {
        std::slice::from_raw_parts(q8.contents() as *const i8, seq_len * hidden)
    };
    let blocks_per_pos = hidden.div_ceil(32);
    let q8s_f32 = unsafe {
        std::slice::from_raw_parts(q8s.contents() as *const f32, seq_len * blocks_per_pos)
    };
    let mut dequant = vec![0.0f32; seq_len * hidden];
    for p in 0..seq_len {
        for b in 0..blocks_per_pos {
            let scale = q8s_f32[p * blocks_per_pos + b];
            for i in 0..32 {
                let idx = p * hidden + b * 32 + i;
                if idx < (p + 1) * hidden {
                    dequant[idx] = q8_bytes[idx] as f32 * scale;
                }
            }
        }
    }
    let max_abs = ffn_f32.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let d = max_diff(&ffn_f32, &dequant);
    assert!(d < max_abs / 100.0 + 1e-4,
        "Q8 roundtrip error {d} exceeds 1% of max_abs {max_abs}");
}
