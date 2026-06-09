//! RotorQuant ↔ cudarc FFI bridge smoke test.
//!
//! Phase 1 of the long-context arc (`Step 4` in `project_gpu_attn_arc`):
//! prove that the vendored RotorQuant CUDA kernels are reachable from
//! `larql-compute` and produce a roundtrip-correct f16 → packed-quantized
//! → f32 result on real hardware. This locks down the FFI boundary —
//! future PRs wire the same calls into the qwen35 KV-cache path.
//!
//! Test is `#[ignore]` by default and only runs under `--features cuda`
//! plus an attached CUDA backend. Run on a GPU host with:
//!
//!     cargo test -p larql-compute --features cuda --test test_rotorquant_bridge -- --ignored

#![cfg(feature = "cuda")]

use cudarc::driver::{DevicePtr, DevicePtrMut};
use half::f16;
use larql_rotorquant::{
    copy_f16_to_quantized_device, dequantize_to_f32_device, quantized_device_len_bytes,
    CUDA_BLOCK_ELEMENTS,
};

/// Synthetic f16 input: 4 blocks worth (`4 * CUDA_BLOCK_ELEMENTS`),
/// values in [-1, 1] with a deterministic pattern so failures
/// reproduce.
fn synth_f16(elements: usize, seed: u64) -> Vec<f16> {
    let mut out = Vec::with_capacity(elements);
    let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
    for _ in 0..elements {
        s = s.wrapping_mul(2654435761).wrapping_add(1);
        let v = ((s >> 11) as i32 % 2048) as f32 / 2048.0; // (-1, 1)
        out.push(f16::from_f32(v));
    }
    out
}

#[test]
#[ignore = "requires CUDA hardware + attached driver — invoke with --ignored"]
fn iso3_roundtrip_preserves_cosine_above_0_98() {
    use larql_compute::cuda::CudaBackend;

    let Ok(backend) = CudaBackend::new() else {
        eprintln!("skipping: CUDA driver unavailable");
        return;
    };
    let drv = {
        // Access the driver via an internal accessor — only available
        // because this test lives inside `larql-compute` and can use
        // `CudaBackend::new()`. Tests can borrow the stream the same
        // way `cuda::attn::decode_attention` does.
        backend
    };
    let _ = &drv;
    iso3_roundtrip_inner();
}

fn iso3_roundtrip_inner() {
    use cudarc::driver::CudaContext;
    use larql_rotorquant::KvFormat;

    let ctx = CudaContext::new(0).expect("cuda ctx");
    unsafe {
        ctx.disable_event_tracking();
    }
    let stream = ctx.new_stream().expect("cuda stream");

    let n_blocks = 4_usize;
    let elements = n_blocks * CUDA_BLOCK_ELEMENTS;

    // 1. htod f16 input.
    let src_host = synth_f16(elements, 0xC0FFEE);
    let src_dev = stream.clone_htod(&src_host).expect("htod f16");

    // 2. allocate the packed quantized destination buffer.
    let quant_bytes =
        quantized_device_len_bytes(KvFormat::Iso3, elements).expect("quantized_device_len_bytes");
    let mut quant_dev = unsafe { stream.alloc::<u8>(quant_bytes) }.expect("alloc quant");

    // 3. compress: f16 → packed Iso3.
    {
        let (src_ptr, _guard_s) = src_dev.device_ptr(&stream);
        let (dst_ptr, _guard_d) = quant_dev.device_ptr_mut(&stream);
        unsafe {
            copy_f16_to_quantized_device(
                KvFormat::Iso3,
                src_ptr as *const std::ffi::c_void,
                dst_ptr as *mut std::ffi::c_void,
                elements,
                larql_rotorquant::CudaStream::from_raw(stream.cu_stream() as *mut std::ffi::c_void),
            )
        }
        .expect("compress");
    }

    // 4. allocate f32 dequant destination.
    let mut dequant_dev = unsafe { stream.alloc::<f32>(elements) }.expect("alloc dequant");

    // 5. decompress: packed Iso3 → f32.
    {
        let (src_ptr, _guard_s) = quant_dev.device_ptr(&stream);
        let (dst_ptr, _guard_d) = dequant_dev.device_ptr_mut(&stream);
        unsafe {
            dequantize_to_f32_device(
                KvFormat::Iso3,
                src_ptr as *const std::ffi::c_void,
                dst_ptr as *mut std::ffi::c_void,
                elements,
                larql_rotorquant::CudaStream::from_raw(stream.cu_stream() as *mut std::ffi::c_void),
            )
        }
        .expect("dequant");
    }

    // 6. dtoh + compare.
    stream.synchronize().expect("sync");
    let dequant_host = stream.clone_dtoh(&dequant_dev).expect("dtoh f32");
    assert_eq!(dequant_host.len(), elements);

    // Iso3 is a lossy 3-bit-per-element quantiser; the published
    // accuracy bound is cosine ≥ 0.99 per row, with realistic
    // (non-adversarial) inputs. We assert ≥ 0.98 to leave slack for
    // the synthetic uniform-noise input.
    let src_f32: Vec<f32> = src_host.iter().map(|&v| v.to_f32()).collect();
    let cos = cosine(&src_f32, &dequant_host);
    assert!(
        cos >= 0.98,
        "Iso3 roundtrip cosine {cos:.4} below 0.98 threshold"
    );
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|v| v * v).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|v| v * v).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        return 1.0;
    }
    dot / (na * nb)
}
