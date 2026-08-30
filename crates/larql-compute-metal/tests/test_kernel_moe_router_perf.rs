#![cfg(target_os = "macos")]

//! Router-kernel cost measurement for the GPU-dataflow routing ladder.
//!
//! The ladder's premise is that moving routing GPU-side buys back the
//! ~5.5 ms/token queue-starvation bubble. That only nets out if the GPU
//! router itself is cheap: 24 layers × router must stay far under the
//! recovered bubble. This measures the kernel's GPU-busy cost at the
//! gpt-oss shape (32 experts × hidden 2880), back-to-back in one
//! pre-committed command buffer — the configuration the end state
//! actually runs — and compares the CPU routing work it replaces.
//!
//! Raw-device setup (same idiom as `test_cb_queue_starvation`): the
//! kernel is compiled and dispatched directly so the measurement sees
//! GPU-busy time, not the backend's readback plumbing.
//!
//! `#[ignore]`d: a measurement, not a correctness gate.
//!
//! ```bash
//! cargo test --release -p larql-compute-metal \
//!   --test test_kernel_moe_router_perf -- --ignored --nocapture
//! ```

use larql_compute_metal::shaders;
use metal::{
    CommandBufferRef, CompileOptions, ComputePipelineDescriptor, Device, MTLResourceOptions,
    MTLSize,
};
use objc::{msg_send, sel, sel_impl};

const NUM_EXPERTS: usize = 32;
const HIDDEN: usize = 2880;
const TOP_K: usize = 4;
/// One decode token's worth of router work.
const LAYERS: usize = 24;
/// Measurement repeats (whole-token batches) for a stable mean.
const REPEATS: usize = 50;

fn gpu_window(cmd: &CommandBufferRef) -> (f64, f64) {
    unsafe {
        let start: f64 = msg_send![cmd, GPUStartTime];
        let end: f64 = msg_send![cmd, GPUEndTime];
        (start, end)
    }
}

fn f32_buffer(device: &Device, data: &[f32]) -> metal::Buffer {
    device.new_buffer_with_data(
        data.as_ptr() as *const std::ffi::c_void,
        std::mem::size_of_val(data) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

#[test]
#[ignore = "timing measurement; run explicitly with --ignored --nocapture"]
fn router_kernel_cost_at_gptoss_shape() {
    let device = Device::system_default().expect("Metal device");
    let queue = device.new_command_queue();
    let src = format!("{}{}", shaders::common::HEADER, shaders::moe_router::SHADER);
    let lib = device
        .new_library_with_source(&src, &CompileOptions::new())
        .expect("compile moe_router shader");
    let func = lib.get_function("moe_router_logits", None).expect("fn");
    let desc = ComputePipelineDescriptor::new();
    desc.set_compute_function(Some(&func));
    let pipe = device
        .new_compute_pipeline_state(&desc)
        .expect("pipeline state");

    let w: Vec<f32> = (0..NUM_EXPERTS * HIDDEN)
        .map(|i| ((i as f32) * 0.0003).sin() * 0.05)
        .collect();
    let bias: Vec<f32> = (0..NUM_EXPERTS).map(|e| (e as f32 * 0.7).sin()).collect();
    let x: Vec<f32> = (0..HIDDEN).map(|i| ((i as f32) * 0.013).sin()).collect();

    let w_buf = f32_buffer(&device, &w);
    let x_buf = f32_buffer(&device, &x);
    let bias_buf = f32_buffer(&device, &bias);
    let out_buf = device.new_buffer(
        (NUM_EXPERTS * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let e_u32 = NUM_EXPERTS as u32;
    let h_u32 = HIDDEN as u32;
    let has_bias: u32 = 1;
    let num_tgs = (NUM_EXPERTS as u64).div_ceil(shaders::moe_router::ROWS_PER_TG);

    let encode_token = |cmd: &CommandBufferRef| {
        let enc = cmd.new_compute_command_encoder();
        for _ in 0..LAYERS {
            enc.set_compute_pipeline_state(&pipe);
            enc.set_buffer(0, Some(&w_buf), 0);
            enc.set_buffer(1, Some(&x_buf), 0);
            enc.set_buffer(2, Some(&bias_buf), 0);
            enc.set_buffer(3, Some(&out_buf), 0);
            enc.set_bytes(4, 4, &e_u32 as *const u32 as *const _);
            enc.set_bytes(5, 4, &h_u32 as *const u32 as *const _);
            enc.set_bytes(6, 4, &has_bias as *const u32 as *const _);
            enc.dispatch_thread_groups(
                MTLSize::new(num_tgs, 1, 1),
                MTLSize::new(shaders::moe_router::THREADS_PER_TG, 1, 1),
            );
        }
        enc.end_encoding();
    };

    // Warmup: pipeline/JIT costs belong to no arm.
    for _ in 0..3 {
        let cmd = queue.new_command_buffer();
        encode_token(cmd);
        cmd.commit();
        cmd.wait_until_completed();
    }

    let mut per_dispatch_us = Vec::with_capacity(REPEATS);
    for _ in 0..REPEATS {
        let cmd = queue.new_command_buffer();
        encode_token(cmd);
        cmd.commit();
        cmd.wait_until_completed();
        let (start, end) = gpu_window(cmd);
        per_dispatch_us.push((end - start) * 1e6 / LAYERS as f64);
    }
    per_dispatch_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = per_dispatch_us[REPEATS / 2];
    let p90 = per_dispatch_us[REPEATS * 9 / 10];

    // Reference arm: the CPU routing work the kernel replaces (projection
    // + softmax + top-k through the production oracle), timed on host.
    let cpu_us = {
        let iters = 2000;
        let t = std::time::Instant::now();
        let mut sink = 0.0f32;
        for _ in 0..iters {
            let logits = larql_compute::cpu::ops::moe::moe_router_logits(
                &x,
                std::hint::black_box(&w),
                &bias,
                NUM_EXPERTS,
            );
            let mut probs = logits;
            larql_compute::cpu::ops::moe::moe_softmax(&mut probs);
            let mut idx: Vec<usize> = (0..NUM_EXPERTS).collect();
            idx.sort_unstable_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
            sink += probs[idx[TOP_K - 1]];
        }
        std::hint::black_box(sink);
        t.elapsed().as_secs_f64() * 1e6 / iters as f64
    };

    let token_gpu_ms = median * LAYERS as f64 / 1e3;
    println!("\n=== moe_router_logits cost, E={NUM_EXPERTS} H={HIDDEN} ===");
    println!("GPU per dispatch     median {median:7.2} us   p90 {p90:7.2} us");
    println!("GPU per token ({LAYERS} layers)  {token_gpu_ms:7.3} ms");
    println!("CPU routing replaced        {cpu_us:7.2} us/layer");
    println!("budget check: {LAYERS} routers = {token_gpu_ms:.3} ms vs ~5.5 ms bubble recovered");

    // Bind the claim, loosely: the whole token's router work must cost
    // well under the bubble it exists to recover. 1 ms catches only
    // order-of-magnitude regressions; the printout carries the number.
    assert!(
        token_gpu_ms < 1.0,
        "GPU router costs {token_gpu_ms:.3} ms/token — eats the scheduling win"
    );
}

/// Occupancy diagnosis: sweep E at fixed H. Flat per-dispatch time ⇒
/// the kernel is latency/occupancy-bound at production E=32 (4 TGs) and
/// a split-H geometry would pay; linear scaling ⇒ it is already
/// bandwidth-bound and 23 µs is the floor for this read.
#[test]
#[ignore = "timing measurement; run explicitly with --ignored --nocapture"]
fn router_kernel_occupancy_probe() {
    let device = Device::system_default().expect("Metal device");
    let queue = device.new_command_queue();
    let src = format!("{}{}", shaders::common::HEADER, shaders::moe_router::SHADER);
    let lib = device
        .new_library_with_source(&src, &CompileOptions::new())
        .expect("compile moe_router shader");
    let func = lib.get_function("moe_router_logits", None).expect("fn");
    let desc = ComputePipelineDescriptor::new();
    desc.set_compute_function(Some(&func));
    let pipe = device
        .new_compute_pipeline_state(&desc)
        .expect("pipeline state");

    println!("\n=== moe_router_logits scaling probe, H={HIDDEN} ===");
    for e in [32usize, 64, 128, 256, 512] {
        let w: Vec<f32> = (0..e * HIDDEN)
            .map(|i| ((i as f32) * 0.0003).sin() * 0.05)
            .collect();
        let x: Vec<f32> = (0..HIDDEN).map(|i| ((i as f32) * 0.013).sin()).collect();
        let w_buf = f32_buffer(&device, &w);
        let x_buf = f32_buffer(&device, &x);
        // Two arms per shape. SERIAL: every dispatch writes the same out
        // buffer, so hazard tracking forces one-at-a-time execution —
        // kernel time PLUS the dependency-boundary cost decode's chain
        // pays. INDEPENDENT: distinct out buffers, dispatches free to
        // overlap — amortized kernel throughput with the boundary hidden.
        // The spread between the arms IS the boundary cost.
        let out_shared = device.new_buffer((e * 4) as u64, MTLResourceOptions::StorageModeShared);
        let outs_indep: Vec<metal::Buffer> = (0..LAYERS)
            .map(|_| device.new_buffer((e * 4) as u64, MTLResourceOptions::StorageModeShared))
            .collect();

        let e_u32 = e as u32;
        let h_u32 = HIDDEN as u32;
        let has_bias: u32 = 0;
        let num_tgs = (e as u64).div_ceil(shaders::moe_router::ROWS_PER_TG);

        let encode_batch = |cmd: &CommandBufferRef, independent: bool| {
            let enc = cmd.new_compute_command_encoder();
            for out_indep in outs_indep.iter().take(LAYERS) {
                let out = if independent { out_indep } else { &out_shared };
                enc.set_compute_pipeline_state(&pipe);
                enc.set_buffer(0, Some(&w_buf), 0);
                enc.set_buffer(1, Some(&x_buf), 0);
                enc.set_buffer(2, Some(&x_buf), 0); // has_bias=0: never read
                enc.set_buffer(3, Some(out), 0);
                enc.set_bytes(4, 4, &e_u32 as *const u32 as *const _);
                enc.set_bytes(5, 4, &h_u32 as *const u32 as *const _);
                enc.set_bytes(6, 4, &has_bias as *const u32 as *const _);
                enc.dispatch_thread_groups(
                    MTLSize::new(num_tgs, 1, 1),
                    MTLSize::new(shaders::moe_router::THREADS_PER_TG, 1, 1),
                );
            }
            enc.end_encoding();
        };

        let measure = |independent: bool| -> f64 {
            for _ in 0..3 {
                let cmd = queue.new_command_buffer();
                encode_batch(cmd, independent);
                cmd.commit();
                cmd.wait_until_completed();
            }
            let mut us = Vec::with_capacity(REPEATS);
            for _ in 0..REPEATS {
                let cmd = queue.new_command_buffer();
                encode_batch(cmd, independent);
                cmd.commit();
                cmd.wait_until_completed();
                let (start, end) = gpu_window(cmd);
                us.push((end - start) * 1e6 / LAYERS as f64);
            }
            us.sort_by(|a, b| a.partial_cmp(b).unwrap());
            us[REPEATS / 2]
        };

        let serial = measure(false);
        let indep = measure(true);
        let mb = (e * HIDDEN * 4) as f64 / 1e6;
        println!(
            "E={e:4}  TGs={num_tgs:3}  serial {serial:7.2} us  \
             independent {indep:7.2} us ({:6.1} GB/s)  boundary {:6.2} us",
            mb / indep * 1e3,
            serial - indep,
        );
    }
}

/// Rung A+B chained: projection → fused selection per layer, 24 layers
/// in one pre-committed command buffer — the routing cost the end-state
/// token actually pays. Budget: comfortably under 1 ms/token against
/// the ~5.5 ms starvation bubble the ladder recovers.
#[test]
#[ignore = "timing measurement; run explicitly with --ignored --nocapture"]
fn fused_route_chain_cost_at_gptoss_shape() {
    let device = Device::system_default().expect("Metal device");
    let queue = device.new_command_queue();
    let src = format!(
        "{}{}{}",
        shaders::common::HEADER,
        shaders::moe_router::SHADER,
        shaders::moe_router_select::SHADER,
    );
    let lib = device
        .new_library_with_source(&src, &CompileOptions::new())
        .expect("compile router+select shaders");
    let make_pipe = |name: &str| {
        let func = lib.get_function(name, None).expect("fn");
        let desc = ComputePipelineDescriptor::new();
        desc.set_compute_function(Some(&func));
        device.new_compute_pipeline_state(&desc).expect("pipeline")
    };
    let logits_pipe = make_pipe("moe_router_logits");
    let select_pipe = make_pipe("moe_router_select");

    let w: Vec<f32> = (0..NUM_EXPERTS * HIDDEN)
        .map(|i| ((i as f32) * 0.0003).sin() * 0.05)
        .collect();
    let bias: Vec<f32> = (0..NUM_EXPERTS).map(|e| (e as f32 * 0.7).sin()).collect();
    let x: Vec<f32> = (0..HIDDEN).map(|i| ((i as f32) * 0.013).sin()).collect();

    let w_buf = f32_buffer(&device, &w);
    let x_buf = f32_buffer(&device, &x);
    let bias_buf = f32_buffer(&device, &bias);
    let logits_buf = device.new_buffer(
        (NUM_EXPERTS * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let ids_buf = device.new_buffer((TOP_K * 4) as u64, MTLResourceOptions::StorageModeShared);
    let wsel_buf = device.new_buffer((TOP_K * 4) as u64, MTLResourceOptions::StorageModeShared);

    let e_u32 = NUM_EXPERTS as u32;
    let h_u32 = HIDDEN as u32;
    let k_u32 = TOP_K as u32;
    let one: u32 = 1;
    let zero: u32 = 0;
    let logits_tgs = (NUM_EXPERTS as u64).div_ceil(shaders::moe_router::ROWS_PER_TG);

    let encode_token = |cmd: &CommandBufferRef| {
        let enc = cmd.new_compute_command_encoder();
        for _ in 0..LAYERS {
            enc.set_compute_pipeline_state(&logits_pipe);
            enc.set_buffer(0, Some(&w_buf), 0);
            enc.set_buffer(1, Some(&x_buf), 0);
            enc.set_buffer(2, Some(&bias_buf), 0);
            enc.set_buffer(3, Some(&logits_buf), 0);
            enc.set_bytes(4, 4, &e_u32 as *const u32 as *const _);
            enc.set_bytes(5, 4, &h_u32 as *const u32 as *const _);
            enc.set_bytes(6, 4, &one as *const u32 as *const _);
            enc.dispatch_thread_groups(
                MTLSize::new(logits_tgs, 1, 1),
                MTLSize::new(shaders::moe_router::THREADS_PER_TG, 1, 1),
            );

            enc.set_compute_pipeline_state(&select_pipe);
            enc.set_buffer(0, Some(&logits_buf), 0);
            enc.set_buffer(1, Some(&logits_buf), 0); // has_scale=0: never read
            enc.set_buffer(2, Some(&ids_buf), 0);
            enc.set_buffer(3, Some(&wsel_buf), 0);
            enc.set_bytes(4, 4, &e_u32 as *const u32 as *const _);
            enc.set_bytes(5, 4, &k_u32 as *const u32 as *const _);
            enc.set_bytes(6, 4, &zero as *const u32 as *const _);
            enc.set_bytes(7, 4, &zero as *const u32 as *const _);
            enc.dispatch_thread_groups(
                MTLSize::new(1, 1, 1),
                MTLSize::new(shaders::moe_router_select::TG_THREADS, 1, 1),
            );
        }
        enc.end_encoding();
    };

    for _ in 0..3 {
        let cmd = queue.new_command_buffer();
        encode_token(cmd);
        cmd.commit();
        cmd.wait_until_completed();
    }
    let mut per_layer_us = Vec::with_capacity(REPEATS);
    for _ in 0..REPEATS {
        let cmd = queue.new_command_buffer();
        encode_token(cmd);
        cmd.commit();
        cmd.wait_until_completed();
        let (start, end) = gpu_window(cmd);
        per_layer_us.push((end - start) * 1e6 / LAYERS as f64);
    }
    per_layer_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = per_layer_us[REPEATS / 2];
    let p90 = per_layer_us[REPEATS * 9 / 10];
    let token_ms = median * LAYERS as f64 / 1e3;

    println!("\n=== fused route chain (logits→select), E={NUM_EXPERTS} H={HIDDEN} K={TOP_K} ===");
    println!("per layer (2 dispatches)  median {median:7.2} us   p90 {p90:7.2} us");
    println!("per token ({LAYERS} layers)     {token_ms:7.3} ms");
    println!("budget: < 1 ms/token against the ~5.5 ms bubble recovered");
    assert!(
        token_ms < 1.0,
        "A+B routing costs {token_ms:.3} ms/token — eats the scheduling win"
    );
}
