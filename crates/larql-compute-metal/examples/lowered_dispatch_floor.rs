//! What does one *dependent* dispatch cost inside a single encoder?
//!
//! The lowered token is one command buffer holding every kernel of the
//! stack, serially dependent. Its time splits into bytes moved (the
//! bandwidth floor) and a residual that on every container measured
//! 2026-08-19 is roughly constant across representations — ~10 ms on
//! Gemma 4 26B-A4B whether attention is f16 or NVFP4, ~8 ms on gpt-oss.
//! A constant residual is the signature of per-kernel fixed cost, not
//! bytes. This measures that fixed cost directly: a chain of N tiny
//! dependent dispatches (each reads what the previous wrote) in ONE
//! encoder, GPU span read off the command buffer, divided by N.
//!
//! Run: cargo run --release -p larql-compute-metal --example lowered_dispatch_floor
use metal::foreign_types::ForeignTypeRef;
use objc::{msg_send, sel, sel_impl};

fn gpu_span_ms(cmd: &metal::CommandBufferRef) -> f64 {
    let raw: *mut objc::runtime::Object = cmd.as_ptr() as *mut _;
    // SAFETY: both selectors exist on a completed MTLCommandBuffer.
    unsafe {
        let start: f64 = msg_send![raw, GPUStartTime];
        let end: f64 = msg_send![raw, GPUEndTime];
        (end - start) * 1e3
    }
}

fn main() {
    let Some(gpu) = larql_compute_metal::MetalBackend::new() else {
        eprintln!("no Metal device");
        std::process::exit(2);
    };
    // Vector widths the real lowerings use: gpt-oss hidden 2880, Gemma 4
    // 2816, Granite 8B 4096; and a deliberately large one to see when the
    // per-dispatch cost stops being fixed.
    for &len in &[2880usize, 4096, 65536] {
        let a = gpu.lowering_scratch(len);
        let b = gpu.lowering_scratch(len);
        for &n in &[1usize, 64, 256, 1024] {
            let mut best_gpu = f64::MAX;
            let mut best_wall = f64::MAX;
            for _rep in 0..5 {
                let cmd = gpu.new_lowering_command_buffer();
                let enc = cmd.new_compute_command_encoder();
                // Dependent chain: a+b -> a, then a+b -> b, ...
                for i in 0..n {
                    let (x, y) = if i.is_multiple_of(2) {
                        (&a, &b)
                    } else {
                        (&b, &a)
                    };
                    gpu.encode_residual_add(enc, x, y, y, len, 1.0);
                }
                enc.end_encoding();
                let t = std::time::Instant::now();
                cmd.commit();
                cmd.wait_until_completed();
                let wall = t.elapsed().as_secs_f64() * 1e3;
                best_gpu = best_gpu.min(gpu_span_ms(&cmd));
                best_wall = best_wall.min(wall);
            }
            println!(
                "len {len:>6}  n {n:>5}  gpu {best_gpu:8.3} ms  wall {best_wall:8.3} ms  per-dispatch gpu {:7.2} us",
                best_gpu * 1e3 / n as f64
            );
        }
        // Same chain, one ENCODER per dispatch: the cost of the stage
        // boundary a per-stage profiler must pay.
        for &n in &[64usize, 256, 1024] {
            let mut best_gpu = f64::MAX;
            for _rep in 0..5 {
                let cmd = gpu.new_lowering_command_buffer();
                for i in 0..n {
                    let enc = cmd.new_compute_command_encoder();
                    let (x, y) = if i.is_multiple_of(2) {
                        (&a, &b)
                    } else {
                        (&b, &a)
                    };
                    gpu.encode_residual_add(enc, x, y, y, len, 1.0);
                    enc.end_encoding();
                }
                cmd.commit();
                cmd.wait_until_completed();
                best_gpu = best_gpu.min(gpu_span_ms(&cmd));
            }
            println!(
                "len {len:>6}  n {n:>5}  ONE ENCODER PER DISPATCH  gpu {best_gpu:8.3} ms  per-dispatch {:7.2} us",
                best_gpu * 1e3 / n as f64
            );
        }
        gpu.recycle_lowering_scratch(a);
        gpu.recycle_lowering_scratch(b);
    }
}
