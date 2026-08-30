//! Is the ~200 µs between command buffers Metal's cost, or an empty queue?
//!
//! Decode measures a ~200 µs gap between `GPUEndTime[n]` and
//! `GPUStartTime[n+1]`, 24 times per token, after sleep/wake (~75 µs,
//! `LARQL_SPIN_WAIT`) and explicit residency (nil, `LARQL_RESIDENCY_SET`) were
//! accounted for. Two explanations remain and they imply completely different
//! work:
//!
//! - **Intrinsic**: Metal costs ~200 µs to transition between command buffers,
//!   so the fix is fewer command buffers.
//! - **Starvation**: the queue simply goes empty, because the host must learn
//!   layer N's route before command buffer N+1 can exist. The fix is to remove
//!   the host from the dependency chain — GPU-resident routing — and the
//!   command-buffer count barely matters.
//!
//! This separates them with no model and no decode path: identical command
//! buffers, identical GPU work, varying only *when* they are committed.
//!
//! - `JUST_IN_TIME` reproduces decode's shape: commit one, wait, build the
//!   next. The queue is empty whenever the host is thinking.
//! - `PRE_QUEUED` calls `enqueue()` on every buffer up front to reserve queue
//!   position, but still encodes and commits just-in-time — the shape decode
//!   could adopt WITHOUT giving up host routing.
//! - `PRE_COMMITTED` builds and commits all of them first, then waits once.
//!   The queue is never empty.
//!
//! A pre-committed run with near-zero gaps convicts starvation. Gaps that
//! survive pre-commitment convict Metal. `PRE_QUEUED` decides how expensive
//! the fix has to be: measured at -4% of the gap, so reserving order is not a
//! substitute for the work existing — the route has to stop being a host
//! decision.
//!
//! Representative GPU work matters: the earlier empty-command-buffer control
//! (`LARQL_EXTRA_BARRIERS`) measured 15 µs, which is a true statement about
//! empty buffers and a misleading one about loaded ones.

#![cfg(target_os = "macos")]

use metal::objc::{msg_send, sel, sel_impl};
use metal::{
    CommandBufferRef, CompileOptions, ComputePipelineDescriptor, Device, MTLResourceOptions,
    MTLSize,
};

/// Reads `stride`-separated floats so the kernel is memory-bound like the
/// expert matvec it stands in for, rather than ALU-bound.
const SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void touch(device const float* src   [[buffer(0)]],
                  device float*       dst   [[buffer(1)]],
                  constant uint&      n     [[buffer(2)]],
                  uint gid [[thread_position_in_grid]],
                  uint gsz [[threads_per_grid]]) {
    float acc = 0.0f;
    for (uint i = gid; i < n; i += gsz) { acc += src[i]; }
    dst[gid] = acc;
}
"#;

/// Floats read per command buffer — sized so one buffer takes roughly the
/// ~460 µs of GPU time a real decode layer takes.
const FLOATS: usize = 22 * 1024 * 1024;
const THREADS: u64 = 65536;
const N_BUFFERS: usize = 24;

fn gpu_window(cmd: &CommandBufferRef) -> (f64, f64) {
    unsafe {
        let start: f64 = msg_send![cmd, GPUStartTime];
        let end: f64 = msg_send![cmd, GPUEndTime];
        (start, end)
    }
}

/// Mean inter-buffer gap in µs, plus mean GPU-busy per buffer in µs.
fn gaps_us(windows: &[(f64, f64)]) -> (f64, f64) {
    let mut gap_total = 0.0;
    for w in windows.windows(2) {
        let gap = (w[1].0 - w[0].1) * 1e6;
        if gap > 0.0 {
            gap_total += gap;
        }
    }
    let busy: f64 = windows.iter().map(|(s, e)| (e - s) * 1e6).sum();
    (
        gap_total / (windows.len() - 1) as f64,
        busy / windows.len() as f64,
    )
}

#[test]
#[ignore = "timing experiment; run explicitly with --ignored --nocapture"]
fn precommitted_command_buffers_close_the_gap_that_just_in_time_leaves() {
    let device = Device::system_default().expect("Metal device");
    let queue = device.new_command_queue();
    let lib = device
        .new_library_with_source(SHADER, &CompileOptions::new())
        .expect("compile");
    let func = lib.get_function("touch", None).expect("fn");
    let desc = ComputePipelineDescriptor::new();
    desc.set_compute_function(Some(&func));
    let pipe = device
        .new_compute_pipeline_state(&desc)
        .expect("pipeline state");

    let bytes = (FLOATS * std::mem::size_of::<f32>()) as u64;
    let src = device.new_buffer(bytes, MTLResourceOptions::StorageModeShared);
    let dst = device.new_buffer(
        THREADS * std::mem::size_of::<f32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let n = FLOATS as u32;

    let encode = |cmd: &CommandBufferRef| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipe);
        enc.set_buffer(0, Some(&src), 0);
        enc.set_buffer(1, Some(&dst), 0);
        enc.set_bytes(
            2,
            std::mem::size_of::<u32>() as u64,
            &n as *const u32 as *const _,
        );
        enc.dispatch_threads(
            MTLSize::new(THREADS, 1, 1),
            MTLSize::new(pipe.thread_execution_width(), 1, 1),
        );
        enc.end_encoding();
    };

    // Warm: first submission pays one-time pipeline/residency costs that
    // belong to neither arm.
    for _ in 0..3 {
        let cmd = queue.new_command_buffer();
        encode(cmd);
        cmd.commit();
        cmd.wait_until_completed();
    }

    // ── C: JUST IN TIME — decode's shape. Queue empty while host works. ──
    let mut jit = Vec::with_capacity(N_BUFFERS);
    for _ in 0..N_BUFFERS {
        let cmd = queue.new_command_buffer().to_owned();
        encode(&cmd);
        cmd.commit();
        cmd.wait_until_completed();
        jit.push(gpu_window(&cmd));
    }

    // ── A: PRE-QUEUED — `enqueue()` reserves each buffer's place in the
    // queue up front, but encoding and commit still happen just-in-time,
    // exactly as decode must because the route is not known earlier.
    //
    // This is the arm that decides how big the fix has to be. If reserving
    // queue position alone closes the gap, larql can keep CPU routing and
    // pre-reserve slots. If it does not, the work genuinely has to exist
    // earlier, which means the route has to stop being a host decision.
    let mut prequeued_cmds = Vec::with_capacity(N_BUFFERS);
    for _ in 0..N_BUFFERS {
        let cmd = queue.new_command_buffer().to_owned();
        cmd.enqueue();
        prequeued_cmds.push(cmd);
    }
    let mut prequeued = Vec::with_capacity(N_BUFFERS);
    for cmd in &prequeued_cmds {
        encode(cmd);
        cmd.commit();
        cmd.wait_until_completed();
        prequeued.push(gpu_window(cmd));
    }

    // ── B: PRE-COMMITTED — same buffers, same work, queue never empty. ──
    let mut pre_cmds = Vec::with_capacity(N_BUFFERS);
    for _ in 0..N_BUFFERS {
        let cmd = queue.new_command_buffer().to_owned();
        encode(&cmd);
        cmd.commit();
        pre_cmds.push(cmd);
    }
    pre_cmds
        .last()
        .expect("at least one buffer")
        .wait_until_completed();
    let pre: Vec<(f64, f64)> = pre_cmds.iter().map(|c| gpu_window(c)).collect();

    let (jit_gap, jit_busy) = gaps_us(&jit);
    let (pre_gap, pre_busy) = gaps_us(&pre);
    let (enq_gap, enq_busy) = gaps_us(&prequeued);

    println!("\n=== command-buffer transition cost, {N_BUFFERS} buffers ===");
    println!("just-in-time   gap {jit_gap:8.1} us   gpu-busy/cb {jit_busy:8.1} us");
    println!("pre-queued     gap {enq_gap:8.1} us   gpu-busy/cb {enq_busy:8.1} us   (enqueue early, commit late)");
    println!("pre-committed  gap {pre_gap:8.1} us   gpu-busy/cb {pre_busy:8.1} us");
    println!(
        "\nenqueue() alone recovers {:.0}% of the just-in-time gap",
        (jit_gap - enq_gap) / (jit_gap - pre_gap) * 100.0
    );
    println!(
        "\nper-token over 24 layers: just-in-time {:.2} ms, pre-committed {:.2} ms",
        jit_gap * 24.0 / 1000.0,
        pre_gap * 24.0 / 1000.0
    );
    println!(
        "verdict: {}",
        if pre_gap < jit_gap * 0.5 {
            "STARVATION — the gap is an empty queue, not Metal's transition cost"
        } else {
            "INTRINSIC — the gap survives pre-commitment; it is Metal's own cost"
        }
    );

    // The GPU work must be comparable between arms or the gaps are not
    // comparable either — a control on the control.
    assert!(
        (jit_busy - pre_busy).abs() < jit_busy * 0.25,
        "arms did different amounts of GPU work: jit {jit_busy:.1} us vs pre {pre_busy:.1} us"
    );
    // Sized to stand in for a decode layer; a wildly different figure means
    // the experiment is not representative of what it claims to model.
    assert!(
        (100.0..3000.0).contains(&jit_busy),
        "kernel is not decode-layer-sized: {jit_busy:.1} us"
    );
}
