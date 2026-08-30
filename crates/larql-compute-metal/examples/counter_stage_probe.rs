//! Do stage-boundary timestamp samples resolve, and in what units?
//! Chain of dependent dispatches, one encoder each, start/end sampled.
use metal::foreign_types::ForeignTypeRef;
use objc::{msg_send, sel, sel_impl};

fn main() {
    let Some(gpu) = larql_compute_metal::MetalBackend::new() else {
        std::process::exit(2)
    };
    let device = metal::Device::system_default().expect("device");
    let set = device
        .counter_sets()
        .into_iter()
        .find(|s| s.name() == "timestamp")
        .expect("timestamp counter set");
    let n = 256usize;
    let desc = metal::CounterSampleBufferDescriptor::new();
    desc.set_counter_set(&set);
    desc.set_sample_count((2 * n) as u64);
    desc.set_storage_mode(metal::MTLStorageMode::Shared);
    let sb = device
        .new_counter_sample_buffer_with_descriptor(&desc)
        .expect("sample buffer");
    let len = 2880usize;
    let a = gpu.lowering_scratch(len);
    let b = gpu.lowering_scratch(len);
    let cmd = gpu.new_lowering_command_buffer();
    for i in 0..n {
        let pass = metal::ComputePassDescriptor::new();
        let att = pass
            .sample_buffer_attachments()
            .object_at(0)
            .expect("attachment 0");
        att.set_sample_buffer(&sb);
        att.set_start_of_encoder_sample_index((2 * i) as u64);
        att.set_end_of_encoder_sample_index((2 * i + 1) as u64);
        let enc = cmd.compute_command_encoder_with_descriptor(pass);
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
    let raw: *mut objc::runtime::Object = cmd.as_ptr() as *mut _;
    let (gs, ge): (f64, f64) =
        unsafe { (msg_send![raw, GPUStartTime], msg_send![raw, GPUEndTime]) };
    // resolveCounterRange: → NSData of MTLCounterResultTimestamp {u64}
    let sbp: *mut objc::runtime::Object = sb.as_ptr() as *mut _;
    let range = metal::NSRange::new(0, (2 * n) as u64);
    let data: *mut objc::runtime::Object = unsafe { msg_send![sbp, resolveCounterRange: range] };
    assert!(!data.is_null(), "resolve returned nil");
    let bytes: *const u8 = unsafe { msg_send![data, bytes] };
    let length: usize = unsafe { msg_send![data, length] };
    let ts: &[u64] = unsafe { std::slice::from_raw_parts(bytes as *const u64, length / 8) };
    println!("gpu span {:.3} ms, samples {}", (ge - gs) * 1e3, ts.len());
    let mut sum = 0u64;
    let mut gaps = 0i128;
    for i in 0..n {
        let (s, e) = (ts[2 * i], ts[2 * i + 1]);
        sum += e.saturating_sub(s);
        if i > 0 {
            gaps += s as i128 - ts[2 * i - 1] as i128;
        }
    }
    println!(
        "first start {} last end {} => total {} units; in-encoder sum {} units; gaps {} units",
        ts[0],
        ts[2 * n - 1],
        ts[2 * n - 1] - ts[0],
        sum,
        gaps
    );
    println!(
        "if units are ns: total {:.3} ms, per-stage {:.2} us, per-gap {:.2} us",
        (ts[2 * n - 1] - ts[0]) as f64 / 1e6,
        sum as f64 / 1e3 / n as f64,
        gaps as f64 / 1e3 / (n - 1) as f64
    );
    for i in [0usize, 1, 2, 100, 255] {
        println!("  stage {i}: {} units", ts[2 * i + 1] - ts[2 * i]);
    }
}
