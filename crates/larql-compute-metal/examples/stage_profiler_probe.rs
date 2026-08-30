//! Does `StageProfiler::new` succeed, and at what sample-buffer sizes?
use larql_compute_metal::lowering::profile::StageProfiler;
fn main() {
    two_alive();
    let gpu = larql_compute_metal::MetalBackend::new().expect("gpu");
    let device = gpu.device_ref();
    for &runs in &[16usize, 256, 1024, 2048, 4096, 8192] {
        let cmd = gpu.new_lowering_command_buffer();
        let ok = StageProfiler::new(&device, cmd, runs).is_some();
        println!(
            "max_runs {runs:>5} (samples {:>6}) -> {}",
            2 * runs,
            if ok { "ok" } else { "NONE" }
        );
    }
}

/// Two profilers alive at once (the look-ahead step holds one while the
/// current token's is outstanding): does the second allocation succeed?
fn two_alive() {
    let gpu = larql_compute_metal::MetalBackend::new().expect("gpu");
    let device = gpu.device_ref();
    let a = StageProfiler::new(&device, gpu.new_lowering_command_buffer(), 2048);
    let b = StageProfiler::new(&device, gpu.new_lowering_command_buffer(), 2048);
    println!("two alive at 2048 runs: {} / {}", a.is_some(), b.is_some());
    let c = StageProfiler::new(&device, gpu.new_lowering_command_buffer(), 1024);
    println!("third at 1024 runs: {}", c.is_some());
}
