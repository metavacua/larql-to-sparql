//! What counter sampling does this device actually support?
//!
//! A counter *set* existing is not the capability — an M3 Max exposes a
//! "timestamp" set and then asserts on
//! `sampleCountersInBuffer:atSampleIndex:withBarrier:`. The supported
//! sampling points must be queried directly.
use metal::MTLCounterSamplingPoint as P;

fn main() {
    let Some(device) = metal::Device::system_default() else {
        eprintln!("no Metal device");
        std::process::exit(2);
    };
    println!("device: {}", device.name());
    for (name, point) in [
        ("AtStageBoundary", P::AtStageBoundary),
        ("AtDrawBoundary", P::AtDrawBoundary),
        ("AtDispatchBoundary", P::AtDispatchBoundary),
        ("AtTileDispatchBoundary", P::AtTileDispatchBoundary),
        ("AtBlitBoundary", P::AtBlitBoundary),
    ] {
        println!("  {name:<24} {}", device.supports_counter_sampling(point));
    }
}
