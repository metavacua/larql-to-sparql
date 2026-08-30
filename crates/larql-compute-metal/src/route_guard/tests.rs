use super::*;

#[cfg(target_os = "macos")]
#[test]
fn a_fresh_guard_reads_zero_and_deltas_are_relative_to_the_last_read() {
    let Some(device) = metal::Device::system_default() else {
        return;
    };
    let g = RouteGuard::new(&device);
    assert_eq!(g.total(), 0);
    assert_eq!(g.take_new_refusals(), 0);
    // Stand in for the gather kernel bumping the shared counter.
    // SAFETY: 4-byte shared buffer owned by `g`; no GPU work in flight.
    unsafe { std::ptr::write_volatile(g.counter.contents() as *mut u32, 3) };
    assert_eq!(g.total(), 3);
    assert_eq!(g.take_new_refusals(), 3);
    assert_eq!(g.take_new_refusals(), 0, "a delta is consumed once");
    unsafe { std::ptr::write_volatile(g.counter.contents() as *mut u32, 5) };
    assert_eq!(g.take_new_refusals(), 2);
}
