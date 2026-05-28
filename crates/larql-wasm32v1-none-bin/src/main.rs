// On wasm32 targets there is no OS and no std — use no_std + alloc.
// On native (cargo check / dev builds) this is a normal std binary that
// exercises the Rust functions directly (the ABI module is wasm32-only).
#![cfg_attr(target_arch = "wasm32", no_std)]
#![cfg_attr(target_arch = "wasm32", no_main)]

#[cfg(target_arch = "wasm32")]
extern crate alloc;

#[cfg(target_arch = "wasm32")]
#[global_allocator]
static ALLOC: dlmalloc::GlobalDlmalloc = dlmalloc::GlobalDlmalloc;

#[cfg(target_arch = "wasm32")]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}

#[cfg(not(target_arch = "wasm32"))]
use larql_wasm32v1_none_lib::{gate, linalg};

// ── Native entry point ────────────────────────────────────────────────────────

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    // Native smoke: exercise the Rust functions directly (the ABI module is wasm32-only).
    let a = [1.0_f32, 0.0, 0.0];
    let b = [0.0_f32, 1.0, 0.0];
    let _ = linalg::dot(&a, &b);
    let _ = linalg::norm(&a);
    let _ = linalg::cosine(&a, &b);
    let idx = gate::index::GateIndex::empty();
    let _ = gate::knn::gate_knn(&idx, 0, &a, 1);
}

// ── wasm32 entry point — ABI round-trip smoke tests ──────────────────────────

#[cfg(target_arch = "wasm32")]
#[no_mangle]
pub extern "C" fn _start() {
    smoke_linalg_abi();
    smoke_gate_knn_abi();
}

// --- linalg ABI smoke --------------------------------------------------------

#[cfg(target_arch = "wasm32")]
fn smoke_linalg_abi() {
    use larql_wasm32v1_none_lib::abi::{alloc, dealloc, solve, solution_ptr, solution_len};

    // dot([1,0,0], [1,0,0]) → 1.0
    let req = build_dot_req(&[1.0f32, 0.0, 0.0], &[1.0f32, 0.0, 0.0]);
    let ptr = alloc(req.len() as u32);
    write_to_guest(ptr, &req);
    assert_eq!(solve(ptr, req.len() as u32), 0);
    assert!((read_scalar(solution_ptr(), solution_len()) - 1.0).abs() < 1e-6);
    dealloc(ptr, req.len() as u32);

    // norm([3,4]) → 5.0
    let req = build_norm_req(&[3.0f32, 4.0]);
    let ptr = alloc(req.len() as u32);
    write_to_guest(ptr, &req);
    solve(ptr, req.len() as u32);
    assert!((read_scalar(solution_ptr(), solution_len()) - 5.0).abs() < 1e-4);
    dealloc(ptr, req.len() as u32);

    // cosine([1,0], [1,0]) → 1.0
    let req = build_cosine_req(&[1.0f32, 0.0], &[1.0f32, 0.0]);
    let ptr = alloc(req.len() as u32);
    write_to_guest(ptr, &req);
    solve(ptr, req.len() as u32);
    assert!((read_scalar(solution_ptr(), solution_len()) - 1.0).abs() < 1e-5);
    dealloc(ptr, req.len() as u32);
}

// --- gate_knn ABI smoke ------------------------------------------------------

#[cfg(target_arch = "wasm32")]
fn smoke_gate_knn_abi() {
    use larql_wasm32v1_none_lib::abi::{alloc, dealloc, solve, solution_ptr, solution_len};

    // hidden_size=2, 1 layer, 2 features ([1,0] and [0,1]) as F32
    let gate_data: alloc::vec::Vec<u8> = [1.0f32, 0.0, 0.0, 1.0]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    let req = build_gate_knn_req(2, 1, &[(true, 2, 0, &gate_data)], 0, &[1.0, 0.0], 1);
    let ptr = alloc(req.len() as u32);
    write_to_guest(ptr, &req);
    assert_eq!(solve(ptr, req.len() as u32), 0);
    let sol = read_bytes(solution_ptr(), solution_len() as usize);

    assert_eq!(sol[0], 0);
    let n = u32::from_le_bytes(sol[1..5].try_into().unwrap_or([0; 4]));
    assert_eq!(n, 1, "expected 1 result from gate_knn");
    dealloc(ptr, req.len() as u32);
}

// ── Request builders (wasm32 only) ───────────────────────────────────────────

#[cfg(target_arch = "wasm32")]
fn f32_vec_bytes(v: &[f32]) -> alloc::vec::Vec<u8> {
    let mut b = alloc::vec::Vec::with_capacity(4 + v.len() * 4);
    b.extend_from_slice(&(v.len() as u32).to_le_bytes());
    for x in v {
        b.extend_from_slice(&x.to_le_bytes());
    }
    b
}

#[cfg(target_arch = "wasm32")]
fn build_dot_req(a: &[f32], b: &[f32]) -> alloc::vec::Vec<u8> {
    let mut r = alloc::vec![0x01u8];
    r.extend(f32_vec_bytes(a));
    r.extend(f32_vec_bytes(b));
    r
}

#[cfg(target_arch = "wasm32")]
fn build_norm_req(a: &[f32]) -> alloc::vec::Vec<u8> {
    let mut r = alloc::vec![0x02u8];
    r.extend(f32_vec_bytes(a));
    r
}

#[cfg(target_arch = "wasm32")]
fn build_cosine_req(a: &[f32], b: &[f32]) -> alloc::vec::Vec<u8> {
    let mut r = alloc::vec![0x03u8];
    r.extend(f32_vec_bytes(a));
    r.extend(f32_vec_bytes(b));
    r
}

/// Build a gate_knn request.
/// `layers`: `(has_data, num_features, dtype_byte [0=F32,1=F16], gate_data)`
#[cfg(target_arch = "wasm32")]
fn build_gate_knn_req(
    hidden_size: u32,
    num_layers: u32,
    layers: &[(bool, u32, u8, &[u8])],
    query_layer: u32,
    query: &[f32],
    k: u32,
) -> alloc::vec::Vec<u8> {
    let mut r = alloc::vec![0x04u8];
    r.extend_from_slice(&hidden_size.to_le_bytes());
    r.extend_from_slice(&num_layers.to_le_bytes());
    for &(has_data, num_features, dtype, data) in layers {
        r.push(has_data as u8);
        if has_data {
            r.extend_from_slice(&num_features.to_le_bytes());
            r.push(dtype);
            r.extend_from_slice(data);
        }
    }
    r.extend_from_slice(&query_layer.to_le_bytes());
    r.extend(f32_vec_bytes(query));
    r.extend_from_slice(&k.to_le_bytes());
    r
}

// ── Memory helpers ───────────────────────────────────────────────────────────

#[cfg(target_arch = "wasm32")]
fn write_to_guest(ptr: i32, data: &[u8]) {
    let dest = unsafe { core::slice::from_raw_parts_mut(ptr as *mut u8, data.len()) };
    dest.copy_from_slice(data);
}

#[cfg(target_arch = "wasm32")]
fn read_bytes(ptr: i32, len: usize) -> alloc::vec::Vec<u8> {
    unsafe { core::slice::from_raw_parts(ptr as *const u8, len) }.to_vec()
}

#[cfg(target_arch = "wasm32")]
fn read_scalar(sol_ptr: i32, sol_len: u32) -> f32 {
    let bytes = read_bytes(sol_ptr, sol_len as usize);
    f32::from_le_bytes(bytes[1..5].try_into().unwrap_or([0u8; 4]))
}
