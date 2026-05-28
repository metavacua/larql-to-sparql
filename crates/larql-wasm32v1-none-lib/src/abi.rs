//! Host-guest ABI — alloc-write-solve-read pattern matching `model-compute`.
//!
//! Exported wasm32 symbols (same names as `model-compute/src/wasm/session.rs`):
//!
//! | Export         | Signature               | Purpose                         |
//! |---------------|-------------------------|---------------------------------|
//! | `alloc`       | `(i32) -> i32`          | Guest malloc; returns pointer   |
//! | `dealloc`     | `(i32, u32) -> ()`      | Guest free                      |
//! | `solve`       | `(i32, u32) -> u32`     | Dispatch request; 0 = ok        |
//! | `solution_ptr`| `() -> i32`             | Pointer to last response        |
//! | `solution_len`| `() -> u32`             | Length of last response         |
//!
//! # Wire protocol
//!
//! All integers are little-endian. Requests start with a one-byte opcode:
//!
//! | Opcode | Operation |
//! |--------|-----------|
//! | `0x01` | `dot(a, b) -> f32`                       |
//! | `0x02` | `norm(a) -> f32`                         |
//! | `0x03` | `cosine(a, b) -> f32`                    |
//! | `0x04` | `gate_knn(index, layer, query, k) -> [(u32, f32)]` |
//!
//! ## Opcode 0x01 / 0x03  (dot / cosine)
//! ```text
//! [0..4]        n_a    u32  — length of vector a
//! [4..4+n_a*4]  a      f32  — vector a
//! [4+n_a*4..]   n_b    u32
//! [..]          b      f32  — vector b
//! ```
//!
//! ## Opcode 0x02  (norm)
//! ```text
//! [0..4]     n    u32
//! [4..4+n*4] a    f32
//! ```
//!
//! ## Opcode 0x04  (gate_knn)
//! ```text
//! [0..4]   hidden_size   u32
//! [4..8]   num_layers    u32
//! for each layer i in 0..num_layers:
//!   [pos]         has_data  u8   (0 = empty, 1 = present)
//!   if has_data:
//!     [pos+1..+5] num_features  u32
//!     [pos+5]     dtype         u8  (0 = F32, 1 = F16)
//!     [pos+6..]   gate_data     raw bytes
//!                   F32: num_features * hidden_size * 4 bytes
//!                   F16: num_features * hidden_size * 2 bytes
//! [m..m+4]       query_layer  u32
//! [m+4..m+8]     query_len    u32
//! [m+8..m+8+q*4] query        f32  (query_len values)
//! [end-4..end]   k            u32
//! ```
//!
//! ## Responses
//! All responses start with a status byte (0 = ok, 1 = error).
//!
//! * Scalar result (dot/norm/cosine): `[status, f32_le]`
//! * gate_knn result: `[status, n_results_u32, (feature_u32, score_f32)...]`
//! * Error: `[1, ...utf8_message]`

use ::alloc::alloc::{alloc as heap_alloc, dealloc as heap_dealloc, Layout};

// ── Static output buffer ─────────────────────────────────────────────────────
// Single-threaded wasm context; no sync needed.
static mut SOLUTION_PTR: *mut u8 = core::ptr::null_mut();
static mut SOLUTION_LEN: usize = 0;

// ── Exported ABI ─────────────────────────────────────────────────────────────

/// Allocate `size` bytes in guest linear memory; returns the pointer.
/// Returns 0 on OOM. Returns 1 (non-null sentinel) for size == 0.
#[no_mangle]
pub extern "C" fn alloc(size: u32) -> i32 {
    if size == 0 {
        return 1;
    }
    let Ok(layout) = Layout::from_size_align(size as usize, 1) else {
        return 0;
    };
    unsafe { heap_alloc(layout) as i32 }
}

/// Free memory previously returned by `alloc(size)`.
#[no_mangle]
pub extern "C" fn dealloc(ptr: i32, size: u32) {
    if ptr <= 1 || size == 0 {
        return;
    }
    let Ok(layout) = Layout::from_size_align(size as usize, 1) else {
        return;
    };
    unsafe { heap_dealloc(ptr as *mut u8, layout) };
}

/// Process a binary request encoded at `(ptr, len)` in guest memory.
/// Writes the response to the static output buffer (read via `solution_ptr` / `solution_len`).
/// Returns 0 on success (response may still be an error payload — check status byte).
#[no_mangle]
pub extern "C" fn solve(ptr: i32, len: u32) -> u32 {
    let input = unsafe { core::slice::from_raw_parts(ptr as *const u8, len as usize) };
    // into_boxed_slice() guarantees capacity == len, so free_solution() can
    // reconstruct the exact layout via Box::from_raw without UB.
    let boxed = wire::dispatch(input).into_boxed_slice();
    let response_len = boxed.len();
    let response_ptr = ::alloc::boxed::Box::into_raw(boxed) as *mut u8;
    unsafe {
        free_solution();
        SOLUTION_PTR = response_ptr;
        SOLUTION_LEN = response_len;
    }
    0
}

/// Returns the pointer to the response buffer written by the last `solve` call.
#[no_mangle]
pub extern "C" fn solution_ptr() -> i32 {
    unsafe { SOLUTION_PTR as i32 }
}

/// Returns the byte length of the response buffer written by the last `solve` call.
#[no_mangle]
pub extern "C" fn solution_len() -> u32 {
    unsafe { SOLUTION_LEN as u32 }
}

// ── Internal helpers ─────────────────────────────────────────────────────────

unsafe fn free_solution() {
    if !SOLUTION_PTR.is_null() {
        // Reconstruct the Box<[u8]> we created via into_boxed_slice() + into_raw().
        // The slice fat-pointer (ptr, len) gives Box the correct layout for dealloc.
        let slice = core::slice::from_raw_parts_mut(SOLUTION_PTR, SOLUTION_LEN);
        drop(::alloc::boxed::Box::from_raw(slice as *mut [u8]));
        SOLUTION_PTR = core::ptr::null_mut();
        SOLUTION_LEN = 0;
    }
}

// ── Wire protocol implementation ─────────────────────────────────────────────

mod wire {
    use alloc::vec::Vec;
    use crate::{gate::{self, decode::StorageDtype, index::GateIndex}, linalg};

    pub(super) fn dispatch(input: &[u8]) -> Vec<u8> {
        let Some(&opcode) = input.first() else {
            return err(b"empty request");
        };
        let payload = &input[1..];
        match opcode {
            0x01 => op_dot(payload),
            0x02 => op_norm(payload),
            0x03 => op_cosine(payload),
            0x04 => op_gate_knn(payload),
            _ => err(b"unknown opcode"),
        }
    }

    // ── Linalg ──────────────────────────────────────────────────────────────

    fn op_dot(payload: &[u8]) -> Vec<u8> {
        (|| -> Option<Vec<u8>> {
            let (a, pos) = read_f32_vec_prefixed(payload, 0)?;
            let (b, _) = read_f32_vec_prefixed(payload, pos)?;
            Some(scalar_ok(linalg::dot(&a, &b)))
        })()
        .unwrap_or_else(|| err(b"dot: malformed"))
    }

    fn op_norm(payload: &[u8]) -> Vec<u8> {
        (|| -> Option<Vec<u8>> {
            let (a, _) = read_f32_vec_prefixed(payload, 0)?;
            Some(scalar_ok(linalg::norm(&a)))
        })()
        .unwrap_or_else(|| err(b"norm: malformed"))
    }

    fn op_cosine(payload: &[u8]) -> Vec<u8> {
        (|| -> Option<Vec<u8>> {
            let (a, pos) = read_f32_vec_prefixed(payload, 0)?;
            let (b, _) = read_f32_vec_prefixed(payload, pos)?;
            Some(scalar_ok(linalg::cosine(&a, &b)))
        })()
        .unwrap_or_else(|| err(b"cosine: malformed"))
    }

    // ── Gate KNN ────────────────────────────────────────────────────────────

    fn op_gate_knn(payload: &[u8]) -> Vec<u8> {
        (|| -> Option<Vec<u8>> {
            let (hidden_size, pos) = read_u32(payload, 0)?;
            let (num_layers, mut pos) = read_u32(payload, pos)?;

            let mut index = GateIndex::new(num_layers as usize, hidden_size as usize);

            for layer in 0..num_layers as usize {
                let (has_data, next) = read_u8(payload, pos)?;
                pos = next;
                if has_data == 0 {
                    continue;
                }
                let (num_features, next) = read_u32(payload, pos)?;
                let (dtype_byte, next) = read_u8(payload, next)?;
                pos = next;
                let dtype = match dtype_byte {
                    0 => StorageDtype::F32,
                    1 => StorageDtype::F16,
                    _ => return None,
                };
                let bytes_per_elem: usize = if dtype_byte == 0 { 4 } else { 2 };
                let data_len = (num_features as usize)
                    .checked_mul(hidden_size as usize)?
                    .checked_mul(bytes_per_elem)?;
                let gate_data = payload.get(pos..pos.checked_add(data_len)?)?;
                index.load_layer(layer, gate_data, dtype);
                pos = pos.checked_add(data_len)?;
            }

            let (query_layer, next) = read_u32(payload, pos)?;
            let (query, next) = read_f32_vec_prefixed(payload, next)?;
            let (k, _) = read_u32(payload, next)?;

            let results = gate::knn::gate_knn(&index, query_layer as usize, &query, k as usize);

            let mut out: Vec<u8> = Vec::with_capacity(1 + 4 + results.len() * 8);
            out.push(0); // ok
            out.extend_from_slice(&(results.len() as u32).to_le_bytes());
            for (feature, score) in results {
                out.extend_from_slice(&(feature as u32).to_le_bytes());
                out.extend_from_slice(&score.to_le_bytes());
            }
            Some(out)
        })()
        .unwrap_or_else(|| err(b"gate_knn: malformed"))
    }

    // ── Wire helpers ─────────────────────────────────────────────────────────

    fn read_u8(buf: &[u8], pos: usize) -> Option<(u8, usize)> {
        Some((*buf.get(pos)?, pos + 1))
    }

    fn read_u32(buf: &[u8], pos: usize) -> Option<(u32, usize)> {
        let bytes: [u8; 4] = buf.get(pos..pos + 4)?.try_into().ok()?;
        Some((u32::from_le_bytes(bytes), pos + 4))
    }

    /// Read a u32 length-prefixed f32 vector at `pos`.
    fn read_f32_vec_prefixed(buf: &[u8], pos: usize) -> Option<(Vec<f32>, usize)> {
        let (n, pos) = read_u32(buf, pos)?;
        read_f32_vec(buf, pos, n as usize)
    }

    fn read_f32_vec(buf: &[u8], pos: usize, n: usize) -> Option<(Vec<f32>, usize)> {
        let byte_len = n.checked_mul(4)?;
        let end = pos.checked_add(byte_len)?;
        let data = buf.get(pos..end)?;
        let floats: Vec<f32> = (0..n)
            .map(|i| {
                let s = i * 4;
                f32::from_le_bytes(data[s..s + 4].try_into().unwrap_or([0u8; 4]))
            })
            .collect();
        Some((floats, end))
    }

    fn scalar_ok(v: f32) -> Vec<u8> {
        let mut out = Vec::with_capacity(5);
        out.push(0); // ok
        out.extend_from_slice(&v.to_le_bytes());
        out
    }

    fn err(msg: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(1 + msg.len());
        out.push(1); // error
        out.extend_from_slice(msg);
        out
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;
    use super::wire;

    fn encode_f32_vec(v: &[f32]) -> Vec<u8> {
        let mut b = Vec::with_capacity(4 + v.len() * 4);
        b.extend_from_slice(&(v.len() as u32).to_le_bytes());
        for x in v {
            b.extend_from_slice(&x.to_le_bytes());
        }
        b
    }

    fn read_scalar_ok(response: &[u8]) -> f32 {
        assert_eq!(response[0], 0, "expected ok status");
        assert_eq!(response.len(), 5);
        f32::from_le_bytes(response[1..5].try_into().unwrap())
    }

    #[test]
    fn dot_round_trip() {
        let mut req = alloc::vec![0x01u8];
        req.extend(encode_f32_vec(&[1.0, 0.0, 0.0]));
        req.extend(encode_f32_vec(&[1.0, 0.0, 0.0]));
        let resp = wire::dispatch(&req);
        let v = read_scalar_ok(&resp);
        assert!((v - 1.0).abs() < 1e-6, "expected dot=1.0, got {v}");
    }

    #[test]
    fn norm_round_trip() {
        let mut req = alloc::vec![0x02u8];
        req.extend(encode_f32_vec(&[3.0, 4.0]));
        let resp = wire::dispatch(&req);
        let v = read_scalar_ok(&resp);
        assert!((v - 5.0).abs() < 1e-5, "expected norm=5.0, got {v}");
    }

    #[test]
    fn cosine_identical_vectors() {
        let mut req = alloc::vec![0x03u8];
        req.extend(encode_f32_vec(&[1.0, 2.0, 3.0]));
        req.extend(encode_f32_vec(&[1.0, 2.0, 3.0]));
        let resp = wire::dispatch(&req);
        let v = read_scalar_ok(&resp);
        assert!((v - 1.0).abs() < 1e-5, "expected cosine=1.0, got {v}");
    }

    #[test]
    fn gate_knn_empty_index_returns_empty() {
        use crate::gate::decode::StorageDtype;
        // Build a gate_knn request with one layer, no data
        let hidden_size = 2u32;
        let num_layers = 1u32;
        let mut req = alloc::vec![0x04u8];
        req.extend_from_slice(&hidden_size.to_le_bytes());
        req.extend_from_slice(&num_layers.to_le_bytes());
        req.push(0); // layer 0: has_data = false
        // query
        req.extend_from_slice(&(0u32).to_le_bytes()); // query_layer = 0
        req.extend(encode_f32_vec(&[1.0, 0.0])); // query
        req.extend_from_slice(&(5u32).to_le_bytes()); // k = 5
        let resp = wire::dispatch(&req);
        assert_eq!(resp[0], 0); // ok
        let n = u32::from_le_bytes(resp[1..5].try_into().unwrap());
        assert_eq!(n, 0, "empty index should return 0 results");
        let _ = StorageDtype::default(); // ensure decode module is reachable
    }

    #[test]
    fn gate_knn_one_layer_two_features() {
        // Build a GateIndex with hidden_size=2, 2 features: [1,0] and [0,1]
        let hidden_size = 2u32;
        let num_layers = 1u32;

        // gate data: [1.0, 0.0, 0.0, 1.0] as F32
        let gate_data: Vec<u8> = [1.0f32, 0.0, 0.0, 1.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let num_features = 2u32;

        let mut req = alloc::vec![0x04u8];
        req.extend_from_slice(&hidden_size.to_le_bytes());
        req.extend_from_slice(&num_layers.to_le_bytes());
        req.push(1); // has_data = true
        req.extend_from_slice(&num_features.to_le_bytes());
        req.push(0); // dtype = F32
        req.extend_from_slice(&gate_data);
        // query = [1.0, 0.0], k = 1, layer = 0
        req.extend_from_slice(&(0u32).to_le_bytes()); // query_layer
        req.extend(encode_f32_vec(&[1.0, 0.0]));      // query
        req.extend_from_slice(&(1u32).to_le_bytes()); // k

        let resp = wire::dispatch(&req);
        assert_eq!(resp[0], 0, "expected ok status");
        let n = u32::from_le_bytes(resp[1..5].try_into().unwrap());
        assert_eq!(n, 1);
        let feature = u32::from_le_bytes(resp[5..9].try_into().unwrap());
        let score = f32::from_le_bytes(resp[9..13].try_into().unwrap());
        assert_eq!(feature, 0);
        assert!((score - 1.0).abs() < 1e-5, "expected score≈1.0, got {score}");
    }

    #[test]
    fn error_on_unknown_opcode() {
        let resp = wire::dispatch(&[0xFFu8]);
        assert_eq!(resp[0], 1, "expected error status for unknown opcode");
    }

    #[test]
    fn error_on_empty_request() {
        let resp = wire::dispatch(&[]);
        assert_eq!(resp[0], 1, "expected error on empty request");
    }
}
