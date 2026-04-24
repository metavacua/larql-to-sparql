//! Shared ABI for LARQL WASM expert modules.
//!
//! Each expert is a `wasm32-wasip1` cdylib that exposes two C-ABI functions:
//!
//!   extern "C" fn larql_call(
//!       op_ptr: u32, op_len: u32,
//!       args_ptr: u32, args_len: u32,
//!   ) -> u32   /* pointer to JSON ExpertResult, or 0 = expert does not handle this op */
//!
//!   extern "C" fn larql_metadata() -> u32  /* pointer to JSON ExpertMetadata */
//!
//! All payloads on the WASM boundary are UTF-8 JSON. The operation name is a
//! language-neutral identifier (e.g. `"gcd"`, `"base64_encode"`); args are a
//! JSON object; the result is a typed JSON value. No natural-language parsing
//! happens inside experts.

#![no_std]
extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;
use serde::{Deserialize, Serialize};

pub use serde_json;
pub use serde_json::{json, Value};

// ── Result types ─────────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize)]
pub struct ExpertResult {
    pub value: Value,
    pub confidence: f32,
    pub latency_ns: u64,
    pub expert_id: String,
    pub op: String,
}

/// One op the expert handles, plus its argument key schema.
///
/// `args` lists the JSON object keys the op reads. Hosts use this to render
/// useful prompts (e.g. `gcd(a, b)` instead of just `gcd`) so models emit
/// the right arg keys instead of guessing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpSpec {
    pub name: String,
    pub args: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ExpertMetadata {
    pub id: String,
    pub tier: u8,
    pub description: String,
    pub version: String,
    /// Language-neutral op identifiers this expert responds to, with their
    /// argument key schemas.
    pub ops: Vec<OpSpec>,
}

// ── WASM ABI helpers ─────────────────────────────────────────────────────────

/// Allocate `len` bytes inside the WASM module's linear memory and return the
/// pointer. The host calls this to obtain a buffer it can write into before
/// invoking `larql_call`.
///
/// # Safety
///
/// The caller (the host) must later free the buffer by calling `larql_dealloc`
/// with the exact same `len`.
#[no_mangle]
pub unsafe extern "C" fn larql_alloc(len: u32) -> u32 {
    let layout = alloc::alloc::Layout::from_size_align(len as usize, 1).unwrap();
    alloc::alloc::alloc(layout) as u32
}

/// Free a buffer previously returned by `larql_alloc`.
///
/// # Safety
///
/// `ptr` must point to a buffer obtained from `larql_alloc` with the same
/// `len`. Double-free or size mismatch is undefined behaviour.
#[no_mangle]
pub unsafe extern "C" fn larql_dealloc(ptr: u32, len: u32) {
    let layout = alloc::alloc::Layout::from_size_align(len as usize, 1).unwrap();
    alloc::alloc::dealloc(ptr as *mut u8, layout);
}

/// Serialise `result` into WASM linear memory; return pointer to null-terminated JSON.
pub fn write_result(result: &ExpertResult) -> u32 {
    let json = serde_json::to_string(result).unwrap_or_default();
    write_cstring(&json)
}

pub fn write_metadata(meta: &ExpertMetadata) -> u32 {
    let json = serde_json::to_string(meta).unwrap_or_default();
    write_cstring(&json)
}

fn write_cstring(s: &str) -> u32 {
    let bytes = s.as_bytes();
    unsafe {
        let layout = alloc::alloc::Layout::from_size_align(bytes.len() + 1, 1).unwrap();
        let ptr = alloc::alloc::alloc(layout);
        core::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len());
        *ptr.add(bytes.len()) = 0;
        ptr as u32
    }
}

/// Read a UTF-8 string from WASM linear memory.
///
/// # Safety
///
/// `(ptr, len)` must describe a valid UTF-8 buffer inside the current WASM
/// module's linear memory that remains live for the duration of the returned
/// reference. In practice this is only called from inside the generated
/// `larql_call` trampoline, which is invoked by the host after writing into
/// a buffer obtained via `larql_alloc`.
pub unsafe fn read_str(ptr: u32, len: u32) -> &'static str {
    let slice = core::slice::from_raw_parts(ptr as *const u8, len as usize);
    core::str::from_utf8_unchecked(slice)
}

// ── Arg helpers (typed lookups into serde_json::Value objects) ───────────────

pub fn arg_f64(args: &Value, key: &str) -> Option<f64> {
    let v = args.get(key)?;
    v.as_f64().or_else(|| v.as_str()?.parse::<f64>().ok())
}

pub fn arg_i64(args: &Value, key: &str) -> Option<i64> {
    let v = args.get(key)?;
    v.as_i64()
        .or_else(|| v.as_f64().map(|x| x as i64))
        .or_else(|| v.as_str()?.parse::<i64>().ok())
}

pub fn arg_u64(args: &Value, key: &str) -> Option<u64> {
    let v = args.get(key)?;
    v.as_u64()
        .or_else(|| v.as_f64().map(|x| x as u64))
        .or_else(|| v.as_str()?.parse::<u64>().ok())
}

pub fn arg_bool(args: &Value, key: &str) -> Option<bool> {
    args.get(key)?.as_bool()
}

pub fn arg_str<'a>(args: &'a Value, key: &str) -> Option<&'a str> {
    args.get(key)?.as_str()
}

pub fn arg_list_f64(args: &Value, key: &str) -> Option<Vec<f64>> {
    let arr = args.get(key)?.as_array()?;
    arr.iter().map(|v| v.as_f64()).collect()
}

// ── Macro for WASM exports ───────────────────────────────────────────────────

/// Implement the two required WASM exports for an expert.
///
/// Usage:
/// ```ignore
/// expert_exports!(
///     id = "arithmetic",
///     tier = 1,
///     description = "Arithmetic and number theory",
///     version = "0.1.0",
///     ops = [
///         ("add",      ["a", "b"]),
///         ("gcd",      ["a", "b"]),
///         ("is_prime", ["n"]),
///     ],
///     dispatch = my_dispatch
/// );
///
/// fn my_dispatch(op: &str, args: &expert_interface::Value) -> Option<expert_interface::Value> {
///     match op {
///         "add" => Some(expert_interface::json!(
///             args.get("a")?.as_f64()? + args.get("b")?.as_f64()?
///         )),
///         _ => None,
///     }
/// }
/// ```
#[macro_export]
macro_rules! expert_exports {
    (
        id = $id:literal,
        tier = $tier:expr,
        description = $desc:literal,
        version = $version:literal,
        ops = [$( ($op:literal, [$($arg:literal),* $(,)?]) ),* $(,)?],
        dispatch = $dispatch:ident $(,)?
    ) => {
        #[no_mangle]
        pub unsafe extern "C" fn larql_call(
            op_ptr: u32,
            op_len: u32,
            args_ptr: u32,
            args_len: u32,
        ) -> u32 {
            let op = $crate::read_str(op_ptr, op_len);
            let args_str = $crate::read_str(args_ptr, args_len);
            let args: $crate::Value = $crate::serde_json::from_str(args_str)
                .unwrap_or($crate::Value::Null);
            match $dispatch(op, &args) {
                Some(value) => {
                    let result = $crate::ExpertResult {
                        value,
                        confidence: 1.0,
                        latency_ns: 0,
                        expert_id: ::std::string::String::from($id),
                        op: ::std::string::String::from(op),
                    };
                    $crate::write_result(&result)
                }
                None => 0,
            }
        }

        #[no_mangle]
        pub extern "C" fn larql_metadata() -> u32 {
            let meta = $crate::ExpertMetadata {
                id: ::std::string::String::from($id),
                tier: $tier,
                description: ::std::string::String::from($desc),
                version: ::std::string::String::from($version),
                ops: ::std::vec![
                    $(
                        $crate::OpSpec {
                            name: ::std::string::String::from($op),
                            args: ::std::vec![$(::std::string::String::from($arg)),*],
                        }
                    ),*
                ],
            };
            $crate::write_metadata(&meta)
        }
    };
}
