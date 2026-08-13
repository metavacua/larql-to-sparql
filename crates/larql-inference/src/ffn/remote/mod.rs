//! Remote FFN backend — dispatches FFN computation to a `larql-server` over HTTP.
//!
//! Wire protocol: POST `/v1/walk-ffn` with `full_output: true`. The server
//! runs the architecture-correct WalkFfn path (gate KNN → activation → up
//! gather → down projection) and returns the hidden-size FFN output per
//! layer. See [`crate::ffn::FfnBackend`] for the trait and
//! `crates/larql-server/src/routes/walk_ffn.rs` for the endpoint.
//!
//! The residual is sent row-major as `seq_len × hidden` floats; output
//! mirrors the shape. One HTTP round trip per `forward()` call.
//!
//! # Wire format
//!
//! By default `RemoteWalkBackend` uses the binary wire format
//! (`Content-Type: application/x-larql-ffn`), which eliminates JSON float
//! serialization overhead (~0.5 ms/hop on a Gemma 3 4B hidden layer).
//!
//! ## Binary request — single layer
//! ```text
//! 0       4     layer_index (u32 LE)
//! 4       4     seq_len (u32 LE)
//! 8       4     flags (u32 LE, bit 0 = full_output = 1)
//! 12      4     top_k (u32 LE, unused in full_output mode)
//! 16      N×4   residual (f32[] LE)
//! ```
//!
//! Asymmetric direction codecs (DEC funnel v0.5 §3 DEC-1A): the request
//! `Content-Type` declares the INBOUND residual format and `Accept` the
//! RETURN format, independently. f16 (`F16_CT`) and i8 (`I8_CT`) request
//! frames keep the header bytes above and encode the residual payload
//! exactly like the corresponding response dtype (f16: `u16 LE` halves;
//! i8: per-position `[scale f32][zero f32][i8×hidden]` symmetric blocks).
//!
//! ## Binary request — batch
//! ```text
//! 0       4     BATCH_MARKER = 0xFFFFFFFF
//! 4       4     num_layers (u32 LE)
//! 8       K×4   layer_indices (u32[] LE)
//! 8+K*4   4     seq_len (u32 LE)
//! 12+K*4  4     flags (u32 LE)
//! 16+K*4  4     top_k (u32 LE)
//! 20+K*4  N×4   residual (f32[] LE)
//! ```
//!
//! ## Binary response — single layer
//! ```text
//! 0       4     layer (u32 LE)
//! 4       4     seq_len (u32 LE)
//! 8       4     latency_ms (f32 LE)
//! 12      N×4   output (f32[] LE)
//! ```
//!
//! ## Binary response — batch
//! ```text
//! 0       4     BATCH_MARKER = 0xFFFFFFFF
//! 4       4     num_results (u32 LE)
//! 8       4     latency_ms (f32 LE)
//! Per result:
//!   0     4     layer (u32 LE)
//!   4     4     seq_len (u32 LE)
//!   8     4     num_output_floats (u32 LE)
//!   12    M×4   output (f32[] LE)
//! ```

pub mod codec;
// reqwest HTTP client.
#[cfg(not(target_arch = "wasm32"))]
mod http;
pub mod q8k_wire;
// LayerShardedBackend wraps http::RemoteWalkBackend directly -- native,
// not portable (correcting an earlier survey miss: no keyword hit on
// `reqwest::`/`std::fs` in this file itself, but it's still coupled).
#[cfg(not(target_arch = "wasm32"))]
pub mod sharded;
pub mod timing;

pub use codec::{
    decode_binary_request, decode_binary_request_as, decode_binary_request_f16,
    decode_binary_request_i8, decode_single_response, encode_binary_output,
    encode_binary_output_f16, encode_binary_output_i8, encode_binary_request,
    encode_binary_request_as, encode_json_full_output, DecodedFfnRequest, FfnEntry, FfnOutput,
    RemoteLatencyStats, WireFormat, BATCH_MARKER, BINARY_CT, F16_CT, I8_CT,
};
#[cfg(not(target_arch = "wasm32"))]
pub use http::{
    RemoteFfnConfig, RemoteFfnError, RemoteWalkBackend, WirePreference, STATS_PATH, WALK_FFN_PATH,
    WALK_FFN_Q8K_PATH,
};
pub use q8k_wire::{
    decode_q8k_batch_request, decode_q8k_batch_response, decode_q8k_batch_response_entries,
    encode_q8k_batch_request, encode_q8k_batch_response, Q8KRequestEntry, Q8K_BATCH_CT,
};
#[cfg(not(target_arch = "wasm32"))]
pub use sharded::LayerShardedBackend;
pub use timing::{
    append_timing_trailer, split_timing_trailer, timing_requested, TIMING_HEADER,
    TIMING_HEADER_VALUE, TIMING_TRAILER_LEN, TIMING_TRAILER_MAGIC,
};
