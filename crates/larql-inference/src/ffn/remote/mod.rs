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
mod http;
pub mod q8k_wire;
pub mod sharded;

pub use codec::RemoteLatencyStats;
pub use http::{RemoteFfnConfig, RemoteFfnError, RemoteWalkBackend, WirePreference};
pub use q8k_wire::{
    decode_q8k_batch_request, decode_q8k_batch_response, encode_q8k_batch_request,
    encode_q8k_batch_response, Q8KRequestEntry, Q8K_BATCH_CT,
};
pub use sharded::LayerShardedBackend;
