//! Teacher-forced scoring through the remote-FFN decode path — the model
//! half of the C6 wire-fidelity gate (docs/dec-funnel.md §3 DEC-1A).
//!
//! [`score_forced_with_remote_ffn`] walks a fixed token sequence through the
//! SAME per-layer remote dispatch that production decode uses
//! (`decode_token_with_moe` + pre-norm → wire → post-norm, mirroring
//! [`super::remote_ffn::generate_with_remote_ffn`]'s streaming loop), but
//! never samples: every input token is forced from the corpus and the
//! full-vocabulary lm_head scores at each position are handed to the caller,
//! which computes `-log2 p(target)`. Because nothing is sampled and the
//! arithmetic is fixed given weights + wire, the resulting bits/char is
//! deterministic — thermal state, battery, and machine load cannot move it
//! (unlike the `dec-bench replay` timing numbers).
//!
//! Wire honesty: the f32/f16/i8 arms ride the connection's configured
//! request/response codecs via `LayerShardedBackend::forward` (see
//! `RemoteFfnConfig::with_wire_formats`). The q8k arm rides
//! `forward_single_q8k` and **errors loudly** if the server lacks the q8k
//! endpoint — the production decode loop's silent f32 fallback would let a
//! drift arm labelled q8k quietly score f32 bytes, which is exactly the kind
//! of corrupted-run-with-no-signal the DEC review bans.

use super::setup::{build_grid_pipeline_setup, reset_and_preallocate_grid_kv, RemotePatch};
use crate::ffn::moe_remote::RemoteMoeError;
use crate::ffn::{FfnBackend, LayerShardedBackend};
use crate::forward::apply_norm;
use crate::layer_graph::generate::lm_head::backend_lm_head_scores;
use larql_compute::cpu::ops::q4k_q8k_dot::quantize_x_to_q8k;
use larql_compute::prelude::*;
use larql_models::ModelWeights;
use larql_vindex::VectorIndex;

/// Score `token_ids` teacher-forced with the FFN remoted at the connection's
/// wire config (or the q8k wire when `use_q8k_wire`).
///
/// For each position `t` in `0..len-1` the token `token_ids[t]` is fed
/// through one decode step (local attention on `backend`, per-layer FFN
/// round-trips on `remote`) and `on_logits(t, raw_scores)` receives the
/// full-vocabulary raw lm_head scores for predicting `token_ids[t + 1]`.
/// Scores are raw — the caller applies the arch's logits scaling / softcap,
/// matching `larql shannon`'s scorer.
///
/// Full-context: the KV cache is reset once at the start and grows over the
/// whole sequence, so every scored position's context was itself computed at
/// the wire config under test (drift compounds through context — the honest
/// end-to-end number).
///
/// Returns the number of scored positions (`token_ids.len() - 1`).
pub fn score_forced_with_remote_ffn<F>(
    weights: &ModelWeights,
    token_ids: &[u32],
    index: &VectorIndex,
    backend: &dyn ComputeBackend,
    remote: &LayerShardedBackend,
    use_q8k_wire: bool,
    mut on_logits: F,
) -> Result<usize, RemoteMoeError>
where
    F: FnMut(usize, &[f32]) -> Result<(), String>,
{
    if token_ids.len() < 2 {
        return Err(RemoteMoeError::BadResponse(
            "forced scoring needs at least two tokens (one scored target)".into(),
        ));
    }
    if use_q8k_wire
        && !weights
            .hidden_size
            .is_multiple_of(crate::ffn::Q4K_Q8K_SUPERBLOCK_ELEMS)
    {
        return Err(RemoteMoeError::BadResponse(format!(
            "q8k wire requires hidden ({}) to be a multiple of {} — \
             this model cannot run the q8k drift arm",
            weights.hidden_size,
            crate::ffn::Q4K_Q8K_SUPERBLOCK_ELEMS
        )));
    }
    let arch = &*weights.arch;
    let norm_offset = arch.norm_weight_offset();
    let setup = build_grid_pipeline_setup(weights, index, RemotePatch::Ffn)?;
    let layers = setup.layers;
    let hidden = setup.hidden;
    let intermediate = setup.intermediate;

    reset_and_preallocate_grid_kv(weights, backend);

    let n_targets = token_ids.len() - 1;
    // Q8K dispatch failures are surfaced through this cell because `moe_fn`
    // must return a Vec<f32>; the step bails as soon as the flag is set.
    let wire_err: std::cell::RefCell<Option<String>> = std::cell::RefCell::new(None);

    for (pos, &tok_id) in token_ids[..n_targets].iter().enumerate() {
        let tok_embed = crate::forward::embed_tokens_pub(weights, &[tok_id]);
        let x_tok: Vec<f32> = tok_embed.as_slice().unwrap_or(&[]).to_vec();

        let mut moe_fn = |layer: usize, h_post_attn: &[f32]| -> Vec<f32> {
            // Pre-norm once, matching the production remote-FFN contract
            // (chrishayuk/larql#114: the server expects pre-normed input).
            let h_normed = super::remote_ffn::apply_norm_for_ffn(weights, h_post_attn, layer);
            let raw_out = if use_q8k_wire {
                let q8k = quantize_x_to_q8k(&h_normed);
                match remote.forward_single_q8k(layer, &q8k) {
                    Some(v) => v,
                    None => {
                        // Loud error, never a silent f32 fallback — the
                        // arm's label must match the bytes that scored.
                        *wire_err.borrow_mut() = Some(format!(
                            "q8k wire unavailable for layer {layer} (server missing \
                             /v1/walk-ffn-q8k or request failed) — refusing to fall \
                             back to f32 inside a q8k drift arm"
                        ));
                        vec![0.0f32; hidden]
                    }
                }
            } else {
                let x = ndarray::Array2::from_shape_vec((1, hidden), h_normed)
                    .expect("shape must match hidden");
                remote.forward(layer, &x).row(0).to_vec()
            };
            super::remote_ffn::apply_post_ffn_norm(weights, &raw_out, layer)
        };

        let h_vec = backend
            .decode_token_with_moe(&layers, &x_tok, hidden, intermediate, &mut moe_fn)
            .ok_or_else(|| {
                RemoteMoeError::BadResponse(
                    "decode_token_with_moe returned None — the remote-FFN forced \
                     scorer needs the fused GPU decode path (pass --metal)"
                        .into(),
                )
            })?;
        if let Some(msg) = wire_err.borrow_mut().take() {
            return Err(RemoteMoeError::BadResponse(msg));
        }

        let h_arr = ndarray::Array2::from_shape_vec((1, hidden), h_vec)
            .map_err(|e| RemoteMoeError::BadResponse(e.to_string()))?;
        let h_normed = apply_norm(weights, &h_arr, arch.final_norm_key(), norm_offset);
        let scores = backend_lm_head_scores(weights, &h_normed.row(0).to_owned(), backend);
        if scores.is_empty() {
            return Err(RemoteMoeError::BadResponse(
                "lm_head produced no scores (empty lm_head or hidden-size mismatch)".into(),
            ));
        }
        on_logits(pos, &scores).map_err(RemoteMoeError::BadResponse)?;
    }

    Ok(n_targets)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffn::moe_remote::RemoteMoeError;
    use crate::test_utils::{make_test_vindex, make_test_weights};
    use larql_compute::CpuBackend;

    /// Minimal one-shot `/v1/stats` HTTP server so a real
    /// `LayerShardedBackend` can be constructed without a live larql-server.
    /// Serves every incoming request the same stats JSON, then exits when
    /// the listener is dropped (thread is detached; accept errors end it).
    fn spawn_stats_server(hidden: usize) -> String {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { break };
                let mut buf = [0u8; 1024];
                let _ = stream.read(&mut buf);
                let body = format!("{{\"hidden_size\": {hidden}}}");
                let resp = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n\
                     content-length: {}\r\nconnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = stream.write_all(resp.as_bytes());
            }
        });
        format!("http://{addr}")
    }

    fn connect_test_remote(hidden: usize) -> LayerShardedBackend {
        let url = spawn_stats_server(hidden);
        LayerShardedBackend::connect(&url, std::time::Duration::from_secs(5))
            .expect("connect to test stats server")
    }

    #[test]
    fn errors_on_fewer_than_two_tokens() {
        let w = make_test_weights();
        let idx = make_test_vindex(&w);
        let remote = connect_test_remote(w.hidden_size);
        let result = score_forced_with_remote_ffn(
            &w,
            &[42u32],
            &idx,
            &CpuBackend,
            &remote,
            false,
            |_, _| Ok(()),
        );
        match result {
            Err(RemoteMoeError::BadResponse(msg)) => {
                assert!(msg.contains("at least two tokens"), "got: {msg}");
            }
            other => panic!("expected BadResponse, got {other:?}"),
        }
    }

    #[test]
    fn errors_when_vindex_has_no_attn_weights() {
        // make_test_vindex has no attention weights, so the setup guard
        // fails cleanly (same surface as `generate_with_remote_moe`'s
        // no-attention test) — before any FFN dispatch.
        let w = make_test_weights();
        let idx = make_test_vindex(&w);
        let remote = connect_test_remote(w.hidden_size);
        let result = score_forced_with_remote_ffn(
            &w,
            &[1u32, 2u32],
            &idx,
            &CpuBackend,
            &remote,
            false,
            |_, _| Ok(()),
        );
        match result {
            Err(RemoteMoeError::BadResponse(msg)) => {
                assert!(msg.contains("no attention weights"), "got: {msg}");
            }
            other => panic!("expected BadResponse, got {other:?}"),
        }
    }

    #[test]
    fn q8k_arm_rejects_non_superblock_hidden() {
        // The synthetic fixture's hidden size is deliberately small (not a
        // multiple of 256), so the q8k guard must fire before dispatch.
        let w = make_test_weights();
        assert!(
            !w.hidden_size
                .is_multiple_of(crate::ffn::Q4K_Q8K_SUPERBLOCK_ELEMS),
            "fixture unexpectedly q8k-compatible; pick a different guard test"
        );
        let idx = make_test_vindex(&w);
        let remote = connect_test_remote(w.hidden_size);
        let result = score_forced_with_remote_ffn(
            &w,
            &[1u32, 2u32],
            &idx,
            &CpuBackend,
            &remote,
            true,
            |_, _| Ok(()),
        );
        match result {
            Err(RemoteMoeError::BadResponse(msg)) => {
                assert!(msg.contains("q8k wire requires hidden"), "got: {msg}");
            }
            other => panic!("expected BadResponse, got {other:?}"),
        }
    }
}
