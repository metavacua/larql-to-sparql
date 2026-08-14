//! Per-layer residual capture across the three production forward paths.
//!
//! Each `ResidualCapture::*` constructor drives the corresponding backend
//! once with its existing per-layer dump hook (file-based env-var, owned
//! by `vindex/kquant_forward.rs` / `metal/ops/full_pipeline.rs` /
//! `metal/decode/mod.rs`), then reads the resulting `.f32` blobs into a
//! typed in-memory `Vec<Vec<f32>>`. The temp dir is cleaned up on drop —
//! callers don't need to know it ever existed.
//!
//! Why thread file-system: the dump hooks are already wired into the
//! backends and exercised end-to-end (the `examples/residual_diff`
//! interactive tool uses them). Replacing the env-var mechanism with a
//! direct callback would touch every backend forward path; not worth
//! the churn for the test ergonomics win this module gives. If a future
//! refactor moves to direct callbacks, `run_with_dump_dir` can become a
//! callback adapter without changing the public surface.

use std::path::Path;

use larql_models::ModelWeights;
use larql_vindex::{GateIndex, VectorIndex};

use crate::forward::dump_config::{
    cpu_layer_file, decode_layer_file, metal_layer_h_out_file, ENV_CPU_DUMP_LAYERS,
    ENV_DECODE_DUMP_LAYERS, ENV_METAL_DUMP_LAYERS,
};
use crate::layer_graph::generate::generate;
use crate::layer_graph::pipeline_layer::DEFAULT_GPU_KV_CACHE_MAX_SEQ;
use crate::layer_graph::CachedLayerGraph;

/// Per-layer end-of-layer hidden state. `layers[l]` is the residual
/// after layer l completes (post post_ffn norm + post-FFN residual +
/// PLE + layer_scalar).
///
/// For prefill captures, each `layers[l]` is `seq_len * hidden` floats
/// in row-major `[seq_len, hidden]`. For decode captures, each is
/// `hidden` floats (one position only — KV-cached single-token decode).
#[derive(Debug, Clone)]
pub struct ResidualCapture {
    /// Per-layer hidden states. Length = `num_layers`.
    pub layers: Vec<Vec<f32>>,
    /// Hidden size of the model.
    pub hidden_size: usize,
    /// Sequence length covered. `1` for decode captures.
    pub seq_len: usize,
}

impl ResidualCapture {
    /// Number of layers captured. Cheap accessor for tests.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Slice the last-position row out of a prefill capture's layer.
    /// Returns `&[f32]` of length `hidden_size`. Use this to compare a
    /// CPU prefill at length N+1 against a Metal decode capture at the
    /// same effective sequence length — they're shape-compatible after
    /// this slice.
    pub fn last_position(&self, layer: usize) -> &[f32] {
        let v = &self.layers[layer];
        let start = (self.seq_len.saturating_sub(1)) * self.hidden_size;
        &v[start..start + self.hidden_size]
    }

    /// Build a decode-style single-position capture from `self` by
    /// projecting each prefill layer down to its last row. Useful for
    /// comparing `CPU prefill(N+1)` directly against `metal_decode(N, id)`
    /// without the caller juggling indices.
    pub fn project_to_last_position(&self) -> Self {
        let layers = (0..self.layers.len())
            .map(|l| self.last_position(l).to_vec())
            .collect();
        Self {
            layers,
            hidden_size: self.hidden_size,
            seq_len: 1,
        }
    }
}

impl ResidualCapture {
    /// CPU full prefill via `predict_kquant_hidden`. Drives the per-layer
    /// dump hook (`LARQL_CPU_DUMP_LAYERS=<dir>`) at file `cpu_layer_NN.f32`
    /// per layer, then reads them back into a `Vec<Vec<f32>>`.
    pub fn cpu_prefill(
        weights: &mut ModelWeights,
        ids: &[u32],
        index: &VectorIndex,
    ) -> Result<Self, String> {
        let hidden = weights.hidden_size;
        let num_layers = weights.num_layers;
        let seq_len = ids.len();

        let dir = run_with_dump_dir(ENV_CPU_DUMP_LAYERS, || {
            let _ = crate::vindex::predict_kquant_hidden(weights, ids, index, None);
        })?;

        let layers = (0..num_layers)
            .map(|l| {
                let path = dir.path().join(cpu_layer_file(l));
                read_f32_vec(&path)
                    .ok_or_else(|| format!("CPU dump missing for layer {l} at {}", path.display()))
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            layers,
            hidden_size: hidden,
            seq_len,
        })
    }

    /// Metal prefill on `prefix_ids` followed by a single
    /// KV-cached `decode_token(new_id)`. The capture reflects the
    /// per-layer output of the *decode step* — one position per layer
    /// (`hidden_size` floats). Uses the dump hook
    /// `LARQL_DECODE_DUMP_LAYERS=<dir>` plumbed into
    /// `decode_token_with_moe_fn` (`metal/decode/mod.rs`).
    ///
    /// Designed to be paired with a CPU prefill of length
    /// `prefix_ids.len() + 1` and projected to `last_position` — the
    /// two should match modulo float noise if KV-cached decode produces
    /// the same hidden state as a fresh prefill at the new position.
    pub fn metal_decode(
        weights: &mut ModelWeights,
        prefix_ids: &[u32],
        new_id: u32,
        index: &VectorIndex,
        backend: &dyn larql_compute::ComputeBackend,
    ) -> Result<Self, String> {
        let hidden = weights.hidden_size;
        let num_layers = weights.num_layers;
        let arch = &*weights.arch;

        // Reset + per-layer-shape KV cache (Gemma 4 has asymmetric
        // sliding/global geometry; uniform allocation would silently
        // truncate global layers).
        backend.reset_kv_cache();
        let kv_shapes: Vec<(usize, usize)> = (0..num_layers)
            .map(|l| (arch.num_kv_heads_for_layer(l), arch.head_dim_for_layer(l)))
            .collect();
        backend.preallocate_kv_cache_per_layer(&kv_shapes, DEFAULT_GPU_KV_CACHE_MAX_SEQ);

        // Build pipeline layers — same wiring `layer_graph::generate` uses.
        let gate_index: &dyn GateIndex = index;
        let (q4_ffn, ffn_is_q4k) = if let Some(m) = gate_index.interleaved_kquant_mmap_ref() {
            (Some(m), true)
        } else {
            (gate_index.interleaved_q4_mmap_ref(), false)
        };
        let q4_ffn_mmap = q4_ffn.ok_or("no Q4 FFN mmap available for decode capture")?;
        let intermediate = gate_index.num_features(0);
        let ffn_format = if ffn_is_q4k {
            larql_compute::QuantFormat::Q4_K
        } else {
            larql_compute::QuantFormat::Q4_0
        };
        let q4_ffn_per_matrix = ffn_format
            .packed_matrix_bytes(intermediate, hidden)
            .ok_or("unsupported Q4 FFN format for decode capture")?;
        let layers = crate::layer_graph::pipeline_layer::build_pipeline_layers(
            weights,
            index,
            0..num_layers,
            q4_ffn_mmap,
            q4_ffn_per_matrix,
            ffn_format,
        );

        let softcap = arch.attn_logit_softcapping().unwrap_or(0.0);
        let qk_norm_val = arch.attn_q_norm_key(0).is_some();

        // Prefill the cache. We don't care about its hidden output —
        // only the KV cache state for the subsequent decode step.
        let h_embed = crate::forward::embed_tokens_pub(weights, prefix_ids);
        let prefill_x: Vec<f32> = h_embed.as_slice().unwrap().to_vec();
        backend
            .prefill_kquant(
                &layers,
                &prefill_x,
                hidden,
                intermediate,
                prefix_ids.len(),
                qk_norm_val,
                softcap,
            )
            .ok_or("Metal prefill_kquant returned None")?;

        // Decode one token, with the per-layer dump hook active.
        let dec_embed = crate::forward::embed_tokens_pub(weights, &[new_id]);
        let dec_x: Vec<f32> = dec_embed.row(0).to_vec();
        let dir = run_with_dump_dir(ENV_DECODE_DUMP_LAYERS, || {
            let _ = backend.decode_token(&layers, &dec_x, hidden, intermediate);
        })?;

        let layer_dumps = (0..num_layers)
            .map(|l| {
                let path = dir.path().join(decode_layer_file(l));
                read_f32_vec(&path).ok_or_else(|| {
                    format!("decode dump missing for layer {l} at {}", path.display())
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            layers: layer_dumps,
            hidden_size: hidden,
            seq_len: 1,
        })
    }

    /// Metal `prefill(prefix_ids)` followed by a sequential chain of
    /// `decode_token(id)` calls for each id in `new_ids`. Captures the
    /// per-layer hidden state of the **last** decode step. Pair with
    /// `cpu_prefill(prefix_ids ++ new_ids)` projected to last position
    /// to verify that the KV cache state written during step k stays
    /// correct for the read at step k+1 — that's not validated by
    /// `metal_decode` (single step) which only sees the initial KV
    /// state from prefill.
    pub fn metal_decode_steps(
        weights: &mut ModelWeights,
        prefix_ids: &[u32],
        new_ids: &[u32],
        index: &VectorIndex,
        backend: &dyn larql_compute::ComputeBackend,
    ) -> Result<Self, String> {
        if new_ids.is_empty() {
            return Err("metal_decode_steps requires at least one new_id".to_string());
        }
        let hidden = weights.hidden_size;
        let num_layers = weights.num_layers;
        let arch = &*weights.arch;

        backend.reset_kv_cache();
        let kv_shapes: Vec<(usize, usize)> = (0..num_layers)
            .map(|l| (arch.num_kv_heads_for_layer(l), arch.head_dim_for_layer(l)))
            .collect();
        backend.preallocate_kv_cache_per_layer(&kv_shapes, DEFAULT_GPU_KV_CACHE_MAX_SEQ);

        let gate_index: &dyn GateIndex = index;
        let (q4_ffn, ffn_is_q4k) = if let Some(m) = gate_index.interleaved_kquant_mmap_ref() {
            (Some(m), true)
        } else {
            (gate_index.interleaved_q4_mmap_ref(), false)
        };
        let q4_ffn_mmap = q4_ffn.ok_or("no Q4 FFN mmap available for decode capture")?;
        let intermediate = gate_index.num_features(0);
        let ffn_format = if ffn_is_q4k {
            larql_compute::QuantFormat::Q4_K
        } else {
            larql_compute::QuantFormat::Q4_0
        };
        let q4_ffn_per_matrix = ffn_format
            .packed_matrix_bytes(intermediate, hidden)
            .ok_or("unsupported Q4 FFN format for decode capture")?;
        let layers = crate::layer_graph::pipeline_layer::build_pipeline_layers(
            weights,
            index,
            0..num_layers,
            q4_ffn_mmap,
            q4_ffn_per_matrix,
            ffn_format,
        );

        let softcap = arch.attn_logit_softcapping().unwrap_or(0.0);
        let qk_norm_val = arch.attn_q_norm_key(0).is_some();

        let h_embed = crate::forward::embed_tokens_pub(weights, prefix_ids);
        let prefill_x: Vec<f32> = h_embed.as_slice().unwrap().to_vec();
        backend
            .prefill_kquant(
                &layers,
                &prefill_x,
                hidden,
                intermediate,
                prefix_ids.len(),
                qk_norm_val,
                softcap,
            )
            .ok_or("Metal prefill_kquant returned None")?;

        // Decode all but the last id without the dump hook (cheaper —
        // we only need per-layer state of the final step). Then decode
        // the last id with the dump hook active.
        for &id in &new_ids[..new_ids.len() - 1] {
            let dec_embed = crate::forward::embed_tokens_pub(weights, &[id]);
            let dec_x: Vec<f32> = dec_embed.row(0).to_vec();
            let _ = backend.decode_token(&layers, &dec_x, hidden, intermediate);
        }

        let last_id = *new_ids.last().unwrap();
        let dec_embed = crate::forward::embed_tokens_pub(weights, &[last_id]);
        let dec_x: Vec<f32> = dec_embed.row(0).to_vec();
        let dir = run_with_dump_dir(ENV_DECODE_DUMP_LAYERS, || {
            let _ = backend.decode_token(&layers, &dec_x, hidden, intermediate);
        })?;

        let layer_dumps = (0..num_layers)
            .map(|l| {
                let path = dir.path().join(decode_layer_file(l));
                read_f32_vec(&path).ok_or_else(|| {
                    format!("decode dump missing for layer {l} at {}", path.display())
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            layers: layer_dumps,
            hidden_size: hidden,
            seq_len: 1,
        })
    }

    /// Metal full prefill via `prefill_kquant`. Drives the per-layer dump
    /// hook (`LARQL_METAL_DUMP_LAYERS=<dir>`) at `metal_layer_NN_h_out.f32`
    /// per layer.
    ///
    /// Uses `generate(max_tokens=1)` to drive prefill — that's the same
    /// entry point production code takes, so we're testing the path
    /// users actually run, not a hand-stitched approximation.
    pub fn metal_prefill(
        weights: &mut ModelWeights,
        ids: &[u32],
        index: &VectorIndex,
        backend: &dyn larql_compute::ComputeBackend,
    ) -> Result<Self, String> {
        let hidden = weights.hidden_size;
        let num_layers = weights.num_layers;
        let seq_len = ids.len();

        // We need a tokenizer for `generate`. Build a minimal one from
        // the vindex if the caller hasn't already loaded it — avoiding
        // putting the tokenizer in the public signature keeps the API
        // symmetrical with `cpu_prefill`.
        let dir = run_with_dump_dir(ENV_METAL_DUMP_LAYERS, || {
            let cached = CachedLayerGraph::from_residuals(Vec::new());
            // generate() also drives the embed→prefill→sample chain,
            // including the per-layer dump hook for Metal.
            let dummy_tok = build_dummy_tokenizer();
            let _ = generate(
                weights,
                &dummy_tok,
                ids,
                1,
                index,
                backend,
                &cached,
                0..num_layers,
            );
        })?;

        let layers = (0..num_layers)
            .map(|l| {
                let path = dir.path().join(metal_layer_h_out_file(l));
                read_f32_vec(&path).ok_or_else(|| {
                    format!(
                        "Metal prefill dump missing for layer {l} at {}",
                        path.display()
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            layers,
            hidden_size: hidden,
            seq_len,
        })
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Serialises every dump-directory env-var window in this module tree.
///
/// Shared with [`super::stages::run_with_two_env_vars`] rather than kept
/// private: both helpers point process-global variables at per-call
/// tempdirs, and two *different* helpers racing corrupts a capture just
/// as thoroughly as two calls to the same one. One lock for the whole
/// mechanism is the only version that is obviously correct.
///
/// Env vars are process-global, so two concurrent captures would each
/// point the *same* variable at their own tempdir. This is not
/// hypothetical — it produced two distinct failures in
/// `test_cpu_metal_parity`, which `cargo test` runs as four threads in
/// one process:
///
/// - the loser's dump landed in the winner's directory, so its own
///   directory was empty → "Metal prefill dump missing for layer 0";
/// - or the loser read files the winner had written **for a different
///   model**, giving a residual comparison of cos ≈ 0.005 — an
///   apparently catastrophic kernel regression that was nothing of the
///   sort.
///
/// The previous version documented the hazard ("racing `cargo test
/// --test-threads=N` would stomp; tests in this suite run with
/// `--test-threads=1` upstream") but nothing enforced it, and the
/// default invocation violates it. The invariant belongs with the
/// function that owns the global, not with each caller.
pub(super) static DUMP_DIR_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Set the named env var to a fresh tempdir, run `f`, and return the
/// tempdir guard so the caller can read files before it drops. The
/// previous value is restored before returning.
///
/// The env var is mutated under [`DUMP_DIR_ENV_LOCK`], so concurrent
/// callers queue rather than redirect each other's dumps. Reads of the
/// returned directory need no lock: each call gets its own tempdir, and
/// `f` has finished writing to it by the time this returns.
fn run_with_dump_dir(env_var: &str, f: impl FnOnce()) -> Result<tempfile::TempDir, String> {
    let dir = tempfile::tempdir().map_err(|e| format!("tempdir: {e}"))?;
    // A panicking caller poisons the lock; the guarded state is just the
    // env var, which is restored below either way, so recover rather
    // than cascade one test's failure into every other test's.
    let _guard = DUMP_DIR_ENV_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let prev = std::env::var(env_var).ok();
    std::env::set_var(env_var, dir.path());
    f();
    match prev {
        Some(v) => std::env::set_var(env_var, v),
        None => std::env::remove_var(env_var),
    }
    Ok(dir)
}

/// Read a flat `f32` little-endian file. Returns `None` on any I/O
/// error or non-multiple-of-4 file size — caller surfaces a friendly
/// error.
fn read_f32_vec(path: &Path) -> Option<Vec<f32>> {
    let bytes = std::fs::read(path).ok()?;
    if !bytes.len().is_multiple_of(4) {
        return None;
    }
    Some(
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    )
}

/// Build a minimal `tokenizers::Tokenizer` for the captures that need
/// to call `generate()` but don't actually use the tokenizer for
/// anything other than its decode-sample step (the dump hooks fire
/// before sampling). `generate()` decodes the first generated token
/// id back to a string for its return value; we don't care about that
/// string here. A trivially-built tokenizer with an empty vocab won't
/// work because `generate` calls `decode([id], true)` which goes
/// through the model — but for our use we just need *something* that
/// won't panic on construction.
///
/// In practice we don't end up here: `metal_prefill` is called with
/// the same ids the user just tokenised, and the caller's tokenizer
/// would do. We thread the construction through to avoid a 4-arg
/// public signature.
fn build_dummy_tokenizer() -> tokenizers::Tokenizer {
    // BPE builder requires a vocab. Use the smallest possible model.
    use tokenizers::models::wordpiece::WordPiece;
    let model = WordPiece::default();
    tokenizers::Tokenizer::new(model)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn last_position_returns_correct_slice() {
        let cap = ResidualCapture {
            layers: vec![
                // [3, 4] flat: pos 0 = [1,1,1,1], pos 1 = [2,2,2,2], pos 2 = [3,3,3,3]
                vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0],
            ],
            hidden_size: 4,
            seq_len: 3,
        };
        assert_eq!(cap.last_position(0), &[3.0, 3.0, 3.0, 3.0]);
    }

    #[test]
    fn project_to_last_position_drops_other_rows() {
        let cap = ResidualCapture {
            layers: vec![vec![1.0, 1.0, 2.0, 2.0], vec![10.0, 10.0, 20.0, 20.0]],
            hidden_size: 2,
            seq_len: 2,
        };
        let dec = cap.project_to_last_position();
        assert_eq!(dec.layers, vec![vec![2.0, 2.0], vec![20.0, 20.0]]);
        assert_eq!(dec.seq_len, 1);
        assert_eq!(dec.hidden_size, 2);
    }

    #[test]
    fn run_with_dump_dir_restores_prior_env() {
        const ENV: &str = "LARQL_TEST_RESID_DUMP_DIR_RESTORE";
        std::env::set_var(ENV, "previous");

        // Observe the state *inside* `f` — that is the window the dump hooks
        // actually run in, and the only place the claim "the tempdir existed
        // and the var pointed at it" is checkable. The previous version
        // asserted `dir.path().exists() || !dir.path().exists()`, which is a
        // tautology: it can never fail, so it pinned nothing. Same shape as
        // the plausibility-only timestamp tests in `docs/k3-funnel.md` §4.5.
        let mut seen_var = String::new();
        let mut existed_during = false;
        let dir = run_with_dump_dir(ENV, || {
            seen_var = std::env::var(ENV).unwrap_or_default();
            existed_during = std::path::Path::new(&seen_var).is_dir();
        })
        .unwrap();

        assert!(existed_during, "the tempdir must exist while `f` runs");
        assert_eq!(
            seen_var,
            dir.path().to_string_lossy(),
            "`f` must see the var pointing at this call's tempdir"
        );
        // And afterwards the prior value is back.
        assert_eq!(std::env::var(ENV).unwrap(), "previous");
        std::env::remove_var("LARQL_TEST_RESID_DUMP_DIR_RESTORE");
    }

    #[test]
    fn run_with_dump_dir_clears_when_no_prior_value() {
        std::env::remove_var("LARQL_TEST_RESID_DUMP_DIR_NONE");
        let _ = run_with_dump_dir("LARQL_TEST_RESID_DUMP_DIR_NONE", || {}).unwrap();
        assert!(std::env::var("LARQL_TEST_RESID_DUMP_DIR_NONE").is_err());
    }

    /// `cpu_prefill` end to end on a synthetic Q4K model.
    ///
    /// The four capture constructors were entirely untested: they drive a real
    /// forward pass and read the resulting `.f32` dumps back, so the only way
    /// to cover them is to actually run one. The CPU path needs no GPU, so it
    /// can run everywhere — the three `metal_*` constructors cannot, which is
    /// why this file sits outside the crate's coverage `include_globs`.
    ///
    /// Asserts the shape contract the comparison code depends on: one entry
    /// per layer, each `seq_len * hidden` floats, all finite.
    #[test]
    fn cpu_prefill_captures_every_layer() {
        let mut weights = crate::test_utils::make_test_q4k_weights();
        let index = crate::test_utils::make_test_q4k_vindex(&weights);
        let ids: Vec<u32> = vec![1, 2, 3];

        let capture = ResidualCapture::cpu_prefill(&mut weights, &ids, &index)
            .expect("cpu prefill capture on the synthetic Q4K fixture");

        assert_eq!(capture.num_layers(), weights.num_layers);
        assert_eq!(capture.seq_len, ids.len());
        assert_eq!(capture.hidden_size, weights.hidden_size);
        for (l, layer) in capture.layers.iter().enumerate() {
            assert_eq!(
                layer.len(),
                ids.len() * weights.hidden_size,
                "layer {l} has the wrong element count"
            );
            assert!(
                layer.iter().all(|v| v.is_finite()),
                "layer {l} has non-finite values"
            );
        }
    }

    /// `project_to_last_position` on a real capture must agree with
    /// `last_position` for every layer — the two are used interchangeably by
    /// the parity suites when comparing a prefill against a decode step.
    #[test]
    fn projecting_a_real_capture_matches_last_position() {
        let mut weights = crate::test_utils::make_test_q4k_weights();
        let index = crate::test_utils::make_test_q4k_vindex(&weights);
        let capture = ResidualCapture::cpu_prefill(&mut weights, &[1, 2, 3], &index)
            .expect("cpu prefill capture");

        let projected = capture.project_to_last_position();
        assert_eq!(projected.seq_len, 1);
        for l in 0..capture.num_layers() {
            assert_eq!(projected.layers[l], capture.last_position(l).to_vec());
        }
    }

    /// `metal_prefill` on the same synthetic Q4K fixture.
    ///
    /// Gated on macOS because it needs a real Metal device; the three
    /// `metal_*` constructors are why this file cannot reach the 90% floor in
    /// a Linux CI coverage run, and why it sits outside `include_globs`. The
    /// test still earns its place — it means the constructor is exercised
    /// somewhere rather than nowhere.
    #[test]
    #[cfg(all(feature = "gpu", target_os = "macos"))]
    fn metal_prefill_captures_every_layer() {
        let Some(backend) = larql_compute_metal::MetalBackend::new() else {
            eprintln!("skip: no Metal device");
            return;
        };
        let mut weights = crate::test_utils::make_test_q4k_weights();
        let index = crate::test_utils::make_test_q4k_vindex(&weights);
        let ids: Vec<u32> = vec![1, 2, 3];

        let capture = ResidualCapture::metal_prefill(&mut weights, &ids, &index, &backend)
            .expect("metal prefill capture on the synthetic Q4K fixture");

        assert_eq!(capture.num_layers(), weights.num_layers);
        assert_eq!(capture.hidden_size, weights.hidden_size);
        for (l, layer) in capture.layers.iter().enumerate() {
            assert!(
                layer.iter().all(|v| v.is_finite()),
                "layer {l} has non-finite values"
            );
        }
    }

    /// A panic while the dump-dir lock is held must not cascade.
    ///
    /// `run_with_dump_dir` recovers from a poisoned lock via
    /// `into_inner()` rather than unwrapping. The guarded state is just an
    /// env var, which is restored on every path, so a poisoned lock carries
    /// no corrupt data — whereas propagating the poison would turn one
    /// failing test into every subsequent capture failing too, which is
    /// exactly the cascade this module was fixed to remove.
    #[test]
    fn a_poisoned_lock_does_not_cascade() {
        const ENV: &str = "LARQL_TEST_RESID_DUMP_DIR_POISON";

        // Poison it: panic inside a thread while holding the guard.
        let poisoned = std::thread::spawn(|| {
            let _g = DUMP_DIR_ENV_LOCK.lock().unwrap();
            panic!("deliberate panic to poison the lock");
        })
        .join();
        assert!(poisoned.is_err(), "the helper thread must have panicked");
        assert!(DUMP_DIR_ENV_LOCK.is_poisoned(), "lock should be poisoned");

        // The next caller still works.
        let mut saw = String::new();
        let dir = run_with_dump_dir(ENV, || {
            saw = std::env::var(ENV).unwrap_or_default();
        })
        .expect("a poisoned lock must not stop a capture");
        assert_eq!(saw, dir.path().to_string_lossy());
        assert!(std::env::var(ENV).is_err(), "env var restored");
    }

    /// Concurrent callers must each observe **their own** tempdir for the
    /// whole of `f`, never a sibling's.
    ///
    /// This is the regression that mattered: `cargo test` runs the four
    /// `test_cpu_metal_parity` cases as threads in one process, and the
    /// unsynchronised version let one capture's dump directory be
    /// redirected mid-run by another. The symptoms were a missing dump
    /// ("Metal prefill dump missing for layer 0") and, worse, a *silent*
    /// cross-model comparison reported as cos ≈ 0.005 — a parity failure
    /// that looked exactly like a catastrophic kernel bug.
    ///
    /// Note that the two tests above cannot catch this: both are
    /// single-threaded, and the hazard only exists under concurrency.
    #[test]
    fn concurrent_callers_never_see_each_others_dump_dir() {
        const ENV: &str = "LARQL_TEST_RESID_DUMP_DIR_CONCURRENT";
        const THREADS: usize = 8;

        let mismatches = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        std::thread::scope(|scope| {
            for _ in 0..THREADS {
                let mismatches = std::sync::Arc::clone(&mismatches);
                scope.spawn(move || {
                    let dir = run_with_dump_dir(ENV, || {
                        // Inside `f` the variable must name the directory
                        // this call created — the window the dump hooks
                        // actually read it in.
                        let seen = std::env::var(ENV).unwrap_or_default();
                        // Re-read after a yield so a racing setter has a
                        // chance to land, the way a long capture would.
                        std::thread::yield_now();
                        let seen_again = std::env::var(ENV).unwrap_or_default();
                        if seen != seen_again {
                            mismatches.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        }
                        std::fs::write(std::path::Path::new(&seen).join("marker"), b"x").ok();
                    })
                    .expect("tempdir");
                    // The marker this call wrote must be in its own dir.
                    if !dir.path().join("marker").exists() {
                        mismatches.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                });
            }
        });

        assert_eq!(
            mismatches.load(std::sync::atomic::Ordering::Relaxed),
            0,
            "a concurrent caller observed another's dump directory"
        );
        std::env::remove_var(ENV);
    }

    #[test]
    fn read_f32_vec_decodes_le_floats() {
        use std::io::Write;
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let bytes: Vec<u8> = [1.0f32, 2.5, -3.25]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        tmp.as_file().write_all(&bytes).unwrap();
        let v = read_f32_vec(tmp.path()).unwrap();
        assert_eq!(v, vec![1.0, 2.5, -3.25]);
    }

    #[test]
    fn read_f32_vec_rejects_non_multiple_of_four() {
        use std::io::Write;
        let tmp = tempfile::NamedTempFile::new().unwrap();
        tmp.as_file().write_all(&[1u8, 2, 3]).unwrap(); // 3 bytes
        assert!(read_f32_vec(tmp.path()).is_none());
    }

    #[test]
    fn read_f32_vec_returns_none_on_missing_file() {
        let p = PathBuf::from("/nonexistent/path/that/cant/exist/xyz.f32");
        assert!(read_f32_vec(&p).is_none());
    }
}
