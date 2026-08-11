//! Recomputing K/V from stored pre-layer residuals — the operation the
//! residual-stream engines exist to make cheap, plus the walk-KV selection
//! gates and diagnostics around it.
//!
//! Prefill and the decode step live in [`super::prefill`] and [`super::step`].

// DISCREPANCY vs the round-1 survey (which classified this file
// WHOLESALE_NATIVE and gated `pub mod compute;` at mod.rs): the mod.rs
// re-export split explicitly wants `kv_memory_bytes_for_seq` portable,
// which has zero native markers — pure arithmetic over `WeightsView`.
// Doing the real per-item split here instead of gating the whole file:
// `recompute_kv` and every private helper feeding it (VectorIndex
// param, `std::env`/`thread_local!`, and `eprintln!` in the diag path,
// which also has no core/alloc equivalent) are native.
// `last_row` is pure arithmetic too, but unlike `kv_memory_bytes_for_seq`
// both its callers (prefill.rs::rs_prefill, walk.rs) are native-gated, so
// it is dead code on wasm32 and gated accordingly below (Algorithm A
// dead-code classification, not a native marker on `last_row` itself).
#[cfg(not(target_arch = "wasm32"))]
use larql_compute::{dot_proj_gpu, ComputeBackend, QuantFormat};
#[cfg(not(target_arch = "wasm32"))]
use larql_vindex::VectorIndex;
// s!/ArrayBase/ArrayView1/Ix2 are only used inside native-gated
// functions (recompute_kv and its helpers, plus last_row); Array2/Data
// likewise have no remaining portable use now that last_row is native-only.
#[cfg(not(target_arch = "wasm32"))]
use ndarray::{s, Array2, ArrayBase, ArrayView1, Data, Ix2};
#[cfg(not(target_arch = "wasm32"))]
use std::cell::RefCell;
#[cfg(not(target_arch = "wasm32"))]
use std::cmp::Ordering;

#[cfg(not(target_arch = "wasm32"))]
use larql_inference::attention::apply_rope_partial_at;
#[cfg(not(target_arch = "wasm32"))]
use larql_inference::forward::{add_bias, apply_norm};
#[cfg(not(target_arch = "wasm32"))]
use larql_inference::residual::{rms_norm_heads_no_weight, rms_norm_qk_for_arch};

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Copy)]
enum KvProjection {
    K,
    V,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone)]
struct WalkKvSelection {
    select_layer: usize,
    top_k: usize,
    seq_len: usize,
    k_indices: Vec<Vec<usize>>,
    v_indices: Vec<Vec<usize>>,
}

#[cfg(not(target_arch = "wasm32"))]
thread_local! {
    static WALK_KV_SELECTION: RefCell<Option<WalkKvSelection>> = const { RefCell::new(None) };
    /// Per-thread override for `LARQL_MARKOV_*` env vars consulted by
    /// the walk-KV helpers below. Tests set entries here to exercise
    /// the env-gated branches without mutating the process-global env
    /// (which would race other parallel tests in the same crate that
    /// also call `recompute_kv`). Production code is unaffected — when
    /// the thread-local is empty the helpers fall through to
    /// `std::env::var`.
    static MARKOV_ENV_OVERRIDE: RefCell<std::collections::HashMap<&'static str, Option<String>>> =
        RefCell::new(std::collections::HashMap::new());
}

/// Read an env var subject to thread-local overrides (test-only escape
/// hatch — see `MARKOV_ENV_OVERRIDE`). An override of `Some(value)`
/// behaves like the env var being set to that value; `None` behaves
/// like the var being unset. With no override the helper delegates to
/// the real process env, so production callers see no change.
#[cfg(not(target_arch = "wasm32"))]
fn read_markov_env(key: &'static str) -> Option<String> {
    let overridden = MARKOV_ENV_OVERRIDE.with(|o| {
        o.borrow()
            .get(key)
            .map(|v| (true, v.clone()))
            .unwrap_or((false, None))
    });
    if overridden.0 {
        overridden.1
    } else {
        std::env::var(key).ok()
    }
}

#[cfg(test)]
pub(crate) fn set_markov_env_override(key: &'static str, value: Option<&str>) {
    MARKOV_ENV_OVERRIDE.with(|o| {
        o.borrow_mut().insert(key, value.map(|s| s.to_string()));
    });
}

#[cfg(test)]
pub(crate) fn clear_markov_env_overrides() {
    MARKOV_ENV_OVERRIDE.with(|o| o.borrow_mut().clear());
}

/// Recompute K/V from stored pre-layer residuals using `backend` for projection matmuls.
///
/// `index: Some(idx)` enables the Q4K-native fast path: per-row Q4K matvec
/// directly against the vindex's Q4K bytes, skipping the dequant-to-f32
/// step that's otherwise 8× the memory bandwidth. Quant-agnostic — the
/// backend's `quant_matvec` inspects the format byte and dispatches to
/// the right kernel (Q4K today; Q6K / future formats slot in
/// automatically). `None` keeps the f32 fallback for legacy callers.
#[cfg(not(target_arch = "wasm32"))]
pub fn recompute_kv(
    weights: larql_inference::WeightsView,
    h_stored: &Array2<f32>,
    layer: usize,
    abs_start: usize,
    backend: &dyn ComputeBackend,
    index: Option<&VectorIndex>,
) -> Option<(Array2<f32>, Array2<f32>)> {
    let arch = &*weights.arch;
    let head_dim = arch.head_dim_for_layer(layer);
    let num_kv = arch.num_kv_heads_for_layer(layer);
    let norm_offset = arch.norm_weight_offset();
    let qk_offset = arch.qk_norm_weight_offset();
    let qk_norm_off = if qk_offset != 0.0 {
        qk_offset
    } else {
        norm_offset
    };

    let h_norm = apply_norm(
        &weights,
        h_stored,
        &arch.input_layernorm_key(layer),
        norm_offset,
    );

    let kv_dim = num_kv * head_dim;
    let hidden = weights.hidden_size;
    let seq_len = h_norm.shape()[0];

    let walk_kv_top_k = markov_walk_kv_top_k(layer, kv_dim);
    let walk_kv_select_at = markov_walk_kv_select_at();
    let should_cache_selection = walk_kv_select_at
        .is_some_and(|select_layer| select_layer == layer)
        && markov_walk_kv_requested_top_k(kv_dim).is_some();

    if should_cache_selection {
        if let Some((w_k, w_v)) = attn_kv_projection_weights(weights, layer) {
            let top_k = markov_walk_kv_requested_top_k(kv_dim)?;
            cache_walk_kv_selection(layer, top_k, &h_norm, w_k, w_v);
        }
    }

    // Q4K-native path: per-row matvec on the vindex's raw Q4K bytes.
    // Saves the dequant-to-f32 cost (8× memory bandwidth) when the
    // backend supports Q4K matvec and the vindex has Q4K attn data.
    //
    // Disabled when the experimental walk-KV path is active: that path
    // intentionally replaces the projection matmul with row-wise top-K
    // projection against the f32 tensor rows below.
    let q4k_path = if walk_kv_top_k.is_none() && !markov_kv_force_f32_projection() {
        index
            .and_then(|idx| idx.attn_kquant_layer_data(layer))
            .filter(|_| backend.supports_quant(::larql_compute::QuantFormat::Q4_K))
    } else {
        None
    };

    let used_q4k_projection = q4k_path.is_some();
    let (mut k, mut v) = if let Some(attn_data) = q4k_path {
        // attn_data: [(Q, fmt), (K, fmt), (V, fmt), (O, fmt)]
        let (k_bytes, k_fmt) = attn_data[1];
        let (v_bytes, v_fmt) = attn_data[2];
        let k_format = parse_quant_format(k_fmt)?;
        let v_format = parse_quant_format(v_fmt)?;

        let mut k_out = Array2::<f32>::zeros((seq_len, kv_dim));
        let mut v_out = Array2::<f32>::zeros((seq_len, kv_dim));
        for row_idx in 0..seq_len {
            let x_row = h_norm.row(row_idx);
            let x_slice = x_row.as_slice()?;
            let k_row = backend.quant_matvec(k_format, k_bytes, x_slice, kv_dim, hidden)?;
            let v_row = backend.quant_matvec(v_format, v_bytes, x_slice, kv_dim, hidden)?;
            k_out
                .row_mut(row_idx)
                .iter_mut()
                .zip(k_row.iter())
                .for_each(|(o, &i)| *o = i);
            v_out
                .row_mut(row_idx)
                .iter_mut()
                .zip(v_row.iter())
                .for_each(|(o, &i)| *o = i);
        }
        (k_out, v_out)
    } else {
        // f32 fallback: read dequantised weights from `weights.tensors`.
        let (w_k, w_v) = attn_kv_projection_weights(weights, layer)?;
        let (k, v) = if let Some(top_k) = walk_kv_top_k {
            let cached = walk_kv_select_at
                .filter(|&select_layer| select_layer != layer)
                .and_then(|select_layer| {
                    let k = walk_project_cached_topk(
                        &h_norm,
                        w_k,
                        top_k,
                        select_layer,
                        KvProjection::K,
                    )?;
                    let v = walk_project_cached_topk(
                        &h_norm,
                        w_v,
                        top_k,
                        select_layer,
                        KvProjection::V,
                    )?;
                    Some((k, v))
                });
            let (k, v) = if let Some(pair) = cached {
                pair
            } else {
                (
                    walk_project_topk(&h_norm, w_k, top_k)?,
                    walk_project_topk(&h_norm, w_v, top_k)?,
                )
            };
            (k, v)
        } else {
            let k = dot_proj_gpu(&h_norm, w_k, Some(backend));
            let v = dot_proj_gpu(&h_norm, w_v, Some(backend));
            (k, v)
        };
        (k, v)
    };

    if markov_walk_kv_diag_enabled() && markov_walk_kv_diag_layer(layer) {
        if let Some((w_k, w_v)) = attn_kv_projection_weights(weights, layer) {
            let dense_k = dot_proj_gpu(&h_norm, w_k, Some(backend));
            let dense_v = dot_proj_gpu(&h_norm, w_v, Some(backend));
            let walk_k = walk_project_topk(&h_norm, w_k, kv_dim)?;
            let walk_v = walk_project_topk(&h_norm, w_v, kv_dim)?;
            let path = if used_q4k_projection { "q4k" } else { "f32" };
            print_walk_kv_diag(layer, path, "K", "actual_vs_f32", &k, &dense_k);
            print_walk_kv_diag(layer, path, "V", "actual_vs_f32", &v, &dense_v);
            print_walk_kv_diag(layer, path, "K", "f32_vs_walk_full", &dense_k, &walk_k);
            print_walk_kv_diag(layer, path, "V", "f32_vs_walk_full", &dense_v, &walk_v);
            print_walk_kv_diag(layer, path, "K", "actual_vs_walk_full", &k, &walk_k);
            print_walk_kv_diag(layer, path, "V", "actual_vs_walk_full", &v, &walk_v);
        }
    }

    if let Some(bias) = arch
        .attn_k_bias_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        add_bias(&mut k, bias);
    }
    if let Some(bias) = arch
        .attn_v_bias_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        add_bias(&mut v, bias);
    }
    if arch.has_v_norm() {
        v = rms_norm_heads_no_weight(&v, num_kv, head_dim);
    }
    let k_normed = match arch
        .attn_k_norm_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        Some(norm_w) => rms_norm_qk_for_arch(&k, norm_w, num_kv, head_dim, qk_norm_off, arch),
        None => k,
    };
    let k_rope = apply_rope_partial_at(
        &k_normed,
        num_kv,
        head_dim,
        arch.rope_base_for_layer(layer),
        arch.rotary_fraction_for_layer(layer),
        abs_start,
    );
    Some((k_rope, v))
}

/// Type alias for an attention K/V projection weight pair as stored in
/// `weights.tensors` (Arc-shared, `Ix2`). Used by `attn_kv_projection_weights`
/// to keep its signature readable; the clippy `type_complexity` lint
/// triggers on the inline tuple form.
#[cfg(not(target_arch = "wasm32"))]
type AttnKvWeightPair<'a> = (
    &'a ArrayBase<ndarray::OwnedArcRepr<f32>, Ix2>,
    &'a ArrayBase<ndarray::OwnedArcRepr<f32>, Ix2>,
);

#[cfg(not(target_arch = "wasm32"))]
fn attn_kv_projection_weights<'a>(
    weights: larql_inference::WeightsView<'a>,
    layer: usize,
) -> Option<AttnKvWeightPair<'a>> {
    let arch = &*weights.arch;
    let w_k = weights.tensor(&arch.attn_k_key(layer))?;
    let v_from_k = !weights.has_tensor(&arch.attn_v_key(layer));
    let w_v = if v_from_k {
        w_k
    } else {
        weights.tensor(&arch.attn_v_key(layer))?
    };
    Some((w_k, w_v))
}

/// Experimental Markov-KV walk gate.
///
/// Set `LARQL_MARKOV_WALK_KV_TOPK=N` to replace the K/V projection
/// matmul with row-wise top-K projection. By default it applies to all
/// layers; restrict it with `LARQL_MARKOV_WALK_KV_LAYERS=5-20,26`.
#[cfg(not(target_arch = "wasm32"))]
fn markov_walk_kv_top_k(layer: usize, kv_dim: usize) -> Option<usize> {
    let top_k = markov_walk_kv_requested_top_k(kv_dim)?;
    if let Some(select_layer) = markov_walk_kv_select_at() {
        if layer == select_layer {
            return None;
        }
    }
    if let Some(spec) = read_markov_env("LARQL_MARKOV_WALK_KV_LAYERS") {
        if !layer_in_spec(&spec, layer) {
            return None;
        }
    }
    Some(top_k)
}

#[cfg(not(target_arch = "wasm32"))]
fn markov_walk_kv_requested_top_k(kv_dim: usize) -> Option<usize> {
    let raw = read_markov_env("LARQL_MARKOV_WALK_KV_TOPK")?;
    let top_k = raw.trim().parse::<usize>().ok()?;
    if top_k == 0 {
        return None;
    }
    Some(top_k.min(kv_dim))
}

#[cfg(not(target_arch = "wasm32"))]
fn markov_walk_kv_select_at() -> Option<usize> {
    read_markov_env("LARQL_MARKOV_WALK_KV_SELECT_AT")?
        .trim()
        .parse()
        .ok()
}

#[cfg(not(target_arch = "wasm32"))]
fn markov_walk_kv_diag_enabled() -> bool {
    read_markov_env("LARQL_MARKOV_WALK_KV_DIAG")
        .is_some_and(|v| matches!(v.trim(), "1" | "true" | "TRUE" | "yes" | "on"))
}

#[cfg(not(target_arch = "wasm32"))]
fn markov_kv_force_f32_projection() -> bool {
    read_markov_env("LARQL_MARKOV_KV_FORCE_F32")
        .is_some_and(|v| matches!(v.trim(), "1" | "true" | "TRUE" | "yes" | "on"))
}

/// In-place hot-K/V append on the resident walk's steady state (default ON).
/// When enabled, step 2+ appends the new K/V row into the doubling-capacity
/// `hot_kv` buffer and attends over views — O(L) total cache copy vs the
/// owned-concat path's O(L²). Set `LARQL_MARKOV_INPLACE_KV=0` to fall back to
/// the owned concat: the reference the parity test A/Bs against, and a
/// production escape hatch. Both paths are bit-identical (proven by
/// `run_..._inplace ≡ run_..._q4k_direct` at the compute level and the
/// engine-level A/B test). Shared with the codec twin (same mechanism, one
/// toggle for both residual engines).
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn markov_inplace_kv_enabled() -> bool {
    !matches!(
        read_markov_env("LARQL_MARKOV_INPLACE_KV").as_deref(),
        Some("0") | Some("false") | Some("off") | Some("no")
    )
}

#[cfg(not(target_arch = "wasm32"))]
fn markov_walk_kv_diag_layer(layer: usize) -> bool {
    // Env-var absent → true (diag applies to all layers); present → check
    // the comma-list. This was a `map_or(true, ..)` while the workspace
    // pinned MSRV 1.80, since `is_none_or` stabilised in 1.82; the MSRV is
    // 1.88 now, so it says what it means.
    read_markov_env("LARQL_MARKOV_WALK_KV_LAYERS").is_none_or(|spec| layer_in_spec(&spec, layer))
}

#[cfg(not(target_arch = "wasm32"))]
fn layer_in_spec(spec: &str, layer: usize) -> bool {
    spec.split(',').any(|part| {
        let part = part.trim();
        if part.is_empty() {
            return false;
        }
        if let Some((start, end)) = part.split_once('-') {
            let Some(start) = start.trim().parse::<usize>().ok() else {
                return false;
            };
            let Some(end) = end.trim().parse::<usize>().ok() else {
                return false;
            };
            return start <= layer && layer <= end;
        }
        part.parse::<usize>() == Ok(layer)
    })
}

#[cfg(not(target_arch = "wasm32"))]
fn cache_walk_kv_selection<SK, SV>(
    select_layer: usize,
    top_k: usize,
    x: &Array2<f32>,
    w_k: &ArrayBase<SK, Ix2>,
    w_v: &ArrayBase<SV, Ix2>,
) where
    SK: Data<Elem = f32>,
    SV: Data<Elem = f32>,
{
    let k_indices = walk_select_topk_indices(x, w_k, top_k);
    let v_indices = walk_select_topk_indices(x, w_v, top_k);
    let selection = WalkKvSelection {
        select_layer,
        top_k,
        seq_len: x.shape()[0],
        k_indices,
        v_indices,
    };
    WALK_KV_SELECTION.with(|slot| {
        *slot.borrow_mut() = Some(selection);
    });
}

#[cfg(not(target_arch = "wasm32"))]
fn walk_select_topk_indices<S>(
    x: &Array2<f32>,
    weights: &ArrayBase<S, Ix2>,
    top_k: usize,
) -> Vec<Vec<usize>>
where
    S: Data<Elem = f32>,
{
    (0..x.shape()[0])
        .map(|row_idx| {
            let pairs = walk_select_topk_scores(x.row(row_idx), weights, top_k);
            pairs.into_iter().map(|(idx, _)| idx).collect()
        })
        .collect()
}

#[cfg(not(target_arch = "wasm32"))]
fn walk_project_topk<S>(
    x: &Array2<f32>,
    weights: &ArrayBase<S, Ix2>,
    top_k: usize,
) -> Option<Array2<f32>>
where
    S: Data<Elem = f32>,
{
    let seq_len = x.shape()[0];
    let hidden = x.shape()[1];
    let rows = weights.shape()[0];
    if weights.shape()[1] != hidden || top_k == 0 {
        return None;
    }

    let mut out = Array2::<f32>::zeros((seq_len, rows));
    for row_idx in 0..seq_len {
        for (out_idx, score) in walk_select_topk_scores(x.row(row_idx), weights, top_k) {
            out[[row_idx, out_idx]] = score;
        }
    }
    Some(out)
}

#[cfg(not(target_arch = "wasm32"))]
fn walk_select_topk_scores<S>(
    x_row: ArrayView1<'_, f32>,
    weights: &ArrayBase<S, Ix2>,
    top_k: usize,
) -> Vec<(usize, f32)>
where
    S: Data<Elem = f32>,
{
    let rows = weights.shape()[0];
    let k = top_k.min(rows);
    let mut scores: Vec<(usize, f32)> = (0..rows)
        .map(|out_idx| (out_idx, dot_rows(x_row, weights.row(out_idx))))
        .collect();
    if k < scores.len() {
        scores.select_nth_unstable_by(k, compare_abs_desc);
        scores.truncate(k);
    }
    scores
}

#[cfg(not(target_arch = "wasm32"))]
fn walk_project_cached_topk<S>(
    x: &Array2<f32>,
    weights: &ArrayBase<S, Ix2>,
    top_k: usize,
    select_layer: usize,
    projection: KvProjection,
) -> Option<Array2<f32>>
where
    S: Data<Elem = f32>,
{
    let seq_len = x.shape()[0];
    let hidden = x.shape()[1];
    let rows = weights.shape()[0];
    if weights.shape()[1] != hidden || top_k == 0 {
        return None;
    }

    let indices = WALK_KV_SELECTION.with(|slot| {
        let borrowed = slot.borrow();
        let selection = borrowed.as_ref()?;
        if selection.select_layer != select_layer
            || selection.top_k != top_k.min(rows)
            || selection.seq_len != seq_len
        {
            return None;
        }
        Some(match projection {
            KvProjection::K => selection.k_indices.clone(),
            KvProjection::V => selection.v_indices.clone(),
        })
    })?;

    let mut out = Array2::<f32>::zeros((seq_len, rows));
    for row_idx in 0..seq_len {
        let x_row = x.row(row_idx);
        for &out_idx in indices.get(row_idx)? {
            if out_idx >= rows {
                return None;
            }
            out[[row_idx, out_idx]] = dot_rows(x_row, weights.row(out_idx));
        }
    }
    Some(out)
}

#[cfg(not(target_arch = "wasm32"))]
fn compare_abs_desc(a: &(usize, f32), b: &(usize, f32)) -> Ordering {
    b.1.abs().partial_cmp(&a.1.abs()).unwrap_or(Ordering::Equal)
}

#[cfg(not(target_arch = "wasm32"))]
fn dot_rows(a: ArrayView1<'_, f32>, b: ArrayView1<'_, f32>) -> f32 {
    a.iter().zip(b.iter()).map(|(x, w)| x * w).sum()
}

// `eprintln!` has no core/alloc equivalent under wasm32v1-none (no
// stderr without std) — native regardless of the VectorIndex question.
#[cfg(not(target_arch = "wasm32"))]
fn print_walk_kv_diag(
    layer: usize,
    path: &str,
    projection: &str,
    label: &str,
    a: &Array2<f32>,
    b: &Array2<f32>,
) {
    let (max_abs, rms, cos) = array_diff_stats(a, b);
    eprintln!(
        "[walk-kv-diag] layer={layer:02} path={path} proj={projection} cmp={label} max_abs={max_abs:.6e} rms={rms:.6e} cos={cos:.9}"
    );
}

#[cfg(not(target_arch = "wasm32"))]
fn array_diff_stats(a: &Array2<f32>, b: &Array2<f32>) -> (f64, f64, f64) {
    if a.shape() != b.shape() {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let mut max_abs = 0.0f64;
    let mut sum_sq_diff = 0.0f64;
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;
    let mut n = 0usize;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let x = x as f64;
        let y = y as f64;
        let diff = x - y;
        max_abs = max_abs.max(diff.abs());
        sum_sq_diff += diff * diff;
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
        n += 1;
    }
    let rms = if n == 0 {
        0.0
    } else {
        (sum_sq_diff / n as f64).sqrt()
    };
    let denom = norm_a.sqrt() * norm_b.sqrt();
    let cos = if denom == 0.0 { 1.0 } else { dot / denom };
    (max_abs, rms, cos)
}

#[cfg(not(target_arch = "wasm32"))]
fn parse_quant_format(fmt: &str) -> Option<QuantFormat> {
    match fmt {
        "Q4_K" => Some(QuantFormat::Q4_K),
        "Q4_KF" => Some(QuantFormat::Q4_KF),
        "Q6_K" => Some(QuantFormat::Q6_K),
        _ => None,
    }
}

/// Equivalent Standard KV memory in bytes for `seq_len` tokens (FP16).
pub fn kv_memory_bytes_for_seq(weights: larql_inference::WeightsView, seq_len: usize) -> usize {
    let arch = &*weights.arch;
    (0..weights.num_layers)
        .map(|l| {
            let kv_dim = arch.num_kv_heads_for_layer(l) * arch.head_dim_for_layer(l);
            seq_len * kv_dim * 2 * 2
        })
        .sum()
}

// Native-only: its callers (prefill.rs::rs_prefill, walk.rs) are both
// native-gated -- rs_prefill directly, walk.rs as a whole module.
#[cfg(not(target_arch = "wasm32"))]
pub(super) fn last_row(h: &Array2<f32>) -> Array2<f32> {
    let last = h.shape()[0] - 1;
    h.slice(s![last..=last, ..]).to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_compute::CpuBackend;
    use larql_inference::test_utils::make_test_weights;

    // ── recompute_kv ──────────────────────────────────────────────────────────

    #[test]
    fn recompute_kv_returns_some_with_valid_weights() {
        let weights = make_test_weights();
        let h = Array2::from_elem((3, weights.hidden_size), 0.5f32);
        let result = recompute_kv(
            larql_inference::WeightsView::dense(&weights),
            &h,
            0,
            0,
            &CpuBackend,
            None,
        );
        assert!(
            result.is_some(),
            "recompute_kv should return Some with valid weights"
        );
    }

    #[test]
    fn recompute_kv_output_shape_correct() {
        let weights = make_test_weights();
        let seq_len = 4;
        let h = Array2::from_elem((seq_len, weights.hidden_size), 1.0f32);
        let (k, v) = recompute_kv(
            larql_inference::WeightsView::dense(&weights),
            &h,
            0,
            0,
            &CpuBackend,
            None,
        )
        .unwrap();
        let kv_dim = weights.num_kv_heads * weights.head_dim;
        assert_eq!(k.shape(), &[seq_len, kv_dim], "K shape mismatch");
        assert_eq!(v.shape(), &[seq_len, kv_dim], "V shape mismatch");
    }

    #[test]
    fn recompute_kv_output_is_finite() {
        let weights = make_test_weights();
        let h = Array2::from_elem((2, weights.hidden_size), 0.1f32);
        let (k, v) = recompute_kv(
            larql_inference::WeightsView::dense(&weights),
            &h,
            0,
            0,
            &CpuBackend,
            None,
        )
        .unwrap();
        assert!(
            k.iter().all(|v| v.is_finite()),
            "K contains non-finite values"
        );
        assert!(
            v.iter().all(|v| v.is_finite()),
            "V contains non-finite values"
        );
    }

    #[test]
    fn recompute_kv_abs_start_shifts_rope() {
        let weights = make_test_weights();
        let h = Array2::from_elem((1, weights.hidden_size), 0.5f32);
        // Different abs_start should produce different RoPE-applied K
        let (k0, _) = recompute_kv(
            larql_inference::WeightsView::dense(&weights),
            &h,
            0,
            0,
            &CpuBackend,
            None,
        )
        .unwrap();
        let (k5, _) = recompute_kv(
            larql_inference::WeightsView::dense(&weights),
            &h,
            0,
            5,
            &CpuBackend,
            None,
        )
        .unwrap();
        let diff: f32 = k0.iter().zip(k5.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(
            diff > 0.0,
            "RoPE at different positions should produce different K"
        );
    }

    #[test]
    fn walk_project_topk_full_k_matches_dense_projection() {
        let x = Array2::from_shape_vec((2, 3), vec![1.0, -2.0, 0.5, 0.25, 0.75, -1.0]).unwrap();
        let w = Array2::from_shape_vec(
            (4, 3),
            vec![
                0.5, 1.0, -0.5, -1.0, 0.25, 0.75, 0.0, 2.0, 1.0, 1.5, -0.5, 0.25,
            ],
        )
        .unwrap();
        let walked = walk_project_topk(&x, &w, 4).unwrap();
        let dense = dot_proj_gpu(&x, &w, None);
        let max_diff = walked
            .iter()
            .zip(dense.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(max_diff < 1e-6, "max_diff={max_diff}");
    }

    #[test]
    fn walk_project_topk_keeps_largest_absolute_outputs_per_row() {
        let x = Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap();
        let w = Array2::from_shape_vec(
            (4, 3),
            vec![1.0, 0.0, 0.0, 0.0, -3.0, 0.0, 0.0, 0.0, 2.0, -2.0, 0.0, 0.0],
        )
        .unwrap();
        let walked = walk_project_topk(&x, &w, 2).unwrap();
        let non_zero: Vec<usize> = walked
            .row(0)
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| (v != 0.0).then_some(i))
            .collect();
        assert_eq!(non_zero, vec![1, 2]);
        assert_eq!(walked[[0, 1]], -6.0);
        assert_eq!(walked[[0, 2]], 6.0);
    }

    #[test]
    fn walk_project_cached_topk_reuses_selector_layer_indices() {
        WALK_KV_SELECTION.with(|slot| {
            *slot.borrow_mut() = None;
        });
        let x = Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap();
        let selector_w_k = Array2::from_shape_vec(
            (4, 3),
            vec![1.0, 0.0, 0.0, 0.0, -3.0, 0.0, 0.0, 0.0, 2.0, -2.0, 0.0, 0.0],
        )
        .unwrap();
        let selector_w_v = selector_w_k.clone();
        cache_walk_kv_selection(4, 2, &x, &selector_w_k, &selector_w_v);

        let later_w = Array2::from_shape_vec(
            (4, 3),
            vec![
                10.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 30.0, 40.0, 0.0, 0.0,
            ],
        )
        .unwrap();
        let walked =
            walk_project_cached_topk(&x, &later_w, 2, 4, KvProjection::K).expect("cached walk");
        let non_zero: Vec<usize> = walked
            .row(0)
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| (v != 0.0).then_some(i))
            .collect();
        assert_eq!(non_zero, vec![1, 2]);
        assert_eq!(walked[[0, 1]], 40.0);
        assert_eq!(walked[[0, 2]], 90.0);
    }

    #[test]
    fn markov_walk_kv_layer_spec_accepts_ranges_and_singletons() {
        assert!(layer_in_spec("5-20", 5));
        assert!(layer_in_spec("5-20", 20));
        assert!(layer_in_spec(" 2, 5-7, 26 ", 6));
        assert!(layer_in_spec(" 2, 5-7, 26 ", 26));
        assert!(!layer_in_spec("5-20", 4));
        assert!(!layer_in_spec("5-20", 21));
        assert!(!layer_in_spec("x-y, 30", 29));
    }

    #[test]
    fn kv_memory_bytes_for_seq_scales_linearly() {
        let weights = make_test_weights();
        let one = kv_memory_bytes_for_seq(larql_inference::WeightsView::dense(&weights), 1);
        let ten = kv_memory_bytes_for_seq(larql_inference::WeightsView::dense(&weights), 10);
        assert!(one > 0);
        assert_eq!(ten, one * 10, "kv memory must scale linearly with seq len");
    }

    // ── parse_quant_format pure helper (lines 384-391) ───────────────────

    #[test]
    fn parse_quant_format_recognises_q4k_q4kf_q6k() {
        assert!(matches!(
            parse_quant_format("Q4_K"),
            Some(QuantFormat::Q4_K)
        ));
        assert!(matches!(
            parse_quant_format("Q4_KF"),
            Some(QuantFormat::Q4_KF)
        ));
        assert!(matches!(
            parse_quant_format("Q6_K"),
            Some(QuantFormat::Q6_K)
        ));
    }

    #[test]
    fn parse_quant_format_unknown_returns_none() {
        assert!(parse_quant_format("Q8_0").is_none());
        assert!(parse_quant_format("F16").is_none());
        assert!(parse_quant_format("").is_none());
        assert!(parse_quant_format("Q4").is_none());
        assert!(parse_quant_format("nonsense").is_none());
    }

    // ── Pure helpers ────────────────────────────────────────────────────────

    #[test]
    fn dot_rows_basic_arithmetic() {
        let a = ndarray::arr1(&[1.0f32, 2.0, 3.0]);
        let b = ndarray::arr1(&[4.0f32, 5.0, 6.0]);
        // 1*4 + 2*5 + 3*6 = 32
        assert!((dot_rows(a.view(), b.view()) - 32.0).abs() < 1e-6);
    }

    #[test]
    fn compare_abs_desc_orders_by_absolute_magnitude() {
        let a = (0usize, -5.0f32);
        let b = (1usize, 3.0f32);
        // |a| > |b| so a comes before b under descending sort.
        assert_eq!(compare_abs_desc(&a, &b), Ordering::Less);
        assert_eq!(compare_abs_desc(&b, &a), Ordering::Greater);
        // Tie: NaN/Equal fallback returns Equal.
        let c = (2usize, 5.0f32);
        let d = (3usize, -5.0f32);
        assert_eq!(compare_abs_desc(&c, &d), Ordering::Equal);
    }

    #[test]
    fn array_diff_stats_identical_arrays_returns_zero_diff_and_unit_cos() {
        // Identical arrays → max_abs=0, rms=0, cos=1.
        let a = Array2::<f32>::from_elem((2, 3), 1.5);
        let b = a.clone();
        let (max_abs, rms, cos) = array_diff_stats(&a, &b);
        assert!(max_abs.abs() < 1e-12);
        assert!(rms.abs() < 1e-12);
        assert!(
            (cos - 1.0).abs() < 1e-9,
            "cos should be 1 for identical, got {cos}"
        );
    }

    #[test]
    fn array_diff_stats_reports_max_abs_and_rms() {
        let a = Array2::<f32>::from_shape_vec((1, 3), vec![0.0, 0.0, 0.0]).unwrap();
        let b = Array2::<f32>::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap();
        let (max_abs, rms, cos) = array_diff_stats(&a, &b);
        // max_abs = 3, rms = sqrt(((-1)^2 + (-2)^2 + (-3)^2) / 3) = sqrt(14/3)
        assert!((max_abs - 3.0).abs() < 1e-9);
        assert!((rms - (14.0_f64 / 3.0).sqrt()).abs() < 1e-9);
        // a is all zeros so cosine has denom=0 → returns 1.0 sentinel.
        assert!((cos - 1.0).abs() < 1e-9, "all-zeros a → cos sentinel = 1");
    }

    #[test]
    fn array_diff_stats_mismatched_shape_returns_nan_tuple() {
        let a = Array2::<f32>::zeros((2, 3));
        let b = Array2::<f32>::zeros((3, 2));
        let (max_abs, rms, cos) = array_diff_stats(&a, &b);
        assert!(max_abs.is_nan() && rms.is_nan() && cos.is_nan());
    }

    #[test]
    fn layer_in_spec_accepts_singleton_and_ranges() {
        // Direct test of the spec parser. Covers "5", "5-7", "1,5-7,9"
        // forms — the helper layered under markov_walk_kv_diag_layer
        // and markov_walk_kv_top_k env-var paths.
        assert!(layer_in_spec("5", 5));
        assert!(!layer_in_spec("5", 6));
        assert!(layer_in_spec("5-7", 5));
        assert!(layer_in_spec("5-7", 6));
        assert!(layer_in_spec("5-7", 7));
        assert!(!layer_in_spec("5-7", 8));
        assert!(layer_in_spec("1,5-7,9", 1));
        assert!(layer_in_spec("1,5-7,9", 6));
        assert!(layer_in_spec("1,5-7,9", 9));
        assert!(!layer_in_spec("1,5-7,9", 3));
    }

    #[test]
    fn layer_in_spec_rejects_malformed_input() {
        // Non-numeric pieces should not crash and should return false.
        assert!(!layer_in_spec("abc", 5));
        assert!(!layer_in_spec("", 5));
    }

    #[test]
    fn print_walk_kv_diag_runs_without_panicking() {
        // Pure logging helper. The body just prints diagnostic stats;
        // exercising it produces console output but no observable
        // state change. Coverage credit for the function body.
        let a = Array2::<f32>::from_elem((2, 4), 1.0f32);
        let b = Array2::<f32>::from_elem((2, 4), 0.5f32);
        print_walk_kv_diag(0, "test_path", "K", "test_label", &a, &b);
    }

    // ── Env-var-gated walk-KV paths ───────────────────────────────────────────
    //
    // These tests cover the `LARQL_MARKOV_WALK_KV_*` /
    // `LARQL_MARKOV_KV_FORCE_F32` paths in `recompute_kv` and the
    // `markov_walk_kv_*` helpers. Production reads via
    // `read_markov_env`, which consults the per-thread
    // `MARKOV_ENV_OVERRIDE` map *before* `std::env::var`. Tests inject
    // values through `set_markov_env_override` — no process-global env
    // mutation, no `#[serial]` needed, no race with other parallel
    // tests that also call `recompute_kv`.

    #[test]
    fn markov_walk_kv_requested_top_k_parses_clamps_and_rejects_zero() {
        clear_markov_env_overrides();
        assert_eq!(markov_walk_kv_requested_top_k(32), None);
        set_markov_env_override("LARQL_MARKOV_WALK_KV_TOPK", Some("8"));
        assert_eq!(markov_walk_kv_requested_top_k(32), Some(8));
        assert_eq!(
            markov_walk_kv_requested_top_k(4),
            Some(4),
            "clamp to kv_dim"
        );
        set_markov_env_override("LARQL_MARKOV_WALK_KV_TOPK", Some("0"));
        assert_eq!(markov_walk_kv_requested_top_k(32), None);
        set_markov_env_override("LARQL_MARKOV_WALK_KV_TOPK", Some("abc"));
        assert_eq!(markov_walk_kv_requested_top_k(32), None);
        clear_markov_env_overrides();
    }

    #[test]
    fn markov_walk_kv_select_at_parses_layer_index() {
        clear_markov_env_overrides();
        assert_eq!(markov_walk_kv_select_at(), None);
        set_markov_env_override("LARQL_MARKOV_WALK_KV_SELECT_AT", Some("7"));
        assert_eq!(markov_walk_kv_select_at(), Some(7));
        set_markov_env_override("LARQL_MARKOV_WALK_KV_SELECT_AT", Some("bad"));
        assert_eq!(markov_walk_kv_select_at(), None);
        clear_markov_env_overrides();
    }

    #[test]
    fn markov_walk_kv_diag_enabled_accepts_truthy_strings() {
        clear_markov_env_overrides();
        assert!(!markov_walk_kv_diag_enabled());
        for val in ["1", "true", "TRUE", "yes", "on"] {
            set_markov_env_override("LARQL_MARKOV_WALK_KV_DIAG", Some(val));
            assert!(markov_walk_kv_diag_enabled(), "should accept {val}");
        }
        for val in ["0", "false", "no"] {
            set_markov_env_override("LARQL_MARKOV_WALK_KV_DIAG", Some(val));
            assert!(!markov_walk_kv_diag_enabled(), "should reject {val}");
        }
        clear_markov_env_overrides();
    }

    #[test]
    fn markov_kv_force_f32_projection_reads_env() {
        clear_markov_env_overrides();
        assert!(!markov_kv_force_f32_projection());
        set_markov_env_override("LARQL_MARKOV_KV_FORCE_F32", Some("1"));
        assert!(markov_kv_force_f32_projection());
        set_markov_env_override("LARQL_MARKOV_KV_FORCE_F32", Some("no"));
        assert!(!markov_kv_force_f32_projection());
        clear_markov_env_overrides();
    }

    #[test]
    fn markov_walk_kv_diag_layer_respects_layers_spec() {
        clear_markov_env_overrides();
        assert!(markov_walk_kv_diag_layer(0));
        assert!(markov_walk_kv_diag_layer(99));
        set_markov_env_override("LARQL_MARKOV_WALK_KV_LAYERS", Some("3-5"));
        assert!(markov_walk_kv_diag_layer(4));
        assert!(!markov_walk_kv_diag_layer(0));
        clear_markov_env_overrides();
    }

    #[test]
    fn markov_walk_kv_top_k_honours_layers_and_select_at_gates() {
        clear_markov_env_overrides();
        assert_eq!(markov_walk_kv_top_k(0, 32), None);
        set_markov_env_override("LARQL_MARKOV_WALK_KV_TOPK", Some("4"));
        set_markov_env_override("LARQL_MARKOV_WALK_KV_LAYERS", Some("5-7"));
        assert_eq!(markov_walk_kv_top_k(0, 32), None);
        assert_eq!(markov_walk_kv_top_k(6, 32), Some(4));
        set_markov_env_override("LARQL_MARKOV_WALK_KV_LAYERS", None);
        set_markov_env_override("LARQL_MARKOV_WALK_KV_SELECT_AT", Some("6"));
        assert_eq!(markov_walk_kv_top_k(6, 32), None);
        assert_eq!(markov_walk_kv_top_k(7, 32), Some(4));
        clear_markov_env_overrides();
    }

    #[test]
    fn recompute_kv_force_f32_disables_q4k_path() {
        clear_markov_env_overrides();
        set_markov_env_override("LARQL_MARKOV_KV_FORCE_F32", Some("1"));
        let weights = make_test_weights();
        let h = Array2::from_elem((2, weights.hidden_size), 0.5f32);
        let (k, v) = recompute_kv(
            larql_inference::WeightsView::dense(&weights),
            &h,
            0,
            0,
            &CpuBackend,
            None,
        )
        .unwrap();
        let kv_dim = weights.num_kv_heads * weights.head_dim;
        assert_eq!(k.shape(), &[2, kv_dim]);
        assert_eq!(v.shape(), &[2, kv_dim]);
        clear_markov_env_overrides();
    }

    #[test]
    fn recompute_kv_topk_routes_through_walk_projection() {
        clear_markov_env_overrides();
        set_markov_env_override("LARQL_MARKOV_WALK_KV_TOPK", Some("2"));
        let weights = make_test_weights();
        let h = Array2::from_elem((2, weights.hidden_size), 0.25f32);
        let result = recompute_kv(
            larql_inference::WeightsView::dense(&weights),
            &h,
            0,
            0,
            &CpuBackend,
            None,
        );
        assert!(result.is_some());
        clear_markov_env_overrides();
    }

    #[test]
    fn recompute_kv_select_at_uses_cached_indices_on_later_layers() {
        clear_markov_env_overrides();
        set_markov_env_override("LARQL_MARKOV_WALK_KV_TOPK", Some("2"));
        set_markov_env_override("LARQL_MARKOV_WALK_KV_SELECT_AT", Some("0"));
        let weights = make_test_weights();
        let h = Array2::from_elem((2, weights.hidden_size), 0.25f32);
        // Layer 0: should_cache_selection fires, populates
        // WALK_KV_SELECTION; layer 1: walk_project_cached_topk reads it.
        let _ = recompute_kv(
            larql_inference::WeightsView::dense(&weights),
            &h,
            0,
            0,
            &CpuBackend,
            None,
        );
        if weights.num_layers >= 2 {
            let result = recompute_kv(
                larql_inference::WeightsView::dense(&weights),
                &h,
                1,
                0,
                &CpuBackend,
                None,
            );
            assert!(result.is_some());
        }
        clear_markov_env_overrides();
    }

    #[test]
    fn recompute_kv_diag_fires_when_enabled() {
        clear_markov_env_overrides();
        set_markov_env_override("LARQL_MARKOV_WALK_KV_DIAG", Some("1"));
        let weights = make_test_weights();
        let h = Array2::from_elem((1, weights.hidden_size), 0.5f32);
        let result = recompute_kv(
            larql_inference::WeightsView::dense(&weights),
            &h,
            0,
            0,
            &CpuBackend,
            None,
        );
        assert!(result.is_some());
        clear_markov_env_overrides();
    }
}
