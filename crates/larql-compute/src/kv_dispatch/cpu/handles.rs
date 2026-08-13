//! CPU-side handle types (`CpuKvHandle`, `CpuResidualHandle`,
//! `CpuQ4kCacheHandle`) and the dispatch-time downcast helpers. Split
//! from `kv_dispatch/cpu.rs` — see `cpu/mod.rs` for the module doc.

use ndarray::Array2;

use crate::attention::SharedKV;
use crate::kv_dispatch::{KvHandle, KvHandleInner, ResidualHandle, ResidualHandleInner};

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

// ─── CpuKvHandle ────────────────────────────────────────────────────────────

/// Single-layer K/V cache held in host memory as growable row-major
/// buffers (`rows × kv_dim` valid prefix of each Vec). Appending one row
/// per decode step is amortised O(kv_dim) — the previous `SharedKV`-tuple
/// representation re-allocated and copied the WHOLE cache per layer per
/// step (clone + zeros + two assigns ≈ 3 full-cache copies), which the
/// decode sample showed as the dominant O(ctx) serial sink.
pub struct CpuKvHandle {
    /// Layer index this handle was minted for. Carried for debugging
    /// / future trait surface; not consulted by the current append /
    /// attend paths (the trait already takes `layer` per call).
    #[allow(dead_code)]
    layer: usize,
    pub(super) kv_dim: usize,
    pub(super) k_buf: Vec<f32>,
    pub(super) v_buf: Vec<f32>,
    pub(super) rows: usize,
}

impl CpuKvHandle {
    pub(super) fn new(layer: usize, kv_dim: usize) -> Self {
        Self {
            layer,
            kv_dim,
            k_buf: Vec::new(),
            v_buf: Vec::new(),
            rows: 0,
        }
    }

    /// Append one K/V row in place (amortised O(kv_dim) — Vec doubling).
    pub(super) fn append_row(&mut self, k_row: &[f32], v_row: &[f32]) {
        debug_assert_eq!(k_row.len(), self.kv_dim);
        debug_assert_eq!(v_row.len(), self.kv_dim);
        self.k_buf.extend_from_slice(k_row);
        self.v_buf.extend_from_slice(v_row);
        self.rows += 1;
    }

    /// Views over the valid prefix — zero-copy access for the attend half.
    pub(super) fn views(
        &self,
    ) -> Option<(ndarray::ArrayView2<'_, f32>, ndarray::ArrayView2<'_, f32>)> {
        if self.rows == 0 {
            return None;
        }
        let n = self.rows * self.kv_dim;
        let k = ndarray::ArrayView2::from_shape((self.rows, self.kv_dim), &self.k_buf[..n])
            .expect("k_buf prefix matches rows × kv_dim");
        let v = ndarray::ArrayView2::from_shape((self.rows, self.kv_dim), &self.v_buf[..n])
            .expect("v_buf prefix matches rows × kv_dim");
        Some((k, v))
    }

    /// Replace the buffers from an owned `SharedKV` — prefill and the f32
    /// fallback path still produce whole arrays.
    pub(super) fn replace_state(&mut self, kv: SharedKV) {
        let (k, v) = kv;
        debug_assert_eq!(k.shape()[1], self.kv_dim);
        self.rows = k.shape()[0];
        self.k_buf.clear();
        self.v_buf.clear();
        match k.as_slice() {
            Some(s) => self.k_buf.extend_from_slice(s),
            None => self.k_buf.extend(k.iter().copied()),
        }
        match v.as_slice() {
            Some(s) => self.v_buf.extend_from_slice(s),
            None => self.v_buf.extend(v.iter().copied()),
        }
    }

    /// Materialise an owned `SharedKV` copy (host reads, f32 fallback).
    pub(super) fn to_shared(&self) -> Option<SharedKV> {
        if self.rows == 0 {
            return None;
        }
        let n = self.rows * self.kv_dim;
        let k = Array2::from_shape_vec((self.rows, self.kv_dim), self.k_buf[..n].to_vec()).ok()?;
        let v = Array2::from_shape_vec((self.rows, self.kv_dim), self.v_buf[..n].to_vec()).ok()?;
        Some((k, v))
    }

    /// Remove the most recently appended row (failure-path undo: the q4k
    /// attend half can in principle fail AFTER the project half appended —
    /// the handle must be left exactly as before the step so engine
    /// fallbacks that reuse it see consistent state).
    pub(super) fn pop_row(&mut self) {
        if self.rows > 0 {
            self.rows -= 1;
            self.k_buf.truncate(self.rows * self.kv_dim);
            self.v_buf.truncate(self.rows * self.kv_dim);
        }
    }
}

impl KvHandleInner for CpuKvHandle {
    fn cached_len(&self) -> usize {
        self.rows
    }

    fn kv_dim(&self) -> usize {
        self.kv_dim
    }

    fn backend_name(&self) -> &'static str {
        "cpu"
    }

    fn as_any(&self) -> &dyn core::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn core::any::Any {
        self
    }
}

/// Downcast helper — backend implementations use this to retrieve the
/// concrete handle type from an opaque `KvHandle`. Panics if the
/// handle was allocated by a different backend.
pub(super) fn cpu_handle(h: &KvHandle) -> &CpuKvHandle {
    h.as_inner()
        .as_any()
        .downcast_ref::<CpuKvHandle>()
        .unwrap_or_else(|| {
            panic!(
                "CpuBackend::KvDispatch received a foreign handle (backend={}); \
                 handles must be allocated by the same backend that consumes them",
                h.backend_name()
            )
        })
}

pub(super) fn cpu_handle_mut(h: &mut KvHandle) -> &mut CpuKvHandle {
    let name = h.backend_name();
    h.as_inner_mut()
        .as_any_mut()
        .downcast_mut::<CpuKvHandle>()
        .unwrap_or_else(|| {
            panic!(
                "CpuBackend::KvDispatch received a foreign handle (backend={name}); \
                 handles must be allocated by the same backend that consumes them"
            )
        })
}

// ─── CpuResidualHandle ──────────────────────────────────────────────────────

/// Host-resident residual upload. CPU has no device memory to manage,
/// so this is just a flat `Vec<f32>` wrapper. Storing flat matches
/// what `forward_from_layer` consumes (`&[f32]` interpreted as
/// `[seq_len, hidden]` row-major).
pub struct CpuResidualHandle {
    pub(super) flat: Vec<f32>,
    pub(super) shape: (usize, usize),
}

impl ResidualHandleInner for CpuResidualHandle {
    fn shape(&self) -> (usize, usize) {
        self.shape
    }

    fn backend_name(&self) -> &'static str {
        "cpu"
    }

    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}

pub(super) fn cpu_residual(r: &ResidualHandle) -> &CpuResidualHandle {
    r.as_inner()
        .as_any()
        .downcast_ref::<CpuResidualHandle>()
        .unwrap_or_else(|| {
            panic!(
                "CpuBackend::KvDispatch received a foreign residual handle (backend={}); \
                 handles must be allocated by the same backend that consumes them",
                r.backend_name()
            )
        })
}

// ─── CpuQ4kCacheHandle — Q4K cached-decode handle ──────────────────────────
//
// Wraps the production `CpuKvCache` (per-layer K/V) so it can flow through
// the dispatch trait's `KvHandle` shape. Cache populated by
// `cached_prefill_q4k`; consumed by `cached_decode_step_q4k`.
//
// One handle per engine (not per layer), unlike the legacy `CpuKvHandle`
// (one per layer for the f32 per-layer dispatch path). The two shapes
// coexist because they serve different dispatch granularities.

pub struct CpuQ4kCacheHandle {
    pub(super) cache: crate::kquant_forward::CpuKvCache,
}

impl KvHandleInner for CpuQ4kCacheHandle {
    /// Rows in the **first populated layer**.
    ///
    /// This assumes every layer holds the same count, which is true only
    /// while nothing rewrites layers independently. It holds today
    /// because this is the *coarse* whole-model handle: engines reach
    /// per-layer row surgery through `PerLayerKvAccess`, which is not
    /// offered on the coarse path at all, so a heterogeneous coarse
    /// cache is currently unreachable. If that changes, this must become
    /// a `KvExtent`-shaped report rather than a scalar — see
    /// `larql_inference::kv_engine::KvExtent`.
    fn cached_len(&self) -> usize {
        self.cache
            .iter()
            .filter_map(|o| o.as_ref())
            .map(|(k, _)| k.shape()[0])
            .next()
            .unwrap_or(0)
    }

    fn kv_dim(&self) -> usize {
        self.cache
            .iter()
            .filter_map(|o| o.as_ref())
            .map(|(k, _)| k.shape()[1])
            .next()
            .unwrap_or(0)
    }

    fn backend_name(&self) -> &'static str {
        "cpu-q4k"
    }

    /// One handle, every layer — so sum the layers instead of taking the
    /// trait default's single-layer estimate. `cached_len`/`kv_dim` above
    /// deliberately report layer-0's shape (the dispatch surface asks for
    /// "the" cache geometry), and multiplying those alone undercounted a
    /// 28-layer model's K/V by 28×.
    fn resident_bytes(&self) -> usize {
        self.cache
            .iter()
            .filter_map(|o| o.as_ref())
            .map(|(k, v)| {
                (k.shape()[0] * k.shape()[1] + v.shape()[0] * v.shape()[1])
                    * core::mem::size_of::<f32>()
            })
            .sum()
    }

    fn as_any(&self) -> &dyn core::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn core::any::Any {
        self
    }
}

/// Non-panicking downcast for the coarse-decode path. Returns `None`
/// when the handle is not a `CpuQ4kCacheHandle` (e.g. a per-layer
/// `CpuKvHandle` from the per-layer dispatch path) — the coarse intent
/// contract lets callers interpret `None` as "this backend cannot
/// coarse-decode this handle" and fall back, rather than aborting.
/// Invariant: a `Some` return is the same handle `coarse_prefill`
/// minted on this backend.
pub(super) fn try_cpu_q4k_cache_mut(h: &mut KvHandle) -> Option<&mut CpuQ4kCacheHandle> {
    h.as_inner_mut()
        .as_any_mut()
        .downcast_mut::<CpuQ4kCacheHandle>()
}

#[cfg(test)]
#[path = "handles_resident_tests.rs"]
mod resident_bytes_tests;
