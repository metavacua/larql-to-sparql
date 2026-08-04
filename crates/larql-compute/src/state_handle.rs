//! State handles — opaque references to per-layer state rows and slabs
//! that may live on different devices or remote nodes.
//!
//! **Status:** 📝 Trait surface draft for W10 Phase A (2026-05-18).
//! No implementations yet — this file defines the API contract that
//! `larql-compute`'s CPU backend, `larql-compute-metal`'s GPU backend,
//! and a future `larql-compute-grid` remote backend will satisfy.
//!
//! ## Motivation
//!
//! Today's hot path on Metal moves per-layer `(h_in, k_new, v_new)`
//! GPU → CPU every decode step:
//!
//! 1. Metal kernel writes the rows to staging buffers (W7 blit fusion
//!    inside one command buffer).
//! 2. `decode_token_with_state_dump` drains, reads back into
//!    `Vec<Vec<f32>>` (`DecodeStateDump`).
//! 3. `coarse_decode_step_with_state` wraps each `Vec<f32>` as
//!    `Array2::from_shape_vec((1, dim), vec)` into `PerLayerDecodeState`.
//! 4. Engine's `decode_step_via_dispatch` calls `append_row` to fold
//!    each row into `rs.stored[layer]` and `rs.hot_kv[layer]`.
//!
//! Steps 2–4 are eager — they pay readback + wrap + memcpy whether or
//! not the engine ever reads the row before window-clip. The handle
//! abstraction defers materialisation until a consumer actually needs
//! bytes on the local CPU.
//!
//! The same shape composes with two future deployments:
//!
//! - **Layer-sharded grid** (`larql-grid`): a row produced on node A
//!   for layer L can stay on node A; the engine sees a
//!   `RemoteStateHandle` and only fetches when this node needs to read.
//! - **Remote FFN** (`--ffn http://...`): residuals can stay on the
//!   FFN node across multiple layers; the engine's slab holds a
//!   `RemoteSlabHandle` for those layers.
//!
//! ## Composition with StatePolicy
//!
//! See `crates/larql-kv/docs/state-policy.md`. Each engine declares
//! whether a slab holds **canonical** state (defines the continuation
//! point — `materialise()` must round-trip losslessly) or
//! **derivative** state (a cache the engine can rebuild from canonical
//! — `materialise()` MAY recompute from canonical). The role tag
//! lives on the slab, not on individual rows: rows are just data, the
//! slab knows what kind of state it represents.
//!
//! This prevents the **PCA-90 inversion** class of bug — "refresh
//! derivative more often" is a free knob; "refresh canonical more
//! often" is state intervention — at the API boundary, instead of
//! relying on convention.
//!
//! ## Composition with LayerEngine
//!
//! See `crates/larql-inference/docs/specs/layer-engine.md`. The
//! handle surface lives *below* LayerEngine — each `KvEngine_L` still
//! owns its own slabs at its layer; LayerEngine just chooses which
//! engine runs where. Two composition points matter:
//!
//! - **[`SlabRole`] is the static kind; LayerEngine's §4.2 skip rule
//!   is the dynamic query.** A `Derivative` tag is necessary for a
//!   LayerEngine to elide a K/V append at L, but not sufficient: the
//!   engine's own State Policy must also report
//!   `permits_no_append_at(L)` for that decode step. The two
//!   compose; this trait surface answers the static half.
//! - **[`RowLocation`] is what LayerEngine's §6.2 backend refusal
//!   reads.** A LayerEngine with heterogeneous slab locations (e.g.
//!   `L0–12 = CompiledLookup` on a cache server, `L13+ =
//!   MarkovResidual` on a GPU node) is enumerable through this enum;
//!   backends decline LayerEngines whose location set they can't
//!   serve at construction, not silently mid-decode.

use ndarray::Array2;

/// Where a row or slab physically lives.
///
/// Used as a batching hint by engines that want to coalesce work
/// across layers (e.g. "if all rows are LocalGpu, issue one drain
/// instead of N"; "if any row is Remote, coalesce the fetches into a
/// single gRPC call"). Engines should not branch on the backend name
/// inside `LocalGpu` for correctness — that's an implementation
/// detail of the device's handle type.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RowLocation {
    /// Resident in this process's CPU memory. `materialise()` is a
    /// zero-cost borrow or a single memcpy.
    LocalCpu,
    /// Resident on a local GPU. `materialise()` triggers a GPU → CPU
    /// readback. `backend` names the GPU API ("metal", "vulkan", …)
    /// for diagnostics only.
    LocalGpu { backend: &'static str },
    /// Resident on a peer node in a `larql-grid` deployment.
    /// `materialise()` triggers a network fetch. `node_id` identifies
    /// the peer; opaque to engines.
    Remote { node_id: u64 },
}

/// Whether a slab holds canonical or derivative state.
///
/// See `crates/larql-kv/docs/state-policy.md` §2.1 / §2.2 for the
/// formal definitions; the short version:
///
/// - **Canonical**: discarding it loses the conversation. Examples:
///   `MarkovResidualEngine`'s residual stream, `TurboQuantEngine`'s
///   compressed K/V (destructive), `StandardEngine`'s K/V tensors,
///   `WindowedCheckpointEngine`'s in-window K/V.
/// - **Derivative**: discardable. The engine can rebuild it from
///   canonical state + model weights without changing its output
///   distribution. Example: `MarkovResidualEngine`'s hot K/V cache
///   (W2 — reprojectable from `rs.stored`).
///
/// Materialise semantics flow from this:
///
/// - On a `Canonical` slab, [`SlabHandle::materialise_range`] MUST
///   round-trip losslessly. If a transport is lossy (e.g. a future
///   remote backend that compresses on the wire), that's a contract
///   change — bumps `exact_logits` to `bounded_KL(ε)` — and requires
///   a spec PR, not just an engine PR.
/// - On a `Derivative` slab, [`SlabHandle::materialise_range`] MAY
///   recompute from canonical state. Implementations are free to
///   discard the cached bytes and ask the engine to re-project.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SlabRole {
    /// State that defines the engine's continuation point. Lossless
    /// materialisation required.
    Canonical,
    /// Cache reconstructible from canonical state. Free to discard.
    Derivative,
}

/// A 2D chunk of f32s representing one layer's state in one of three
/// slots (`h_in`, `k_new`, or `v_new`). Opaque to consumers — the
/// data may live on GPU or on a remote node until materialised.
///
/// **Shape duality.** [`PerLayerDecodeState`](crate::PerLayerDecodeState)
/// is filled by both prefill (one handle per layer, shape
/// `[seq_len, dim]`) and decode-step (one handle per layer, shape
/// `[1, dim]`). The handle reports its actual shape via
/// [`shape`](Self::shape); engines must not assume single-row.
///
/// `StateHandle` is one trait for all three slot kinds because the
/// materialisation semantics are identical; the semantic identity
/// ("this is K vs V vs h_in") is fixed by which field of
/// [`PerLayerDecodeState`](crate::PerLayerDecodeState) the handle is
/// stored in. K/V handles have `cols = kv_dim_for_layer`; h_in
/// handles have `cols = hidden_size`.
///
/// ## Performance contract
///
/// - `shape()` and `location()` are O(1). They must not trigger I/O.
/// - [`to_array`](Self::to_array) is the borrow-equivalent — read
///   into a freshly-owned `Array2`. Triggers GPU readback or remote
///   fetch on non-local handles. Engines use this when they want a
///   read but plan to keep the handle around.
/// - [`into_array`](Self::into_array) is the consuming variant. On
///   local-CPU handles it moves the internal `Array2` out without a
///   copy (this is the W10 invariant that keeps CPU hot path at
///   today's allocation count). On non-local handles it materialises
///   then yields.
///
/// Engines on the W10 hot path drain `PerLayerDecodeState`'s vectors
/// into the engine slab via [`into_array`](Self::into_array), so the
/// CPU happy path has zero extra allocation vs the pre-W10 design.
pub trait StateHandle: Send + Sync {
    /// 2D shape `(rows, cols)`. Constant across the handle's
    /// lifetime.
    fn shape(&self) -> (usize, usize);

    /// Where the chunk lives. Engines use this as a batching hint;
    /// implementations must NOT trigger I/O to answer.
    fn location(&self) -> RowLocation;

    /// Read the chunk into a freshly-owned `Array2`. Triggers
    /// readback on non-local handles. Returns a new allocation each
    /// call.
    fn to_array(&self) -> Array2<f32>;

    /// Consume the handle and yield its `Array2`. CPU-local handles
    /// move their internal buffer out without a copy; non-local
    /// handles materialise then yield.
    fn into_array(self: Box<Self>) -> Array2<f32>;
}

/// A growable per-layer buffer of state rows. Engines hold one slab
/// per `(layer, slot)` — e.g. `rs.stored[layer]` (h_in slab) and
/// `rs.hot_kv[layer].0` (K slab), `.1` (V slab).
///
/// **Phase status (2026-05-18):** Not wired in Phase A. Engines still
/// use `Array2<f32>` for their per-layer slabs and consume incoming
/// [`StateHandle`]s via [`StateHandle::into_array`]. `SlabHandle`
/// lands in Phase B (Metal: slab IS the kv cache, append is a length
/// bump; CPU: slab is a doubling-capacity `Array2<f32>` wrapper) and
/// Phase C (residual slab on GPU).
///
/// ## Role
///
/// The slab declares its [`SlabRole`] at construction. Engines pin
/// roles per slab kind; the role flows into
/// [`materialise_range`](SlabHandle::materialise_range) correctness
/// obligations:
///
/// - `Canonical` slab → `materialise_range` must round-trip losslessly.
/// - `Derivative` slab → `materialise_range` MAY recompute from
///   canonical (engine's choice; the slab impl exposes whichever it
///   has).
///
/// ## Location
///
/// The slab declares its [`RowLocation`] — typically the same device
/// as the rows it holds. Heterogeneous slabs (rows from different
/// devices) are an extension point: today the trait assumes one
/// location per slab.
///
/// ## Append contract
///
/// [`append`](Self::append) consumes a `Box<dyn StateHandle>`. The
/// slab is free to:
///
/// - Move the row's bytes into its growing buffer (CPU happy path).
/// - Inspect the row's location and skip the copy when the row
///   already lives in the slab's storage (Metal happy path — the
///   kernel wrote into the kv cache, the slab IS the kv cache, so
///   append is a length bump).
/// - Materialise + copy (cross-device fallback).
///
/// The `dim` of the appended row MUST match the slab's row dim. Slab
/// implementations may assert this.
pub trait SlabHandle: Send + Sync {
    /// Width of each row in f32 elements. Constant across the slab's
    /// lifetime.
    fn row_dim(&self) -> usize;

    /// Number of rows currently held.
    fn len(&self) -> usize;

    /// `true` when [`len`](Self::len) is zero.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether the slab holds canonical or derivative state. Fixed at
    /// construction; see [`SlabRole`].
    fn role(&self) -> SlabRole;

    /// Where the slab physically lives.
    fn location(&self) -> RowLocation;

    /// Append one chunk (typically a single row from a decode-step
    /// state handle). See type-level docs for the consume semantics.
    /// Panics (or returns an error in a future fallible variant) if
    /// the appended chunk's column count does not equal
    /// [`row_dim`](Self::row_dim).
    fn append(&mut self, chunk: Box<dyn StateHandle>);

    /// Read rows `start..end` as a contiguous `Array2<f32>` of shape
    /// `(end - start, row_dim())`.
    ///
    /// On a `Canonical` slab this must be a faithful representation
    /// of the slab's bytes (lossless w.r.t. the canonical form — e.g.
    /// codec-encoded rows materialise as the codec bytes, NOT
    /// re-encoded from a decoded form).
    ///
    /// On a `Derivative` slab the implementation may recompute the
    /// range from canonical state if doing so is cheaper than reading
    /// the cached bytes.
    ///
    /// Panics if `end > self.len()` or `start > end`.
    fn materialise_range(&self, start: usize, end: usize) -> Array2<f32>;

    /// Remove the oldest `n` rows and return them as a snapshot. Used
    /// by window-clip eviction to hand rows off to a cold-tier store.
    ///
    /// The returned snapshot owns its bytes — the slab's storage MAY
    /// be reused for future appends. Snapshot is an opaque type that
    /// can be materialised on demand (the eviction handoff to a cold
    /// tier may itself be cross-device).
    ///
    /// Panics if `n > self.len()`.
    fn evict_oldest(&mut self, n: usize) -> Box<dyn SlabSnapshot>;
}

/// Opaque snapshot of evicted rows. Produced by
/// [`SlabHandle::evict_oldest`]; consumed by cold-tier stores.
///
/// The snapshot's role inherits from its source slab: canonical-slab
/// evictions produce canonical snapshots that must round-trip
/// losslessly into the cold tier; derivative-slab evictions produce
/// derivative snapshots (which a cold tier may legitimately discard).
pub trait SlabSnapshot: Send + Sync {
    /// Number of rows in the snapshot.
    fn len(&self) -> usize;

    /// `true` when the snapshot is empty (zero rows).
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Width of each row.
    fn row_dim(&self) -> usize;

    /// Inherited role of the rows.
    fn role(&self) -> SlabRole;

    /// Materialise to a contiguous `Array2<f32>`. Same correctness
    /// obligations as [`SlabHandle::materialise_range`].
    fn materialise(&self) -> Array2<f32>;
}

/// CPU-resident [`StateHandle`] backed by an owned `Array2<f32>`.
///
/// **Performance contract:** [`into_array`](StateHandle::into_array)
/// moves the inner buffer out without a copy — this is the W10
/// invariant that keeps the CPU hot path at the pre-W10 allocation
/// count. [`to_array`](StateHandle::to_array) clones (1 allocation),
/// used by engines that want to read but keep the handle live.
///
/// Producers (`predict_kquant_prefill_with_state`,
/// `predict_kquant_decode_step_direct_with_state`, and Metal's
/// readback path) wrap their fresh `Array2<f32>` rows in this type
/// before pushing into [`crate::PerLayerDecodeState`]. The Metal
/// path uses it too at Phase A — the bytes already live on CPU after
/// the readback, so a `CpuStateHandle` is the correct wrapping. Phase
/// B will introduce a `MetalStateHandle` that defers the readback.
pub struct CpuStateHandle {
    array: Array2<f32>,
}

impl CpuStateHandle {
    /// Wrap an owned `Array2<f32>`. Zero-cost; the array is moved in.
    pub fn new(array: Array2<f32>) -> Self {
        Self { array }
    }

    /// Boxed convenience for producers that push directly into
    /// [`crate::PerLayerDecodeState`]'s `Vec<Box<dyn StateHandle>>`.
    pub fn boxed(array: Array2<f32>) -> Box<dyn StateHandle> {
        Box::new(Self::new(array))
    }
}

impl StateHandle for CpuStateHandle {
    fn shape(&self) -> (usize, usize) {
        let s = self.array.shape();
        (s[0], s[1])
    }

    fn location(&self) -> RowLocation {
        RowLocation::LocalCpu
    }

    fn to_array(&self) -> Array2<f32> {
        self.array.clone()
    }

    fn into_array(self: Box<Self>) -> Array2<f32> {
        self.array
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_handle_round_trip_preserves_bytes() {
        let arr = Array2::<f32>::from_shape_vec((2, 3), vec![1., 2., 3., 4., 5., 6.]).unwrap();
        let h: Box<dyn StateHandle> = CpuStateHandle::boxed(arr.clone());
        assert_eq!(h.shape(), (2, 3));
        assert_eq!(h.location(), RowLocation::LocalCpu);
        assert_eq!(h.to_array(), arr);
        assert_eq!(h.into_array(), arr);
    }

    #[test]
    fn cpu_handle_into_array_moves_without_clone() {
        // We can't directly observe the move on a stable API, but we
        // can verify the public contract: into_array yields the
        // shape-matched array, distinct allocation from to_array's
        // clone path.
        let arr = Array2::<f32>::zeros((4, 8));
        let h = CpuStateHandle::boxed(arr);
        let owned = h.into_array();
        assert_eq!(owned.shape(), &[4, 8]);
    }

    #[test]
    fn cpu_handle_to_array_is_independent_of_handle() {
        let mut arr = Array2::<f32>::zeros((1, 4));
        arr[[0, 0]] = 1.0;
        let h: Box<dyn StateHandle> = CpuStateHandle::boxed(arr.clone());
        let copy = h.to_array();
        // Mutating the clone does not affect the handle's clone (no
        // aliasing through `to_array`).
        let mut copy = copy;
        copy[[0, 0]] = 99.0;
        assert_eq!(h.to_array()[[0, 0]], 1.0);
    }

    #[test]
    fn row_location_variants_are_distinct() {
        assert_ne!(
            RowLocation::LocalCpu,
            RowLocation::LocalGpu { backend: "metal" }
        );
        assert_ne!(
            RowLocation::LocalGpu { backend: "metal" },
            RowLocation::LocalGpu { backend: "vulkan" }
        );
        assert_ne!(RowLocation::LocalCpu, RowLocation::Remote { node_id: 7 });
    }

    /// `SlabRole` round-trips. The enum is used as a static tag on
    /// trait impls; pinning equality + clone here catches a future
    /// `#[derive]` drift.
    #[test]
    fn slab_role_equality_and_clone() {
        assert_eq!(SlabRole::Canonical, SlabRole::Canonical);
        assert_ne!(SlabRole::Canonical, SlabRole::Derivative);
        let copied = SlabRole::Canonical;
        assert_eq!(copied, SlabRole::Canonical);
    }

    // ── Stub trait impls to cover the default-method bodies ──────────
    //
    // The trait surface lives ahead of any concrete impl (Phase A spec
    // draft). To exercise the defaults `is_empty` on `SlabHandle` /
    // `SlabSnapshot` we provide minimal stubs whose required methods
    // are stubbed; the defaults compose on top.

    struct StubSlab {
        rows: usize,
    }

    impl SlabHandle for StubSlab {
        fn row_dim(&self) -> usize {
            4
        }
        fn len(&self) -> usize {
            self.rows
        }
        fn role(&self) -> SlabRole {
            SlabRole::Canonical
        }
        fn location(&self) -> RowLocation {
            RowLocation::LocalCpu
        }
        fn append(&mut self, _chunk: Box<dyn StateHandle>) {
            self.rows += 1;
        }
        fn materialise_range(&self, _start: usize, _end: usize) -> Array2<f32> {
            Array2::<f32>::zeros((0, self.row_dim()))
        }
        fn evict_oldest(&mut self, _n: usize) -> Box<dyn SlabSnapshot> {
            Box::new(StubSnapshot { rows: 0 })
        }
    }

    struct StubSnapshot {
        rows: usize,
    }

    impl SlabSnapshot for StubSnapshot {
        fn len(&self) -> usize {
            self.rows
        }
        fn row_dim(&self) -> usize {
            4
        }
        fn role(&self) -> SlabRole {
            SlabRole::Canonical
        }
        fn materialise(&self) -> Array2<f32> {
            Array2::<f32>::zeros((self.rows, self.row_dim()))
        }
    }

    #[test]
    fn slab_handle_is_empty_default_uses_len() {
        let slab = StubSlab { rows: 0 };
        assert!(slab.is_empty(), "len=0 → is_empty");
        let slab = StubSlab { rows: 3 };
        assert!(!slab.is_empty(), "len=3 → !is_empty");
    }

    #[test]
    fn slab_snapshot_is_empty_default_uses_len() {
        let snap = StubSnapshot { rows: 0 };
        assert!(snap.is_empty());
        let snap = StubSnapshot { rows: 5 };
        assert!(!snap.is_empty());
        assert_eq!(snap.len(), 5);
        assert_eq!(snap.row_dim(), 4);
        assert_eq!(snap.role(), SlabRole::Canonical);
        assert_eq!(snap.materialise().shape(), &[5, 4]);
    }

    #[test]
    fn slab_handle_drives_required_surface() {
        let mut slab = StubSlab { rows: 0 };
        assert_eq!(slab.row_dim(), 4);
        assert_eq!(slab.role(), SlabRole::Canonical);
        assert_eq!(slab.location(), RowLocation::LocalCpu);
        let handle = CpuStateHandle::boxed(Array2::<f32>::zeros((1, 4)));
        slab.append(handle);
        assert_eq!(slab.len(), 1);
        let mat = slab.materialise_range(0, 0);
        assert_eq!(mat.shape(), &[0, 4]);
        let evicted = slab.evict_oldest(0);
        assert_eq!(evicted.len(), 0);
    }
}
