//! `EmbedMatrix` — `ModelWeights.embed` storage that's either a
//! heap-backed `ArcArray2<f32>` (the legacy form, used by every
//! caller pre-Arc-2) or an mmap-backed view (Arc 2; skips the 2 GB
//! `to_vec` copy when `embeddings.bin` is f32 on disk).
//!
//! ## Why a newtype, not `Option<MmappedEmbed>` alongside the
//! existing field
//!
//! A second optional field would let the heap copy and the mmap view
//! drift out of sync — every consumer would have to remember to
//! check `embed_mmap` first and fall through to `embed`, with the
//! invariant "exactly one of these is populated" unverifiable at the
//! type level. The newtype enforces the single-source-of-truth at the
//! type system.
//!
//! ## Why not change `WeightArray` itself
//!
//! `WeightArray = ArcArray2<f32>` is used by every dense tensor in
//! `ModelWeights` (norms, lm_head, etc.). The mmap optimisation only
//! matters for `embed` (the 2 GB tensor). Generalising would force
//! every consumer to handle "what if this is mmap-backed?" — most
//! tensors are <100 MB and the RAM win wouldn't justify the
//! complexity.
//!
//! ## Hot-path API
//!
//! - [`EmbedMatrix::row`] returns an `ArrayView1<'_, f32>` for any
//!   variant. Heap dispatches to the inner `ArcArray2::row`; mmap
//!   constructs a view over `bytemuck::cast_slice` of the relevant
//!   bytes (zero-copy).
//! - [`EmbedMatrix::shape`] / [`EmbedMatrix::nrows`] /
//!   [`EmbedMatrix::ncols`] mirror `ArcArray2`'s shape queries.
//! - [`EmbedMatrix::as_array`] materialises an `ArcArray2<f32>` on
//!   demand (`Cow::Borrowed` for the Heap variant, `Cow::Owned` +
//!   one 2 GB allocation for the Mmap variant). Reserved for cold
//!   paths that need `.dot()` / `.view()` / scalar multiplication —
//!   walker / capture / introspection. The hot inference forward
//!   uses `.row()` and never triggers materialisation.

use std::borrow::Cow;
use std::sync::Arc;

use memmap2::Mmap;
use ndarray::{ArcArray2, ArrayView1};

/// `embed` storage on [`ModelWeights`](crate::ModelWeights). See the
/// module docs for the design rationale.
#[derive(Clone)]
pub enum EmbedMatrix {
    /// Owned f32 `[vocab, hidden]` matrix on the heap. The pre-Arc-2
    /// form — used by every caller that loaded weights via
    /// `load_gguf` / `load_safetensors`, and by the vindex loader
    /// when `embeddings.bin` is f16 (decode is mandatory).
    Heap(ArcArray2<f32>),

    /// Mmap-backed view over an f32 `embeddings.bin`. The Arc 2
    /// optimisation — saves the ~2 GB `to_vec` copy that
    /// `decode_floats(&mmap, F32)` would otherwise pay.
    Mmap(MmappedEmbed),
}

impl Default for EmbedMatrix {
    /// Empty `0×0` heap matrix — what the legacy
    /// `ArcArray2::from_shape_vec((0, 0), Vec::new())` produces.
    /// Used as the placeholder when `embed_quant` is the active form.
    fn default() -> Self {
        Self::empty()
    }
}

impl EmbedMatrix {
    /// `0×0` heap matrix — placeholder used when `embed_quant` is
    /// the active form (lazy-quant lm_head path; PR #144) and the
    /// dense `embed` field is unused.
    pub fn empty() -> Self {
        Self::Heap(
            ArcArray2::from_shape_vec((0, 0), Vec::new()).expect("0×0 shape is always valid"),
        )
    }

    /// Wrap an existing `ArcArray2`. Use from loaders that already
    /// materialise the matrix into a heap Vec (safetensors,
    /// `load_gguf`, f16 `embeddings.bin`).
    pub fn from_array(arr: ArcArray2<f32>) -> Self {
        Self::Heap(arr)
    }

    /// Wrap a memory-mapped `embeddings.bin` (f32, vocab×hidden
    /// floats). Caller MUST verify the mmap length equals
    /// `vocab * hidden * 4` and that the f32 byte layout is native
    /// little-endian (which it is on every platform larql currently
    /// builds for; document if that ever changes).
    pub fn from_mmap(mmap: Arc<Mmap>, vocab: usize, hidden: usize) -> Self {
        debug_assert_eq!(
            mmap.len(),
            vocab * hidden * 4,
            "MmappedEmbed: mmap length {} != vocab {vocab} × hidden {hidden} × 4",
            mmap.len()
        );
        Self::Mmap(MmappedEmbed {
            mmap,
            vocab,
            hidden,
        })
    }

    /// Per-row view. Hot-path accessor — Heap dispatches to the
    /// inner `ArcArray2::row`; Mmap constructs an `ArrayView1` over
    /// `bytemuck::cast_slice` of the row's bytes (zero-copy).
    ///
    /// Returns an `ArrayView1` with the same semantics as
    /// `ArcArray2::row` so existing call sites work unchanged.
    pub fn row(&self, idx: usize) -> ArrayView1<'_, f32> {
        match self {
            Self::Heap(a) => a.row(idx),
            Self::Mmap(m) => {
                let start = idx * m.hidden * std::mem::size_of::<f32>();
                let end = start + m.hidden * std::mem::size_of::<f32>();
                let bytes = &m.mmap[start..end];
                let floats: &[f32] = bytemuck::cast_slice(bytes);
                ArrayView1::from(floats)
            }
        }
    }

    /// Shape `&[vocab, hidden]`. Matches `ArcArray2::shape` so
    /// existing `.shape()[0]` / `.shape()[1]` indexing works
    /// unchanged.
    pub fn shape(&self) -> [usize; 2] {
        match self {
            Self::Heap(a) => [a.nrows(), a.ncols()],
            Self::Mmap(m) => [m.vocab, m.hidden],
        }
    }

    /// Number of vocab rows.
    pub fn nrows(&self) -> usize {
        match self {
            Self::Heap(a) => a.nrows(),
            Self::Mmap(m) => m.vocab,
        }
    }

    /// Hidden dimension.
    pub fn ncols(&self) -> usize {
        match self {
            Self::Heap(a) => a.ncols(),
            Self::Mmap(m) => m.hidden,
        }
    }

    /// Materialise as an `ArcArray2<f32>`. Heap is a cheap refcount
    /// bump (`Cow::Borrowed`); Mmap pays a `[vocab × hidden × 4]`
    /// allocation + memcpy (`Cow::Owned`).
    ///
    /// Reserved for cold paths that genuinely need an owned matrix
    /// (`.dot()`, `.view()`, scalar multiplication for embed-scale).
    /// The hot inference forward uses [`row`] and never triggers
    /// the Mmap materialisation.
    pub fn as_array(&self) -> Cow<'_, ArcArray2<f32>> {
        match self {
            Self::Heap(a) => Cow::Borrowed(a),
            Self::Mmap(m) => {
                let bytes: &[u8] = &m.mmap;
                let floats: Vec<f32> = bytemuck::cast_slice::<u8, f32>(bytes).to_vec();
                let arr = ArcArray2::from_shape_vec((m.vocab, m.hidden), floats)
                    .expect("MmappedEmbed shape invariant guarantees vocab×hidden");
                Cow::Owned(arr)
            }
        }
    }

    /// Whether this is mmap-backed (i.e. the Arc 2 RAM optimisation
    /// is active). Cold-path callers may want to log or branch on
    /// this; the hot path doesn't care.
    pub fn is_mmap(&self) -> bool {
        matches!(self, Self::Mmap(_))
    }
}

/// Mmap-backed embed payload — held inside the [`EmbedMatrix::Mmap`]
/// variant. The `Arc<Mmap>` keeps the underlying `embeddings.bin`
/// pages mapped for the lifetime of any [`EmbedMatrix`] clone.
#[derive(Clone)]
pub struct MmappedEmbed {
    mmap: Arc<Mmap>,
    vocab: usize,
    hidden: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn heap_with_floats(floats: &[f32], vocab: usize, hidden: usize) -> EmbedMatrix {
        assert_eq!(floats.len(), vocab * hidden);
        EmbedMatrix::from_array(
            ArcArray2::from_shape_vec((vocab, hidden), floats.to_vec()).expect("valid shape"),
        )
    }

    fn mmap_with_floats(floats: &[f32], vocab: usize, hidden: usize) -> EmbedMatrix {
        use std::io::Write;
        assert_eq!(floats.len(), vocab * hidden);
        let mut tmp = tempfile::NamedTempFile::new().expect("tempfile");
        for f in floats {
            tmp.write_all(&f.to_le_bytes()).expect("write");
        }
        tmp.flush().expect("flush");
        let file = tmp.reopen().expect("reopen");
        let mmap = Arc::new(unsafe { Mmap::map(&file).expect("mmap") });
        // Keep the NamedTempFile alive by leaking it for the test —
        // the mmap+arc keeps the file path live anyway.
        std::mem::forget(tmp);
        EmbedMatrix::from_mmap(mmap, vocab, hidden)
    }

    #[test]
    fn heap_and_mmap_row_view_produce_same_bytes() {
        let (vocab, hidden) = (8usize, 4usize);
        let floats: Vec<f32> = (0..vocab * hidden).map(|i| i as f32 * 0.013).collect();
        let h = heap_with_floats(&floats, vocab, hidden);
        let m = mmap_with_floats(&floats, vocab, hidden);

        assert_eq!(h.shape(), m.shape());
        assert_eq!(h.nrows(), m.nrows());
        assert_eq!(h.ncols(), m.ncols());
        for r in 0..vocab {
            let h_row: Vec<f32> = h.row(r).iter().copied().collect();
            let m_row: Vec<f32> = m.row(r).iter().copied().collect();
            assert_eq!(h_row, m_row, "row {r}");
        }
    }

    #[test]
    fn as_array_borrows_for_heap_owns_for_mmap() {
        let (vocab, hidden) = (4usize, 3usize);
        let floats: Vec<f32> = (0..vocab * hidden).map(|i| i as f32).collect();
        let h = heap_with_floats(&floats, vocab, hidden);
        let m = mmap_with_floats(&floats, vocab, hidden);

        match h.as_array() {
            Cow::Borrowed(_) => {}
            Cow::Owned(_) => panic!("Heap variant must yield Cow::Borrowed"),
        }
        match m.as_array() {
            Cow::Owned(_) => {}
            Cow::Borrowed(_) => panic!("Mmap variant must yield Cow::Owned"),
        }
    }

    #[test]
    fn default_is_zero_by_zero_heap_placeholder() {
        let e = EmbedMatrix::default();
        assert_eq!(e.shape(), [0, 0]);
        assert!(!e.is_mmap());
    }
}
