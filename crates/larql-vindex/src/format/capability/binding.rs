//! Semantic bindings and physical identity (spec §9.1, §11).
//!
//! Two identities, deliberately separate, because they answer different
//! questions and deduplicate on different keys:
//!
//! ```text
//! RepresentationIdentity  which catalogue declaration was selected
//! PhysicalStorageId       which bytes on disk that resolves to
//! ```
//!
//! Folding the variant into physical identity would defeat the deduplication
//! it exists for: two catalogue declarations may name the same extent, and
//! residency, byte accounting and mmap planning care about the extent, not
//! about how many names point at it.
//!
//! ```text
//! same bytes, two semantic uses         → one residency allocation
//! same bytes, two permitted views       → one physical component, two bindings
//! same logical tensor, duplicated bytes → two physical components
//! ```
//!
//! Two declarations naming one extent while disagreeing about format, shape,
//! packing or fidelity is a **catalogue defect**, not two components — the
//! bytes cannot be both.

use super::authority::Fidelity;
use super::component::{ComponentContract, TensorKind};

/// A byte extent. The deduplication key for residency and accounting.
///
/// Deliberately excludes the variant: identity is about *which bytes*, and two
/// catalogue names for one extent are one allocation.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PhysicalStorageId {
    pub file_digest: String,
    pub offset: u64,
    pub length: u64,
}

impl PhysicalStorageId {
    pub fn new(file_digest: impl Into<String>, offset: u64, length: u64) -> Self {
        Self {
            file_digest: file_digest.into(),
            offset,
            length,
        }
    }

    pub fn end(&self) -> u64 {
        self.offset + self.length
    }

    /// Whether two extents are the same bytes. Only exact identity is safe to
    /// deduplicate — partial overlap stays distinct unless a storage format
    /// explicitly defines the relationship.
    pub fn is_same_extent(&self, other: &Self) -> bool {
        self == other
    }

    /// Whether two extents overlap without being identical. Such a pair must
    /// **not** be deduplicated; it is either a legitimate sub-range or a
    /// catalogue error, and neither is "the same component".
    pub fn overlaps_partially(&self, other: &Self) -> bool {
        self.file_digest == other.file_digest
            && self != other
            && self.offset < other.end()
            && other.offset < self.end()
    }

    pub fn describe(&self) -> String {
        format!(
            "{}..{} of {}",
            self.offset,
            self.end(),
            &self.file_digest[..self.file_digest.len().min(12)]
        )
    }
}

/// Which catalogue declaration was selected.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RepresentationIdentity {
    pub region_set: String,
    pub variant: String,
}

impl RepresentationIdentity {
    pub fn new(region_set: impl Into<String>, variant: impl Into<String>) -> Self {
        Self {
            region_set: region_set.into(),
            variant: variant.into(),
        }
    }

    pub fn describe(&self) -> String {
        format!("{}::{}", self.region_set, self.variant)
    }
}

/// How stored bytes are accessed to satisfy a semantic role.
///
/// A bounded vocabulary, not arbitrary tensor transformation. It exists
/// because embedding-as-LM-head is not the same *access* as a dedicated head:
/// the embedding is indexed by token, the projection contracts over hidden, so
/// the same bytes serve both only through a transpose. Without recording the
/// view, a plan knows which bytes it reads and not how they perform the role.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComponentView {
    Direct,
    Transpose,
    Slice { dim: usize, start: u32, len: u32 },
}

/// Why a view cannot be applied to the stored shape.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ViewError {
    #[error("cannot transpose a {kind}: transpose is defined for matrices only")]
    TransposeOfNonMatrix { kind: &'static str },
    #[error("slice dim {dim} is out of range for a shape with {rank} dimensions")]
    SliceDimOutOfRange { dim: usize, rank: usize },
    #[error("slice {start}..{end} is out of range for dimension {dim} of length {extent}")]
    SliceOutOfRange {
        dim: usize,
        start: u32,
        end: u32,
        extent: u32,
    },
}

impl ComponentView {
    /// The contract the *semantic role* sees, after this view is applied.
    ///
    /// Contract checking must happen on this side of the view. A dedicated
    /// head is `[hidden, vocab]`; an embedding is `[vocab, hidden]`; comparing
    /// the requirement against raw storage would reject a perfectly valid
    /// tied-weight model.
    pub fn apply_to(&self, storage: &ComponentContract) -> Result<ComponentContract, ViewError> {
        match self {
            Self::Direct => Ok(storage.clone()),
            Self::Transpose => {
                if storage.kind != TensorKind::Matrix || storage.shape.len() != 2 {
                    return Err(ViewError::TransposeOfNonMatrix {
                        kind: storage.kind.name(),
                    });
                }
                Ok(ComponentContract {
                    shape: vec![storage.shape[1], storage.shape[0]],
                    kind: TensorKind::Matrix,
                })
            }
            Self::Slice { dim, start, len } => {
                let rank = storage.shape.len();
                let extent = *storage
                    .shape
                    .get(*dim)
                    .ok_or(ViewError::SliceDimOutOfRange { dim: *dim, rank })?;
                let end = start + len;
                if end > extent {
                    return Err(ViewError::SliceOutOfRange {
                        dim: *dim,
                        start: *start,
                        end,
                        extent,
                    });
                }
                let mut shape = storage.shape.clone();
                shape[*dim] = *len;
                Ok(ComponentContract {
                    shape,
                    kind: storage.kind,
                })
            }
        }
    }

    /// Whether this view changes any stored value.
    ///
    /// None of them do. A transpose reorders access, it does not alter
    /// numbers, so **views never affect authority**. They affect reference
    /// support, kernel eligibility and access efficiency — a kernel that
    /// serves a dedicated head may not accept transposed embedding access, and
    /// the loader must expose that route at Reference maturity rather than
    /// silently materialising a repacked copy at decode time.
    pub const fn alters_values(&self) -> bool {
        false
    }

    pub fn describe(&self) -> String {
        match self {
            Self::Direct => "direct".into(),
            Self::Transpose => "transposed".into(),
            Self::Slice { dim, start, len } => {
                format!("slice dim {dim} [{start}..{}]", start + len)
            }
        }
    }
}

/// One semantic use of one physical extent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComponentBinding {
    /// Which declaration was selected.
    pub representation: RepresentationIdentity,
    /// Which bytes that resolves to — the deduplication key.
    pub storage: PhysicalStorageId,
    /// How those bytes perform this role.
    pub view: ComponentView,
    /// Fidelity against the source checkpoint. Unaffected by the view.
    pub fidelity: Fidelity,
}

impl ComponentBinding {
    pub fn describe(&self) -> String {
        format!(
            "{} ({}, {})",
            self.representation.describe(),
            self.storage.describe(),
            self.view.describe()
        )
    }
}

/// Deduplicate bindings to the physical extents they read.
///
/// Two semantic uses of one tied tensor produce two bindings and one extent.
/// Byte accounting, residency estimates, mmap planning, checksum work and
/// remote-transfer planning all consume this side; execution and kernel
/// binding consume the bindings.
pub fn physical_extents(bindings: &[ComponentBinding]) -> Vec<PhysicalStorageId> {
    let mut out: Vec<PhysicalStorageId> = bindings.iter().map(|b| b.storage.clone()).collect();
    out.sort();
    out.dedup();
    out
}

/// Total bytes an execution reads, counting each extent once.
pub fn total_bytes(bindings: &[ComponentBinding]) -> u64 {
    physical_extents(bindings).iter().map(|s| s.length).sum()
}

/// Two declarations naming one extent while disagreeing about it.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error(
    "representations '{a}' and '{b}' name the same extent ({extent}) but disagree on {field} — \
     the bytes cannot be both"
)]
pub struct CatalogueConflict {
    pub a: String,
    pub b: String,
    pub extent: String,
    pub field: &'static str,
}

/// Detect declarations that share an extent but describe it differently.
///
/// Not two components — a catalogue defect. Resolution must surface it rather
/// than pick one, because either declaration could be the wrong one.
pub fn catalogue_conflicts(bindings: &[ComponentBinding]) -> Vec<CatalogueConflict> {
    let mut out = Vec::new();
    for (i, a) in bindings.iter().enumerate() {
        for b in bindings.iter().skip(i + 1) {
            if !a.storage.is_same_extent(&b.storage) {
                continue;
            }
            if a.fidelity != b.fidelity {
                out.push(CatalogueConflict {
                    a: a.representation.describe(),
                    b: b.representation.describe(),
                    extent: a.storage.describe(),
                    field: "fidelity",
                });
            }
        }
    }
    out
}
