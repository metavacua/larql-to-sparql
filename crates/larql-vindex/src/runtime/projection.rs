//! The gate/up projection of one expert, in whichever arrangement it was stored.
//!
//! This is where "the executor follows the bound recipe" is actually cashed
//! out. A gated MLP needs a gate row and an up row per intermediate unit;
//! whether those live in two regions or in two halves of one is a storage
//! decision, and it is resolved here, once, behind an interface that yields
//! the same two rows either way.
//!
//! Everything downstream — activation, the elementwise product, the down
//! projection, the reduction — is then written once and cannot acquire a
//! layout assumption, because it never sees the layout.
//!
//! ```text
//! Decomposed   gate[i]     up[i]            two regions, row i of each
//! Fused        gate_up[i]  gate_up[N + i]   one region, halves in sequence
//! ```
//!
//! The fused halves are **concatenated, not interleaved**. Reading them
//! interleaved yields a correctly-shaped tensor of wrong values — a failure
//! that survives all the way to the logits, which is why the arrangement is a
//! bound property rather than something inferred at execution time.

use crate::format::lyrw2::region_role::RegionRole;

use super::consts::FUSED_PROJECTION_HALVES;
use super::error::ExecutionError;
use super::tensor::BoundTensor;

/// Which physical arrangement a projection was stored in.
///
/// Defined once, here, and used by the executor, its diagnostics and the
/// fixtures alike. The role names come from the §6.5 registry rather than
/// being spelled again: `gate`/`up` versus `gate_up_fused` is the registry's
/// vocabulary, and a second copy of it drifts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ProjectionArrangement {
    /// Separate `gate` and `up` regions.
    Decomposed,
    /// One concatenated `gate_up_fused` region.
    Fused,
}

impl ProjectionArrangement {
    pub const ALL: [Self; 2] = [Self::Decomposed, Self::Fused];

    /// The roles this arrangement stores, in region order.
    pub fn roles(self) -> Vec<RegionRole> {
        match self {
            Self::Decomposed => vec![RegionRole::Gate, RegionRole::Up],
            Self::Fused => vec![RegionRole::GateUpFused],
        }
    }

    /// Named from the roles it stores, so the registry stays the one source.
    pub fn name(self) -> String {
        self.roles()
            .iter()
            .map(|r| r.name())
            .collect::<Vec<_>>()
            .join("+")
    }
}

/// How one expert's gate and up weights are arranged in storage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BoundProjection<'a> {
    /// Separate `gate` and `up` regions, each `[intermediate, hidden]`.
    Decomposed {
        gate: BoundTensor<'a>,
        up: BoundTensor<'a>,
    },
    /// One `[2 * intermediate, hidden]` region: all gate rows, then all up rows.
    Fused { gate_up: BoundTensor<'a> },
}

impl BoundProjection<'_> {
    /// Intermediate units this projection produces.
    pub fn intermediate_dim(&self) -> usize {
        match self {
            Self::Decomposed { gate, .. } => gate.rows(),
            Self::Fused { gate_up } => gate_up.rows() / FUSED_PROJECTION_HALVES,
        }
    }

    /// Input width this projection contracts over.
    pub fn hidden_dim(&self) -> usize {
        match self {
            Self::Decomposed { gate, .. } => gate.cols(),
            Self::Fused { gate_up } => gate_up.cols(),
        }
    }

    /// Check the arrangement is internally consistent and matches the layer.
    ///
    /// A fused region with an odd row count is the clearest possible signal
    /// that a decomposed region was bound as fused; catching it here turns a
    /// silent half-wrong expert into a named refusal.
    pub fn validate(&self, intermediate: usize, hidden: usize) -> Result<(), ExecutionError> {
        match self {
            Self::Decomposed { gate, up } => {
                gate.require_matrix(intermediate, hidden)?;
                up.require_matrix(intermediate, hidden)
            }
            Self::Fused { gate_up } => {
                gate_up.require_matrix(intermediate * FUSED_PROJECTION_HALVES, hidden)
            }
        }
    }

    /// Row `unit` of the gate projection.
    pub fn gate_row_into(&self, unit: usize, out: &mut [f32]) -> Result<(), ExecutionError> {
        match self {
            Self::Decomposed { gate, .. } => gate.row_into(unit, out),
            Self::Fused { gate_up } => gate_up.row_into(unit, out),
        }
    }

    /// Row `unit` of the up projection.
    pub fn up_row_into(&self, unit: usize, out: &mut [f32]) -> Result<(), ExecutionError> {
        match self {
            Self::Decomposed { up, .. } => up.row_into(unit, out),
            // The second half. Not `2 * unit + 1` — the halves are
            // concatenated, and the interleaved reading is the classic
            // silent-wrong-values bug this layout invites.
            Self::Fused { gate_up } => gate_up.row_into(self.intermediate_dim() + unit, out),
        }
    }

    /// Which arrangement this is, for diagnostics and trace output.
    pub const fn arrangement(&self) -> ProjectionArrangement {
        match self {
            Self::Decomposed { .. } => ProjectionArrangement::Decomposed,
            Self::Fused { .. } => ProjectionArrangement::Fused,
        }
    }

    pub fn describe(&self) -> String {
        let operands = match self {
            Self::Decomposed { gate, up } => format!("{}, {}", gate.describe(), up.describe()),
            Self::Fused { gate_up } => gate_up.describe(),
        };
        format!("{}({operands})", self.arrangement().name())
    }
}
