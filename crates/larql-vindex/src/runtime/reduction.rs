//! How selected experts' outputs combine into one vector.
//!
//! Currently one kind, named rather than assumed. `WeightedSum` is what every
//! programme in the registry uses today, but writing it as `sum(w_i * y_i)`
//! inline would make the next reduction — a shared-expert bank that adds
//! unweighted, or a normalised combine — a change to the executor instead of a
//! change to the bound recipe.

use super::axis::Axis;
use super::error::ExecutionError;

/// The combining rule for a bank's selected expert outputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundReduction {
    /// `sum_i weight_i * output_i` — the routed-MoE combine.
    WeightedSum,
}

impl BoundReduction {
    pub const fn name(self) -> &'static str {
        match self {
            Self::WeightedSum => "weighted_sum",
        }
    }

    /// Accumulate one expert's contribution into `accumulator`.
    pub fn accumulate(
        self,
        accumulator: &mut [f32],
        output: &[f32],
        weight: f32,
    ) -> Result<(), ExecutionError> {
        if accumulator.len() != output.len() {
            return Err(ExecutionError::DimensionMismatch {
                operand: format!("{} reduction", self.name()),
                axis: Axis::OutputWidth,
                expected: accumulator.len(),
                found: output.len(),
            });
        }
        match self {
            Self::WeightedSum => {
                for (acc, &value) in accumulator.iter_mut().zip(output) {
                    *acc += weight * value;
                }
            }
        }
        Ok(())
    }
}
