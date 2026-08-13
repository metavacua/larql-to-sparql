//! Latent transforms around a bank (spec §5, `latent-moe-v1`).
//!
//! A direct routed MoE reads the residual and writes the residual. A latent
//! one does not: it projects into a narrower routed space, runs the bank
//! there, and projects back.
//!
//! ```text
//! direct    residual → bank → residual
//! latent    residual → routed_input → bank → routed_output → residual
//! ```
//!
//! Modelling that as an ordered stage list rather than two optional tensors
//! keeps the direct case as the empty list — no `Option` to forget, no branch
//! that silently skips a projection when one was bound.

use super::axis::Axis;
use super::error::ExecutionError;
use super::kernels::dot;
use super::tensor::BoundTensor;

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

/// Where a transform sits relative to the bank.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TransformStage {
    /// Applied to the residual before routing and expert execution.
    RoutedInput,
    /// Applied to the bank's reduced output before it rejoins the residual.
    RoutedOutput,
}

impl TransformStage {
    pub const fn name(self) -> &'static str {
        match self {
            Self::RoutedInput => "routed_input",
            Self::RoutedOutput => "routed_output",
        }
    }
}

/// A bound projection applied at one stage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundTransform<'a> {
    pub stage: TransformStage,
    /// `[out_dim, in_dim]`.
    pub weight: BoundTensor<'a>,
}

impl BoundTransform<'_> {
    pub fn in_dim(&self) -> usize {
        self.weight.cols()
    }

    pub fn out_dim(&self) -> usize {
        self.weight.rows()
    }

    /// Apply the projection to `input`.
    pub fn apply(&self, input: &[f32]) -> Result<Vec<f32>, ExecutionError> {
        if input.len() != self.in_dim() {
            return Err(ExecutionError::DimensionMismatch {
                operand: self.weight.describe(),
                axis: Axis::InputWidth,
                expected: self.in_dim(),
                found: input.len(),
            });
        }
        let mut out = vec![0.0f32; self.out_dim()];
        let mut row = vec![0.0f32; self.in_dim()];
        for (r, slot) in out.iter_mut().enumerate() {
            self.weight.row_into(r, &mut row)?;
            *slot = dot(&row, input);
        }
        Ok(out)
    }

    pub fn describe(&self) -> String {
        format!(
            "{} {} [{}→{}]",
            self.stage.name(),
            self.weight.describe(),
            self.in_dim(),
            self.out_dim()
        )
    }
}

/// Apply every transform bound to a stage, in order.
pub fn apply_stage(
    transforms: &[BoundTransform<'_>],
    stage: TransformStage,
    input: &[f32],
) -> Result<Vec<f32>, ExecutionError> {
    let mut current = input.to_vec();
    for transform in transforms.iter().filter(|t| t.stage == stage) {
        current = transform.apply(&current)?;
    }
    Ok(current)
}
