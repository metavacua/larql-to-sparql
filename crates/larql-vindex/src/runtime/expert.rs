//! One expert's gated MLP.
//!
//! ```text
//! y = down · ( act(gate · x) ⊙ (up · x) )
//! ```
//!
//! Written once, against the projection interface rather than against storage,
//! so a fused and a decomposed expert reach identical arithmetic. If this
//! function ever needs to know which arrangement it got, the abstraction has
//! failed and the two routes can diverge.

use larql_compute::Activation;

use super::bank::BoundExpert;
use super::error::ExecutionError;
use super::kernels::{activate, dot};

/// Run one expert over `input`, returning its unweighted output.
pub fn forward(
    expert: &BoundExpert<'_>,
    input: &[f32],
    intermediate: usize,
    activation: Activation,
) -> Result<Vec<f32>, ExecutionError> {
    let hidden = input.len();
    let mut gate_row = vec![0.0f32; hidden];
    let mut up_row = vec![0.0f32; hidden];
    let mut gated = vec![0.0f32; intermediate];

    for (unit, slot) in gated.iter_mut().enumerate() {
        expert.projection.gate_row_into(unit, &mut gate_row)?;
        expert.projection.up_row_into(unit, &mut up_row)?;
        *slot = dot(&gate_row, input);
        // Activation on the gate half only; the up half enters unactivated.
        // Applying it to the product instead is a plausible-looking variant
        // that produces wrong values with no shape error anywhere.
        *slot = activate_one(activation, *slot) * dot(&up_row, input);
    }

    let out_dim = expert.down.rows();
    let mut out = vec![0.0f32; out_dim];
    let mut down_row = vec![0.0f32; intermediate];
    for (r, slot) in out.iter_mut().enumerate() {
        expert.down.row_into(r, &mut down_row)?;
        *slot = dot(&down_row, &gated);
    }
    Ok(out)
}

/// Single-element activation, sharing the elementwise implementation.
fn activate_one(activation: Activation, x: f32) -> f32 {
    let mut one = [x];
    activate(activation, &mut one);
    one[0]
}
