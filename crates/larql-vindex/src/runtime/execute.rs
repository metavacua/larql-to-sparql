//! Executing a bound MoE operation.
//!
//! ```text
//! residual → routed_input → router → selection → experts → reduction
//!          → routed_output → residual delta
//! ```
//!
//! One implementation, generic over its trace sink. The fixture path and the
//! token path differ only in which sink is instantiated, so a fixture that
//! passes is evidence about the code Gemma runs rather than about a parallel
//! instrumented copy of it.

use super::axis::Axis;
use super::consts::{OPERAND_OPERATION_OUTPUT, OPERAND_RESIDUAL};
use super::error::ExecutionError;
use super::expert_kernel::BankKernel;
use super::inputs::MoeInputs;
use super::kernels::{dot, renormalize, softmax, top_k_with_margin};
use super::operation::BoundMoeOperation;
use super::router::{BoundRouter, RouterKernel, SelectedExpert};
use super::trace::{CollectedTrace, NoTrace, TraceSink};
use super::transform::{apply_stage, TransformStage};
use larql_compute::MoeTopKWeightPolicy;

/// Run the operation over one token's residual, returning its contribution.
///
/// The returned vector is the **delta**, not the updated residual: adding it
/// back is the caller's business, because whether a norm sits between the two
/// is a property of the surrounding block rather than of the MoE operation.
pub fn execute(
    operation: &BoundMoeOperation<'_>,
    inputs: MoeInputs<'_>,
) -> Result<Vec<f32>, ExecutionError> {
    execute_with(operation, inputs, &mut NoTrace)
}

/// Run the operation and record every internal checkpoint.
pub fn execute_traced(
    operation: &BoundMoeOperation<'_>,
    inputs: MoeInputs<'_>,
) -> Result<(Vec<f32>, CollectedTrace), ExecutionError> {
    let mut trace = CollectedTrace::default();
    let out = execute_with(operation, inputs, &mut trace)?;
    Ok((out, trace))
}

/// The single implementation. `sink` compiles away entirely for [`NoTrace`].
///
/// Routed-input transforms apply to `inputs.bank`. `inputs.router` is used as
/// supplied, because the router's input is produced by the surrounding block's
/// norms and scales, not by a projection this operation owns.
pub fn execute_with<S: TraceSink>(
    operation: &BoundMoeOperation<'_>,
    inputs: MoeInputs<'_>,
    sink: &mut S,
) -> Result<Vec<f32>, ExecutionError> {
    if inputs.bank.len() != operation.residual_dim {
        return Err(ExecutionError::DimensionMismatch {
            operand: OPERAND_RESIDUAL.into(),
            axis: Axis::Width,
            expected: operation.residual_dim,
            found: inputs.bank.len(),
        });
    }

    let bank_input = apply_stage(
        &operation.transforms,
        TransformStage::RoutedInput,
        inputs.bank,
    )?;
    sink.routed_input(&bank_input);

    let mut reduced = vec![0.0f32; operation.bank_input_dim()];
    for bank in &operation.banks {
        let selected = select(&operation.router, inputs.router, bank.population(), sink)?;
        // Opened before the expert loop, not inside it. The incumbent kernel
        // quantises the bank input to Q8_K once and shares it across the
        // bank's selected experts; quantising per expert would be binding a
        // differently-shaped call than production makes.
        let mut kernel = BankKernel::open(bank, &bank_input)?;

        // Experts run in parallel; the reduction stays serial and in
        // selection order. Splitting it this way is what makes the speed-up
        // free of numerical consequence: each expert writes only its own
        // disjoint slot, and the float additions below happen in exactly the
        // order the serial loop used, so every bit-exactness claim in the
        // ladder still holds. Summing inside the pool would reorder them.
        let experts = selected
            .iter()
            .map(|c| bank.expert(c.expert_id, operation.router.population()))
            .collect::<Result<Vec<_>, _>>()?;
        let hidden = bank.hidden_dim;
        let mut outs = vec![0.0f32; experts.len() * hidden];
        kernel.run_many_into(&experts, bank, &bank_input, hidden, &mut outs)?;

        for (choice, out) in selected.iter().zip(outs.chunks(hidden)) {
            sink.expert_output(choice.expert_id, out);
            operation
                .reduction
                .accumulate(&mut reduced, out, choice.weight)?;
        }
    }
    sink.reduced(&reduced);

    let delta = apply_stage(
        &operation.transforms,
        TransformStage::RoutedOutput,
        &reduced,
    )?;
    if delta.len() != operation.residual_dim {
        return Err(ExecutionError::DimensionMismatch {
            operand: OPERAND_OPERATION_OUTPUT.into(),
            axis: Axis::Width,
            expected: operation.residual_dim,
            found: delta.len(),
        });
    }
    sink.residual_delta(&delta);
    Ok(delta)
}

/// Score the population with whichever kernel was bound.
///
/// The reference reads row by row through the operand's view, so it serves any
/// layout. The incumbent kernel takes the bytes directly and refuses rather
/// than accepting a reconstruction.
fn score(router: &BoundRouter<'_>, input: &[f32]) -> Result<Vec<f32>, ExecutionError> {
    match router.kernel {
        RouterKernel::Reference => {
            let mut scores = vec![0.0f32; router.population()];
            let mut row = vec![0.0f32; router.hidden_dim()];
            for (e, slot) in scores.iter_mut().enumerate() {
                router.weight.row_into(e, &mut row)?;
                *slot = dot(&row, input);
            }
            softmax(&mut scores);
            Ok(scores)
        }
        RouterKernel::Incumbent => {
            let weights = router.weight.as_f32_slice().map_err(|reason| {
                ExecutionError::KernelOperandUnsuitable {
                    kernel: RouterKernel::Incumbent.name(),
                    operand: router.weight.describe(),
                    reason,
                }
            })?;
            // The incumbent's own functions, not a lookalike.
            let mut scores = larql_compute::cpu::ops::moe::moe_score_experts(
                input,
                weights,
                router.population(),
                router.hidden_dim(),
            );
            larql_compute::cpu::ops::moe::moe_softmax(&mut scores);
            Ok(scores)
        }
    }
}

/// Score the population, select top-k, and apply the weight policies.
fn select<S: TraceSink>(
    router: &BoundRouter<'_>,
    input: &[f32],
    population: usize,
    sink: &mut S,
) -> Result<Vec<SelectedExpert>, ExecutionError> {
    let scores = score(router, input)?;
    sink.router_scores(&scores);

    // Before ordering, not after. A non-finite score still takes a position
    // under a total order, so leaving it in would let sort mechanics make a
    // routing decision.
    if let Some((expert, value)) = scores
        .iter()
        .position(|s| !s.is_finite())
        .map(|i| (i, scores[i]))
    {
        return Err(ExecutionError::NonFiniteRouterScore { expert, value });
    }

    let (chosen, margin) = top_k_with_margin(&scores, router.top_k);
    sink.selection_margin(margin);
    let mut weights: Vec<f32> = chosen.iter().map(|(_, w)| *w).collect();
    if router.selected_weight == MoeTopKWeightPolicy::RenormalizedSoftmax {
        renormalize(&mut weights);
    }

    // Per-expert scale multiplies *after* renormalisation, so the selected
    // weights need not sum to one afterwards. Applying it before would let
    // renormalisation divide the learned scale back out.
    let per_expert = match router.scaling.scales() {
        Some(scales) => Some(scales.to_vec()?),
        None => None,
    };

    let mut selected = Vec::with_capacity(chosen.len());
    for ((index, raw_score), weight) in chosen.into_iter().zip(weights) {
        let expert_id = u32::try_from(index).map_err(|_| ExecutionError::ExpertOutOfRange {
            expert: u32::MAX,
            population,
        })?;
        let weight = match &per_expert {
            Some(scale) => weight * scale.get(index).copied().unwrap_or(1.0),
            None => weight,
        };
        selected.push(SelectedExpert {
            expert_id,
            weight,
            raw_score,
        });
    }
    sink.selection(&selected);
    Ok(selected)
}
