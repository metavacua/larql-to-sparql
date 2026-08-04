//! Tests for `expert_kernel` — binding the production expert kernel.
//!
//! The headline is [`the_bound_kernel_reproduces_the_incumbent_call_bit_for_bit`].
//! Everything else exists so that a bit-identical result cannot be a
//! bit-identical *mistake*: two calls of one function agree whatever operands
//! they are handed, so the refusals below are what establish that the operands
//! reaching it are the right ones.

use larql_compute::cpu::ops::moe::{
    quantize_x_to_q8k, run_single_expert_q4k_q8k_into, ExpertScratch,
};
use larql_compute::cpu::ops::q4_common::dequantize_q4_k;
use larql_compute::{Activation, MoeTopKWeightPolicy};

use crate::format::capability::binding::{ComponentView, RepresentationIdentity};
use crate::format::capability::component::ComponentContract;
use crate::format::capability::coordinate::BankCoordinate;
use crate::format::lyrw2::region_format::RegionFormat;

use crate::runtime::axis::Axis;
use crate::runtime::bank::{BoundBankOperation, BoundExpert};
use crate::runtime::consts::{COL_DIM, FUSED_PROJECTION_HALVES};
use crate::runtime::error::{ExecutionError, OperandUnsuitability};
use crate::runtime::execute::execute_traced;
use crate::runtime::expert_kernel::ExpertKernel;
use crate::runtime::inputs::MoeInputs;
use crate::runtime::operation::BoundMoeOperation;
use crate::runtime::projection::BoundProjection;
use crate::runtime::reduction::BoundReduction;
use crate::runtime::router::{BoundExpertScaling, BoundRouter, RouterKernel};
use crate::runtime::tensor::BoundTensor;

use super::support::{q4k_bytes, Q4K_BLOCK_ELEMS};

/// One super-block wide, so the Q8_K activation quantiser is satisfied and the
/// fixture stays small enough to reason about.
const HIDDEN: usize = Q4K_BLOCK_ELEMS;
/// Half a super-block, so `down` is stored padded — the shape every real
/// k-quant MoE layer has, and the one a kernel binding gets wrong.
const INTERMEDIATE: usize = Q4K_BLOCK_ELEMS / 2;
const INTERMEDIATE_PADDED: usize = Q4K_BLOCK_ELEMS;
const POPULATION: usize = 2;
const TOP_K: usize = 1;
const LAYER: u32 = 0;
const BANK_ID: u16 = 0;
/// Gemma's activation, and one of the two the incumbent kernel implements.
const ACTIVATION: Activation = Activation::GeluTanh;

const VARIANT: &str = "test";
const GATE_UP_REGION_SET: &str = "gate_up_fused";
const DOWN_REGION_SET: &str = "down";
const ROUTER_REGION_SET: &str = "router";

const GATE_UP_SEED: usize = 1;
const DOWN_SEED: usize = 7;

/// The reference dequantises to f32 while the incumbent keeps an integer dot
/// against a Q8_K activation, so the two agree to quantisation noise, not to
/// the bit. A band, not a target.
const KERNEL_TOLERANCE: f32 = 5e-2;

fn identity(region_set: &str) -> RepresentationIdentity {
    RepresentationIdentity::new(region_set, VARIANT)
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn direct<'a>(
    region_set: &str,
    bytes: &'a [u8],
    format: RegionFormat,
    rows: u32,
    cols: u32,
) -> BoundTensor<'a> {
    BoundTensor::direct(
        identity(region_set),
        bytes,
        format,
        ComponentContract::matrix(rows, cols),
    )
    .expect("well-formed fixture operand")
}

fn column_prefix<'a>(
    region_set: &str,
    bytes: &'a [u8],
    format: RegionFormat,
    rows: u32,
    stored_cols: u32,
    role_cols: u32,
) -> BoundTensor<'a> {
    BoundTensor::new(
        identity(region_set),
        bytes,
        format,
        ComponentContract::matrix(rows, stored_cols),
        ComponentView::Slice {
            dim: COL_DIM,
            start: 0,
            len: role_cols,
        },
    )
    .expect("well-formed fixture operand")
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Q4_K bytes, plus the same values dequantised, so both kernels can be bound
/// over one set of weights.
struct Fixture {
    gate_up: Vec<u8>,
    gate_up_f32: Vec<u8>,
    down: Vec<u8>,
    down_f32: Vec<u8>,
    router: Vec<u8>,
    input: Vec<f32>,
}

impl Fixture {
    fn new() -> Self {
        let gate_up = q4k_bytes(FUSED_PROJECTION_HALVES * INTERMEDIATE, HIDDEN, GATE_UP_SEED);
        let down = q4k_bytes(HIDDEN, INTERMEDIATE_PADDED, DOWN_SEED);
        let gate_up_f32 = f32_bytes(&dequantize_q4_k(
            &gate_up,
            FUSED_PROJECTION_HALVES * INTERMEDIATE * HIDDEN,
        ));
        let down_f32 = f32_bytes(&dequantize_q4_k(&down, HIDDEN * INTERMEDIATE_PADDED));

        // A signed input, so the kernel exercises both halves of the 4-bit
        // range rather than a one-sided ramp.
        let input: Vec<f32> = (0..HIDDEN).map(|i| (i % 11) as f32 * 0.1 - 0.5).collect();

        // Expert 0's router row *is* the input, so its logit is ‖x‖² while
        // expert 1's is zero. An earlier fixture filled row 0 with ones, which
        // scored `sum(x)` — and this input sums to approximately zero, so the
        // two logits tied and selection fell to whichever side the float
        // residue landed. A fixture that is accidentally testing a tie is not
        // testing the kernel.
        let mut router = vec![0.0f32; POPULATION * HIDDEN];
        router[..HIDDEN].copy_from_slice(&input);
        Self {
            gate_up,
            gate_up_f32,
            down,
            down_f32,
            router: f32_bytes(&router),
            input,
        }
    }

    /// The store's own Q4_K blocks, `down` bound at its stored padded width
    /// with the role seeing the live columns.
    fn q4k_expert(&self) -> BoundExpert<'_> {
        BoundExpert {
            expert_id: 0,
            projection: BoundProjection::Fused {
                gate_up: direct(
                    GATE_UP_REGION_SET,
                    &self.gate_up,
                    RegionFormat::Q4K,
                    (FUSED_PROJECTION_HALVES * INTERMEDIATE) as u32,
                    HIDDEN as u32,
                ),
            },
            down: column_prefix(
                DOWN_REGION_SET,
                &self.down,
                RegionFormat::Q4K,
                HIDDEN as u32,
                INTERMEDIATE_PADDED as u32,
                INTERMEDIATE as u32,
            ),
        }
    }

    /// The same weights dequantised, for the reference kernel.
    fn f32_expert(&self) -> BoundExpert<'_> {
        BoundExpert {
            expert_id: 0,
            projection: BoundProjection::Fused {
                gate_up: direct(
                    GATE_UP_REGION_SET,
                    &self.gate_up_f32,
                    RegionFormat::F32,
                    (FUSED_PROJECTION_HALVES * INTERMEDIATE) as u32,
                    HIDDEN as u32,
                ),
            },
            down: column_prefix(
                DOWN_REGION_SET,
                &self.down_f32,
                RegionFormat::F32,
                HIDDEN as u32,
                INTERMEDIATE_PADDED as u32,
                INTERMEDIATE as u32,
            ),
        }
    }

    fn operation<'a>(
        &'a self,
        expert: BoundExpert<'a>,
        kernel: ExpertKernel,
    ) -> BoundMoeOperation<'a> {
        self.operation_with(expert, kernel, ACTIVATION)
    }

    fn operation_with<'a>(
        &'a self,
        expert: BoundExpert<'a>,
        kernel: ExpertKernel,
        activation: Activation,
    ) -> BoundMoeOperation<'a> {
        BoundMoeOperation {
            router: BoundRouter {
                weight: direct(
                    ROUTER_REGION_SET,
                    &self.router,
                    RegionFormat::F32,
                    POPULATION as u32,
                    HIDDEN as u32,
                ),
                top_k: TOP_K,
                selected_weight: MoeTopKWeightPolicy::RenormalizedSoftmax,
                scaling: BoundExpertScaling::None,
                kernel: RouterKernel::default(),
            },
            transforms: Vec::new(),
            banks: vec![BoundBankOperation {
                bank: BankCoordinate::new(LAYER, BANK_ID),
                experts: vec![expert],
                intermediate_dim: INTERMEDIATE,
                hidden_dim: HIDDEN,
                activation,
                kernel,
            }],
            reduction: BoundReduction::WeightedSum,
            residual_dim: HIDDEN,
        }
    }

    /// The incumbent's own function, called directly on the same bytes.
    fn incumbent_expert_output(&self) -> Vec<f32> {
        let q8k = quantize_x_to_q8k(&self.input);
        let mut scratch = ExpertScratch::new(HIDDEN, INTERMEDIATE, INTERMEDIATE_PADDED);
        run_single_expert_q4k_q8k_into(
            &mut scratch,
            &q8k,
            &self.gate_up,
            &self.down,
            INTERMEDIATE,
            ACTIVATION,
        )
        .to_vec()
    }

    fn run(&self, operation: &BoundMoeOperation<'_>) -> Vec<f32> {
        let (_, trace) = execute_traced(operation, MoeInputs::shared(&self.input))
            .expect("the fixture executes");
        trace
            .expert_output(0)
            .expect("expert 0 was selected")
            .to_vec()
    }
}

// ── The claim ──────────────────────────────────────────────────────────────

#[test]
fn the_bound_kernel_reproduces_the_incumbent_call_bit_for_bit() {
    // The rung's whole point, at unit scale and without a checkpoint: the same
    // Q4_K bytes, the same activation, the same production function — reached
    // through a bound operation rather than by calling it directly.
    let fixture = Fixture::new();
    let operation = fixture.operation(fixture.q4k_expert(), ExpertKernel::IncumbentQ4kQ8k);
    operation.validate().expect("the fixture binds");

    let bound = fixture.run(&operation);
    let incumbent = fixture.incumbent_expert_output();

    assert_eq!(bound.len(), HIDDEN);
    assert_eq!(
        bound,
        incumbent,
        "bound kernel diverged from the incumbent call by {}",
        max_abs_diff(&bound, &incumbent)
    );
}

#[test]
fn the_bound_kernel_is_not_producing_zeros() {
    // Guards the guard. The incumbent's short-slab branch zeroes its output and
    // returns successfully, so an all-zero agreement would be two failures
    // agreeing rather than a parity result.
    let fixture = Fixture::new();
    let operation = fixture.operation(fixture.q4k_expert(), ExpertKernel::IncumbentQ4kQ8k);
    let bound = fixture.run(&operation);
    assert!(
        bound.iter().any(|v| v.abs() > f32::EPSILON),
        "an all-zero expert output means the kernel took its refusal branch"
    );
    assert!(bound.iter().all(|v| v.is_finite()));
}

#[test]
fn the_reference_kernel_agrees_with_the_bound_one_within_quantisation_noise() {
    // The independent leg. Two calls of one function agree whatever they are
    // handed; this says the operands were the right ones, because a swapped
    // gate/up half or a mis-strided `down` would not land inside a band.
    let fixture = Fixture::new();
    let bound =
        fixture.run(&fixture.operation(fixture.q4k_expert(), ExpertKernel::IncumbentQ4kQ8k));
    let reference = fixture.run(&fixture.operation(fixture.f32_expert(), ExpertKernel::Reference));

    let diff = max_abs_diff(&bound, &reference);
    assert!(
        diff <= KERNEL_TOLERANCE,
        "kernels disagree by {diff}, beyond quantisation noise"
    );
}

#[test]
fn the_two_kernels_are_not_trivially_identical() {
    // Guards the tolerance. If the reference somehow ran the same arithmetic,
    // the band above would be vacuous and would keep passing through a real
    // regression.
    let fixture = Fixture::new();
    let bound =
        fixture.run(&fixture.operation(fixture.q4k_expert(), ExpertKernel::IncumbentQ4kQ8k));
    let reference = fixture.run(&fixture.operation(fixture.f32_expert(), ExpertKernel::Reference));
    assert_ne!(
        bound, reference,
        "an integer-dot kernel and an f32 reference should not be bit-identical"
    );
}

// ── Naming ─────────────────────────────────────────────────────────────────

#[test]
fn each_kernel_has_a_distinct_name_and_the_reference_is_the_default() {
    assert_eq!(ExpertKernel::default(), ExpertKernel::Reference);
    assert_ne!(
        ExpertKernel::Reference.name(),
        ExpertKernel::IncumbentQ4kQ8k.name()
    );
    assert!(ExpertKernel::IncumbentQ4kQ8k.name().contains("q4k"));
}

#[test]
fn the_bank_reports_which_kernel_it_bound() {
    let fixture = Fixture::new();
    let operation = fixture.operation(fixture.q4k_expert(), ExpertKernel::IncumbentQ4kQ8k);
    assert!(operation
        .describe()
        .contains(ExpertKernel::IncumbentQ4kQ8k.name()));
}

// ── Refusals: the activation ───────────────────────────────────────────────

#[test]
fn an_activation_the_kernel_does_not_implement_is_refused() {
    // The most dangerous case, because nothing else would signal it: the
    // incumbent's loop branches on GeluTanh and falls through to SiLU, so a
    // ReLU bank would run SiLU and return finite plausible values.
    let fixture = Fixture::new();
    let operation = fixture.operation_with(
        fixture.q4k_expert(),
        ExpertKernel::IncumbentQ4kQ8k,
        Activation::ReLU,
    );
    let err = operation.validate().unwrap_err();
    assert!(
        matches!(err, ExecutionError::KernelActivationUnsupported { .. }),
        "{err}"
    );
    assert!(err.to_string().contains("ReLU"), "{err}");
}

#[test]
fn both_activations_the_kernel_implements_are_accepted() {
    let fixture = Fixture::new();
    for activation in [Activation::Silu, Activation::GeluTanh] {
        fixture
            .operation_with(
                fixture.q4k_expert(),
                ExpertKernel::IncumbentQ4kQ8k,
                activation,
            )
            .validate()
            .unwrap_or_else(|e| panic!("{activation:?} should bind: {e}"));
    }
}

#[test]
fn the_reference_kernel_accepts_every_activation() {
    // The counter-case: the restriction belongs to the incumbent kernel, not to
    // the runtime.
    let fixture = Fixture::new();
    fixture
        .operation_with(
            fixture.f32_expert(),
            ExpertKernel::Reference,
            Activation::ReLU,
        )
        .validate()
        .expect("the reference implements all four");
}

// ── Refusals: the operands ─────────────────────────────────────────────────

#[test]
fn a_decomposed_projection_is_refused_by_arrangement() {
    // Remedy: bind the fused variant, or a kernel that takes two regions.
    // Stitching them into one temporary would make the parity result a
    // statement about the stitching.
    let fixture = Fixture::new();
    let half = fixture.gate_up.len() / FUSED_PROJECTION_HALVES;
    let expert = BoundExpert {
        expert_id: 0,
        projection: BoundProjection::Decomposed {
            gate: direct(
                GATE_UP_REGION_SET,
                &fixture.gate_up[..half],
                RegionFormat::Q4K,
                INTERMEDIATE as u32,
                HIDDEN as u32,
            ),
            up: direct(
                GATE_UP_REGION_SET,
                &fixture.gate_up[half..],
                RegionFormat::Q4K,
                INTERMEDIATE as u32,
                HIDDEN as u32,
            ),
        },
        down: fixture.q4k_expert().down,
    };
    let err = fixture
        .operation(expert, ExpertKernel::IncumbentQ4kQ8k)
        .validate()
        .unwrap_err();
    assert!(
        matches!(
            err,
            ExecutionError::KernelOperandUnsuitable {
                reason: OperandUnsuitability::Arrangement { .. },
                ..
            }
        ),
        "{err}"
    );
}

#[test]
fn dequantised_operands_are_refused_by_format() {
    // Remedy: bind the Q4_K variant, or the reference kernel. Reading f32 bytes
    // as super-blocks is exactly the substitution this refuses.
    let fixture = Fixture::new();
    let err = fixture
        .operation(fixture.f32_expert(), ExpertKernel::IncumbentQ4kQ8k)
        .validate()
        .unwrap_err();
    assert!(
        matches!(
            err,
            ExecutionError::KernelOperandUnsuitable {
                reason: OperandUnsuitability::ElementFormat { .. },
                ..
            }
        ),
        "{err}"
    );
}

#[test]
fn a_down_region_padded_to_the_wrong_width_is_refused_by_the_kernel() {
    // Shape-valid — the role still exposes `INTERMEDIATE` columns — but stored
    // at a width the kernel would not stride by. Only the kernel binding can
    // catch this one, which is why the stored extent travels with the operand.
    let fixture = Fixture::new();
    let wide = INTERMEDIATE_PADDED + Q4K_BLOCK_ELEMS;
    let bytes = q4k_bytes(HIDDEN, wide, DOWN_SEED);
    let expert = BoundExpert {
        expert_id: 0,
        down: column_prefix(
            DOWN_REGION_SET,
            &bytes,
            RegionFormat::Q4K,
            HIDDEN as u32,
            wide as u32,
            INTERMEDIATE as u32,
        ),
        ..fixture.q4k_expert()
    };
    let err = fixture
        .operation(expert, ExpertKernel::IncumbentQ4kQ8k)
        .validate()
        .unwrap_err();
    assert!(
        matches!(
            err,
            ExecutionError::DimensionMismatch { axis, expected, found, .. }
                if axis == Axis::InputWidth && expected == INTERMEDIATE_PADDED && found == wide
        ),
        "{err}"
    );
}

#[test]
fn a_bank_input_that_is_not_whole_blocks_is_refused_at_execution() {
    // The activation-side counterpart. The incumbent guards this by falling
    // back to its f32 path; VINDEX3 refuses, because a silent change of kernel
    // is a silent change of answer.
    //
    // It fires at execution rather than at bind because the bank input is a
    // property of the token, not of the binding.
    let ragged = HIDDEN - 1;
    let gate_up = q4k_bytes(FUSED_PROJECTION_HALVES * INTERMEDIATE, HIDDEN, GATE_UP_SEED);
    let down = q4k_bytes(HIDDEN, INTERMEDIATE_PADDED, DOWN_SEED);
    let router = f32_bytes(&vec![1.0f32; POPULATION * ragged]);
    let operation = BoundMoeOperation {
        router: BoundRouter {
            weight: direct(
                ROUTER_REGION_SET,
                &router,
                RegionFormat::F32,
                POPULATION as u32,
                ragged as u32,
            ),
            top_k: TOP_K,
            selected_weight: MoeTopKWeightPolicy::RenormalizedSoftmax,
            scaling: BoundExpertScaling::None,
            kernel: RouterKernel::default(),
        },
        transforms: Vec::new(),
        banks: vec![BoundBankOperation {
            bank: BankCoordinate::new(LAYER, BANK_ID),
            experts: vec![BoundExpert {
                expert_id: 0,
                projection: BoundProjection::Fused {
                    gate_up: direct(
                        GATE_UP_REGION_SET,
                        &gate_up,
                        RegionFormat::Q4K,
                        (FUSED_PROJECTION_HALVES * INTERMEDIATE) as u32,
                        ragged as u32,
                    ),
                },
                down: column_prefix(
                    DOWN_REGION_SET,
                    &down,
                    RegionFormat::Q4K,
                    ragged as u32,
                    INTERMEDIATE_PADDED as u32,
                    INTERMEDIATE as u32,
                ),
            }],
            intermediate_dim: INTERMEDIATE,
            hidden_dim: ragged,
            activation: ACTIVATION,
            kernel: ExpertKernel::IncumbentQ4kQ8k,
        }],
        reduction: BoundReduction::WeightedSum,
        residual_dim: ragged,
    };

    let input = vec![0.1f32; ragged];
    let err = execute_traced(&operation, MoeInputs::shared(&input)).unwrap_err();
    assert!(
        matches!(
            err,
            ExecutionError::KernelOperandUnsuitable {
                reason: OperandUnsuitability::BlockAlignment { found, .. },
                ..
            } if found == ragged
        ),
        "{err}"
    );
}

// ── The session ────────────────────────────────────────────────────────────

#[test]
fn every_expert_in_a_bank_reads_the_one_quantised_activation() {
    // Two experts over identical weights, both selected. Each must reproduce
    // the incumbent's output for the *bank's* input — which is what the shared
    // Q8_K session buys, and what a per-expert quantisation would only
    // accidentally match.
    let fixture = Fixture::new();
    let mut operation = fixture.operation(fixture.q4k_expert(), ExpertKernel::IncumbentQ4kQ8k);
    let second = BoundExpert {
        expert_id: 1,
        ..fixture.q4k_expert()
    };
    operation.banks[0].experts.push(second);
    operation.router.top_k = POPULATION;
    operation.validate().expect("both experts bind");

    let (_, trace) = execute_traced(&operation, MoeInputs::shared(&fixture.input))
        .expect("the fixture executes");
    let incumbent = fixture.incumbent_expert_output();
    assert_eq!(trace.expert_outputs.len(), POPULATION);
    for output in &trace.expert_outputs {
        assert_eq!(
            output.values, incumbent,
            "expert {} read a different activation",
            output.expert_id
        );
    }
}

// ── The kernel checks its own operands, even unvalidated ───────────────────
//
// `execute` does not call `validate` — re-checking a binding per token is the
// resolution creep the bound object exists to prevent. So the kernel's operand
// checks have to hold on their own, and these reach them by executing an
// operation that was never validated. Every one of them would otherwise be a
// stride read at the wrong offset producing well-shaped noise.

/// Run without validating first, and return the refusal.
fn execute_unvalidated(fixture: &Fixture, expert: BoundExpert<'_>) -> ExecutionError {
    let operation = fixture.operation(expert, ExpertKernel::IncumbentQ4kQ8k);
    execute_traced(&operation, MoeInputs::shared(&fixture.input))
        .expect_err("the kernel should refuse this operand")
}

#[test]
fn a_gate_up_slab_of_the_wrong_height_is_refused_by_the_kernel() {
    // A fused region holds both halves. One that is only `INTERMEDIATE` rows
    // tall would have the kernel read the gate half as gate+up, and the up half
    // from past the end of the region.
    let fixture = Fixture::new();
    let bytes = q4k_bytes(INTERMEDIATE, HIDDEN, GATE_UP_SEED);
    let err = execute_unvalidated(
        &fixture,
        BoundExpert {
            expert_id: 0,
            projection: BoundProjection::Fused {
                gate_up: direct(
                    GATE_UP_REGION_SET,
                    &bytes,
                    RegionFormat::Q4K,
                    INTERMEDIATE as u32,
                    HIDDEN as u32,
                ),
            },
            ..fixture.q4k_expert()
        },
    );
    assert!(
        matches!(
            err,
            ExecutionError::DimensionMismatch { axis, expected, found, .. }
                if axis == Axis::Rows
                    && expected == FUSED_PROJECTION_HALVES * INTERMEDIATE
                    && found == INTERMEDIATE
        ),
        "{err}"
    );
}

#[test]
fn a_gate_up_slab_that_contracts_the_wrong_width_is_refused_by_the_kernel() {
    // The kernel strides gate_up by `hidden / block` super-blocks per row, so a
    // region storing a different width is read at the wrong offset from row one.
    let fixture = Fixture::new();
    let narrow = HIDDEN + Q4K_BLOCK_ELEMS;
    let bytes = q4k_bytes(FUSED_PROJECTION_HALVES * INTERMEDIATE, narrow, GATE_UP_SEED);
    let err = execute_unvalidated(
        &fixture,
        BoundExpert {
            expert_id: 0,
            projection: BoundProjection::Fused {
                gate_up: direct(
                    GATE_UP_REGION_SET,
                    &bytes,
                    RegionFormat::Q4K,
                    (FUSED_PROJECTION_HALVES * INTERMEDIATE) as u32,
                    narrow as u32,
                ),
            },
            ..fixture.q4k_expert()
        },
    );
    assert!(
        matches!(
            err,
            ExecutionError::DimensionMismatch { axis, expected, found, .. }
                if axis == Axis::InputWidth && expected == HIDDEN && found == narrow
        ),
        "{err}"
    );
}

#[test]
fn a_down_region_of_the_wrong_height_is_refused_by_the_kernel() {
    // `down` contracts the intermediate axis away and produces the bank's
    // output width. A shorter one silently produces a shorter residual delta.
    let fixture = Fixture::new();
    let short = HIDDEN - Q4K_BLOCK_ELEMS / 2;
    let bytes = q4k_bytes(short, INTERMEDIATE_PADDED, DOWN_SEED);
    let err = execute_unvalidated(
        &fixture,
        BoundExpert {
            expert_id: 0,
            down: column_prefix(
                DOWN_REGION_SET,
                &bytes,
                RegionFormat::Q4K,
                short as u32,
                INTERMEDIATE_PADDED as u32,
                INTERMEDIATE as u32,
            ),
            ..fixture.q4k_expert()
        },
    );
    assert!(
        matches!(
            err,
            ExecutionError::DimensionMismatch { axis, expected, found, .. }
                if axis == Axis::OutputWidth && expected == HIDDEN && found == short
        ),
        "{err}"
    );
}

#[test]
fn a_down_region_exposing_the_wrong_live_width_is_refused_by_the_kernel() {
    // Stored correctly, but the role claims more live columns than the bank
    // means. The kernel passes `intermediate` to the incumbent as the count of
    // activation values to compute, so this is the difference between the
    // padding being inert and being read as data.
    let fixture = Fixture::new();
    let wrong = INTERMEDIATE + 1;
    let err = execute_unvalidated(
        &fixture,
        BoundExpert {
            expert_id: 0,
            down: column_prefix(
                DOWN_REGION_SET,
                &fixture.down,
                RegionFormat::Q4K,
                HIDDEN as u32,
                INTERMEDIATE_PADDED as u32,
                wrong as u32,
            ),
            ..fixture.q4k_expert()
        },
    );
    assert!(
        matches!(
            err,
            ExecutionError::DimensionMismatch { axis, expected, found, .. }
                if axis == Axis::Columns && expected == INTERMEDIATE && found == wrong
        ),
        "{err}"
    );
}

#[test]
fn a_decomposed_projection_is_refused_at_execution_too() {
    let fixture = Fixture::new();
    let half = fixture.gate_up.len() / FUSED_PROJECTION_HALVES;
    let err = execute_unvalidated(
        &fixture,
        BoundExpert {
            expert_id: 0,
            projection: BoundProjection::Decomposed {
                gate: direct(
                    GATE_UP_REGION_SET,
                    &fixture.gate_up[..half],
                    RegionFormat::Q4K,
                    INTERMEDIATE as u32,
                    HIDDEN as u32,
                ),
                up: direct(
                    GATE_UP_REGION_SET,
                    &fixture.gate_up[half..],
                    RegionFormat::Q4K,
                    INTERMEDIATE as u32,
                    HIDDEN as u32,
                ),
            },
            ..fixture.q4k_expert()
        },
    );
    assert!(
        matches!(
            err,
            ExecutionError::KernelOperandUnsuitable {
                reason: OperandUnsuitability::Arrangement { .. },
                ..
            }
        ),
        "{err}"
    );
}
