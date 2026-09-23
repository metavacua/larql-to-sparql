//! Gemma one-layer parity — VINDEX3 bound over VINDEX2's own expert bytes.
//!
//! The first real-model step. It holds *everything* constant except the
//! execution path:
//!
//! ```text
//! one VINDEX2 index on disk
//!         ↓
//! the same Q4_K expert bytes, the same f32 router, the same activation
//!    ↓                                    ↓
//! incumbent MoE path              BoundMoeOperation bound over those bytes
//!    ↓                                    ↓
//!         compare, checkpoint by checkpoint
//! ```
//!
//! # Why not extract a VINDEX3 container first
//!
//! Because then a mismatch would have two candidate causes — the executor, or
//! the re-extraction — and telling those apart is the entire point. Binding
//! over the incumbent's own bytes makes any divergence unambiguously about
//! execution: routing policy, region interpretation, activation, or reduction.
//!
//! # The ladder
//!
//! Rung by rung, so agreement at the bottom cannot conceal compensating
//! differences higher up:
//!
//! ```text
//! 1-5   router     scores, selection, margin, pre-norm and final weights
//! 6-8   experts    per-expert outputs, the weighted reduction, the block
//! ```
//!
//! Rungs 1-5 bind `larql-compute`'s scoring functions; rungs 6-8 bind its
//! Q4_K × Q8_K expert kernel over the store's own super-blocks. Both are
//! *bindings*, not reimplementations — writing a lookalike here and calling the
//! agreement "parity" would prove only that two similar loops agree.
//!
//! # Three paths, not two
//!
//! ```text
//! incumbent    cpu_moe_forward and the functions underneath it
//! bound        VINDEX3 over the same Q4_K bytes, production kernels
//! reference    VINDEX3 over the same weights dequantised, oracle kernels
//! ```
//!
//! The third exists because two calls of one function agree whatever operands
//! they are handed. A bit-identical result says the handover was faithful; only
//! an independently-computed answer says the operands were the right ones.
//!
//! # What is deliberately *not* claimed
//!
//! That the reference agrees to the bit. It dequantises to f32 where the
//! incumbent keeps an integer dot against a Q8_K activation, so the two differ
//! by quantisation noise by construction. That leg is judged against a
//! *relative* band, because expert outputs are residual-scale values rather
//! than probabilities and an absolute figure would have to be re-derived per
//! layer.
//!
//! Nor is the *schedule* claimed. The incumbent sums its experts through rayon
//! or the spin pool depending on configuration, and a tree reduction is not
//! required to agree bit-for-bit with a sequential one. Rung 7 therefore
//! compares against the incumbent's own per-expert outputs summed in selection
//! order, which isolates the combine from the scheduling; rung 8 reports the
//! end-to-end block figure without asserting it. Rung 8 came out bit-identical
//! on the spin-pool path — which accumulates in selection order — and that is a
//! measurement of one configuration, not a promise about every one.
//!
//! # Capturing the activation
//!
//! ```text
//! LARQL_CPU_DUMP_LAYERS=/tmp/gemma_dump \
//!   larql run <vindex> "The capital of France is" -n 1
//! ```
//!
//! writes `cpu_layer_NN_h_post_attn.f32` — the real residual entering each
//! layer's FFN/MoE block.
//!
//! Usage:
//! ```text
//! cargo run --release -p larql-vindex --example vindex3_gemma_layer_parity -- \
//!   --vindex <path> --dump /tmp/gemma_dump [--layer 5]
//! ```

use larql_compute::cpu::ops::moe::{
    cpu_moe_forward, moe_expert_input, moe_post_expert_output, moe_route_from_router_input,
    moe_router_input, moe_score_experts, moe_softmax, quantize_x_to_q8k,
    run_single_expert_q4k_q8k_into, ExpertScratch,
};
use larql_compute::cpu::ops::q4_common::dequantize_q4_k;
use larql_compute::pipeline_layer::build_moe_weights;
use larql_compute::MoeLayerWeights;

use larql_vindex::format::capability::binding::{ComponentView, RepresentationIdentity};
use larql_vindex::format::capability::component::ComponentContract;
use larql_vindex::format::capability::coordinate::BankCoordinate;
use larql_vindex::format::lyrw2::region_format::RegionFormat;
use larql_vindex::format::lyrw2::region_role::RegionRole;
use larql_vindex::runtime::consts::{COL_DIM, FUSED_PROJECTION_HALVES};
use larql_vindex::runtime::{
    execute_traced, BoundBankOperation, BoundExpert, BoundExpertScaling, BoundMoeOperation,
    BoundProjection, BoundReduction, BoundRouter, BoundTensor, ExpertKernel, MoeInputs,
    RouterKernel,
};

/// Band for the router's gate weights. They are softmax probabilities, so an
/// absolute figure is meaningful: the quantities being compared are all in
/// `[0, 1]` and sum to one.
const WEIGHT_TOLERANCE: f32 = 1e-3;

/// Band for the oracle leg, as a fraction of the reference's own magnitude.
///
/// Relative, and not the weight tolerance, because expert outputs are not
/// probabilities — they are residual-scale values whose magnitude is a property
/// of the layer. Judging them against a band derived from softmax outputs
/// compares the right direction on the wrong object.
///
/// The size is derivable rather than fitted. Both paths read *identical*
/// weights: the reference dequantises exactly the Q4_K blocks the kernel reads,
/// so the weight side contributes nothing. The whole difference is the
/// activation side — Q8_K stores `d = amax / 127`, bounding each element's
/// representation error at `d / 2`, and the intermediate is quantised a second
/// time before `down`. Two such roundings carried through a 2816-wide
/// contraction put the expected disagreement in the low percent. Five percent
/// is a band around that, not a target: a swapped gate/up half or a mis-strided
/// `down` misses it by orders of magnitude rather than by a factor.
const ORACLE_RELATIVE_BAND: f32 = 0.05;

/// Reference magnitude below which a ratio stops being reportable.
///
/// The denominator of the oracle figure is `max|reference|`. Near zero that
/// division amplifies a rounding difference into a number that looks
/// catastrophic and compares to nothing, so below this floor the comparison
/// switches to an absolute one and says so. Named, and checked against the
/// reference rather than the found values, so a layer's percentage stays
/// comparable to every other layer's.
const ORACLE_SCALE_FLOOR: f32 = 1e-6;
/// Selection is discrete. Any difference is a real disagreement.
const DEFAULT_LAYER: usize = 5;
const VARIANT: &str = "vindex2-bytes";
const ROUTER_REGION_SET: &str = "router";
const PER_EXPERT_SCALE_REGION_SET: &str = "router_per_expert_scale";
/// The single bank this one-layer harness binds.
const BANK_ID: u16 = 0;
/// Column the ladder's verdicts line up in.
const LADDER_LABEL_WIDTH: usize = 24;

fn arg(name: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

/// Read an f32 dump and return its final row — the token being decoded.
fn last_row(path: &str, width: usize) -> Result<Vec<f32>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    let values: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    if !values.len().is_multiple_of(width) || values.is_empty() {
        return Err(format!(
            "{path}: {} values is not a whole number of {width}-wide rows",
            values.len()
        ));
    }
    let rows = values.len() / width;
    Ok(values[(rows - 1) * width..].to_vec())
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn report(label: &str, a: &[f32], b: &[f32]) -> bool {
    if a.len() != b.len() {
        println!(
            "  {label:<LADDER_LABEL_WIDTH$} LENGTH {} vs {}",
            a.len(),
            b.len()
        );
        return false;
    }
    let diff = max_abs_diff(a, b);
    let ok = diff <= WEIGHT_TOLERANCE;
    println!(
        "  {label:<LADDER_LABEL_WIDTH$} max|Δ| = {diff:.3e}  {}",
        if ok { "within tolerance" } else { "OUTSIDE" }
    );
    ok
}

/// Largest absolute value in a vector — the scale a difference is judged against.
fn magnitude(values: &[f32]) -> f32 {
    values.iter().fold(0.0f32, |m, v| m.max(v.abs()))
}

/// Compare two vectors relative to the **reference's** own magnitude.
///
/// An absolute band would have to be re-derived for every layer, because a
/// residual-scale quantity's size is a property of the layer rather than of the
/// arithmetic. The ratio is the same question asked in units that travel.
///
/// # The denominator is fixed, and it is stated
///
/// `max|reference|`, never `max|found|` and never the larger of the two. A
/// denominator that moved with the thing being measured would let a kernel that
/// inflates its output report a *smaller* percentage for a larger error, and
/// would make two layers' figures incomparable. The oracle is the fixed point;
/// the bound path is what is being judged against it.
///
/// # Zero-near-zero is a separate verdict, not a divide
///
/// Below [`ORACLE_SCALE_FLOOR`] the ratio stops meaning anything — a reference
/// of 1e-9 turns any rounding difference into thousands of percent — so this
/// reports the absolute difference against the floor and labels it, rather than
/// dividing and emitting a number that looks like the others but is not
/// comparable to them. A layer whose output is genuinely near zero should read
/// as such rather than as a catastrophic relative error.
fn report_relative(label: &str, reference: &[f32], found: &[f32]) -> bool {
    if reference.len() != found.len() {
        println!(
            "  {label:<LADDER_LABEL_WIDTH$} LENGTH {} vs {}",
            reference.len(),
            found.len()
        );
        return false;
    }
    let diff = max_abs_diff(reference, found);
    let scale = magnitude(reference);
    if scale < ORACLE_SCALE_FLOOR {
        let ok = diff < ORACLE_SCALE_FLOOR;
        println!(
            "  {label:<LADDER_LABEL_WIDTH$} max|Δ| = {diff:.3e}, reference magnitude \
             {scale:.3e} below the {ORACLE_SCALE_FLOOR:.0e} floor — absolute, not a ratio  {}",
            if ok { "within floor" } else { "OUTSIDE" }
        );
        return ok;
    }
    let relative = diff / scale;
    let ok = relative <= ORACLE_RELATIVE_BAND;
    println!(
        "  {label:<LADDER_LABEL_WIDTH$} max|Δ| = {diff:.3e} of {scale:.3e} = {:.2}%  {}",
        relative * 100.0,
        if ok { "within band" } else { "OUTSIDE" }
    );
    ok
}

/// Bind one expert's Q4_K bytes as dequantised f32.
///
/// Materialising rather than binding Q4_K directly: the reference decoder
/// implements the directly-readable encodings only, and a quantised region is
/// a missing kernel rather than bad bytes. The incumbent's own fallback path
/// dequantises the same way, which is what makes the two comparable at all.
fn dequantised(
    role: RegionRole,
    bytes: &[u8],
    rows: usize,
    cols: usize,
) -> Result<(Vec<u8>, ComponentContract), String> {
    let values = dequantize_q4_k(bytes, rows * cols);
    if values.len() < rows * cols {
        return Err(format!(
            "{}: dequantised {} of {} expected elements",
            role.name(),
            values.len(),
            rows * cols
        ));
    }
    let raw: Vec<u8> = values[..rows * cols]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    Ok((raw, ComponentContract::matrix(rows as u32, cols as u32)))
}

fn tensor<'a>(
    region_set: &str,
    bytes: &'a [u8],
    format: RegionFormat,
    contract: ComponentContract,
) -> Result<BoundTensor<'a>, String> {
    BoundTensor::direct(
        RepresentationIdentity::new(region_set, VARIANT),
        bytes,
        format,
        contract,
    )
    .map_err(|e| e.to_string())
}

/// Bind a stored matrix whose trailing columns are quantisation padding.
///
/// The role sees `[rows, keep]`; the bytes remain `[rows, stored_cols]`. No
/// repacking, no copy — the view resolves the difference at read time.
///
/// Both kernels read this one binding and take from it what each needs: the
/// reference reads the `keep` live columns and never touches the padding,
/// while the Q4_K kernel — which decodes whole super-blocks and cannot stop
/// inside one — reads all `stored_cols` and lets the zero-padded activation
/// cancel the rest. That is why the view belongs to the operand rather than
/// being something each kernel is told separately.
fn sliced_tensor<'a>(
    region_set: &str,
    bytes: &'a [u8],
    format: RegionFormat,
    storage: ComponentContract,
    keep: usize,
) -> Result<BoundTensor<'a>, String> {
    BoundTensor::new(
        RepresentationIdentity::new(region_set, VARIANT),
        bytes,
        format,
        storage,
        ComponentView::Slice {
            dim: COL_DIM,
            start: 0,
            len: keep as u32,
        },
    )
    .map_err(|e| e.to_string())
}

/// Owns the dequantised expert buffers so bound tensors can borrow them.
struct ExpertBuffers {
    expert_id: u32,
    gate_up: Vec<u8>,
    down: Vec<u8>,
    gate_up_contract: ComponentContract,
    down_contract: ComponentContract,
}

/// Assemble the operation around one bank of experts.
///
/// Written once and called twice. Two hand-written literals meant to differ in
/// exactly one field is how a parity harness acquires a second difference
/// nobody notices.
fn bind_operation<'a>(
    layer: usize,
    hidden: usize,
    moe: &MoeLayerWeights<'_>,
    router_bytes: &'a [u8],
    scale_bytes: &'a [u8],
    experts: Vec<BoundExpert<'a>>,
    kernel: ExpertKernel,
) -> Result<BoundMoeOperation<'a>, String> {
    Ok(BoundMoeOperation {
        router: BoundRouter {
            weight: tensor(
                ROUTER_REGION_SET,
                router_bytes,
                RegionFormat::F32,
                ComponentContract::matrix(moe.num_experts as u32, hidden as u32),
            )?,
            top_k: moe.top_k,
            selected_weight: moe.routing_policy.selected_weight,
            scaling: if scale_bytes.is_empty() {
                BoundExpertScaling::None
            } else {
                BoundExpertScaling::PerExpert {
                    scales: tensor(
                        PER_EXPERT_SCALE_REGION_SET,
                        scale_bytes,
                        RegionFormat::F32,
                        ComponentContract::vector(moe.num_experts as u32),
                    )?,
                }
            },
            // Rung 1: bind the production scoring kernel, not a lookalike.
            kernel: RouterKernel::Incumbent,
        },
        transforms: Vec::new(),
        banks: vec![BoundBankOperation {
            bank: BankCoordinate::new(layer as u32, BANK_ID),
            experts,
            intermediate_dim: moe.intermediate_size,
            hidden_dim: hidden,
            activation: moe.activation,
            kernel,
        }],
        reduction: BoundReduction::WeightedSum,
        residual_dim: hidden,
    })
}

/// Run the incumbent's own expert kernel over each selected expert, in
/// selection order, sharing one Q8_K activation exactly as production does.
fn incumbent_expert_outputs(
    moe: &MoeLayerWeights<'_>,
    expert_input: &[f32],
    selected: &[usize],
) -> Result<Vec<Vec<f32>>, String> {
    let hidden = expert_input.len();
    let inter = moe.intermediate_size;
    let q8k = quantize_x_to_q8k(expert_input);
    let mut scratch = ExpertScratch::new(hidden, inter, moe.inter_padded());
    selected
        .iter()
        .map(|&e| {
            let gate_up = *moe
                .experts_gate_up
                .get(e)
                .ok_or_else(|| format!("expert {e} has no gate_up bytes"))?;
            let down = *moe
                .experts_down
                .get(e)
                .ok_or_else(|| format!("expert {e} has no down bytes"))?;
            Ok(run_single_expert_q4k_q8k_into(
                &mut scratch,
                &q8k,
                gate_up,
                down,
                inter,
                moe.activation,
            )
            .to_vec())
        })
        .collect()
}

/// `sum_i weight_i * output_i`, in selection order.
///
/// The incumbent's parallel paths reach this same sum through a rayon
/// tree-reduce or a spin-pool slot scan, neither of which fixes an order. So
/// the reduction is compared against *this* — the incumbent's own per-expert
/// outputs, combined in the order VINDEX3 combines them — which isolates the
/// combine from the schedule. Blaming a scheduling difference on the reduction
/// is exactly the false diagnosis a ladder exists to prevent.
fn weighted_sum(outputs: &[Vec<f32>], weights: &[f32], width: usize) -> Vec<f32> {
    let mut acc = vec![0.0f32; width];
    for (out, &w) in outputs.iter().zip(weights) {
        for (slot, &v) in acc.iter_mut().zip(out) {
            *slot += w * v;
        }
    }
    acc
}

/// Print one ladder rung and say whether it was bit-identical.
fn rung(step: usize, label: &str, a: &[f32], b: &[f32]) -> bool {
    let identical = a == b;
    let verdict = if identical {
        "BIT-IDENTICAL".to_string()
    } else if a.len() == b.len() {
        format!("max|Δ| = {:.3e}", max_abs_diff(a, b))
    } else {
        format!("LENGTH {} vs {}", a.len(), b.len())
    };
    println!("  {step} {label:<LADDER_LABEL_WIDTH$} {verdict}");
    identical
}

fn main() -> Result<(), String> {
    let vindex = arg("--vindex").ok_or("set --vindex <path>")?;
    let dump = arg("--dump").ok_or("set --dump <dir> (LARQL_CPU_DUMP_LAYERS output)")?;
    let layer: usize = arg("--layer")
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_LAYER);

    println!("Gemma one-layer parity — VINDEX2 bytes, two execution paths");
    println!("  vindex  {vindex}");
    println!("  layer   {layer}");

    // ── Load the incumbent's weights ───────────────────────────────────────
    let mut callbacks = larql_vindex::SilentLoadCallbacks;
    let weights =
        larql_vindex::load_model_weights_kquant(std::path::Path::new(&vindex), &mut callbacks)
            .map_err(|e| format!("load weights: {e}"))?;
    let arch = &*weights.arch;
    let hidden = weights.hidden_size;
    let norm_offset = arch.norm_weight_offset();
    let eps = arch.norm_eps();

    let moe: MoeLayerWeights<'_> = build_moe_weights(&weights, arch, layer)
        .ok_or_else(|| format!("layer {layer} is not an MoE layer"))?;
    println!(
        "  shape   hidden {hidden}, {} experts, top-{}, intermediate {}",
        moe.num_experts, moe.top_k, moe.intermediate_size
    );

    // ── The real activation ────────────────────────────────────────────────
    let h = last_row(
        &larql_compute::forward::dump_config::cpu_layer_h_post_attn_path(&dump, layer),
        hidden,
    )?;
    println!("  input   real h_post_attn, last token of the prompt");

    // ── Incumbent: the two inputs and the routing decision ─────────────────
    let expert_input = moe_expert_input(&h, &moe, norm_offset, eps);
    let router_in = moe_router_input(&h, &expert_input, &moe, norm_offset, eps);
    let (incumbent_ids, incumbent_weights) = moe_route_from_router_input(&router_in, &moe);
    println!(
        "\n  incumbent routes on a {} vector",
        if expert_input == router_in {
            "shared"
        } else {
            "separate"
        }
    );

    // ── Bind the selected experts' bytes as a VINDEX3 operation ────────────
    //
    // Only the selected experts: a bank holding a subset of the population is
    // a legitimate shard, and materialising all of them would cost gigabytes to
    // no purpose. If the VINDEX3 router disagrees about the selection it will
    // ask for an expert that is not bound and fail loudly, which is the right
    // failure.
    //
    // Two bindings of the *same* experts:
    //
    //   bound       the store's own Q4_K super-blocks → the production kernel
    //   reference   the same weights dequantised to f32 → the oracle
    let inter = moe.intermediate_size;
    let inter_padded = moe.inter_padded();
    let mut buffers = Vec::new();
    let mut q4k_experts: Vec<BoundExpert<'_>> = Vec::new();
    for &e in &incumbent_ids {
        let gate_up_bytes = *moe
            .experts_gate_up
            .get(e)
            .ok_or_else(|| format!("expert {e} has no gate_up bytes"))?;
        let down_bytes = *moe
            .experts_down
            .get(e)
            .ok_or_else(|| format!("expert {e} has no down bytes"))?;

        // `down` is stored at the *padded* intermediate width: Q4_K rounds 704
        // up to the next 256-multiple (768), while `gate_up` is unpadded
        // because `hidden` is already a multiple. So the two regions disagree
        // about the intermediate axis, and the padding columns are inert.
        //
        // This is precisely what a slice view is for: bind the stored
        // [hidden, 768] and let the role see [hidden, 704]. The incumbent
        // reaches the same place by zero-padding the activation instead — and
        // so does the Q4_K kernel bound here, which is why one binding serves
        // both kernels rather than each needing its own.
        q4k_experts.push(BoundExpert {
            expert_id: e as u32,
            projection: BoundProjection::Fused {
                gate_up: tensor(
                    &RegionRole::GateUpFused.name(),
                    gate_up_bytes,
                    RegionFormat::Q4K,
                    ComponentContract::matrix(
                        (FUSED_PROJECTION_HALVES * inter) as u32,
                        hidden as u32,
                    ),
                )?,
            },
            down: sliced_tensor(
                &RegionRole::Down.name(),
                down_bytes,
                RegionFormat::Q4K,
                ComponentContract::matrix(hidden as u32, inter_padded as u32),
                inter,
            )?,
        });

        let (gate_up, gate_up_contract) = dequantised(
            RegionRole::GateUpFused,
            gate_up_bytes,
            FUSED_PROJECTION_HALVES * inter,
            hidden,
        )?;
        let (down, down_contract) =
            dequantised(RegionRole::Down, down_bytes, hidden, inter_padded)?;
        buffers.push(ExpertBuffers {
            expert_id: e as u32,
            gate_up,
            down,
            gate_up_contract,
            down_contract,
        });
    }

    let router_bytes: Vec<u8> = moe
        .router_proj
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    // Gemma's routing policy is PerExpert, so the learned per-expert scale is
    // part of the recipe. Omitting it left scores bit-identical and normalised
    // weights 7e-4 apart — which the ladder localised to post-processing
    // rather than to the scoring kernel it would otherwise have been blamed on.
    let per_expert_scale_bytes: Vec<u8> = moe
        .router_per_expert_scale
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let f32_experts: Vec<BoundExpert<'_>> = buffers
        .iter()
        .map(|b| -> Result<BoundExpert<'_>, String> {
            Ok(BoundExpert {
                expert_id: b.expert_id,
                projection: BoundProjection::Fused {
                    gate_up: tensor(
                        &RegionRole::GateUpFused.name(),
                        &b.gate_up,
                        RegionFormat::F32,
                        b.gate_up_contract.clone(),
                    )?,
                },
                down: sliced_tensor(
                    &RegionRole::Down.name(),
                    &b.down,
                    RegionFormat::F32,
                    b.down_contract.clone(),
                    inter,
                )?,
            })
        })
        .collect::<Result<_, _>>()?;

    let operation = bind_operation(
        layer,
        hidden,
        &moe,
        &router_bytes,
        &per_expert_scale_bytes,
        q4k_experts,
        // Rung 2: bind the production expert kernel over the stored blocks.
        ExpertKernel::IncumbentQ4kQ8k,
    )?;
    operation.validate().map_err(|e| format!("bind: {e}"))?;
    let reference = bind_operation(
        layer,
        hidden,
        &moe,
        &router_bytes,
        &per_expert_scale_bytes,
        f32_experts,
        ExpertKernel::Reference,
    )?;
    reference
        .validate()
        .map_err(|e| format!("bind reference: {e}"))?;

    println!("  bound   {}", operation.describe());
    println!("  oracle  {}", reference.describe());
    println!(
        "  shard   holds {} of {} experts (full population: {})",
        operation.banks[0].population(),
        operation.router.population(),
        operation.holds_full_population()
    );
    println!("  padding down stores {inter_padded} columns, the role means {inter}");

    // ── Execute both VINDEX3 routes on the identical inputs ────────────────
    let (_, trace) = execute_traced(&operation, MoeInputs::split(&expert_input, &router_in))
        .map_err(|e| format!("execute: {e}"))?;
    let (_, reference_trace) =
        execute_traced(&reference, MoeInputs::split(&expert_input, &router_in))
            .map_err(|e| format!("execute reference: {e}"))?;

    // ── Compare ────────────────────────────────────────────────────────────
    println!("\n== selection (exact match required) ==");
    let vindex3_ids: Vec<usize> = trace.selected_ids().iter().map(|i| *i as usize).collect();
    println!("  incumbent  {incumbent_ids:?}");
    println!("  vindex3    {vindex3_ids:?}");
    let selection_matches = incumbent_ids == vindex3_ids;
    println!(
        "  {}",
        if selection_matches {
            "PASS identical experts, identical order"
        } else {
            "FAIL the two paths route differently"
        }
    );
    if let Some(margin) = trace.selection_margin {
        println!(
            "  margin     {margin:.6}{}",
            if margin == 0.0 {
                "  (an exact tie decided the boundary)"
            } else {
                ""
            }
        );
    }

    // ── The failure ladder ─────────────────────────────────────────────────
    //
    // Rung by rung, so that agreement at the bottom cannot conceal
    // compensating differences higher up. Renormalisation in particular can
    // make two different score vectors produce identical final weights.
    println!(
        "\n== router ladder (kernel: {}) ==",
        operation.router.kernel.name()
    );

    // 1. raw scores, straight from the incumbent's own functions
    let mut incumbent_scores =
        moe_score_experts(&router_in, moe.router_proj, moe.num_experts, hidden);
    moe_softmax(&mut incumbent_scores);
    let scores_identical = incumbent_scores == trace.router_scores;
    println!(
        "  1 raw scores           {}",
        if scores_identical {
            "BIT-IDENTICAL".to_string()
        } else {
            format!(
                "max|Δ| = {:.3e}",
                max_abs_diff(&incumbent_scores, &trace.router_scores)
            )
        }
    );

    // 2. ordered (id, score) pairs
    let incumbent_pairs: Vec<(usize, f32)> = incumbent_ids
        .iter()
        .map(|&e| (e, incumbent_scores[e]))
        .collect();
    let vindex3_pairs: Vec<(usize, f32)> = trace
        .selection
        .iter()
        .map(|s| (s.expert_id as usize, s.raw_score))
        .collect();
    let pairs_identical = incumbent_pairs == vindex3_pairs;
    println!(
        "  2 top-k id/score pairs {}",
        if pairs_identical {
            "BIT-IDENTICAL"
        } else {
            "DIFFER"
        }
    );

    // 3. boundary margin
    println!("  3 boundary margin      {:?}", trace.selection_margin);

    // 4. pre-normalisation selected weights — the softmax probabilities
    let pre_norm: Vec<f32> = incumbent_ids.iter().map(|&e| incumbent_scores[e]).collect();
    let vindex3_pre_norm: Vec<f32> = trace.selection.iter().map(|s| s.raw_score).collect();
    let pre_norm_identical = pre_norm == vindex3_pre_norm;
    println!(
        "  4 pre-norm weights     {}",
        if pre_norm_identical {
            "BIT-IDENTICAL"
        } else {
            "DIFFER"
        }
    );

    // 5. normalised weights, after every policy
    let final_identical = incumbent_weights == trace.gate_weights();
    println!(
        "  5 normalised weights   {}",
        if final_identical {
            "BIT-IDENTICAL".to_string()
        } else {
            format!(
                "max|Δ| = {:.3e}",
                max_abs_diff(&incumbent_weights, &trace.gate_weights())
            )
        }
    );

    let weights_match = report(
        "gate weights (tolerance)",
        &incumbent_weights,
        &trace.gate_weights(),
    );
    let router_bit_identical =
        scores_identical && pairs_identical && pre_norm_identical && final_identical;

    // ── The expert ladder ──────────────────────────────────────────────────
    //
    // Same shape as the router's, and for the same reason: a single end-to-end
    // number would leave a per-expert kernel fault, a reduction fault and a
    // scheduling difference indistinguishable.
    println!(
        "\n== expert ladder (kernel: {}) ==",
        operation.banks[0].kernel.name()
    );

    // 6. per-expert outputs, before any routing weight is applied
    let incumbent_outputs = incumbent_expert_outputs(&moe, &expert_input, &incumbent_ids)?;
    let mut experts_identical = incumbent_outputs.len() == trace.expert_outputs.len();
    for (e, expected) in incumbent_ids.iter().zip(&incumbent_outputs) {
        let Some(found) = trace.expert_output(*e as u32) else {
            println!("  6 expert {e:<15} MISSING from the VINDEX3 trace");
            experts_identical = false;
            continue;
        };
        experts_identical &= expected.as_slice() == found;
    }
    println!(
        "  6 per-expert outputs   {}",
        if experts_identical {
            format!("BIT-IDENTICAL across {} experts", incumbent_outputs.len())
        } else {
            "DIFFER — see the first expert above".to_string()
        }
    );
    // Guards the guard: the incumbent's short-slab branch zeroes its output and
    // returns successfully, so an all-zero agreement would be two failures
    // agreeing rather than a parity result.
    let nonzero = trace.reduced.iter().any(|v| v.abs() > f32::EPSILON);
    println!(
        "    output magnitude     {}",
        if nonzero {
            "non-zero (the kernel ran)"
        } else {
            "ALL ZERO — the kernel took a refusal branch"
        }
    );

    // 7. the weighted reduction, in selection order
    let incumbent_reduced = weighted_sum(&incumbent_outputs, &incumbent_weights, hidden);
    let reduction_identical = rung(7, "weighted reduction", &incumbent_reduced, &trace.reduced);

    // 8. the block's contribution, after the policy norm the surrounding block
    //    owns. Reported, not asserted: `cpu_moe_forward` sums its experts
    //    through whichever parallel schedule is configured.
    let block = moe_post_expert_output(&trace.residual_delta, &moe, norm_offset, eps);
    let incumbent_block = cpu_moe_forward(&h, &moe, norm_offset, eps);
    let block_identical = rung(8, "block output", &incumbent_block, &block);

    // The independent leg. Not a rung: it is a different kernel by design, so
    // a band is the honest reading for it alone — and a *relative* one, since
    // what is being compared is a residual-scale quantity rather than a
    // probability.
    let oracle_match = report_relative(
        "oracle (relative band)",
        &reference_trace.reduced,
        &trace.reduced,
    );

    println!("\n== notes ==");
    if router_bit_identical {
        println!("  Router rung CLOSED: the bound kernel reproduces production scoring exactly.");
    } else {
        println!("  Router rung open — see the first ladder step that differs.");
    }
    if experts_identical && reduction_identical {
        println!("  Expert rung CLOSED: the bound kernel reproduces production experts exactly,");
        println!("  and the combine agrees in selection order.");
    } else {
        println!("  Expert rung open — see the first ladder step that differs.");
    }
    if !block_identical {
        println!("  Rung 8 differing is expected when the incumbent reduces through a tree:");
        println!("  floating-point addition is not associative, and rung 7 already isolated");
        println!("  the combine from the schedule.");
    }

    if selection_matches
        && weights_match
        && experts_identical
        && reduction_identical
        && oracle_match
    {
        println!("\nPARITY: selection, router weights, expert outputs and reduction all agree.");
        Ok(())
    } else {
        Err("parity not established — see above".into())
    }
}
