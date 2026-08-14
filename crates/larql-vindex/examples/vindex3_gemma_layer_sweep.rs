//! Locked-input MoE sweep — every Gemma MoE layer, independently.
//!
//! The single-layer harness proved layer 5. This proves every layer, and
//! deliberately does *not* let them interact:
//!
//! ```text
//! for each MoE layer:
//!     incumbent's captured h_post_attn      ← the same input to both paths
//!        ↓                        ↓
//!     incumbent MoE block     VINDEX3-bound MoE block
//!        ↓                        ↓
//!            classify, per checkpoint
//! ```
//!
//! # Why locked input
//!
//! Because free propagation cannot localise. If layer 7 diverges and layer 8
//! then reads a residual that already differs, layer 8's mismatch says nothing
//! about layer 8. Feeding every layer the incumbent's own captured activation
//! makes each layer an independent measurement, so the first failure is the
//! first *defect* rather than the first symptom. Composition is a separate
//! proof, and it comes after this one passes.
//!
//! # Why the full expert population
//!
//! The single-layer harness binds only the selected experts, which is a
//! legitimate shard. For a sweep that would be a mistake: a layer routing to an
//! expert outside the bound slice would report a residency refusal, and the
//! headline table would end up measuring slice coverage rather than execution.
//! So the principal run binds all of them — free, because Q4_K regions are
//! bound from the mapped bytes and nothing is materialised.
//!
//! `--shard N` binds only the first N experts instead. That run exists to prove
//! the *classifier*: selected-but-absent must come out as a residency refusal
//! and must not indict the execution.
//!
//! # Why the oracle is opt-in
//!
//! The dequantised f32 leg costs ~24 MB per expert. At full population that is
//! ~3 GB per layer, so it cannot ride along. It answers a different question
//! anyway — quantisation fidelity, which is a sample — where the sweep answers
//! implementation parity, which needs coverage. `--oracle-layer N` runs it for
//! one layer, over that layer's selected experts only.
//!
//! # Capturing the activations
//!
//! ```text
//! LARQL_CPU_DUMP_LAYERS=/tmp/gemma_dump \
//!   larql run <vindex> "The capital of France is" -n 1
//! ```
//!
//! Usage:
//! ```text
//! cargo run --release -p larql-vindex --example vindex3_gemma_layer_sweep -- \
//!   --vindex <path> --dump /tmp/gemma_dump [--shard 8] [--oracle-layer 5]
//! ```

use larql_compute::cpu::ops::moe::{
    cpu_moe_forward, moe_expert_input, moe_post_expert_output, moe_route_from_router_input,
    moe_router_input, quantize_x_to_q8k, run_single_expert_q4k_q8k_into, ExpertScratch,
};
use larql_compute::cpu::ops::q4_common::dequantize_q4_k;
use larql_compute::pipeline_layer::build_moe_weights;
use larql_compute::{Activation, MoeLayerWeights};

use larql_vindex::format::capability::binding::{ComponentView, RepresentationIdentity};
use larql_vindex::format::capability::component::ComponentContract;
use larql_vindex::format::capability::coordinate::BankCoordinate;
use larql_vindex::format::lyrw2::region_format::RegionFormat;
use larql_vindex::format::lyrw2::region_role::RegionRole;
use larql_vindex::runtime::consts::{COL_DIM, FUSED_PROJECTION_HALVES};
use larql_vindex::runtime::{
    execute_traced, BoundBankOperation, BoundExpert, BoundExpertScaling, BoundMoeOperation,
    BoundProjection, BoundReduction, BoundRouter, BoundTensor, CollectedTrace, ExecutionError,
    ExpertKernel, MoeInputs, RouterKernel, Verdict,
};

/// Band for the oracle leg, relative to `max|reference|`. See the single-layer
/// harness for the derivation; it is the Q8_K activation rounding, twice,
/// carried through the hidden-width contraction.
const ORACLE_RELATIVE_BAND: f32 = 0.05;
/// Reference magnitude below which a ratio stops being reportable and the
/// comparison switches to an absolute one.
const ORACLE_SCALE_FLOOR: f32 = 1e-6;
/// Softmax probabilities, so an absolute band is meaningful for these alone.
const WEIGHT_TOLERANCE: f32 = 1e-3;

const VARIANT: &str = "vindex2-bytes";
const ROUTER_REGION_SET: &str = "router";
const PER_EXPERT_SCALE_REGION_SET: &str = "router_per_expert_scale";
const BANK_ID: u16 = 0;

const ARG_VINDEX: &str = "--vindex";
const ARG_DUMP: &str = "--dump";
const ARG_SHARD: &str = "--shard";
const ARG_ORACLE_LAYER: &str = "--oracle-layer";

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
    if a.len() != b.len() {
        return f32::INFINITY;
    }
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn magnitude(values: &[f32]) -> f32 {
    values.iter().fold(0.0f32, |m, v| m.max(v.abs()))
}

/// Classify one checkpoint. `asserted` says whether the harness contracts this
/// comparison or merely watches it — see [`Verdict`].
fn classify(a: &[f32], b: &[f32], tolerance: f32, asserted: bool) -> Verdict {
    let diff = max_abs_diff(a, b);
    Verdict::from_comparison(a == b, diff <= tolerance, asserted)
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

/// Bind a stored matrix whose trailing columns are quantisation padding: the
/// role sees `[rows, keep]`, the bytes stay `[rows, stored_cols]`.
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

/// Bind one expert's Q4_K regions straight from the mapped bytes.
fn q4k_expert<'a>(
    expert_id: usize,
    moe: &MoeLayerWeights<'a>,
    hidden: usize,
) -> Result<BoundExpert<'a>, String> {
    let gate_up_bytes = *moe
        .experts_gate_up
        .get(expert_id)
        .ok_or_else(|| format!("expert {expert_id} has no gate_up bytes"))?;
    let down_bytes = *moe
        .experts_down
        .get(expert_id)
        .ok_or_else(|| format!("expert {expert_id} has no down bytes"))?;
    let inter = moe.intermediate_size;
    Ok(BoundExpert {
        expert_id: expert_id as u32,
        projection: BoundProjection::Fused {
            gate_up: tensor(
                &RegionRole::GateUpFused.name(),
                gate_up_bytes,
                RegionFormat::Q4K,
                ComponentContract::matrix((FUSED_PROJECTION_HALVES * inter) as u32, hidden as u32),
            )?,
        },
        down: sliced_tensor(
            &RegionRole::Down.name(),
            down_bytes,
            RegionFormat::Q4K,
            ComponentContract::matrix(hidden as u32, moe.inter_padded() as u32),
            inter,
        )?,
    })
}

/// Assemble the bound operation for one layer.
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
            kernel: RouterKernel::Incumbent,
        },
        transforms: Vec::new(),
        banks: vec![BoundBankOperation {
            bank: BankCoordinate::new(layer as u32, BANK_ID),
            experts,
            intermediate_dim: moe.intermediate_size,
            hidden_dim: hidden,
            activation: match moe.gate_rule {
                larql_compute::MoeGateRule::Gated(a) => a,
                rule => panic!("bound MoE kernels are gated-only, layer declares {rule:?}"),
            },
            kernel,
        }],
        reduction: BoundReduction::WeightedSum,
        residual_dim: hidden,
    })
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// The incumbent's own expert kernel over each selected expert, in selection
/// order, sharing one Q8_K activation exactly as production does.
fn incumbent_expert_outputs(
    moe: &MoeLayerWeights<'_>,
    expert_input: &[f32],
    selected: &[usize],
) -> Result<Vec<Vec<f32>>, String> {
    let hidden = expert_input.len();
    let inter = moe.intermediate_size;
    if !hidden.is_multiple_of(256) {
        return Err(format!("hidden {hidden} is not a whole Q8_K super-block"));
    }
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
                moe.expert_mlp(e),
            )
            .to_vec())
        })
        .collect()
}

/// `sum_i weight_i * output_i`, in selection order — the combine isolated from
/// whatever schedule the incumbent's parallel path happened to use.
fn weighted_sum(outputs: &[Vec<f32>], weights: &[f32], width: usize) -> Vec<f32> {
    let mut acc = vec![0.0f32; width];
    for (out, &w) in outputs.iter().zip(weights) {
        for (slot, &v) in acc.iter_mut().zip(out) {
            *slot += w * v;
        }
    }
    acc
}

/// One layer's classification, per checkpoint.
struct LayerReport {
    layer: usize,
    population: usize,
    resident: usize,
    top_k: usize,
    intermediate: usize,
    inter_padded: usize,
    activation: Activation,
    selection: Verdict,
    weights: Verdict,
    experts: Verdict,
    reduction: Verdict,
    block: Verdict,
    refusal: Option<ExecutionError>,
}

impl LayerReport {
    /// The layer's outcome: the weakest claim any checkpoint reached.
    ///
    /// `Verdict` orders weakest-last precisely so this is a maximum rather than
    /// a hand-written precedence table that could disagree with itself.
    fn verdict(&self) -> Verdict {
        if let Some(err) = &self.refusal {
            return Verdict::from(err);
        }
        [
            self.selection,
            self.weights,
            self.experts,
            self.reduction,
            self.block,
        ]
        .into_iter()
        .max()
        .unwrap_or(Verdict::Exact)
    }

    fn print(&self) {
        let padding = if self.inter_padded == self.intermediate {
            "none".to_string()
        } else {
            format!("{}→{}", self.intermediate, self.inter_padded)
        };
        println!(
            "  {:>3}  {:>4}/{:<4} top-{:<2} {:<9} pad {:<9} {:<16} {}",
            self.layer,
            self.resident,
            self.population,
            self.top_k,
            format!("{:?}", self.activation),
            padding,
            self.verdict().name(),
            self.detail()
        );
    }

    /// Checkpoints that came out weaker than their own *strongest attainable*
    /// outcome.
    ///
    /// Not "weaker than `Exact`". The block comparison is unasserted by design,
    /// so its ceiling is `ObservedExact`, and flagging it against `Exact` would
    /// print the same note for a perfect run as for a broken one — a column
    /// that says the same thing in both cases is worse than no column.
    fn detail(&self) -> String {
        if let Some(err) = &self.refusal {
            return err.to_string();
        }
        let below: Vec<&str> = [
            ("selection", self.selection, Verdict::Exact),
            ("weights", self.weights, Verdict::Exact),
            ("experts", self.experts, Verdict::Exact),
            ("reduction", self.reduction, Verdict::Exact),
            ("block", self.block, Verdict::ObservedExact),
        ]
        .into_iter()
        // `Verdict` orders weakest-last, so "greater than" is "weaker than".
        .filter(|(_, got, ceiling)| got > ceiling)
        .map(|(name, _, _)| name)
        .collect();
        if below.is_empty() {
            String::new()
        } else {
            format!("below ceiling at: {}", below.join(", "))
        }
    }
}

/// Compare one layer's two paths and classify every checkpoint.
fn sweep_layer(
    layer: usize,
    hidden: usize,
    moe: &MoeLayerWeights<'_>,
    h: &[f32],
    norm_offset: f32,
    eps: f32,
    shard: Option<usize>,
) -> Result<LayerReport, String> {
    let expert_input = moe_expert_input(h, moe, norm_offset, eps);
    let router_in = moe_router_input(h, &expert_input, moe, norm_offset, eps);
    let (incumbent_ids, incumbent_weights) = moe_route_from_router_input(&router_in, moe);

    let resident = shard.unwrap_or(moe.num_experts).min(moe.num_experts);
    let experts: Vec<BoundExpert<'_>> = (0..resident)
        .map(|e| q4k_expert(e, moe, hidden))
        .collect::<Result<_, _>>()?;

    let router_bytes = f32_bytes(moe.router_proj);
    let scale_bytes = f32_bytes(moe.router_per_expert_scale);
    let operation = bind_operation(
        layer,
        hidden,
        moe,
        &router_bytes,
        &scale_bytes,
        experts,
        ExpertKernel::IncumbentQ4kQ8k,
    )?;

    let mut report = LayerReport {
        layer,
        population: moe.num_experts,
        resident,
        top_k: moe.top_k,
        intermediate: moe.intermediate_size,
        inter_padded: moe.inter_padded(),
        activation: match moe.gate_rule {
            larql_compute::MoeGateRule::Gated(a) => a,
            rule => panic!("bound MoE kernels are gated-only, layer declares {rule:?}"),
        },
        selection: Verdict::Exact,
        weights: Verdict::Exact,
        experts: Verdict::Exact,
        reduction: Verdict::Exact,
        block: Verdict::Exact,
        refusal: None,
    };

    // Binding faults and missing kernels surface here, before any comparison —
    // and they are classified, not counted as mismatches.
    if let Err(err) = operation.validate() {
        report.refusal = Some(err);
        return Ok(report);
    }

    let trace: CollectedTrace =
        match execute_traced(&operation, MoeInputs::split(&expert_input, &router_in)) {
            Ok((_, trace)) => trace,
            Err(err) => {
                report.refusal = Some(err);
                return Ok(report);
            }
        };

    // Selection is discrete: identical or not, and asserted either way.
    let bound_ids: Vec<usize> = trace.selected_ids().iter().map(|i| *i as usize).collect();
    report.selection = Verdict::from_comparison(bound_ids == incumbent_ids, false, true);
    report.weights = classify(
        &incumbent_weights,
        &trace.gate_weights(),
        WEIGHT_TOLERANCE,
        true,
    );

    let incumbent_outputs = incumbent_expert_outputs(moe, &expert_input, &incumbent_ids)?;
    let mut experts_verdict = Verdict::Exact;
    for (e, expected) in incumbent_ids.iter().zip(&incumbent_outputs) {
        let found = trace.expert_output(*e as u32).unwrap_or(&[]);
        experts_verdict = experts_verdict.max(classify(expected, found, 0.0, true));
    }
    report.experts = experts_verdict;

    let incumbent_reduced = weighted_sum(&incumbent_outputs, &incumbent_weights, hidden);
    report.reduction = classify(&incumbent_reduced, &trace.reduced, 0.0, true);

    // Observed, not asserted: `cpu_moe_forward` sums through whichever parallel
    // schedule is configured, and a tree reduction is not required to agree
    // bit-for-bit with a sequential one.
    let bound_block = moe_post_expert_output(&trace.residual_delta, moe, norm_offset, eps);
    let incumbent_block = cpu_moe_forward(h, moe, norm_offset, eps);
    report.block = classify(&incumbent_block, &bound_block, 0.0, false);

    Ok(report)
}

/// The fidelity leg, for one layer, over its selected experts only.
fn oracle_layer(
    layer: usize,
    hidden: usize,
    moe: &MoeLayerWeights<'_>,
    h: &[f32],
    norm_offset: f32,
    eps: f32,
) -> Result<(), String> {
    let expert_input = moe_expert_input(h, moe, norm_offset, eps);
    let router_in = moe_router_input(h, &expert_input, moe, norm_offset, eps);
    let (selected, _) = moe_route_from_router_input(&router_in, moe);
    let inter = moe.intermediate_size;
    let inter_padded = moe.inter_padded();

    // Owned buffers, so the bound tensors have something to borrow.
    let mut buffers: Vec<(u32, Vec<u8>, Vec<u8>)> = Vec::new();
    for &e in &selected {
        let gate_up = dequantize_q4_k(
            moe.experts_gate_up[e],
            FUSED_PROJECTION_HALVES * inter * hidden,
        );
        let down = dequantize_q4_k(moe.experts_down[e], hidden * inter_padded);
        buffers.push((e as u32, f32_bytes(&gate_up), f32_bytes(&down)));
    }
    let experts: Vec<BoundExpert<'_>> = buffers
        .iter()
        .map(|(id, gate_up, down)| -> Result<BoundExpert<'_>, String> {
            Ok(BoundExpert {
                expert_id: *id,
                projection: BoundProjection::Fused {
                    gate_up: tensor(
                        &RegionRole::GateUpFused.name(),
                        gate_up,
                        RegionFormat::F32,
                        ComponentContract::matrix(
                            (FUSED_PROJECTION_HALVES * inter) as u32,
                            hidden as u32,
                        ),
                    )?,
                },
                down: sliced_tensor(
                    &RegionRole::Down.name(),
                    down,
                    RegionFormat::F32,
                    ComponentContract::matrix(hidden as u32, inter_padded as u32),
                    inter,
                )?,
            })
        })
        .collect::<Result<_, _>>()?;

    let router_bytes = f32_bytes(moe.router_proj);
    let scale_bytes = f32_bytes(moe.router_per_expert_scale);
    let reference = bind_operation(
        layer,
        hidden,
        moe,
        &router_bytes,
        &scale_bytes,
        experts,
        ExpertKernel::Reference,
    )?;
    reference
        .validate()
        .map_err(|e| format!("bind reference: {e}"))?;
    let (_, reference_trace) =
        execute_traced(&reference, MoeInputs::split(&expert_input, &router_in))
            .map_err(|e| format!("execute reference: {e}"))?;

    let incumbent_outputs = incumbent_expert_outputs(moe, &expert_input, &selected)?;
    let (_, incumbent_weights) = moe_route_from_router_input(&router_in, moe);
    let production = weighted_sum(&incumbent_outputs, &incumbent_weights, hidden);

    let diff = max_abs_diff(&reference_trace.reduced, &production);
    let scale = magnitude(&reference_trace.reduced);
    if scale < ORACLE_SCALE_FLOOR {
        println!(
            "\n== oracle, layer {layer} ==\n  max|Δ| = {diff:.3e}, reference magnitude \
             {scale:.3e} below the {ORACLE_SCALE_FLOOR:.0e} floor — absolute, not a ratio"
        );
        return Ok(());
    }
    let relative = diff / scale;
    println!(
        "\n== oracle, layer {layer} ==\n  quantisation fidelity, not parity: \
         max|Δ| = {diff:.3e} of {scale:.3e} = {:.2}%  {}",
        relative * 100.0,
        if relative <= ORACLE_RELATIVE_BAND {
            "within band"
        } else {
            "OUTSIDE"
        }
    );
    Ok(())
}

fn main() -> Result<(), String> {
    let vindex = arg(ARG_VINDEX).ok_or(format!("set {ARG_VINDEX} <path>"))?;
    let dump = arg(ARG_DUMP).ok_or(format!("set {ARG_DUMP} <dir>"))?;
    let shard: Option<usize> = arg(ARG_SHARD).and_then(|v| v.parse().ok());
    let oracle: Option<usize> = arg(ARG_ORACLE_LAYER).and_then(|v| v.parse().ok());

    println!("Locked-input MoE sweep — every layer fed the incumbent's own activation");
    println!("  vindex  {vindex}");
    match shard {
        Some(n) => println!("  shard   first {n} experts only — a residency-classification run"),
        None => println!("  shard   full population — the parity run"),
    }

    let mut callbacks = larql_vindex::SilentLoadCallbacks;
    let weights =
        larql_vindex::load_model_weights_kquant(std::path::Path::new(&vindex), &mut callbacks)
            .map_err(|e| format!("load weights: {e}"))?;
    let arch = &*weights.arch;
    let hidden = weights.hidden_size;
    let norm_offset = arch.norm_weight_offset();
    let eps = arch.norm_eps();
    let num_layers = weights.num_layers;

    println!("  layers  {num_layers}, hidden {hidden}\n");
    println!("  lyr  resident   top-k activation pad       verdict");

    let mut reports = Vec::new();
    let mut skipped = Vec::new();
    for layer in 0..num_layers {
        let Some(moe) = build_moe_weights(&weights, arch, layer) else {
            skipped.push((layer, "not an MoE layer".to_string()));
            continue;
        };
        let path = larql_compute::forward::dump_config::cpu_layer_h_post_attn_path(&dump, layer);
        let h = match last_row(&path, hidden) {
            Ok(h) => h,
            Err(e) => {
                skipped.push((layer, e));
                continue;
            }
        };
        let report = sweep_layer(layer, hidden, &moe, &h, norm_offset, eps, shard)?;
        report.print();
        reports.push(report);
    }

    // ── Summary ────────────────────────────────────────────────────────────
    println!("\n== classification ==");
    let mut counts: Vec<(Verdict, usize)> = Vec::new();
    for report in &reports {
        let verdict = report.verdict();
        match counts.iter_mut().find(|(v, _)| *v == verdict) {
            Some((_, n)) => *n += 1,
            None => counts.push((verdict, 1)),
        }
    }
    counts.sort_unstable_by_key(|(v, _)| *v);
    for (verdict, n) in &counts {
        println!("  {:<16} {n} layer(s)", verdict.name());
    }
    if !skipped.is_empty() {
        println!("\n== skipped ==");
        for (layer, why) in &skipped {
            println!("  layer {layer}: {why}");
        }
    }

    if let Some(layer) = oracle {
        let moe = build_moe_weights(&weights, arch, layer)
            .ok_or_else(|| format!("layer {layer} is not an MoE layer"))?;
        let path = larql_compute::forward::dump_config::cpu_layer_h_post_attn_path(&dump, layer);
        let h = last_row(&path, hidden)?;
        oracle_layer(layer, hidden, &moe, &h, norm_offset, eps)?;
    }

    println!("\n== notes ==");
    println!("  exact            every asserted checkpoint bit-identical — a contract");
    println!("  observed_exact   bit-identical but schedule-dependent — a measurement");
    println!("  equivalent       within a named tolerance");
    println!("  residency        selected expert not in the bound bank — NOT a parity failure");
    println!("  unsupported      no bound kernel serves this operand — a gap, not a defect");
    println!("  binding_defect   the binding is wrong");
    println!("  numeric_mismatch executed and disagreed");

    let indicted: Vec<usize> = reports
        .iter()
        .filter(|r| r.verdict().indicts_execution())
        .map(|r| r.layer)
        .collect();
    if indicted.is_empty() {
        println!("\nSWEEP: no layer indicts the execution.");
        Ok(())
    } else {
        Err(format!("layers {indicted:?} indict the execution"))
    }
}
