//! VINDEX3 bound-execution perf gate — the successor to
//! `vindex_storage_dispatch`.
//!
//! That bench asks whether an indirection (`&[u8]` → `Bytes` → `dyn`) costs
//! anything on the byte-fetch path. This one asks the question VINDEX3 raises
//! instead:
//!
//! > **Has resolution leaked into decode?**
//!
//! The whole load path — manifest lookup, capability traversal, alternative
//! selection, variant resolution, authority folding, contract checking —
//! happens once, before a `BoundMoeOperation` exists. If any of it stays
//! reachable from `execute`, per-token cost starts tracking *catalogue* size
//! rather than *work* size, and it does so quietly: the answers stay correct
//! and the model just gets slower as the index gets richer.
//!
//! So the gate is a scaling property, not a throughput number.
//!
//! # What "flat" does and does not mean here
//!
//! A first draft of this bench asserted per-token cost is *flat* in population.
//! That claim is wrong, and measuring it said so: routing legitimately scores
//! every expert before it can take a top-k, so an honest budget is
//!
//! ```text
//! O(population × hidden)      router scoring      unavoidable
//! O(top_k × expert work)      selected experts    must not track population
//! O(catalogue)                resolution          must not appear at all
//! ```
//!
//! The gate is therefore the *shape* of the growth. Scoring is a single
//! `[population, hidden]` matvec while each expert is three
//! `[intermediate, hidden]`-scale passes, so a 64× population increase should
//! buy a small constant factor — not 64×, and not superlinear. A resolution
//! leak, a per-token population scan over experts, or a revalidate call would
//! all bend the line well past that.
//!
//! Absolute times are not the point and are not comparable to the incumbent:
//! this is the reference decoder, which reads every operand element by element
//! on purpose.
//!
//! Run with: `cargo bench -p larql-vindex --bench vindex3_bound_execute`

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use larql_compute::{Activation, MoeTopKWeightPolicy};
use larql_vindex::format::capability::coordinate::BankCoordinate;
use larql_vindex::format::capability::{
    binding::RepresentationIdentity, component::ComponentContract,
};
use larql_vindex::format::lyrw2::region_format::RegionFormat;
use larql_vindex::format::lyrw2::region_role::RegionRole;
use larql_vindex::runtime::MoeInputs;
use larql_vindex::runtime::{
    execute, execute_traced, BoundBankOperation, BoundExpert, BoundExpertScaling,
    BoundMoeOperation, BoundProjection, BoundReduction, BoundRouter, BoundTensor, ExpertKernel,
    ProjectionArrangement, RouterKernel,
};

/// Populations spanning two orders of magnitude. 64× the experts should cost a
/// small constant factor — the router's scoring pass — and nothing more.
const POPULATIONS: [usize; 3] = [8, 64, 512];
const HIDDEN: usize = 64;
const INTERMEDIATE: usize = 48;
const TOP_K: usize = 4;
const VARIANT: &str = "bench";

/// Deterministic weights. Value content is irrelevant to timing; determinism
/// keeps run-to-run comparison meaningful.
fn weight(seed: usize, i: usize) -> f32 {
    (((seed * 31 + i * 17) % 199) as f32) / 199.0 - 0.5
}

fn bytes(count: usize, seed: usize) -> Vec<u8> {
    (0..count)
        .flat_map(|i| weight(seed, i).to_le_bytes())
        .collect()
}

/// Owns every operand for one bound operation.
struct Operands {
    gate: Vec<Vec<u8>>,
    up: Vec<Vec<u8>>,
    gate_up: Vec<Vec<u8>>,
    down: Vec<Vec<u8>>,
    router: Vec<u8>,
}

impl Operands {
    fn new(population: usize) -> Self {
        let proj = INTERMEDIATE * HIDDEN;
        Self {
            gate: (0..population).map(|e| bytes(proj, e)).collect(),
            up: (0..population).map(|e| bytes(proj, e + 1_000)).collect(),
            gate_up: (0..population)
                .map(|e| {
                    let mut v = bytes(proj, e);
                    v.extend(bytes(proj, e + 1_000));
                    v
                })
                .collect(),
            down: (0..population)
                .map(|e| bytes(HIDDEN * INTERMEDIATE, e + 2_000))
                .collect(),
            router: bytes(population * HIDDEN, 7),
        }
    }

    fn operation(&self, arrangement: ProjectionArrangement) -> BoundMoeOperation<'_> {
        let population = self.gate.len();
        let experts = (0..population)
            .map(|e| BoundExpert {
                expert_id: e as u32,
                projection: match arrangement {
                    ProjectionArrangement::Decomposed => BoundProjection::Decomposed {
                        gate: tensor(
                            &RegionRole::Gate.name(),
                            &self.gate[e],
                            INTERMEDIATE,
                            HIDDEN,
                        ),
                        up: tensor(&RegionRole::Up.name(), &self.up[e], INTERMEDIATE, HIDDEN),
                    },
                    ProjectionArrangement::Fused => BoundProjection::Fused {
                        gate_up: tensor(
                            &RegionRole::GateUpFused.name(),
                            &self.gate_up[e],
                            INTERMEDIATE * 2,
                            HIDDEN,
                        ),
                    },
                },
                down: tensor(
                    &RegionRole::Down.name(),
                    &self.down[e],
                    HIDDEN,
                    INTERMEDIATE,
                ),
            })
            .collect();

        BoundMoeOperation {
            router: BoundRouter {
                weight: tensor("router", &self.router, population, HIDDEN),
                top_k: TOP_K,
                selected_weight: MoeTopKWeightPolicy::RenormalizedSoftmax,
                scaling: BoundExpertScaling::None,
                kernel: RouterKernel::default(),
            },
            transforms: Vec::new(),
            banks: vec![BoundBankOperation {
                bank: BankCoordinate::new(0, 0),
                experts,
                intermediate_dim: INTERMEDIATE,
                hidden_dim: HIDDEN,
                activation: Activation::Silu,
                kernel: ExpertKernel::default(),
            }],
            reduction: BoundReduction::WeightedSum,
            residual_dim: HIDDEN,
        }
    }
}

fn tensor<'a>(region_set: &str, data: &'a [u8], rows: usize, cols: usize) -> BoundTensor<'a> {
    BoundTensor::direct(
        RepresentationIdentity::new(region_set, VARIANT),
        data,
        RegionFormat::F32,
        ComponentContract::matrix(rows as u32, cols as u32),
    )
    .expect("bench operands are well-formed")
}

fn residual() -> Vec<f32> {
    (0..HIDDEN).map(|i| weight(99, i)).collect()
}

/// The gate: growth across a 64× population must stay near the router term.
fn population_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("vindex3/execute_by_population");
    for population in POPULATIONS {
        let operands = Operands::new(population);
        let op = operands.operation(ProjectionArrangement::Decomposed);
        op.validate().expect("bench operation is well-formed");
        let input = residual();
        // One token per iteration. Expert work is identical across these
        // points; only the router's scoring pass grows.
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::from_parameter(population),
            &population,
            |b, _| {
                b.iter(|| black_box(execute(&op, MoeInputs::shared(black_box(&input))).unwrap()))
            },
        );
    }
    group.finish();
}

/// Fused and decomposed storage must cost about the same, as well as agreeing.
fn arrangement_parity(c: &mut Criterion) {
    let operands = Operands::new(POPULATIONS[1]);
    let mut group = c.benchmark_group("vindex3/execute_by_arrangement");
    for arrangement in ProjectionArrangement::ALL {
        let op = operands.operation(arrangement);
        let input = residual();
        group.bench_function(BenchmarkId::from_parameter(arrangement.name()), |b| {
            b.iter(|| black_box(execute(&op, MoeInputs::shared(black_box(&input))).unwrap()))
        });
    }
    group.finish();
}

/// The no-op sink must compile away; the collecting one is diagnostic-only.
fn trace_overhead(c: &mut Criterion) {
    let operands = Operands::new(POPULATIONS[1]);
    let op = operands.operation(ProjectionArrangement::Decomposed);
    let input = residual();
    let mut group = c.benchmark_group("vindex3/trace");
    group.bench_function("untraced", |b| {
        b.iter(|| black_box(execute(&op, MoeInputs::shared(black_box(&input))).unwrap()))
    });
    group.bench_function("traced", |b| {
        b.iter(|| black_box(execute_traced(&op, MoeInputs::shared(black_box(&input))).unwrap()))
    });
    group.finish();
}

criterion_group!(
    benches,
    population_scaling,
    arrangement_parity,
    trace_overhead
);
criterion_main!(benches);
