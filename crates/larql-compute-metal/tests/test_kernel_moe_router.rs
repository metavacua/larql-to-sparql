#![cfg(target_os = "macos")]

//! Rung A of the GPU-dataflow routing ladder: router projection parity.
//!
//! GPU arm: `MetalBackend::moe_router_logits` (`shaders/moe_router.rs`).
//! CPU oracle: `larql_compute::cpu::ops::moe::moe_router_logits` — the
//! exact function production routing calls, not a re-implementation.
//!
//! Discipline (controls before parity): the FIRST test perturbs one
//! weight so an expert crosses the top-k boundary and requires the
//! instrument to (a) fail the logit comparison and (b) report the route
//! change — then restores the input and requires green. A parity gate
//! that has never failed proves nothing about the instrument.

#[path = "common/mod.rs"]
mod common;
use common::get_metal;

use larql_compute::cpu::ops::moe::moe_router_logits as cpu_router_logits;

/// gpt-oss-20b router shape: 32 experts over hidden 2880, top-4.
const GPTOSS_EXPERTS: usize = 32;
const GPTOSS_HIDDEN: usize = 2880;
const GPTOSS_TOP_K: usize = 4;

/// Parity bar for the logit comparison. CPU (serial sum) and GPU
/// (simd-tree, stride-32 lanes) accumulate in different orders, so
/// agreement is bounded by f32 rounding: ~sqrt(H)·eps ≈ 7e-6 relative
/// at H=2880. 1e-4 sits comfortably above that noise floor and far
/// below any perturbation that could move an expert across a top-k
/// boundary on realistic routers (the negative control verifies the
/// gap empirically).
const REL_TOL: f32 = 1e-4;

/// Deterministic router fixture. Magnitudes chosen so logits land
/// O(1)–O(10), the range real routers produce after their normed input.
fn synth_router(num_experts: usize, hidden: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let w: Vec<f32> = (0..num_experts * hidden)
        .map(|i| {
            let f = i as f32;
            ((f * 0.0003).sin() + 0.3 * (f * 0.0011).cos()) * 0.05
        })
        .collect();
    let bias: Vec<f32> = (0..num_experts)
        .map(|e| ((e as f32) * 0.7).sin() * 0.5)
        .collect();
    let x: Vec<f32> = (0..hidden)
        .map(|i| ((i as f32) * 0.013).sin() * 0.8)
        .collect();
    (w, bias, x)
}

/// Route under the ladder's tie contract: descending score, then
/// ascending expert id. (The contract itself is rung B's subject; here
/// it only derives routes from logits for the control's route-change
/// check.)
fn route_top_k(logits: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    idx.truncate(k);
    idx
}

/// Largest |cpu - gpu| relative to the largest |cpu| logit.
fn max_rel_diff(cpu: &[f32], gpu: &[f32]) -> f32 {
    assert_eq!(cpu.len(), gpu.len());
    let max_abs = cpu.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-9);
    common::max_diff(cpu, gpu) / max_abs
}

/// NEGATIVE CONTROL — run conceptually first. Perturb ONE weight so the
/// expert ranked k+1 crosses above the expert ranked k, and require the
/// instrument to fail loudly and name the route change; then restore the
/// weight and require green. Calibrates that REL_TOL sits between
/// accumulation noise and a route-moving difference.
#[test]
fn negative_control_boundary_crossing_fails_the_gate() {
    let metal = get_metal();
    let (w, bias, x) = synth_router(GPTOSS_EXPERTS, GPTOSS_HIDDEN);

    let cpu = cpu_router_logits(&x, &w, &bias, GPTOSS_EXPERTS);
    let baseline_route = route_top_k(&cpu, GPTOSS_TOP_K);

    // Experts at the k-th and (k+1)-th rank define the boundary.
    let full_rank = route_top_k(&cpu, GPTOSS_EXPERTS);
    let kth = full_rank[GPTOSS_TOP_K - 1];
    let boundary = full_rank[GPTOSS_TOP_K];

    // Lift `boundary` just past `kth` via a single weight element,
    // scaled through the input value it multiplies. 1.5× the logit gap
    // is decisively across, still ~O(gap) — not a sledgehammer.
    let gap = cpu[kth] - cpu[boundary];
    assert!(gap > 0.0, "fixture degenerate: no top-k boundary gap");
    let j = (0..GPTOSS_HIDDEN)
        .max_by(|&a, &b| x[a].abs().partial_cmp(&x[b].abs()).unwrap())
        .unwrap();
    let mut w_perturbed = w.clone();
    w_perturbed[boundary * GPTOSS_HIDDEN + j] += 1.5 * gap / x[j];

    let gpu_perturbed = metal
        .moe_router_logits(&w_perturbed, &bias, &x, GPTOSS_EXPERTS, GPTOSS_HIDDEN)
        .expect("GPU router dispatch");

    // (a) The comparator must detect the difference…
    let rel = max_rel_diff(&cpu, &gpu_perturbed);
    assert!(
        rel > REL_TOL,
        "instrument BLIND: a top-k-boundary-crossing perturbation moved \
         max rel diff only {rel:.3e}, inside the {REL_TOL:.0e} tolerance — \
         the parity gate below cannot be trusted"
    );

    // (b) …and the difference must be a ROUTE change, reported as one.
    let perturbed_route = route_top_k(&gpu_perturbed, GPTOSS_TOP_K);
    assert_ne!(
        baseline_route, perturbed_route,
        "perturbation crossed the k-th boundary but the derived route did \
         not change — boundary arithmetic is wrong"
    );
    assert!(
        perturbed_route.contains(&boundary),
        "expert {boundary} was lifted across the boundary but is absent \
         from the perturbed route {perturbed_route:?}"
    );

    // Restore: the unmodified input must pass the same gate.
    let gpu = metal
        .moe_router_logits(&w, &bias, &x, GPTOSS_EXPERTS, GPTOSS_HIDDEN)
        .expect("GPU router dispatch");
    let rel = max_rel_diff(&cpu, &gpu);
    assert!(
        rel < REL_TOL,
        "restored input fails the gate (rel={rel:.3e}) — the control \
         perturbation leaked into the baseline arm"
    );
}

/// Parity at the production shape, bias included. The route derived
/// from each arm's logits must agree exactly (rung A's informal
/// preview of rung B; the logit bar is the binding assertion).
#[test]
fn router_logits_parity_at_gptoss_shape() {
    let metal = get_metal();
    let (w, bias, x) = synth_router(GPTOSS_EXPERTS, GPTOSS_HIDDEN);

    let cpu = cpu_router_logits(&x, &w, &bias, GPTOSS_EXPERTS);
    let gpu = metal
        .moe_router_logits(&w, &bias, &x, GPTOSS_EXPERTS, GPTOSS_HIDDEN)
        .expect("GPU router dispatch");

    let rel = max_rel_diff(&cpu, &gpu);
    assert!(
        rel < REL_TOL,
        "CPU/GPU router logits diverge at gpt-oss shape: rel={rel:.3e}"
    );
    assert_eq!(
        route_top_k(&cpu, GPTOSS_TOP_K),
        route_top_k(&gpu, GPTOSS_TOP_K),
        "logits within tolerance but derived top-{GPTOSS_TOP_K} routes \
         differ — scores sit closer than the noise floor; rung B's \
         boundary-margin diagnostic applies"
    );
}

/// Awkward shapes are instruments: E not a multiple of ROWS_PER_TG (8),
/// H off the 128-element unroll and below one 32-lane stride, alternating
/// bias/no-bias. A geometry bug (dead simdgroups writing, tail-loop
/// misses, bias flag misread) shows here before it can hide inside a
/// round production shape.
#[test]
fn router_logits_parity_awkward_shapes() {
    let metal = get_metal();
    // (num_experts, hidden, with_bias)
    let shapes = [
        (30usize, 300usize, true), // E % 8 ≠ 0, H % 128 ≠ 0
        (33, 2883, false),         // one row past a full TG, odd H
        (8, 31, true),             // H below one 32-lane stride: tail loop only
        (129, 512, false),         // one row past 16 full TGs
    ];
    for (e, h, with_bias) in shapes {
        let (w, bias_full, x) = synth_router(e, h);
        let bias: &[f32] = if with_bias { &bias_full } else { &[] };

        let cpu = cpu_router_logits(&x, &w, bias, e);
        let gpu = metal
            .moe_router_logits(&w, bias, &x, e, h)
            .expect("GPU router dispatch");

        assert_eq!(gpu.len(), e, "output length at (E={e}, H={h})");
        let rel = max_rel_diff(&cpu, &gpu);
        assert!(
            rel < REL_TOL,
            "CPU/GPU router logits diverge at (E={e}, H={h}, bias={with_bias}): \
             rel={rel:.3e}"
        );
    }
}

/// Bias joins BEFORE selection — it changes which expert wins, and both
/// arms must agree on the flip. Constructed so the bias reverses the
/// no-bias argmax.
#[test]
fn bias_flips_selection_identically_on_both_arms() {
    let metal = get_metal();
    let e = 16usize;
    let h = 256usize;
    let (w, _, x) = synth_router(e, h);

    let cpu_plain = cpu_router_logits(&x, &w, &[], e);
    let top = route_top_k(&cpu_plain, 2);
    let (winner, runner_up) = (top[0], top[1]);

    // Bias hands the win to the runner-up, decisively.
    let mut bias = vec![0.0f32; e];
    bias[runner_up] = 2.0 * (cpu_plain[winner] - cpu_plain[runner_up]).abs() + 1.0;

    let cpu_biased = cpu_router_logits(&x, &w, &bias, e);
    let gpu_plain = metal
        .moe_router_logits(&w, &[], &x, e, h)
        .expect("GPU router dispatch");
    let gpu_biased = metal
        .moe_router_logits(&w, &bias, &x, e, h)
        .expect("GPU router dispatch");

    assert_eq!(
        route_top_k(&cpu_plain, 1),
        route_top_k(&gpu_plain, 1),
        "no-bias argmax disagrees before the flip is even in play"
    );
    assert_eq!(route_top_k(&cpu_biased, 1)[0], runner_up, "CPU flip");
    assert_eq!(route_top_k(&gpu_biased, 1)[0], runner_up, "GPU flip");
}

/// Shape-mismatch contract mirrors the gemv family: `None`, never a
/// silently wrong dispatch.
#[test]
fn rejects_mismatched_shapes() {
    let metal = get_metal();
    let (w, bias, x) = synth_router(8, 64);
    // wrong W length
    assert!(metal.moe_router_logits(&w[1..], &bias, &x, 8, 64).is_none());
    // wrong x length
    assert!(metal.moe_router_logits(&w, &bias, &x[1..], 8, 64).is_none());
    // wrong bias width (neither empty nor E)
    assert!(metal.moe_router_logits(&w, &bias[..3], &x, 8, 64).is_none());
    // zero experts
    assert!(metal.moe_router_logits(&[], &[], &x, 0, 64).is_none());
}
