//! G5 attribution — where did the projected MXFP4 expert saving go?
//!
//! The composed serve measured Q6_K 14.56 ms/token against MXFP4 13.40
//! (GPU busy 11.56 vs 10.31), a −1.25 ms delta where bytes alone predict
//! more. Everything in the command buffer other than the expert matvecs
//! is identical between the arms, so the question is a kernel question:
//! at the production geometry, what bandwidth does each format's grouped
//! expert kernel realise?
//!
//! Both instruments already exist and share one protocol — grouped
//! dispatch, batched into one command buffer, cold working set:
//!
//! - Q6_K:  `kernel_profile::profile_grouped_experts` (the K3a bench).
//! - MXFP4: `mxfp4_layout::race` arm A (`split / LUT16`), which is the
//!   pipeline `grouped_experts_for(MXFP4)` serves in production
//!   (`Mxfp4Arm::SplitLut16` + `ExpertScaleBinding::SplitE8M0`).
//!
//! Geometry is gpt-oss-20b's routed bank: fused gate/up `[5760, 2880]`
//! and down `[2880, 2880]`, top-4 of 32 experts, 24 MoE layers per
//! token. The per-token prediction each arm implies is printed next to
//! the e2e numbers it must explain.
//!
//! Usage:
//!   cargo run --release -p larql-compute-metal --example bench_moe_expert_format_split

extern crate blas_src;

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("Requires macOS + Metal");
}

#[cfg(target_os = "macos")]
fn main() {
    use larql_compute_metal::diag::{kernel_profile, mxfp4_layout};

    const HIDDEN: usize = 2880;
    const INTER: usize = 2880;
    const GATE_UP_ROWS: usize = 2 * INTER; // fused halves
    const TOP_K: usize = 4;
    const LAYERS: usize = 24;
    const ROOFLINE: f64 = 367.0;
    const BATCH: usize = 24; // dispatches per command buffer, one per layer
    const WARMUP: usize = 3;
    const ITERS: usize = 20;

    const Q6K_BPW: f64 = 210.0 / 256.0 * 8.0; // super-block bytes over elems
    const MXFP4_BPW: f64 = 4.25; // payload nibbles + e8m0 partner stream

    println!("G5 attribution — grouped expert kernels at gpt-oss geometry");
    println!("============================================================");
    println!(
        "  gate_up [{GATE_UP_ROWS}, {HIDDEN}]  down [{HIDDEN}, {INTER}]  top-{TOP_K}, \
         {LAYERS} layers/token"
    );
    println!("  grouped dispatch, batched {BATCH}/cmdbuf, cold payloads");
    println!();

    // Per-token bytes each stage reads in each format.
    let weights_per_expert = |rows: usize, k: usize| (rows * k) as f64;
    let token_bytes = |bpw: f64| {
        (weights_per_expert(GATE_UP_ROWS, HIDDEN) + weights_per_expert(HIDDEN, INTER))
            * (bpw / 8.0)
            * (TOP_K * LAYERS) as f64
    };

    struct Stage {
        name: &'static str,
        n: usize,
        k: usize,
    }
    let stages = [
        Stage {
            name: "gate_up",
            n: GATE_UP_ROWS,
            k: HIDDEN,
        },
        Stage {
            name: "down",
            n: HIDDEN,
            k: INTER,
        },
    ];

    let mut q6k_token_ms = 0.0;
    let mut mxfp4_token_ms = 0.0;
    let mut mxfp4_vec_token_ms = 0.0;
    for s in &stages {
        let (_ungrouped, q6k_gbs) =
            kernel_profile::profile_grouped_experts(s.n, s.k, TOP_K, BATCH, WARMUP, ITERS);
        let arms = mxfp4_layout::race(s.n, s.k, TOP_K, BATCH, WARMUP, ITERS);
        // Arm A (scalar split) and arm A2 (vectorised split) — the
        // production pipeline is A2 with A as the alignment fallback.
        let split = &arms[0];
        let split_vec = &arms[1];
        assert!(
            split.name.contains("split") && split_vec.name.contains("vec"),
            "arm order changed; refusing to report the wrong kernel: {} / {}",
            split.name,
            split_vec.name
        );

        let stage_bytes =
            |bpw: f64| weights_per_expert(s.n, s.k) * (bpw / 8.0) * (TOP_K * LAYERS) as f64;
        let q6k_ms = stage_bytes(Q6K_BPW) / (q6k_gbs * 1e6);
        let mx_ms = stage_bytes(MXFP4_BPW) / (split.gbs * 1e6);
        let mxv_ms = stage_bytes(MXFP4_BPW) / (split_vec.gbs * 1e6);
        q6k_token_ms += q6k_ms;
        mxfp4_token_ms += mx_ms;
        mxfp4_vec_token_ms += mxv_ms;

        println!(
            "  {:<8} Q6_K       {:>6.1} GB/s  eta {:>5.2}   -> {:>6.3} ms/token",
            s.name,
            q6k_gbs,
            q6k_gbs / ROOFLINE,
            q6k_ms
        );
        for (label, arm, ms) in [("MXFP4 a", split, mx_ms), ("MXFP4 a2", split_vec, mxv_ms)] {
            println!(
                "  {:<8} {:<10} {:>6.1} GB/s  eta {:>5.2}   -> {:>6.3} ms/token   (oracle {:.2e})",
                s.name,
                label,
                arm.gbs,
                arm.eta(ROOFLINE),
                ms,
                arm.max_abs_diff
            );
        }
    }

    println!();
    println!("--- per-token expert read predictions ---");
    println!(
        "  Q6_K  reads {:>7.1} MB -> {:.3} ms    MXFP4 reads {:>7.1} MB -> a {:.3} / a2 {:.3} ms",
        token_bytes(Q6K_BPW) / 1e6,
        q6k_token_ms,
        token_bytes(MXFP4_BPW) / 1e6,
        mxfp4_token_ms,
        mxfp4_vec_token_ms,
    );
    println!(
        "  predicted deltas: a {:.3} ms (e2e 1.25), a2-over-a {:.3} ms (e2e 0.46, \
         GPU busy 11.56 -> 10.31 -> 9.93)",
        q6k_token_ms - mxfp4_token_ms,
        mxfp4_token_ms - mxfp4_vec_token_ms
    );
    println!();
    println!("  If the predicted delta lands near 1.25 ms, the kernels are already");
    println!("  realising their bandwidth and the remaining gap to MLX lives at the");
    println!("  token boundary (lm_head). If it is much larger, the MXFP4 kernel is");
    println!("  leaving bandwidth on the table under the real route.");
}
