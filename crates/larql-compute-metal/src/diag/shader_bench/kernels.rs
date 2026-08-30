//! One bench per kernel family.
//!
//! Split out of `shader_bench.rs`; [`super`] composes the pieces.

#[allow(unused_imports)]
use super::*;
use crate::kernels::KernelHandle;
use crate::ops::q4_common::quantize_to_q8;
use crate::MetalBackend;
use larql_compute::cpu::ops::q4_common::{
    quantize_q4_0, quantize_q4_k, quantize_q4_kf, quantize_q6_k,
};
use larql_compute::cpu::ops::q8_matvec::quantize_weights_q8;
use metal::MTLSize;

pub(crate) fn bench_q4_0_matvec(metal: &MetalBackend, cfg: &Config, shape: Shape) -> BenchResult {
    let n = shape.hidden;
    let k = shape.hidden;
    let w = quantize_q4_0(&synth_f32(n * k, 0.21));
    let x = synth_f32(k, 0.31);
    let (q8_x, q8_scales) = quantize_to_q8(&x);
    let bufs = metal.bufs();
    let wb = bufs.uncached_bytes(&w);
    let xb = bufs.transient_from_i8(&q8_x);
    let sb = bufs.transient_from_f32(&q8_scales);
    let ob = bufs.output((n * 4) as u64);
    let kh = &metal.q4.matvec;
    let n_val = n as u32;
    let k_val = k as u32;
    let tgs = (n as u64).div_ceil(kh.rows_per_tg);

    measure_tiled(
        metal,
        cfg,
        "q4_matvec_v4",
        "q4-0-matvec",
        kh,
        format!("N={n} K={k}"),
        w.len() as u64 + q8_x.len() as u64 + (q8_scales.len() * 4) as u64,
        &ob,
        n,
        "checked",
        "Q4_0 x Q8 input matvec",
        |enc| {
            enc.set_compute_pipeline_state(&kh.state);
            enc.set_buffer(0, Some(&wb), 0);
            enc.set_buffer(1, Some(&xb), 0);
            enc.set_buffer(2, Some(&sb), 0);
            enc.set_buffer(3, Some(&ob), 0);
            enc.set_bytes(4, 4, &n_val as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(5, 4, &k_val as *const u32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(tgs, 1, 1),
                MTLSize::new(kh.threads_per_tg, 1, 1),
            );
        },
    )
}

pub(crate) fn bench_q8_matvec(metal: &MetalBackend, cfg: &Config, shape: Shape) -> BenchResult {
    let n = shape.hidden;
    let k = shape.hidden;
    let (w_q8, w_scales) = quantize_weights_q8(&synth_f32(n * k, 0.22), n, k);
    let x = synth_f32(k, 0.32);
    let (x_q8, x_scales) = quantize_to_q8(&x);
    let bufs = metal.bufs();
    let wb = bufs.transient_from_i8(&w_q8);
    let wsb = bufs.transient_from_f32(&w_scales);
    let xb = bufs.transient_from_i8(&x_q8);
    let xsb = bufs.transient_from_f32(&x_scales);
    let ob = bufs.output((n * 4) as u64);
    let kh = &metal.quant.q8_matvec_pipeline;
    let n_val = n as u32;
    let k_val = k as u32;
    let tgs = (n as u64).div_ceil(kh.rows_per_tg);

    measure_tiled(
        metal,
        cfg,
        "q8_matvec",
        "q8-matvec",
        kh,
        format!("N={n} K={k}"),
        w_q8.len() as u64 + (w_scales.len() * 4) as u64,
        &ob,
        n,
        "checked",
        "Q8_0 x Q8 input matvec",
        |enc| {
            enc.set_compute_pipeline_state(&kh.state);
            enc.set_buffer(0, Some(&wb), 0);
            enc.set_buffer(1, Some(&xb), 0);
            enc.set_buffer(2, Some(&wsb), 0);
            enc.set_buffer(3, Some(&xsb), 0);
            enc.set_buffer(4, Some(&ob), 0);
            enc.set_bytes(5, 4, &n_val as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(6, 4, &k_val as *const u32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(tgs, 1, 1),
                MTLSize::new(kh.threads_per_tg, 1, 1),
            );
        },
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn bench_qk_matvec(
    metal: &MetalBackend,
    cfg: &Config,
    shape: Shape,
    name: &'static str,
    family: &'static str,
    kh: &KernelHandle,
    w: &[u8],
    n: usize,
    k: usize,
    note: &'static str,
) -> BenchResult {
    let x = synth_f32(k, 0.41);
    let bufs = metal.bufs();
    let wb = bufs.uncached_bytes(w);
    let xb = bufs.transient_from_f32(&x);
    let ob = bufs.output((n * 4) as u64);
    let n_val = n as u32;
    let k_val = k as u32;
    let tgs = (n as u64).div_ceil(kh.rows_per_tg);

    measure_tiled(
        metal,
        cfg,
        name,
        family,
        kh,
        format!("{} N={n} K={k}", shape.label),
        w.len() as u64,
        &ob,
        n,
        "checked",
        note,
        |enc| {
            enc.set_compute_pipeline_state(&kh.state);
            enc.set_buffer(0, Some(&wb), 0);
            enc.set_buffer(1, Some(&xb), 0);
            enc.set_buffer(2, Some(&ob), 0);
            enc.set_bytes(3, 4, &n_val as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(4, 4, &k_val as *const u32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(tgs, 1, 1),
                MTLSize::new(kh.threads_per_tg, 1, 1),
            );
        },
    )
}

pub(crate) fn bench_gate_up_family(
    metal: &MetalBackend,
    cfg: &Config,
    shape: Shape,
) -> Vec<BenchResult> {
    let n = shape.inter;
    let k = shape.hidden;
    let gate_q4k = quantize_q4_k(&synth_f32(n * k, 0.51));
    let up_q4k = quantize_q4_k(&synth_f32(n * k, 0.52));
    let gate_q4kf = quantize_q4_kf(&synth_f32(n * k, 0.53));
    let up_q4kf = quantize_q4_kf(&synth_f32(n * k, 0.54));
    let mut out = Vec::new();
    for (name, kh, gate, up, sanity, note) in [
        (
            "q4k_ffn_gate_up",
            &metal.ffn.q4k_ffn_gate_up_pipeline,
            gate_q4k.as_slice(),
            up_q4k.as_slice(),
            "checked",
            "baseline Q4_K gate+up",
        ),
        (
            "q4k_ffn_gate_up_8sg",
            &metal.ffn.q4k_ffn_gate_up_8sg_pipeline,
            gate_q4k.as_slice(),
            up_q4k.as_slice(),
            "checked",
            "8-simdgroup Q4_K gate+up candidate/default path",
        ),
        (
            "q4k_ffn_gate_up_f16acc",
            &metal.ffn.q4k_ffn_gate_up_f16acc_pipeline,
            gate_q4k.as_slice(),
            up_q4k.as_slice(),
            "checked",
            "f16 accumulator candidate",
        ),
        (
            "q4k_ffn_gate_up_coop",
            &metal.ffn.q4k_ffn_gate_up_coop_pipeline,
            gate_q4k.as_slice(),
            up_q4k.as_slice(),
            "checked",
            "cooperative scale-load candidate",
        ),
        (
            "q4kf_ffn_gate_up",
            &metal.ffn.q4kf_ffn_gate_up_pipeline,
            gate_q4kf.as_slice(),
            up_q4kf.as_slice(),
            "layout-sensitive",
            "Q4_KF/GGUF-layout gate+up; synthetic Q4_KF may not exercise every row",
        ),
    ] {
        out.push(bench_gate_up(
            metal, cfg, shape, name, kh, gate, up, n, k, sanity, note,
        ));
    }
    out
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn bench_gate_up(
    metal: &MetalBackend,
    cfg: &Config,
    shape: Shape,
    name: &'static str,
    kh: &KernelHandle,
    gate: &[u8],
    up: &[u8],
    n: usize,
    k: usize,
    sanity: &'static str,
    note: &'static str,
) -> BenchResult {
    let x = synth_f32(k, 0.61);
    let bufs = metal.bufs();
    let gb = bufs.uncached_bytes(gate);
    let ub = bufs.uncached_bytes(up);
    let xb = bufs.transient_from_f32(&x);
    let go = bufs.output((n * 4) as u64);
    let uo = bufs.output((n * 4) as u64);
    let n_val = n as u32;
    let k_val = k as u32;
    let tgs = (n as u64).div_ceil(kh.rows_per_tg) * 2;

    measure_tiled(
        metal,
        cfg,
        name,
        "ffn-gate-up",
        kh,
        format!("{} N={n} K={k}", shape.label),
        (gate.len() + up.len()) as u64,
        &go,
        n,
        sanity,
        note,
        |enc| {
            enc.set_compute_pipeline_state(&kh.state);
            enc.set_buffer(0, Some(&gb), 0);
            enc.set_buffer(1, Some(&ub), 0);
            enc.set_buffer(2, Some(&xb), 0);
            enc.set_buffer(3, Some(&go), 0);
            enc.set_buffer(4, Some(&uo), 0);
            enc.set_bytes(5, 4, &n_val as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(6, 4, &k_val as *const u32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(tgs, 1, 1),
                MTLSize::new(kh.threads_per_tg, 1, 1),
            );
        },
    )
}

pub(crate) fn bench_geglu_down_family(
    metal: &MetalBackend,
    cfg: &Config,
    shape: Shape,
) -> Vec<BenchResult> {
    let n = shape.hidden;
    let k = shape.inter;
    let q4k_down = quantize_q4_k(&synth_f32(n * k, 0.71));
    let q6k_down = quantize_q6_k(&synth_f32(n * k, 0.72));
    let gate = synth_f32(k, 0.73);
    let up = synth_f32(k, 0.74);
    vec![
        bench_geglu_down(
            metal,
            cfg,
            shape,
            "q4k_geglu_silu_down",
            "ffn-down",
            &metal.ffn.q4k_geglu_silu_down_pipeline,
            &q4k_down,
            &gate,
            &up,
            "checked",
            "Q4_K fused SiLU GEGLU down",
        ),
        bench_geglu_down(
            metal,
            cfg,
            shape,
            "q4k_geglu_gelu_tanh_down",
            "ffn-down",
            &metal.ffn.q4k_geglu_gelu_tanh_down_pipeline,
            &q4k_down,
            &gate,
            &up,
            "checked",
            "Q4_K fused GELU-tanh GEGLU down",
        ),
        bench_geglu_down(
            metal,
            cfg,
            shape,
            "q6k_geglu_silu_down",
            "ffn-down",
            &metal.ffn.q6k_geglu_silu_down_pipeline,
            &q6k_down,
            &gate,
            &up,
            "checked",
            "Q6_K fused SiLU GEGLU down",
        ),
        bench_geglu_down(
            metal,
            cfg,
            shape,
            "q6k_geglu_gelu_tanh_down",
            "ffn-down",
            &metal.ffn.q6k_geglu_gelu_tanh_down_pipeline,
            &q6k_down,
            &gate,
            &up,
            "checked",
            "Q6_K fused GELU-tanh GEGLU down",
        ),
        bench_geglu_down(
            metal,
            cfg,
            shape,
            "q6k_geglu_gelu_tanh_down_cached",
            "ffn-down",
            &metal.ffn.q6k_geglu_gelu_tanh_down_cached_pipeline,
            &q6k_down,
            &gate,
            &up,
            "checked",
            "Q6_K cached-activation GELU-tanh GEGLU down",
        ),
    ]
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn bench_geglu_down(
    metal: &MetalBackend,
    cfg: &Config,
    shape: Shape,
    name: &'static str,
    family: &'static str,
    kh: &KernelHandle,
    weights: &[u8],
    gate: &[f32],
    up: &[f32],
    sanity: &'static str,
    note: &'static str,
) -> BenchResult {
    let n = shape.hidden;
    let k = shape.inter;
    let bufs = metal.bufs();
    let wb = bufs.uncached_bytes(weights);
    let gb = bufs.transient_from_f32(gate);
    let ub = bufs.transient_from_f32(up);
    let ob = bufs.output((n * 4) as u64);
    let n_val = n as u32;
    let k_val = k as u32;
    let tgs = (n as u64).div_ceil(kh.rows_per_tg);

    measure_tiled(
        metal,
        cfg,
        name,
        family,
        kh,
        format!("{} N={n} K={k}", shape.label),
        weights.len() as u64 + (gate.len() * 8) as u64,
        &ob,
        n,
        sanity,
        note,
        |enc| {
            enc.set_compute_pipeline_state(&kh.state);
            enc.set_buffer(0, Some(&wb), 0);
            enc.set_buffer(1, Some(&gb), 0);
            enc.set_buffer(2, Some(&ub), 0);
            enc.set_buffer(3, Some(&ob), 0);
            enc.set_bytes(4, 4, &n_val as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(5, 4, &k_val as *const u32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(tgs, 1, 1),
                MTLSize::new(kh.threads_per_tg, 1, 1),
            );
        },
    )
}

pub(crate) fn bench_qkv_family(
    metal: &MetalBackend,
    cfg: &Config,
    shape: Shape,
) -> Vec<BenchResult> {
    let q4_q = quantize_q4_k(&synth_f32(shape.q_rows * shape.hidden, 0.81));
    let q4_k = quantize_q4_k(&synth_f32(shape.kv_rows * shape.hidden, 0.82));
    let q4_v = quantize_q4_k(&synth_f32(shape.kv_rows * shape.hidden, 0.83));
    let q6_v = quantize_q6_k(&synth_f32(shape.kv_rows * shape.hidden, 0.84));
    let q4kf_q = quantize_q4_kf(&synth_f32(shape.q_rows * shape.hidden, 0.85));
    let q4kf_k = quantize_q4_kf(&synth_f32(shape.kv_rows * shape.hidden, 0.86));
    let q4kf_v = quantize_q4_kf(&synth_f32(shape.kv_rows * shape.hidden, 0.87));
    vec![
        bench_q4k_qkv(
            metal,
            cfg,
            shape,
            "q4k_qkv_proj",
            &metal.attention.q4k_qkv_proj_pipeline,
            &q4_q,
            &q4_k,
            &q4_v,
            "checked",
            "Q4_K fused QKV projection",
        ),
        bench_q4k_qkv(
            metal,
            cfg,
            shape,
            "q4kf_qkv_proj",
            &metal.attention.q4kf_qkv_proj_pipeline,
            &q4kf_q,
            &q4kf_k,
            &q4kf_v,
            "layout-sensitive",
            "Q4_KF/GGUF fused QKV projection; synthetic Q4_KF may not exercise every row",
        ),
        bench_q4k_q6k_qkv(
            metal,
            cfg,
            shape,
            "q4k_q6k_qkv_proj",
            &metal.attention.q4k_q6k_qkv_proj_pipeline,
            &q4_q,
            &q4_k,
            &q6_v,
            false,
            "checked",
            "mixed Q4_K Q/K + Q6_K V fused QKV projection",
        ),
        bench_q4k_q6k_qkv(
            metal,
            cfg,
            shape,
            "q4k_q6k_qkv_proj_normed",
            &metal.attention.q4k_q6k_qkv_proj_normed_pipeline,
            &q4_q,
            &q4_k,
            &q6_v,
            true,
            "checked",
            "mixed Q4_K/Q6_K fused QKV projection with RMS norm",
        ),
    ]
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn bench_q4k_qkv(
    metal: &MetalBackend,
    cfg: &Config,
    shape: Shape,
    name: &'static str,
    kh: &KernelHandle,
    wq: &[u8],
    wk: &[u8],
    wv: &[u8],
    sanity: &'static str,
    note: &'static str,
) -> BenchResult {
    let x = synth_f32(shape.hidden, 0.91);
    let bufs = metal.bufs();
    let wqb = bufs.uncached_bytes(wq);
    let wkb = bufs.uncached_bytes(wk);
    let wvb = bufs.uncached_bytes(wv);
    let xb = bufs.transient_from_f32(&x);
    let qb = bufs.output((shape.q_rows * 4) as u64);
    let kb = bufs.output((shape.kv_rows * 4) as u64);
    let vb = bufs.output((shape.kv_rows * 4) as u64);
    let q_rows = shape.q_rows as u32;
    let k_rows = shape.kv_rows as u32;
    let v_rows = shape.kv_rows as u32;
    let hidden = shape.hidden as u32;
    let tgs = ((shape.q_rows + 2 * shape.kv_rows) as u64).div_ceil(kh.rows_per_tg);

    measure_tiled(
        metal,
        cfg,
        name,
        "qkv",
        kh,
        format!(
            "{} Q={} K/V={} hidden={}",
            shape.label, shape.q_rows, shape.kv_rows, shape.hidden
        ),
        (wq.len() + wk.len() + wv.len()) as u64,
        &qb,
        shape.q_rows,
        sanity,
        note,
        |enc| {
            enc.set_compute_pipeline_state(&kh.state);
            enc.set_buffer(0, Some(&wqb), 0);
            enc.set_buffer(1, Some(&wkb), 0);
            enc.set_buffer(2, Some(&wvb), 0);
            enc.set_buffer(3, Some(&xb), 0);
            enc.set_buffer(4, Some(&qb), 0);
            enc.set_buffer(5, Some(&kb), 0);
            enc.set_buffer(6, Some(&vb), 0);
            enc.set_bytes(7, 4, &q_rows as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(8, 4, &k_rows as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(9, 4, &v_rows as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(10, 4, &hidden as *const u32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(tgs, 1, 1),
                MTLSize::new(kh.threads_per_tg, 1, 1),
            );
        },
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn bench_q4k_q6k_qkv(
    metal: &MetalBackend,
    cfg: &Config,
    shape: Shape,
    name: &'static str,
    kh: &KernelHandle,
    wq: &[u8],
    wk: &[u8],
    wv: &[u8],
    normed: bool,
    sanity: &'static str,
    note: &'static str,
) -> BenchResult {
    let x = synth_f32(shape.hidden, 0.92);
    let norm_w = vec![1.0f32; shape.hidden];
    let bufs = metal.bufs();
    let wqb = bufs.uncached_bytes(wq);
    let wkb = bufs.uncached_bytes(wk);
    let wvb = bufs.uncached_bytes(wv);
    let xb = bufs.transient_from_f32(&x);
    let nb = bufs.transient_from_f32(&norm_w);
    let qb = bufs.output((shape.q_rows * 4) as u64);
    let kb = bufs.output((shape.kv_rows * 4) as u64);
    let vb = bufs.output((shape.kv_rows * 4) as u64);
    let q_rows = shape.q_rows as u32;
    let k_rows = shape.kv_rows as u32;
    let v_rows = shape.kv_rows as u32;
    let hidden = shape.hidden as u32;
    let eps = larql_compute::RMSNORM_EPSILON_DEFAULT;
    let offset = 0.0f32;
    let tgs = ((shape.q_rows + 2 * shape.kv_rows) as u64).div_ceil(kh.rows_per_tg);

    measure_tiled(
        metal,
        cfg,
        name,
        "qkv",
        kh,
        format!(
            "{} Q={} K/V={} hidden={}",
            shape.label, shape.q_rows, shape.kv_rows, shape.hidden
        ),
        (wq.len() + wk.len() + wv.len()) as u64,
        &qb,
        shape.q_rows,
        sanity,
        note,
        |enc| {
            enc.set_compute_pipeline_state(&kh.state);
            enc.set_buffer(0, Some(&wqb), 0);
            enc.set_buffer(1, Some(&wkb), 0);
            enc.set_buffer(2, Some(&wvb), 0);
            enc.set_buffer(3, Some(&xb), 0);
            if normed {
                enc.set_buffer(4, Some(&nb), 0);
                enc.set_buffer(5, Some(&qb), 0);
                enc.set_buffer(6, Some(&kb), 0);
                enc.set_buffer(7, Some(&vb), 0);
                enc.set_bytes(8, 4, &q_rows as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(9, 4, &k_rows as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(10, 4, &v_rows as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(11, 4, &hidden as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(12, 4, &eps as *const f32 as *const std::ffi::c_void);
                enc.set_bytes(13, 4, &offset as *const f32 as *const std::ffi::c_void);
            } else {
                enc.set_buffer(4, Some(&qb), 0);
                enc.set_buffer(5, Some(&kb), 0);
                enc.set_buffer(6, Some(&vb), 0);
                enc.set_bytes(7, 4, &q_rows as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(8, 4, &k_rows as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(9, 4, &v_rows as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(10, 4, &hidden as *const u32 as *const std::ffi::c_void);
            }
            enc.dispatch_thread_groups(
                MTLSize::new(tgs, 1, 1),
                MTLSize::new(kh.threads_per_tg, 1, 1),
            );
        },
    )
}

pub(crate) fn bench_f32_gemv(metal: &MetalBackend, cfg: &Config, shape: Shape) -> BenchResult {
    let n = shape.lm_rows;
    let k = shape.hidden;
    let weights = synth_f32(n * k, 1.01);
    let x = synth_f32(k, 1.02);
    let bufs = metal.bufs();
    let wb = bufs.get_f32(&weights);
    let xb = bufs.transient_from_f32(&x);
    let ob = bufs.output((n * 4) as u64);
    let kh = &metal.f32_gemv_pipeline;
    let n_val = n as u32;
    let k_val = k as u32;
    let tgs = (n as u64).div_ceil(kh.rows_per_tg);

    measure_tiled(
        metal,
        cfg,
        "f32_gemv",
        "lm-head",
        kh,
        format!("{} N={n} K={k}", shape.label),
        (weights.len() * 4) as u64,
        &ob,
        n,
        "checked",
        "f32 row-per-simdgroup GEMV; Gemma3 profile caps N to avoid multi-GB synthetic allocation",
        |enc| {
            enc.set_compute_pipeline_state(&kh.state);
            enc.set_buffer(0, Some(&wb), 0);
            enc.set_buffer(1, Some(&xb), 0);
            enc.set_buffer(2, Some(&ob), 0);
            enc.set_bytes(3, 4, &n_val as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(4, 4, &k_val as *const u32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(tgs, 1, 1),
                MTLSize::new(kh.threads_per_tg, 1, 1),
            );
        },
    )
}
