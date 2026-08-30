//! Entry point: resolve a shape, run the selected benches.
//!
//! Split out of `shader_bench.rs`; [`super`] composes the pieces.

#[allow(unused_imports)]
use super::*;
use crate::MetalBackend;
use larql_compute::cpu::ops::q4_common::{quantize_q4_k, quantize_q6_k};

pub fn run(cfg: &Config) -> Result<Vec<BenchResult>, String> {
    let shape = Shape::for_profile(cfg.profile);

    println!("Metal shader bench");
    println!(
        "profile={} hidden={} inter={} q_rows={} kv_rows={} lm_rows={} layers={} warmup={} iters={}",
        shape.label,
        shape.hidden,
        shape.inter,
        shape.q_rows,
        shape.kv_rows,
        shape.lm_rows,
        cfg.n_layers,
        cfg.warmup,
        cfg.iters
    );
    println!();

    print_inventory();

    let mut results = inventory_results(cfg.inventory_only);
    if cfg.inventory_only {
        print_inventory_rows(&results);
        if let Some(path) = &cfg.json {
            std::fs::write(path, to_json(&results)).map_err(|e| format!("write json: {e}"))?;
            println!();
            println!("wrote {}", path.display());
        }
        return Ok(results);
    }

    let metal = MetalBackend::new().ok_or("Metal backend unavailable")?;

    results.extend(run_benches(&metal, cfg, shape));
    print_results(&results);

    if let Some(path) = &cfg.compare {
        let baseline = load_baseline(path)?;
        print_compare(&results, &baseline, path, cfg.threshold_pct);
    }

    if let Some(path) = &cfg.json {
        std::fs::write(path, to_json(&results)).map_err(|e| format!("write json: {e}"))?;
        println!();
        println!("wrote {}", path.display());
    }

    Ok(results)
}

pub(crate) fn run_benches(metal: &MetalBackend, cfg: &Config, shape: Shape) -> Vec<BenchResult> {
    let mut out = Vec::new();

    out.push(bench_q4_0_matvec(metal, cfg, shape));
    out.push(bench_q8_matvec(metal, cfg, shape));

    let q4k_w = quantize_q4_k(&synth_f32(shape.hidden * shape.hidden, 0.11));
    let q6k_w = quantize_q6_k(&synth_f32(shape.hidden * shape.inter, 0.12));
    out.push(bench_qk_matvec(
        metal,
        cfg,
        shape,
        "q4k_matvec_active",
        "q4k-matvec",
        &metal.quant.q4k_matvec_pipeline,
        &q4k_w,
        shape.hidden,
        shape.hidden,
        "active production Q4_K matvec handle after env selection",
    ));
    out.push(bench_qk_matvec(
        metal,
        cfg,
        shape,
        "q4k_matvec_4sg",
        "q4k-matvec",
        &metal.quant.q4k_matvec_4sg_pipeline,
        &q4k_w,
        shape.hidden,
        shape.hidden,
        "explicit 4-simdgroup Q4_K variant",
    ));
    out.push(bench_qk_matvec(
        metal,
        cfg,
        shape,
        "q4k_matvec_8sg",
        "q4k-matvec",
        &metal.quant.q4k_matvec_8sg_pipeline,
        &q4k_w,
        shape.hidden,
        shape.hidden,
        "explicit 8-simdgroup Q4_K variant",
    ));
    out.push(bench_qk_matvec(
        metal,
        cfg,
        shape,
        "q4k_matvec_stride32",
        "q4k-matvec",
        &metal.quant.q4k_matvec_stride32_pipeline,
        &q4k_w,
        shape.hidden,
        shape.hidden,
        "LM-head correctness variant at hidden-square shape",
    ));
    out.push(bench_qk_matvec(
        metal,
        cfg,
        shape,
        "q6k_matvec_active",
        "q6k-matvec",
        &metal.quant.q6k_matvec_pipeline,
        &q6k_w,
        shape.hidden,
        shape.inter,
        "active production Q6_K matvec handle after env selection",
    ));
    out.push(bench_qk_matvec(
        metal,
        cfg,
        shape,
        "q6k_matvec_4sg",
        "q6k-matvec",
        &metal.quant.q6k_matvec_4sg_pipeline,
        &q6k_w,
        shape.hidden,
        shape.inter,
        "explicit 4-simdgroup Q6_K variant",
    ));
    out.push(bench_qk_matvec(
        metal,
        cfg,
        shape,
        "q6k_matvec_8sg",
        "q6k-matvec",
        &metal.quant.q6k_matvec_8sg_pipeline,
        &q6k_w,
        shape.hidden,
        shape.inter,
        "explicit 8-simdgroup Q6_K variant",
    ));

    out.extend(bench_gate_up_family(metal, cfg, shape));
    out.extend(bench_geglu_down_family(metal, cfg, shape));
    out.extend(bench_qkv_family(metal, cfg, shape));
    out.push(bench_f32_gemv(metal, cfg, shape));

    out
}
