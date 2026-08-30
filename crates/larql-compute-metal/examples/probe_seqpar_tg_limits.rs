//! What threadgroup width will Metal actually give the seqpar kernels?
//!
//! `seqpar_slices_for` bounds slices by `tg_partial`, but that is only one
//! of two ceilings. The other is per-pipeline:
//! `maxTotalThreadsPerThreadgroup` falls below Metal's nominal 1024 when a
//! kernel's register or threadgroup-memory pressure is high — and the
//! `_long` variants carry `tg_scores[4096]` (16 KB) on top of
//! `tg_partial`. Dispatching past it is a hard failure, not a slow path,
//! so the shipped bound has to come from the device rather than from
//! arithmetic on array sizes.

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("probe_seqpar_tg_limits requires macOS + Metal");
}

#[cfg(target_os = "macos")]
fn main() {
    use larql_compute_metal::MetalBackend;

    let Some(metal) = MetalBackend::new() else {
        eprintln!("no Metal device");
        return;
    };
    let a = &metal.attention;
    const HEAD_DIM: usize = 64;

    let rows: [(&str, &metal::ComputePipelineState); 6] = [
        ("kv_attention", &a.kv_attend_pipeline),
        ("kv_attention_long", &a.kv_attend_long_pipeline),
        ("kv_attention_seqpar", &a.kv_attend_seqpar_pipeline),
        (
            "kv_attention_seqpar_long",
            &a.kv_attend_seqpar_long_pipeline,
        ),
        ("kv_append_attend_fused", &a.kv_append_attend_fused_pipeline),
        (
            "kv_append_attend_fused_seqpar",
            &a.kv_append_attend_fused_seqpar_pipeline,
        ),
    ];

    println!(
        "  {:<32} {:>10} {:>12} {:>16}",
        "pipeline", "max thr", "tg mem (B)", "max slices @64"
    );
    println!("  {}", "-".repeat(74));
    for (name, p) in rows {
        let max_thr = p.max_total_threads_per_threadgroup();
        let tg_mem = p.static_threadgroup_memory_length();
        println!(
            "  {name:<32} {max_thr:>10} {tg_mem:>12} {:>16}",
            max_thr as usize / HEAD_DIM
        );
    }
}
