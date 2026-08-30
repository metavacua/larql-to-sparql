//! The declared kernel inventory and its coverage check.
//!
//! Split out of `shader_bench.rs`; [`super`] composes the pieces.

#[allow(unused_imports)]
use super::*;

pub(crate) fn inventory() -> &'static [InventoryItem] {
    &[
        InventoryItem {
            name: "sgemm",
            family: "dense",
            status: "inventory",
            note: "flat matmul; covered by Criterion matmul bench",
        },
        InventoryItem {
            name: "sgemm_transb",
            family: "dense",
            status: "inventory",
            note: "flat transposed matmul; covered by Criterion matmul bench",
        },
        InventoryItem {
            name: "q4_matvec_v4",
            family: "q4-0-matvec",
            status: "bench",
            note: "benchmarked here",
        },
        InventoryItem {
            name: "q8_matvec",
            family: "q8-matvec",
            status: "bench",
            note: "benchmarked here",
        },
        InventoryItem {
            name: "q4k_matvec",
            family: "q4k-matvec",
            status: "bench",
            note: "benchmarked here",
        },
        InventoryItem {
            name: "q4k_matvec_8sg",
            family: "q4k-matvec",
            status: "bench",
            note: "benchmarked here",
        },
        InventoryItem {
            name: "q4k_matvec_stride32",
            family: "q4k-matvec",
            status: "bench",
            note: "benchmarked here",
        },
        InventoryItem {
            name: "q6k_matvec",
            family: "q6k-matvec",
            status: "bench",
            note: "benchmarked here",
        },
        InventoryItem {
            name: "q6k_matvec_8sg",
            family: "q6k-matvec",
            status: "bench",
            note: "benchmarked here",
        },
        InventoryItem {
            name: "q4k_ffn_gate_up",
            family: "ffn-gate-up",
            status: "bench",
            note: "benchmarked here",
        },
        InventoryItem {
            name: "q4k_ffn_gate_up_8sg",
            family: "ffn-gate-up",
            status: "bench",
            note: "benchmarked here",
        },
        InventoryItem {
            name: "q4k_ffn_gate_up_f16acc",
            family: "ffn-gate-up",
            status: "bench",
            note: "benchmarked here",
        },
        InventoryItem {
            name: "q4k_ffn_gate_up_coop",
            family: "ffn-gate-up",
            status: "bench",
            note: "benchmarked here",
        },
        InventoryItem {
            name: "q4kf_ffn_gate_up",
            family: "ffn-gate-up",
            status: "bench",
            note: "benchmarked here",
        },
        InventoryItem {
            name: "q4k_geglu_silu_down",
            family: "ffn-down",
            status: "bench",
            note: "benchmarked here",
        },
        InventoryItem {
            name: "q4k_geglu_gelu_tanh_down",
            family: "ffn-down",
            status: "bench",
            note: "benchmarked here",
        },
        InventoryItem {
            name: "q6k_geglu_silu_down",
            family: "ffn-down",
            status: "bench",
            note: "benchmarked here",
        },
        InventoryItem {
            name: "q6k_geglu_gelu_tanh_down",
            family: "ffn-down",
            status: "bench",
            note: "benchmarked here",
        },
        InventoryItem {
            name: "q6k_geglu_gelu_tanh_down_cached",
            family: "ffn-down",
            status: "bench",
            note: "benchmarked here",
        },
        InventoryItem {
            name: "q4k_qkv_proj",
            family: "qkv",
            status: "bench",
            note: "benchmarked here",
        },
        InventoryItem {
            name: "q4kf_qkv_proj",
            family: "qkv",
            status: "bench",
            note: "benchmarked here",
        },
        InventoryItem {
            name: "q4k_q6k_qkv_proj",
            family: "qkv",
            status: "bench",
            note: "benchmarked here",
        },
        InventoryItem {
            name: "q4k_q6k_qkv_proj_normed",
            family: "qkv",
            status: "bench",
            note: "benchmarked here",
        },
        InventoryItem {
            name: "f32_gemv",
            family: "lm-head",
            status: "bench",
            note: "benchmarked here",
        },
        InventoryItem {
            name: "f16_gemv",
            family: "lm-head",
            status: "inventory",
            note: "requires synthetic half buffer; not timed in first pass",
        },
        InventoryItem {
            name: "rms_norm",
            family: "norm",
            status: "inventory",
            note: "flat reduction kernel; stage diagnostics cover decode use",
        },
        InventoryItem {
            name: "residual_add",
            family: "residual",
            status: "inventory",
            note: "flat elementwise kernel",
        },
        InventoryItem {
            name: "rms_norm_q8",
            family: "norm+quant",
            status: "inventory",
            note: "flat fused kernel; shape-sensitive q8 staging",
        },
        InventoryItem {
            name: "residual_norm",
            family: "norm",
            status: "inventory",
            note: "flat fused kernel",
        },
        InventoryItem {
            name: "residual_norm_q8",
            family: "norm+quant",
            status: "inventory",
            note: "flat fused kernel",
        },
        InventoryItem {
            name: "residual_norm_store",
            family: "norm",
            status: "inventory",
            note: "flat fused kernel",
        },
        InventoryItem {
            name: "qk_norm",
            family: "norm",
            status: "inventory",
            note: "head-shaped reduction kernel",
        },
        InventoryItem {
            name: "qk_norm_qk",
            family: "norm",
            status: "inventory",
            note: "Q/K paired norm kernel",
        },
        InventoryItem {
            name: "qk_norm_rope_fused",
            family: "attention",
            status: "inventory",
            note: "complex head-shaped fused kernel",
        },
        InventoryItem {
            name: "rope_at_pos",
            family: "rope",
            status: "inventory",
            note: "flat rope kernel",
        },
        InventoryItem {
            name: "rope_at_pos_batched",
            family: "rope",
            status: "inventory",
            note: "flat rope kernel",
        },
        InventoryItem {
            name: "rope_at_pos_batched_qk",
            family: "rope",
            status: "inventory",
            note: "flat Q/K rope kernel",
        },
        InventoryItem {
            name: "kv_attention",
            family: "attention",
            status: "inventory",
            note: "cache-shaped attention kernel",
        },
        InventoryItem {
            name: "kv_cache_append",
            family: "attention",
            status: "inventory",
            note: "cache-write kernel",
        },
        InventoryItem {
            name: "kv_append_attend_fused",
            family: "attention",
            status: "inventory",
            note: "cache-shaped fused attention kernel",
        },
        InventoryItem {
            name: "attn_fused",
            family: "attention",
            status: "inventory",
            note: "experimental fused attention kernel",
        },
        InventoryItem {
            name: "fused_attention",
            family: "attention",
            status: "inventory",
            note: "prefill/attention-shaped kernel",
        },
        InventoryItem {
            name: "post_attn_residual_norm_store",
            family: "norm",
            status: "inventory",
            note: "complex fused decode-stage kernel",
        },
        InventoryItem {
            name: "post_ffn_norm_residual_add",
            family: "norm",
            status: "inventory",
            note: "complex fused decode-stage kernel",
        },
        InventoryItem {
            name: "silu",
            family: "activation",
            status: "inventory",
            note: "flat activation kernel",
        },
        InventoryItem {
            name: "gelu_tanh",
            family: "activation",
            status: "inventory",
            note: "flat activation kernel",
        },
        InventoryItem {
            name: "geglu_silu",
            family: "activation",
            status: "inventory",
            note: "flat activation kernel",
        },
        InventoryItem {
            name: "geglu_gelu_tanh",
            family: "activation",
            status: "inventory",
            note: "flat activation kernel",
        },
        InventoryItem {
            name: "quantize_q8",
            family: "quant",
            status: "inventory",
            note: "flat quantization kernel",
        },
        InventoryItem {
            name: "layer_norm",
            family: "norm",
            status: "inventory",
            note: "LayerNorm reduction kernel",
        },
        InventoryItem {
            name: "layer_norm_no_bias",
            family: "norm",
            status: "inventory",
            note: "LayerNorm reduction kernel",
        },
        InventoryItem {
            name: "v_norm",
            family: "norm",
            status: "inventory",
            note: "V-norm reduction kernel",
        },
        InventoryItem {
            name: "v_norm_batched",
            family: "norm",
            status: "inventory",
            note: "batched V-norm reduction kernel",
        },
        InventoryItem {
            name: "scale_vector",
            family: "residual",
            status: "inventory",
            note: "flat scalar multiply kernel",
        },
        InventoryItem {
            name: "q4_vecmat",
            family: "q4",
            status: "inventory",
            note: "scatter/vector-matrix helper",
        },
        InventoryItem {
            name: "q4_f32_matvec",
            family: "q4",
            status: "inventory",
            note: "transposed f32-input helper",
        },
        InventoryItem {
            name: "q4_sparse_matvec",
            family: "q4",
            status: "inventory",
            note: "experimental sparse helper",
        },
        InventoryItem {
            name: "q4k_matmul",
            family: "q4k-matmul",
            status: "inventory",
            note: "covered by targeted matmul tests; not in decode hot path",
        },
        InventoryItem {
            name: "q8_qkv_proj",
            family: "qkv",
            status: "inventory",
            note: "Q8 fused QKV projection",
        },
        InventoryItem {
            name: "q8_proj_rope",
            family: "qkv",
            status: "inventory",
            note: "Q8 projection+rope helper",
        },
        InventoryItem {
            name: "f32_argmax_partial",
            family: "lm-head",
            status: "inventory",
            note: "partial reduction helper after f32_gemv",
        },
        InventoryItem {
            name: "f32_topk_partial",
            family: "lm-head",
            status: "inventory",
            note: "partial top-k helper after f32_gemv",
        },
        InventoryItem {
            name: "causal_attention",
            family: "attention",
            status: "inventory",
            note: "causal attention kernel",
        },
        InventoryItem {
            name: "turboquant_encode",
            family: "turboquant",
            status: "inventory",
            note: "KV compression utility",
        },
        InventoryItem {
            name: "turboquant_decode",
            family: "turboquant",
            status: "inventory",
            note: "KV decompression utility",
        },
        InventoryItem {
            name: "graph_walk_knn",
            family: "graph-walk",
            status: "inventory",
            note: "KNN graph walk utility",
        },
    ]
}

pub(crate) fn print_inventory() {
    let total = inventory().len();
    let benched = inventory().iter().filter(|i| i.status == "bench").count();
    println!("inventory: {total} shader functions ({benched} timed by this harness)");
    println!();
}

pub(crate) fn inventory_results(include_benched: bool) -> Vec<BenchResult> {
    inventory()
        .iter()
        .filter(|i| include_benched || i.status != "bench")
        .map(|i| BenchResult {
            name: i.name,
            family: i.family,
            status: i.status,
            shape: String::new(),
            rows_per_tg: None,
            threads_per_tg: None,
            bytes_per_call: 0,
            isolated_ms: None,
            isolated_sd_ms: None,
            batched_ms: None,
            batched_gbs: None,
            output_nonzero: None,
            sanity: inventory_sanity(i),
            note: i.note,
        })
        .collect()
}

pub(crate) fn inventory_sanity(i: &InventoryItem) -> &'static str {
    match i.name {
        "q4kf_ffn_gate_up" | "q4kf_qkv_proj" => "layout-sensitive",
        _ if i.status == "bench" => "timed-mode",
        _ => "not-timed",
    }
}

pub(crate) fn print_inventory_rows(results: &[BenchResult]) {
    println!(
        "{:<34} {:<14} {:<10} {:<16} Note",
        "Kernel", "Family", "Status", "Sanity"
    );
    println!("{}", "-".repeat(96));
    for r in results {
        println!(
            "{:<34} {:<14} {:<10} {:<16} {}",
            r.name, r.family, r.status, r.sanity, r.note
        );
    }
}
