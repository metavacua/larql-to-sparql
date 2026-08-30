//! Container ladder **c9** — every MoE layer, so the model *is* a VINDEX3
//! container rather than one layer of one being.
//!
//! c8 proved a real Gemma layer survives the round trip byte for byte. It left
//! two things open, and they are different in kind:
//!
//! ```text
//! quantitative  one layer -> thirty. A loop.
//! qualitative   thirty layers do not fit in RAM the way one did.
//! ```
//!
//! The second is why this is not `vindex3_import_gemma_layer` with a `for`
//! around it. `import_one_layer` returns a `ContainerSpec` that *owns* its
//! segment bytes; at ~421 MB per layer, describing thirty would hold ~12 GB of
//! already-mmapped weights a second time before writing any of them.
//! [`ContainerBuilder`] streams instead — each layer goes straight to its final
//! path and is forgotten — so peak memory is one expert's slice regardless of
//! model size.
//!
//! # What passing here does and does not license
//!
//! It licenses: *this model's routed layers exist as a VINDEX3 container, and
//! every region in it is byte-identical to the VINDEX2 source it came from.*
//!
//! It does **not** license "Gemma runs from VINDEX3". Execution still binds
//! over VINDEX2 bytes (`vindex3_gemma_layer_parity`); what c8 and c9 establish
//! is that those are the same bytes, which is what lets that result carry over.
//! The remaining rungs are the production expert kernel, full-layer residual
//! parity, and greedy token parity through `larql run` — and `extract` still
//! writes VINDEX2 (spec §12.1: new extractions default to VINDEX3 only once
//! the ABI freezes *and* the E0 preservation matrix passes).
//!
//! # Usage
//!
//! ```text
//! cargo run --release --example vindex3_import_gemma_all_layers -- <vindex-dir> [out-dir]
//! ```

use larql_vindex::format::lyrw2::region_layout::RegionLayout;
use larql_vindex::format::lyrw2::region_role::RegionRole;
use larql_vindex::format::vindex3::import::{region_format_for, routed_storage_key};
use larql_vindex::format::vindex3::{
    ContainerBuilder, ExpertScaleStreams, MoeLayerSource, Vindex3Container,
};

use larql_compute::pipeline_layer::build_moe_weights;

/// Bytes per GiB, for reporting only.
const GIB: f64 = (1024 * 1024 * 1024) as f64;

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let vindex = args
        .next()
        .ok_or("usage: vindex3_import_gemma_all_layers <vindex-dir> [out-dir]")?;
    let out = args.next().unwrap_or_else(|| "./vindex3-gemma".to_string());
    let out_dir = std::path::Path::new(&out);

    println!("c9 — import every MoE layer into one VINDEX3 container");
    println!("  vindex  {vindex}");
    println!("  out     {out}\n");

    let mut callbacks = larql_vindex::SilentLoadCallbacks;
    let weights =
        larql_vindex::load_model_weights_kquant(std::path::Path::new(&vindex), &mut callbacks)
            .map_err(|e| format!("load weights: {e}"))?;
    let arch = &*weights.arch;
    let num_layers = weights.num_layers;
    let hidden = weights.hidden_size as u32;

    // ── Write ────────────────────────────────────────────────────────────
    //
    // Non-MoE layers are skipped, not refused: a hybrid model's dense spine is
    // a legitimate part of the architecture, and this rung is about the routed
    // banks. `num_layers` below still records the model's real depth so a
    // reader can tell a routed-only container from a shallower model.
    let mut builder = ContainerBuilder::create(out_dir).map_err(|e| format!("create: {e}"))?;
    let mut imported: Vec<u32> = Vec::new();
    let mut skipped: Vec<u32> = Vec::new();

    for layer in 0..num_layers as u32 {
        let Some(moe) = build_moe_weights(&weights, arch, layer as usize) else {
            skipped.push(layer);
            continue;
        };
        let source = MoeLayerSource {
            layer,
            experts_gate_up: moe.experts_gate_up.clone(),
            experts_down: moe.experts_down.clone(),
            format: region_format_for(moe.expert_data_format)
                .map_err(|e| format!("layer {layer}: {e}"))?,
            // Gemma's expert banks are BF16 / k-quant — scales inline.
            scales: ExpertScaleStreams::Inline,
            // A fact about the **VINDEX2 store being read**, not about the
            // checkpoint: its per-layer writer emits `[all gate | all up]`.
            gate_up_layout: RegionLayout::ContiguousHalves,
            hidden_size: hidden,
            // gate_up is never padded; only down is. See `MoeLayerSource`.
            gate_up_stored_intermediate: moe.intermediate_size as u32,
            down_stored_intermediate: moe.inter_padded() as u32,
            semantic_intermediate: moe.intermediate_size as u32,
            top_k: moe.top_k as u32,
        };
        let size = builder
            .add_moe_layer(&source)
            .map_err(|e| format!("layer {layer}: {e}"))?;
        imported.push(layer);
        println!(
            "  layer {layer:>3}  {} experts, top-{}, {:.2} GiB  (running {:.2} GiB)",
            moe.num_experts,
            moe.top_k,
            size as f64 / GIB,
            builder.bytes_written() as f64 / GIB
        );
    }

    if imported.is_empty() {
        return Err("no MoE layers found — this model has no routed banks to import".into());
    }
    let total = builder.bytes_written();
    builder
        .finish(
            weights.arch.family(),
            weights.arch.family(),
            hidden as usize,
            num_layers,
        )
        .map_err(|e| format!("finish: {e}"))?;

    println!(
        "\n  wrote {} routed layer(s), {:.2} GiB; skipped {} non-MoE layer(s)",
        imported.len(),
        total as f64 / GIB,
        skipped.len()
    );

    // ── Verify ───────────────────────────────────────────────────────────
    //
    // Reopen from disk rather than trusting the builder: the claim is about
    // what a reader finds, not about what the writer intended.
    let container = Vindex3Container::open(out_dir).map_err(|e| format!("reopen: {e}"))?;
    let defects = container.verify();
    if !defects.is_empty() {
        for d in &defects {
            eprintln!("  DEFECT {d}");
        }
        return Err(format!("{} structural defect(s)", defects.len()));
    }
    println!("  verify  no structural defects");

    // ── The gate ─────────────────────────────────────────────────────────
    //
    // Every region of every imported layer, byte for byte, against the source
    // it came from. Re-deriving each layer's source slices here rather than
    // holding them from the write loop keeps the comparison honest: it reads
    // the incumbent afresh, exactly as a consumer of the VINDEX2 file would.
    let mut checked = 0usize;
    for &layer in &imported {
        let moe = build_moe_weights(&weights, arch, layer as usize)
            .ok_or_else(|| format!("layer {layer} stopped being MoE between passes"))?;
        let reader = container
            .segment(&routed_storage_key(layer))
            .map_err(|e| format!("layer {layer} segment: {e}"))?;

        for expert in 0..moe.num_experts as u32 {
            for (role, expected) in [
                (
                    RegionRole::GateUpFused,
                    moe.experts_gate_up[expert as usize],
                ),
                (RegionRole::Down, moe.experts_down[expert as usize]),
            ] {
                let got = reader
                    .region_bytes(0, expert, role)
                    .map_err(|e| format!("layer {layer} expert {expert} {role:?}: {e}"))?
                    .ok_or_else(|| {
                        format!("layer {layer} expert {expert} {role:?} missing from container")
                    })?;
                if got != expected {
                    return Err(format!(
                        "layer {layer} expert {expert} {role:?}: {} bytes on disk differ \
                         from the source's {}",
                        got.len(),
                        expected.len()
                    ));
                }
                checked += 1;
            }
        }
    }

    println!("  bytes   {checked} regions identical to the VINDEX2 source");
    println!(
        "\nc9 PASS — {} routed layers are one VINDEX3 container, byte-identical to source.",
        imported.len()
    );
    println!("Execution still binds VINDEX2 bytes; `extract` still writes VINDEX2.");
    Ok(())
}
