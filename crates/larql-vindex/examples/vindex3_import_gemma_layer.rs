//! Container ladder **c8** — a real Gemma layer that *is* a VINDEX3 container.
//!
//! Every VINDEX3 result before this one bound its operands out of a VINDEX2
//! file. `vindex3_gemma_layer_parity` explains why that came first: with the
//! incumbent's own bytes on both sides, a mismatch could only be execution.
//! Execution is now closed, so this introduces the remaining variable —
//! re-extraction — and checks it the only way that means anything.
//!
//! ```text
//! VINDEX2 index on disk
//!         │
//!         ├─ per-expert gate_up / down byte ranges ──┐
//!         │                                          │  verbatim, no transcode
//!         ▼                                          ▼
//!   incumbent's bytes                    VINDEX3 container on disk
//!         └──────────── compare, byte for byte ───────┘
//! ```
//!
//! # What passing here does and does not license
//!
//! It licenses: *this layer's regions survive a round trip through the VINDEX3
//! container format unchanged, and the container that results is bindable.*
//!
//! It does **not** license "Gemma runs from VINDEX3". The execution comparison
//! is `vindex3_gemma_layer_parity`'s job and it runs over VINDEX2 bytes; this
//! proves the bytes are the same ones, which is what lets that result carry
//! over. Nor is it c9 — one layer, not all of them, and `extract` still writes
//! VINDEX2 (spec §12.1: new extractions default to VINDEX3 only once the ABI
//! freezes *and* the E0 preservation matrix passes).
//!
//! # Usage
//!
//! ```text
//! cargo run --release --example vindex3_import_gemma_layer -- <vindex-dir> [layer] [out-dir]
//! ```

use larql_vindex::format::vindex3::import::{import_one_layer, routed_storage_key, MoeLayerSource};
use larql_vindex::format::vindex3::{write_container, Vindex3Container};

use larql_compute::pipeline_layer::build_moe_weights;

const DEFAULT_LAYER: u32 = 0;

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let vindex = args
        .next()
        .ok_or("usage: vindex3_import_gemma_layer <vindex-dir> [layer] [out-dir]")?;
    let layer: u32 = args
        .next()
        .map(|s| s.parse().map_err(|_| "layer must be a number".to_string()))
        .transpose()?
        .unwrap_or(DEFAULT_LAYER);
    let out = args
        .next()
        .unwrap_or_else(|| format!("./vindex3-gemma-layer-{layer:03}"));

    println!("c8 — import one real Gemma layer into a VINDEX3 container");
    println!("  vindex  {vindex}");
    println!("  layer   {layer}");
    println!("  out     {out}");

    let mut callbacks = larql_vindex::SilentLoadCallbacks;
    let weights =
        larql_vindex::load_model_weights_kquant(std::path::Path::new(&vindex), &mut callbacks)
            .map_err(|e| format!("load weights: {e}"))?;
    let arch = &*weights.arch;
    let moe = build_moe_weights(&weights, arch, layer as usize)
        .ok_or_else(|| format!("layer {layer} is not a MoE layer in this model"))?;

    let hidden = weights.hidden_size as u32;
    let stored = moe.inter_padded() as u32;
    let semantic = moe.intermediate_size as u32;
    println!(
        "  shape   hidden {hidden}, {} experts, top-{}, intermediate {semantic} \
         (stored {stored})",
        moe.num_experts, moe.top_k
    );
    if stored != semantic {
        println!("          padded axis: the container records both, per §6.2/§8");
    }

    // The bytes the incumbent itself reads. Borrowed, not copied — the point
    // of the exercise is that nothing between here and disk transforms them.
    let source = MoeLayerSource {
        layer,
        experts_gate_up: moe.experts_gate_up.clone(),
        experts_down: moe.experts_down.clone(),
        format: region_format_for(moe.expert_data_format)?,
        hidden_size: hidden,
        // gate_up is never padded; only down is. See `MoeLayerSource`.
        gate_up_stored_intermediate: semantic,
        down_stored_intermediate: stored,
        semantic_intermediate: semantic,
        top_k: moe.top_k as u32,
    };

    let out_dir = std::path::Path::new(&out);
    let staging = out_dir.join(".staging.lyrw");
    std::fs::create_dir_all(out_dir).map_err(|e| format!("create {out}: {e}"))?;

    let spec = import_one_layer(
        &source,
        &format!("{}-layer-{layer:03}", weights.arch.family()),
        weights.arch.family(),
        weights.num_layers,
        &staging,
    )
    .map_err(|e| format!("import: {e}"))?;
    let _ = std::fs::remove_file(&staging);

    write_container(out_dir, &spec).map_err(|e| format!("write container: {e}"))?;
    println!("\n  wrote {} segment(s)", spec.segments.len());

    // Reopen from disk rather than trusting the in-memory spec: the claim is
    // about what a reader finds, not about what the writer intended.
    let container = Vindex3Container::open(out_dir).map_err(|e| format!("reopen: {e}"))?;
    let defects = container.verify();
    if !defects.is_empty() {
        for d in &defects {
            eprintln!("  DEFECT {d}");
        }
        return Err(format!("{} structural defect(s)", defects.len()));
    }
    println!("  verify  no structural defects");

    // The gate: every region, byte for byte, against the source it came from.
    let key = routed_storage_key(layer);
    let reader = container
        .segment(&key)
        .map_err(|e| format!("segment {key}: {e}"))?;
    let mut checked = 0usize;
    for expert in 0..moe.num_experts as u32 {
        for (role, expected) in [
            (
                larql_vindex::format::lyrw2::region_role::RegionRole::GateUpFused,
                moe.experts_gate_up[expert as usize],
            ),
            (
                larql_vindex::format::lyrw2::region_role::RegionRole::Down,
                moe.experts_down[expert as usize],
            ),
        ] {
            let got = reader
                .region_bytes(0, expert, role)
                .map_err(|e| format!("expert {expert} {role:?}: {e}"))?
                .ok_or_else(|| format!("expert {expert} {role:?} missing from the container"))?;
            if got != expected {
                return Err(format!(
                    "expert {expert} {role:?}: {} bytes on disk differ from the source's {}",
                    got.len(),
                    expected.len()
                ));
            }
            checked += 1;
        }
    }

    println!("  bytes   {checked} regions identical to the VINDEX2 source");
    println!("\nc8 PASS — layer {layer} is a VINDEX3 container, byte-identical to its source.");
    println!("Not yet c9 (all layers), and `extract` still writes VINDEX2.");
    Ok(())
}

/// Map the source's quantisation to the region format that describes it.
///
/// Thin wrapper over the importer's own mapping so both drivers agree by
/// construction — two copies of this table could drift, and a drift here is a
/// mislabelled region rather than a visible error.
fn region_format_for(
    q: larql_compute::QuantFormat,
) -> Result<larql_vindex::format::lyrw2::region_format::RegionFormat, String> {
    larql_vindex::format::vindex3::import::region_format_for(q).map_err(|e| e.to_string())
}
