//! VINDEX3-Q2 rung 0 — is MXFP4's power-of-two scale the reason 4-bit
//! attention drifted, or is 4 bits simply too few?
//!
//! Q1 quantised Muse-Glimmer's attention to MXFP4 and the 6-token oracle
//! flipped the argmax; the passing preset kept attention wide. That
//! licenses "our OCP MXFP4 group-32 representation is too lossy for this
//! attention", and nothing broader. NVFP4 keeps the same E2M1 elements
//! and changes only the scale — group 16, E4M3 instead of E8M0, plus one
//! fp32 per tensor — so the two formats isolate the scale geometry from
//! the element width.
//!
//! This rung measures **weight reconstruction on the real tensors**,
//! before any kernel exists. It is a screening step, not the verdict:
//! reconstruction error is not a predictive unit, and the 6-token oracle
//! remains the thing that decides. But it is the cheapest test of the
//! stated mechanism, it runs on the actual weights rather than a
//! synthetic distribution, and if NVFP4 shows no advantage here the
//! hypothesis dies without anyone writing a Metal kernel.
//!
//! Both arms call the production codecs — `quantize_mxfp4` is the same
//! function Q1 executes with, `nvfp4::quantize` is the one Q2 will — so
//! this compares the shipped representations, not a model of them.
//!
//! Usage:
//! ```text
//! cargo run --release -p larql-vindex --example q2_scale_geometry -- \
//!     ~/chris-models/Muse-Glimmer-30B [max_tensors_per_role]
//! ```

use std::collections::BTreeMap;
use std::env;
use std::path::{Path, PathBuf};

use larql_models::quant::mxfp4::{dequantize_expert, MXFP4_GROUP_ELEMS};
use larql_models::quant::nvfp4::{self, NVFP4_GROUP_ELEMS};
use larql_vindex::format::vindex3::opplan::exec::weights::{quantize_mxfp4, LoadedWeight};

/// Matrix classes Q1's ladder judged separately, keyed by the **module**
/// a tensor belongs to rather than its bare suffix. Attention is the
/// class under test; the FFN is the control that already passed at 4
/// bits, and the head is where Q1's second rung failed.
///
/// Keying on the module matters: Muse-Glimmer's attention carries its own
/// `self_attn.gate_proj.weight` (the judged attention output gate), which
/// a bare `gate_proj.weight` suffix match files under the FFN. That is
/// exactly the confusion the per-class question exists to avoid.
const ROLES: &[(&str, &[&str])] = &[
    (
        "attention",
        &[
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.gate_proj.weight",
        ],
    ),
    (
        "ffn",
        &[
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ],
    ),
    ("head", &["lm_head.weight"]),
];

/// Relative RMS error: ‖ŵ − w‖ / ‖w‖, the unit Q1 reported drift in.
fn rel_rms(reference: &[f32], approx: &[f32]) -> f64 {
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (r, a) in reference.iter().zip(approx) {
        let d = (*r as f64) - (*a as f64);
        num += d * d;
        den += (*r as f64) * (*r as f64);
    }
    if den == 0.0 {
        0.0
    } else {
        (num / den).sqrt()
    }
}

/// Largest single-element error, in units of the tensor's own amax —
/// the clipping that a power-of-two scale forces shows up here first.
fn max_rel_error(reference: &[f32], approx: &[f32]) -> f64 {
    let amax = reference.iter().fold(0.0f32, |m, v| m.max(v.abs())) as f64;
    if amax == 0.0 {
        return 0.0;
    }
    reference
        .iter()
        .zip(approx)
        .map(|(r, a)| ((*r as f64) - (*a as f64)).abs())
        .fold(0.0f64, f64::max)
        / amax
}

/// **The control arm.** NVFP4 changes two things at once against
/// MXFP4 — the scale *format* (E8M0 → E4M3 + tensor scale) and the group
/// *size* (32 → 16) — and it costs 4.5 bpw against 4.25. Comparing only
/// those two confounds the mechanism with the bit budget: a win could
/// come entirely from the smaller group.
///
/// So this applies the OCP E8M0 rule at an arbitrary group size. At
/// group 16 it costs exactly what NVFP4 costs (4 + 8/16 = 4.5 bpw) and
/// differs from it *only* in the scale format, which is the claim under
/// test. At group 32 it must reproduce the shipped quantiser exactly,
/// and `main` asserts that before trusting the group-16 arm — a control
/// on the control.
fn e8m0_round_trip(values: &[f32], rows: usize, k: usize, group: usize) -> Vec<f32> {
    use larql_models::quant::fp4::{e2m1_to_f32, f32_to_e2m1};
    use larql_models::quant::mxfp4::e8m0_to_f32;
    /// Exponent of E2M1's largest magnitude, `floor(log2 6) = 2`.
    const EMAX: i32 = 2;

    let mut out = vec![0.0f32; rows * k];
    for (row, row_values) in values.chunks_exact(k).enumerate() {
        for (g, chunk) in row_values.chunks_exact(group).enumerate() {
            let amax = chunk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let scale_byte = if amax == 0.0 {
                0u8
            } else {
                ((amax.log2().floor() as i32 - EMAX) + 127).clamp(1, 254) as u8
            };
            let scale = e8m0_to_f32(scale_byte);
            let inv = if scale == 0.0 { 0.0 } else { scale.recip() };
            for (i, v) in chunk.iter().enumerate() {
                let code = f32_to_e2m1(v * inv);
                out[row * k + g * group + i] = scale * e2m1_to_f32(code);
            }
        }
    }
    out
}

fn mxfp4_round_trip(values: &[f32], rows: usize, k: usize) -> Vec<f32> {
    let LoadedWeight::Mxfp4 { packed, scales } =
        quantize_mxfp4(values, rows, k, "measure").expect("mxfp4 quantise")
    else {
        panic!("quantiser must produce the mxfp4 variant");
    };
    let groups = k / MXFP4_GROUP_ELEMS;
    dequantize_expert(
        &packed.as_slice()[..rows * groups * (MXFP4_GROUP_ELEMS / 2)],
        &scales.as_slice()[..rows * groups],
        rows,
        groups,
    )
    .expect("mxfp4 dequantise")
}

fn bf16_slice_to_f32(raw: &[u8]) -> Vec<f32> {
    raw.chunks_exact(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
        .collect()
}

fn shard_paths(dir: &Path) -> Vec<PathBuf> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
        .expect("read model dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    paths.sort();
    paths
}

fn role_of(name: &str) -> Option<&'static str> {
    ROLES
        .iter()
        .find_map(|(role, fragments)| fragments.iter().any(|f| name.ends_with(f)).then_some(*role))
}

/// One role's accumulated comparison across three arms.
#[derive(Default)]
struct RoleStats {
    tensors: usize,
    /// E8M0 group 32 — 4.25 bpw, what Q1 executed.
    mxfp4_rel_rms: Vec<f64>,
    /// E8M0 group 16 — 4.5 bpw, matched to NVFP4's budget.
    e8m0_16_rel_rms: Vec<f64>,
    /// E4M3 group 16 + tensor scale — 4.5 bpw.
    nvfp4_rel_rms: Vec<f64>,
    mxfp4_max_rel: Vec<f64>,
    e8m0_16_max_rel: Vec<f64>,
    nvfp4_max_rel: Vec<f64>,
}

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        0.0
    } else {
        xs.iter().sum::<f64>() / xs.len() as f64
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: {} <model_dir> [max_tensors_per_role]", args[0]);
        std::process::exit(2);
    }
    let dir = PathBuf::from(&args[1]);
    let cap: usize = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(usize::MAX);

    let mut stats: BTreeMap<&'static str, RoleStats> = BTreeMap::new();
    let mut control_checked = false;
    println!(
        "{:<52} {:>6} {:>10} {:>10} {:>10} {:>8}",
        "tensor", "rows", "mx@32", "e8m0@16", "nvfp4@16", "nv/e8m0"
    );

    for path in shard_paths(&dir) {
        let file = std::fs::File::open(&path).expect("open shard");
        // SAFETY: the shard is read-only for the lifetime of this process.
        let mmap = unsafe { memmap2::Mmap::map(&file) }.expect("mmap shard");
        let st = safetensors::SafeTensors::deserialize(&mmap).expect("parse shard");

        for (name, view) in st.tensors() {
            let Some(role) = role_of(&name) else { continue };
            let entry = stats.entry(role).or_default();
            if entry.tensors >= cap {
                continue;
            }
            let shape = view.shape();
            if shape.len() != 2 {
                continue;
            }
            let (rows, k) = (shape[0], shape[1]);
            // Both formats need whole groups; NVFP4's 16 divides MXFP4's
            // 32, so the MXFP4 constraint is the binding one.
            if !k.is_multiple_of(MXFP4_GROUP_ELEMS) || !k.is_multiple_of(NVFP4_GROUP_ELEMS) {
                eprintln!("skip {name}: k={k} not group-aligned");
                continue;
            }
            let values = bf16_slice_to_f32(view.data());
            if values.len() != rows * k {
                eprintln!("skip {name}: dtype is not bf16");
                continue;
            }

            let mx = mxfp4_round_trip(&values, rows, k);
            let e8 = e8m0_round_trip(&values, rows, k, NVFP4_GROUP_ELEMS);
            let nv = nvfp4::round_trip(&values, rows, k).expect("nvfp4 round trip");

            // Control on the control: the local E8M0 rule at group 32 must
            // reproduce the shipped quantiser bit for bit, or the group-16
            // arm is measuring my reimplementation rather than the format.
            if !control_checked {
                let mirror = e8m0_round_trip(&values, rows, k, MXFP4_GROUP_ELEMS);
                assert_eq!(
                    mirror, mx,
                    "the control's E8M0 rule must match the shipped quantiser at group 32"
                );
                eprintln!("control check: E8M0@32 reproduces quantize_mxfp4 exactly\n");
                control_checked = true;
            }

            let (mx_rms, e8_rms, nv_rms) = (
                rel_rms(&values, &mx),
                rel_rms(&values, &e8),
                rel_rms(&values, &nv),
            );
            println!(
                "{name:<52} {rows:>6} {mx_rms:>10.6} {e8_rms:>10.6} {nv_rms:>10.6} {:>8.3}",
                if nv_rms > 0.0 {
                    e8_rms / nv_rms
                } else {
                    f64::NAN
                }
            );

            entry.tensors += 1;
            entry.mxfp4_rel_rms.push(mx_rms);
            entry.e8m0_16_rel_rms.push(e8_rms);
            entry.nvfp4_rel_rms.push(nv_rms);
            entry.mxfp4_max_rel.push(max_rel_error(&values, &mx));
            entry.e8m0_16_max_rel.push(max_rel_error(&values, &e8));
            entry.nvfp4_max_rel.push(max_rel_error(&values, &nv));
        }
    }

    println!("\n{:=<104}", "");
    println!("mean relative RMS error, by matrix class");
    println!(
        "{:<12} {:>7} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "role", "tensors", "mx@32", "e8m0@16", "nvfp4@16", "group win", "format win"
    );
    for (role, s) in &stats {
        let (mx, e8, nv) = (
            mean(&s.mxfp4_rel_rms),
            mean(&s.e8m0_16_rel_rms),
            mean(&s.nvfp4_rel_rms),
        );
        println!(
            "{role:<12} {:>7} {mx:>10.6} {e8:>10.6} {nv:>10.6} {:>10.3} {:>10.3}",
            s.tensors,
            // Halving the group at the same scale format…
            if e8 > 0.0 { mx / e8 } else { f64::NAN },
            // …versus changing the scale format at the same group.
            if nv > 0.0 { e8 / nv } else { f64::NAN },
        );
    }

    println!("\nmean max element error, relative to each tensor's amax");
    println!(
        "{:<12} {:>10} {:>10} {:>10} {:>10}",
        "role", "mx@32", "e8m0@16", "nvfp4@16", "format win"
    );
    for (role, s) in &stats {
        let (mx, e8, nv) = (
            mean(&s.mxfp4_max_rel),
            mean(&s.e8m0_16_max_rel),
            mean(&s.nvfp4_max_rel),
        );
        println!(
            "{role:<12} {mx:>10.6} {e8:>10.6} {nv:>10.6} {:>10.3}",
            if nv > 0.0 { e8 / nv } else { f64::NAN }
        );
    }

    println!(
        "\nbits/weight: mx@32 {:.3}, e8m0@16 {:.3}, nvfp4@16 {:.3}",
        4.0 + 8.0 / MXFP4_GROUP_ELEMS as f64,
        4.0 + 8.0 / NVFP4_GROUP_ELEMS as f64,
        4.0 + 8.0 / NVFP4_GROUP_ELEMS as f64,
    );
    println!(
        "`group win` isolates group size at fixed scale format; `format win`\n\
         isolates scale format at fixed group and fixed bits/weight."
    );
}
