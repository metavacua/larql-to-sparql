//! Compare side of the Gate-B layer diff.
//!
//! Reports cosine similarity, max absolute difference and relative RMS error
//! per capture, and names the **first** layer to drift. First matters more
//! than worst: downstream layers inherit an upstream difference, so the deepest
//! disagreement is usually an echo and the shallowest is the defect.

#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;

use ndarray::Array2;

use super::dump::LayerDumpManifest;
#[cfg(not(target_arch = "wasm32"))]
use super::dump::{read_manifest, read_plane};
#[cfg(not(target_arch = "wasm32"))]
use super::LayerDiffArgs;

// `LayerDelta`/`compare_planes`/`check_comparable`/`first_token_mismatch`/
// `first_drift` below are the portable Gate-B math core (pure, no native
// markers); `diff_dumps`/`print_table`/`run_layer_diff` are native-gated
// (fs-backed dump.rs helpers + println!). See the round-1 gating survey
// for `commands/primary` for the full split rationale.
#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;
// `format!` isn't in `alloc_prelude` and this crate's `extern crate
// alloc;` (main.rs) isn't `#[macro_use]` -- import explicitly (see
// report: flagged as a crate-root gap out of this group's scope).
#[cfg(target_arch = "wasm32")]
use alloc::format;

/// Prefix of the machine-readable line, matching `shannon verify`'s.
#[cfg(not(target_arch = "wasm32"))]
const RESULT_PREFIX: &str = "RESULT ";

/// Per-capture agreement between two dumps.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerDelta {
    /// Capture index: 0 is the embedding output, `i + 1` is post-layer `i`.
    pub capture: usize,
    /// Cosine similarity over the flattened plane.
    pub cos: f64,
    /// Largest single-element absolute difference.
    pub max_abs: f64,
    /// `||a - b|| / ||b||` — scale-free, unlike `max_abs`.
    pub rel_rms: f64,
}

impl LayerDelta {
    /// Human label. Capture 0 is not a layer, and calling it "layer 0" has
    /// off-by-one'd every reader of a per-layer table at least once.
    pub fn label(&self) -> String {
        match self.capture {
            0 => "embed".to_string(),
            n => format!("layer {}", n - 1),
        }
    }
}

/// Cosine, max-abs and relative RMS between two equally-shaped planes.
pub fn compare_planes(a: &Array2<f32>, b: &Array2<f32>) -> LayerDelta {
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;
    let mut max_abs = 0.0f64;
    let mut sq_diff = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let (x, y) = (*x as f64, *y as f64);
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
        let d = (x - y).abs();
        if d > max_abs {
            max_abs = d;
        }
        sq_diff += (x - y) * (x - y);
    }
    // Two zero planes agree perfectly; one zero plane against a nonzero one
    // disagrees completely. Neither is a division.
    let denom = (norm_a.sqrt()) * (norm_b.sqrt());
    let cos = if denom > 0.0 {
        dot / denom
    } else if norm_a == norm_b {
        1.0
    } else {
        0.0
    };
    let rel_rms = if norm_b > 0.0 {
        sq_diff.sqrt() / norm_b.sqrt()
    } else {
        sq_diff.sqrt()
    };
    LayerDelta {
        capture: 0,
        cos,
        max_abs,
        rel_rms,
    }
}

/// Refuse to compare two dumps that did not run the same fixture.
///
/// Differing token ids is the failure §4.7.7 caught end-to-end after the fact:
/// two engines each measured correctly, over different text, and the delta was
/// read as an engine disagreement. Here it is an error before any arithmetic.
pub fn check_comparable(
    a: &LayerDumpManifest,
    b: &LayerDumpManifest,
) -> Result<(), Box<dyn core::error::Error>> {
    if a.token_ids != b.token_ids {
        return Err(format!(
            "dumps ran different token windows ({} vs {} tokens{}) — both sides \
             of a layer diff must score the same fixture",
            a.token_ids.len(),
            b.token_ids.len(),
            first_token_mismatch(&a.token_ids, &b.token_ids)
                .map(|i| format!(", first difference at index {i}"))
                .unwrap_or_default(),
        )
        .into());
    }
    if a.hidden_size != b.hidden_size {
        return Err(format!(
            "hidden_size differs: {} vs {}",
            a.hidden_size, b.hidden_size
        )
        .into());
    }
    if a.planes.len() != b.planes.len() {
        return Err(format!(
            "capture count differs: {} vs {} — the reference must hook the same \
             points (embedding output, then each layer output)",
            a.planes.len(),
            b.planes.len()
        )
        .into());
    }
    Ok(())
}

fn first_token_mismatch(a: &[u32], b: &[u32]) -> Option<usize> {
    a.iter().zip(b.iter()).position(|(x, y)| x != y)
}

/// Compare every capture in two dump directories, in capture order.
#[cfg(not(target_arch = "wasm32"))]
pub fn diff_dumps(
    dir_a: &Path,
    dir_b: &Path,
) -> Result<(LayerDumpManifest, LayerDumpManifest, Vec<LayerDelta>), Box<dyn std::error::Error>> {
    let man_a = read_manifest(dir_a)?;
    let man_b = read_manifest(dir_b)?;
    check_comparable(&man_a, &man_b)?;

    let mut deltas = Vec::with_capacity(man_a.planes.len());
    for (idx, (name_a, name_b)) in man_a.planes.iter().zip(man_b.planes.iter()).enumerate() {
        let plane_a = read_plane(&dir_a.join(name_a), man_a.seq_len, man_a.hidden_size)?;
        let plane_b = read_plane(&dir_b.join(name_b), man_b.seq_len, man_b.hidden_size)?;
        let mut delta = compare_planes(&plane_a, &plane_b);
        delta.capture = idx;
        deltas.push(delta);
    }
    Ok((man_a, man_b, deltas))
}

/// First capture whose cosine falls below `threshold_cos`.
pub fn first_drift(deltas: &[LayerDelta], threshold_cos: f64) -> Option<&LayerDelta> {
    deltas.iter().find(|d| d.cos < threshold_cos)
}

#[cfg(not(target_arch = "wasm32"))]
fn print_table(deltas: &[LayerDelta], threshold_cos: f64) {
    println!();
    println!(
        "{:<10}  {:>12}  {:>12}  {:>12}",
        "capture", "cos", "max_abs", "rel_rms"
    );
    println!("{}", "-".repeat(52));
    for d in deltas {
        let flag = if d.cos < threshold_cos {
            "  <-- drift"
        } else {
            ""
        };
        println!(
            "{:<10}  {:>12.9}  {:>12.6}  {:>12.6}{flag}",
            d.label(),
            d.cos,
            d.max_abs,
            d.rel_rms,
        );
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub fn run_layer_diff(args: LayerDiffArgs) -> Result<(), Box<dyn std::error::Error>> {
    let (man_a, man_b, deltas) = diff_dumps(&args.a, &args.b)?;
    println!(
        "a: {} ({})\nb: {} ({})\n{} tokens, hidden_size {}",
        man_a.engine,
        man_a.model,
        man_b.engine,
        man_b.model,
        man_a.token_ids.len(),
        man_a.hidden_size,
    );
    print_table(&deltas, args.threshold_cos);

    let drift = first_drift(&deltas, args.threshold_cos);
    println!();
    match drift {
        Some(d) => println!(
            "first drift: {} (cos {:.9} < {:.9})",
            d.label(),
            d.cos,
            args.threshold_cos
        ),
        None => println!("no capture below cos {:.9}", args.threshold_cos),
    }

    if args.json {
        println!(
            "{RESULT_PREFIX}{}",
            serde_json::json!({
                "engine_a": man_a.engine,
                "engine_b": man_b.engine,
                "tokens": man_a.token_ids.len(),
                "threshold_cos": args.threshold_cos,
                "first_drift_capture": drift.map(|d| d.capture),
                "first_drift_label": drift.map(|d| d.label()),
                "captures": deltas.iter().map(|d| serde_json::json!({
                    "capture": d.capture,
                    "label": d.label(),
                    "cos": d.cos,
                    "max_abs": d.max_abs,
                    "rel_rms": d.rel_rms,
                })).collect::<Vec<_>>(),
            })
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::primary::shannon_trace::dump::{
        plane_name, write_plane, LayerDumpManifest, MANIFEST_NAME, PLANE_DTYPE,
    };

    fn plane(values: &[f32]) -> Array2<f32> {
        Array2::from_shape_vec((1, values.len()), values.to_vec()).unwrap()
    }

    fn manifest(
        engine: &str,
        token_ids: Vec<u32>,
        hidden_size: usize,
        captures: usize,
    ) -> LayerDumpManifest {
        LayerDumpManifest {
            engine: engine.to_string(),
            model: "test/model".to_string(),
            num_layers: captures.saturating_sub(1),
            seq_len: 1,
            hidden_size,
            token_ids,
            planes: (0..captures).map(plane_name).collect(),
            dtype: PLANE_DTYPE.to_string(),
        }
    }

    fn write_dump(dir: &Path, man: &LayerDumpManifest, planes: &[Vec<f32>]) {
        std::fs::create_dir_all(dir).unwrap();
        for (name, values) in man.planes.iter().zip(planes.iter()) {
            write_plane(&dir.join(name), &plane(values)).unwrap();
        }
        std::fs::write(
            dir.join(MANIFEST_NAME),
            serde_json::to_string_pretty(man).unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn identical_planes_agree_exactly() {
        let a = plane(&[1.0, 2.0, 3.0]);
        let d = compare_planes(&a, &a);
        assert!((d.cos - 1.0).abs() < 1e-12);
        assert_eq!(d.max_abs, 0.0);
        assert_eq!(d.rel_rms, 0.0);
    }

    /// Cosine alone cannot see a pure rescale — a plane at half amplitude is
    /// cos 1.0 against its source. `rel_rms` is what catches it, which is why
    /// the table carries both.
    #[test]
    fn scaled_plane_is_cosine_identical_but_rms_different() {
        let a = plane(&[1.0, 2.0, 3.0]);
        let b = plane(&[0.5, 1.0, 1.5]);
        let d = compare_planes(&a, &b);
        assert!((d.cos - 1.0).abs() < 1e-12, "cos {}", d.cos);
        assert!((d.rel_rms - 1.0).abs() < 1e-9, "rel_rms {}", d.rel_rms);
    }

    #[test]
    fn opposed_planes_have_negative_cosine() {
        let d = compare_planes(&plane(&[1.0, 2.0]), &plane(&[-1.0, -2.0]));
        assert!((d.cos + 1.0).abs() < 1e-12, "cos {}", d.cos);
    }

    /// Two all-zero planes are equal, and a norm of zero must not become a
    /// division by zero or a NaN in the table.
    #[test]
    fn zero_planes_agree_without_dividing_by_zero() {
        let d = compare_planes(&plane(&[0.0, 0.0]), &plane(&[0.0, 0.0]));
        assert_eq!(d.cos, 1.0);
        assert_eq!(d.rel_rms, 0.0);
    }

    #[test]
    fn zero_against_nonzero_is_total_disagreement() {
        let d = compare_planes(&plane(&[0.0, 0.0]), &plane(&[1.0, 2.0]));
        assert_eq!(d.cos, 0.0);
        assert!(d.rel_rms > 0.0);
    }

    #[test]
    fn capture_zero_is_labelled_embed_not_layer_zero() {
        let mut d = compare_planes(&plane(&[1.0]), &plane(&[1.0]));
        d.capture = 0;
        assert_eq!(d.label(), "embed");
        d.capture = 1;
        assert_eq!(d.label(), "layer 0");
        d.capture = 17;
        assert_eq!(d.label(), "layer 16");
    }

    #[test]
    fn first_drift_reports_the_shallowest_not_the_worst() {
        let deltas = vec![
            LayerDelta {
                capture: 0,
                cos: 1.0,
                max_abs: 0.0,
                rel_rms: 0.0,
            },
            LayerDelta {
                capture: 1,
                cos: 0.99,
                max_abs: 1.0,
                rel_rms: 0.1,
            },
            LayerDelta {
                capture: 2,
                cos: 0.10,
                max_abs: 9.0,
                rel_rms: 0.9,
            },
        ];
        assert_eq!(first_drift(&deltas, 0.9999).unwrap().capture, 1);
        assert!(first_drift(&deltas[..1], 0.9999).is_none());
    }

    #[test]
    fn different_token_windows_are_refused_before_any_arithmetic() {
        let a = manifest("rust", vec![1, 2, 3], 4, 2);
        let b = manifest("hf", vec![1, 9, 3], 4, 2);
        let err = check_comparable(&a, &b).unwrap_err().to_string();
        assert!(err.contains("same fixture"), "unhelpful: {err}");
        assert!(err.contains("index 1"), "should locate the mismatch: {err}");
    }

    #[test]
    fn different_hidden_size_is_refused() {
        let a = manifest("rust", vec![1], 4, 2);
        let b = manifest("hf", vec![1], 8, 2);
        assert!(check_comparable(&a, &b)
            .unwrap_err()
            .to_string()
            .contains("hidden_size"));
    }

    #[test]
    fn different_capture_count_is_refused() {
        let a = manifest("rust", vec![1], 4, 3);
        let b = manifest("hf", vec![1], 4, 2);
        assert!(check_comparable(&a, &b)
            .unwrap_err()
            .to_string()
            .contains("capture count"));
    }

    #[test]
    fn diff_dumps_walks_both_directories_in_capture_order() {
        let root = tempfile::tempdir().unwrap();
        let (dir_a, dir_b) = (root.path().join("a"), root.path().join("b"));
        let man_a = manifest("rust", vec![7, 8], 2, 3);
        let man_b = manifest("hf", vec![7, 8], 2, 3);
        write_dump(
            &dir_a,
            &man_a,
            &[vec![1.0, 1.0], vec![1.0, 1.0], vec![1.0, 1.0]],
        );
        write_dump(
            &dir_b,
            &man_b,
            &[vec![1.0, 1.0], vec![1.0, 1.0], vec![-1.0, -1.0]],
        );

        let (a, b, deltas) = diff_dumps(&dir_a, &dir_b).unwrap();
        assert_eq!((a.engine.as_str(), b.engine.as_str()), ("rust", "hf"));
        assert_eq!(deltas.len(), 3);
        assert_eq!(
            deltas.iter().map(|d| d.capture).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert_eq!(first_drift(&deltas, 0.9999).unwrap().capture, 2);
    }

    #[test]
    fn run_layer_diff_reports_a_clean_pair() {
        let root = tempfile::tempdir().unwrap();
        let (dir_a, dir_b) = (root.path().join("a"), root.path().join("b"));
        let man = manifest("rust", vec![7], 2, 2);
        let mut man_b = man.clone();
        man_b.engine = "hf".to_string();
        write_dump(&dir_a, &man, &[vec![1.0, 2.0], vec![3.0, 4.0]]);
        write_dump(&dir_b, &man_b, &[vec![1.0, 2.0], vec![3.0, 4.0]]);

        run_layer_diff(LayerDiffArgs {
            a: dir_a,
            b: dir_b,
            threshold_cos: super::super::DEFAULT_DRIFT_COS,
            json: true,
        })
        .unwrap();
    }
}
