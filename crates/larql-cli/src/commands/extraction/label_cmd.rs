//! `larql label` — produce `feature_labels.json` for a vindex.
//!
//! Given a vindex, a relation catalog, and the source model, this command:
//!   1. captures the residual stream for each relation's subject prompts,
//!   2. routes each subject's residual through the vindex gate (signed KNN),
//!   3. frame-subtracts to keep subject-specific features and matches each to
//!      its object by intersecting the object's token-IDs with the feature's
//!      `down_meta` top-K token-IDs,
//!   4. writes `feature_labels.json` (the per-feature relation labels DESCRIBE
//!      reads).
//!
//! The pure routing/frame-subtraction/match/write logic lives in
//! `larql_vindex::label` (already tested). This shell only loads the model,
//! captures residuals one relation at a time (bounded memory), and writes.

use std::collections::HashMap;
use std::path::PathBuf;

use clap::Args;
use larql_inference::{CaptureCallbacks, CaptureConfig, InferenceModel};
use larql_vindex::label::catalog::Catalog;
use larql_vindex::label::run::{label_catalog, load_subject_residuals};
use larql_vindex::label::writer::write_feature_labels;
use larql_vindex::{SilentLoadCallbacks, VectorIndex};
use ndarray::Array1;

/// No-op capture progress sink (the inference crate's `SilentCallbacks` is
/// not re-exported at the crate root, so use a local no-op of the trait).
struct SilentCaptureCallbacks;
impl CaptureCallbacks for SilentCaptureCallbacks {}

#[derive(Args)]
pub struct LabelArgs {
    /// Vindex directory to label (its gate + down_meta drive the matching).
    #[arg(long)]
    vindex: PathBuf,

    /// Relation catalog JSON: { "<rel>": { pid, template, pairs:[[subj,obj]…] } }.
    #[arg(long)]
    catalog: PathBuf,

    /// Source model (path or HuggingFace ID) used to capture residuals.
    #[arg(long)]
    model: String,

    /// Signed top-k gate features kept per layer when routing a residual.
    #[arg(long, default_value_t = 20)]
    per_layer_k: usize,

    /// A feature routed by more than this fraction of a relation's subjects is
    /// treated as relation frame and excluded (frame-subtraction).
    #[arg(long, default_value_t = 0.5)]
    frame_frac: f32,

    /// Output directory for `feature_labels.json`. Defaults to the vindex dir.
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Capture + route over ALL layers (`0..num_layers`) instead of the
    /// vindex's knowledge band. For the full-depth band ablation comparison.
    #[arg(long, conflicts_with = "band")]
    all_layers: bool,

    /// Explicit inclusive layer band `LO-HI` (e.g. `14-27`) to capture + route
    /// over, overriding the vindex's declared knowledge band.
    #[arg(long, conflicts_with = "all_layers")]
    band: Option<String>,
}

/// Decide which residual layers the producer captures + routes over, and a
/// human description for the log line.
///
/// Precedence (clap makes `--all-layers` and `--band` mutually exclusive):
///   1. `--all-layers` → full depth `0..num_layers`.
///   2. `--band LO-HI` → that explicit inclusive range.
///   3. the vindex's `index.json` `layer_bands.knowledge` `[lo, hi]` (default).
///   4. absent / unparseable `layer_bands.knowledge` → full depth, with a note.
///
/// Restricting CAPTURE to the band restricts routing automatically: routing
/// iterates whatever residual layers exist, so `label_catalog`/`routed_features`
/// need no change — features outside the captured layers simply never appear.
fn resolve_capture_layers(
    index_json: &serde_json::Value,
    num_layers: usize,
    all_layers: bool,
    band: Option<(usize, usize)>,
) -> (Vec<usize>, String) {
    let all = || ((0..num_layers).collect::<Vec<_>>(), format!("all {num_layers} layers"));

    if all_layers {
        return all();
    }
    if let Some((lo, hi)) = band {
        let hi = hi.min(num_layers.saturating_sub(1));
        return (
            (lo..=hi).collect(),
            format!("{lo}..={hi} (explicit --band)"),
        );
    }
    let knowledge = index_json
        .get("layer_bands")
        .and_then(|b| b.get("knowledge"))
        .and_then(|k| k.as_array());
    if let Some(arr) = knowledge {
        if let (Some(lo), Some(hi)) = (
            arr.first().and_then(|v| v.as_u64()),
            arr.get(1).and_then(|v| v.as_u64()),
        ) {
            let lo = lo as usize;
            let hi = (hi as usize).min(num_layers.saturating_sub(1));
            return (
                (lo..=hi).collect(),
                format!("{lo}..={hi} (knowledge band)"),
            );
        }
    }
    let (layers, desc) = all();
    (
        layers,
        format!("{desc} (no layer_bands.knowledge in index.json)"),
    )
}

/// Parse an inclusive `"LO-HI"` band string (e.g. `"14-27"`).
fn parse_band(s: &str) -> Result<(usize, usize), String> {
    let (lo, hi) = s
        .split_once('-')
        .ok_or_else(|| format!("--band expects LO-HI, got '{s}'"))?;
    let lo: usize = lo
        .trim()
        .parse()
        .map_err(|_| format!("--band LO not a number: '{lo}'"))?;
    let hi: usize = hi
        .trim()
        .parse()
        .map_err(|_| format!("--band HI not a number: '{hi}'"))?;
    if lo > hi {
        return Err(format!("--band LO {lo} > HI {hi}"));
    }
    Ok((lo, hi))
}

pub fn run(args: LabelArgs) -> Result<(), Box<dyn std::error::Error>> {
    // 1. Catalog.
    let catalog_str = std::fs::read_to_string(&args.catalog)?;
    let catalog = Catalog::from_json_str(&catalog_str)?;

    // 2. Vindex (gate KNN + feature_meta only — no FFN walk tensors needed).
    eprintln!("Loading vindex: {}", args.vindex.display());
    let mut load_cb = SilentLoadCallbacks;
    let index = VectorIndex::load_vindex(&args.vindex, &mut load_cb)?;

    // 3. Hollow-vindex refusal: if no sampled feature has a down_meta top
    //    token, matching can never fire — refuse before doing model work.
    if !has_any_down_meta_token(&index) {
        return Err(
            "vindex has no down_meta tokens — cannot label; re-extract with metadata".into(),
        );
    }

    // 4. Model.
    eprintln!("Loading model: {}", args.model);
    let model = InferenceModel::load(&args.model)?;

    // 4b. Decide which residual layers to capture + route over. Default is the
    //     vindex's knowledge band (where the working reference + DESCRIBE
    //     consumer live); --all-layers / --band override (band ablation).
    let index_json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(args.vindex.join("index.json"))?)
            .unwrap_or(serde_json::Value::Null);
    let band = match args.band.as_deref() {
        Some(s) => Some(parse_band(s)?),
        None => None,
    };
    let (layers, band_desc) =
        resolve_capture_layers(&index_json, model.num_layers(), args.all_layers, band);
    eprintln!("routing layers: {band_desc}");

    // 5. Capture residuals one relation at a time. Each relation's temp FILE
    //    is dropped after its residuals are loaded, so on-disk capture never
    //    accumulates; the in-RAM `residuals` map, however, retains every
    //    relation's vectors (bounded by catalog size, not by # of relations
    //    held on disk at once).
    //    Residuals are keyed by (relation, subject): a subject's last-token
    //    residual is relation-prompt-specific, so a subject shared across
    //    relations must keep one residual per relation (not be overwritten).
    let mut residuals: HashMap<(String, String), HashMap<usize, Array1<f32>>> = HashMap::new();
    for (rel_name, relation) in catalog.iter() {
        let mut subjects: Vec<String> = relation.pairs.iter().map(|(s, _)| s.clone()).collect();
        subjects.sort();
        subjects.dedup();
        if subjects.is_empty() {
            continue;
        }
        eprintln!(
            "  capturing relation '{rel_name}' ({} subjects)...",
            subjects.len()
        );

        let tmp = tempfile::tempdir()?;
        let config = CaptureConfig {
            layers: layers.clone(),
            prompt_template: Some(relation.template.clone()),
            ..Default::default()
        };
        let mut cb = SilentCaptureCallbacks;
        model.capture(&subjects, &config, tmp.path(), &mut cb)?;

        let captured = load_subject_residuals(&tmp.path().join("residuals.vectors.jsonl"))?;
        for (subject, by_layer) in captured {
            residuals.insert((rel_name.clone(), subject), by_layer);
        }
        // `tmp` drops here → relation's residual file is removed.
    }

    // 6. Label. The matching keys object token-IDs against each feature's
    //    top-K down token-IDs, so the producer must tokenize objects with the
    //    SAME tokenizer the model used. The leading-space BPE variant matters
    //    (most subword tokenizers encode " Paris" differently from "Paris"),
    //    so both variants' ids are unioned.
    let tk = model.tokenizer();
    let tokenize = |s: &str| -> std::collections::HashSet<u32> {
        let mut ids = std::collections::HashSet::new();
        if let Ok(enc) = tk.encode(s, false) {
            ids.extend(enc.get_ids().iter().copied());
        }
        if let Ok(enc) = tk.encode(format!(" {s}"), false) {
            ids.extend(enc.get_ids().iter().copied());
        }
        ids
    };
    let labels = label_catalog(
        &index,
        &catalog,
        &residuals,
        args.per_layer_k,
        args.frame_frac,
        &tokenize,
    );

    // 7. Write.
    if labels.is_empty() {
        eprintln!(
            "warning: 0 feature labels produced — catalog had no matches; \
             feature_labels.json will be empty"
        );
    }
    let output_dir = args.output.unwrap_or(args.vindex);
    write_feature_labels(&output_dir, &labels)?;
    println!(
        "{} feature labels written to {}",
        labels.len(),
        output_dir.join("feature_labels.json").display()
    );

    Ok(())
}

/// Sample (layer, feature) slots and report whether any carries a non-empty
/// `down_meta` top token — the guard against the hollow-vindex
/// (silent-incomplete-extract) case where matching could never fire.
///
/// The per-layer bound is `index.num_features(layer)`, not a hardcoded width:
/// a sparse vindex whose populated features sit above a fixed sample window
/// would otherwise be falsely refused. Sampling stops at the first non-empty
/// token (the common, valid case is cheap) and at a total cap, so a genuinely
/// hollow wide index does not churn unbounded `feature_meta` allocations.
fn has_any_down_meta_token(index: &VectorIndex) -> bool {
    const MAX_SAMPLED: usize = 4096;
    let mut sampled = 0usize;
    for layer in 0..index.num_layers {
        for feat in 0..index.num_features(layer) {
            if let Some(meta) = index.feature_meta(layer, feat) {
                if !meta.top_token.is_empty() {
                    return true;
                }
            }
            sampled += 1;
            if sampled >= MAX_SAMPLED {
                return false;
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::{parse_band, resolve_capture_layers};

    fn index_with_knowledge(lo: u64, hi: u64) -> serde_json::Value {
        serde_json::json!({ "layer_bands": { "knowledge": [lo, hi] } })
    }

    #[test]
    fn defaults_to_knowledge_band_when_present() {
        let idx = index_with_knowledge(14, 27);
        let (layers, desc) = resolve_capture_layers(&idx, 34, false, None);
        assert_eq!(layers, (14..=27).collect::<Vec<_>>());
        assert!(desc.contains("knowledge band"), "desc was {desc}");
    }

    #[test]
    fn all_layers_flag_overrides_to_full_depth() {
        let idx = index_with_knowledge(14, 27);
        let (layers, desc) = resolve_capture_layers(&idx, 34, true, None);
        assert_eq!(layers, (0..34).collect::<Vec<_>>());
        assert!(desc.contains("all 34 layers"), "desc was {desc}");
    }

    #[test]
    fn explicit_band_overrides_knowledge() {
        let idx = index_with_knowledge(14, 27);
        let (layers, desc) = resolve_capture_layers(&idx, 34, false, Some((10, 20)));
        assert_eq!(layers, (10..=20).collect::<Vec<_>>());
        assert!(desc.contains("explicit --band"), "desc was {desc}");
    }

    #[test]
    fn absent_layer_bands_falls_back_to_all_layers() {
        let idx = serde_json::json!({ "num_layers": 8 });
        let (layers, desc) = resolve_capture_layers(&idx, 8, false, None);
        assert_eq!(layers, (0..8).collect::<Vec<_>>());
        assert!(desc.contains("all 8 layers"), "desc was {desc}");
        assert!(desc.contains("no layer_bands.knowledge"), "desc was {desc}");
    }

    #[test]
    fn null_index_falls_back_to_all_layers() {
        // Unreadable / unparseable index.json → Value::Null at the call site.
        let (layers, desc) = resolve_capture_layers(&serde_json::Value::Null, 12, false, None);
        assert_eq!(layers, (0..12).collect::<Vec<_>>());
        assert!(desc.contains("all 12 layers"), "desc was {desc}");
    }

    #[test]
    fn band_high_bound_clamped_to_layer_count() {
        let idx = index_with_knowledge(14, 99);
        let (layers, _) = resolve_capture_layers(&idx, 34, false, None);
        assert_eq!(*layers.last().unwrap(), 33);
    }

    #[test]
    fn parse_band_ok_and_errors() {
        assert_eq!(parse_band("14-27").unwrap(), (14, 27));
        assert!(parse_band("14").is_err());
        assert!(parse_band("x-27").is_err());
        assert!(parse_band("27-14").is_err());
    }
}
