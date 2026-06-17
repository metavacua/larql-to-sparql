//! `larql label` — produce `feature_labels.json` for a vindex.
//!
//! Given a vindex, a relation catalog, and the source model, this command:
//!   1. captures the residual stream for each relation's subject prompts,
//!   2. routes each subject's residual through the vindex gate (signed KNN),
//!   3. frame-subtracts to keep subject-specific features and matches each to
//!      its object via the feature's `down_meta` top token,
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
    let layers: Vec<usize> = (0..model.num_layers()).collect();

    // 5. Capture residuals one relation at a time (bounded memory — each
    //    relation's temp dir is dropped after its residuals are loaded).
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

    // 6. Label.
    let labels = label_catalog(&index, &catalog, &residuals, args.per_layer_k, args.frame_frac);

    // 7. Write.
    let output_dir = args.output.unwrap_or(args.vindex);
    write_feature_labels(&output_dir, &labels)?;
    println!(
        "{} feature labels written to {}",
        labels.len(),
        output_dir.join("feature_labels.json").display()
    );

    Ok(())
}

/// Sample a few (layer, feature) slots and report whether any carries a
/// non-empty `down_meta` top token — the guard against the hollow-vindex
/// (silent-incomplete-extract) case where matching could never fire.
fn has_any_down_meta_token(index: &VectorIndex) -> bool {
    for layer in 0..index.num_layers {
        for feat in 0..64 {
            if let Some(meta) = index.feature_meta(layer, feat) {
                if !meta.top_token.is_empty() {
                    return true;
                }
            }
        }
    }
    false
}
