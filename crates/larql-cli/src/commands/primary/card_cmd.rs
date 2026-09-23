//! `larql card render` — print a Hub model card for a build
//! (docs/vindex-factory.md §9). `build_id` is derived from the recipe
//! the same way `larql recipe build-id` does — never passed in, so it
//! can't drift from what the recipe actually hashes to.

use std::path::{Path, PathBuf};

use clap::{Args, Subcommand};
use larql_factory::{CardInputs, Recipe, SliceSummary, VerificationReport};
use larql_vindex_spec::VindexManifest;

#[derive(Subcommand)]
pub enum CardCommand {
    /// Render a Hub model card from a recipe, its manifest, and a
    /// verification report.
    Render(RenderArgs),
}

#[derive(Args)]
pub struct RenderArgs {
    /// Path to the recipe YAML file.
    #[arg(long)]
    pub recipe: PathBuf,
    /// Path to the vindex-v1 manifest (`index.json`).
    #[arg(long)]
    pub manifest: PathBuf,
    /// Path to a verification report JSON file (§8's VERIFY-A/B output).
    #[arg(long)]
    pub verification: PathBuf,
    /// Path to a slice-summary JSON file: an array of
    /// `{"preset": ..., "size_bytes": ...}`. Omit for a card with no
    /// slice table.
    #[arg(long)]
    pub slices: Option<PathBuf>,
}

pub fn run(cmd: CardCommand) -> Result<(), Box<dyn std::error::Error>> {
    match cmd {
        CardCommand::Render(args) => run_render(args),
    }
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, Box<dyn std::error::Error>> {
    let src =
        std::fs::read_to_string(path).map_err(|e| format!("reading {}: {e}", path.display()))?;
    serde_json::from_str(&src)
        .map_err(|e| format!("parsing {}: {e}", path.display()))
        .map_err(Into::into)
}

fn run_render(args: RenderArgs) -> Result<(), Box<dyn std::error::Error>> {
    let recipe_src = std::fs::read_to_string(&args.recipe)
        .map_err(|e| format!("reading {}: {e}", args.recipe.display()))?;
    let recipe = Recipe::from_yaml(&recipe_src)
        .map_err(|e| format!("parsing {}: {e}", args.recipe.display()))?;

    let manifest: VindexManifest = read_json(&args.manifest)?;
    let verification: VerificationReport = read_json(&args.verification)?;
    let slices: Vec<SliceSummary> = match &args.slices {
        Some(path) => read_json(path)?,
        None => Vec::new(),
    };

    let build_id = larql_factory::build_id(&recipe);
    let inputs = CardInputs {
        recipe: &recipe,
        manifest: &manifest,
        verification: &verification,
        slices: &slices,
        build_id: &build_id,
    };

    println!("{}", larql_factory::render_card(&inputs));
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use tempfile::NamedTempFile;

    use super::*;

    const RECIPE_YAML: &str = r#"
apiVersion: vindex.chukai.io/v1
kind: VindexBuild
metadata:
  name: tiny-model
spec:
  source:
    hf_repo: chrishayuk/tiny-model
    revision: 9b5b1a2f7c3e0d1a4b8f2c6e9d0a3b5c7e1f4a82
    licence: mit
  extractor:
    tool: larql
    version: "0.14.2"
    level: inference
    dtype: f16
  outputs:
    - preset: full
  verify:
    reconstruction:
      layers_sampled: 1
      seed: 1
      max_abs_diff: 0.0
      min_cosine: 1.0
    logit_match:
      prompt_set: verify/tiny-smoke.jsonl
      top1_agreement_min: 0.99
      bits_per_char_drift_max: 0.005
    from_hub: true
  publish:
    hub:
      owner: chrishayuk
      repo_template: "{model_slug}-vindex{slice_suffix}"
      repo_type: dataset
      visibility: private-until-verified
  budget:
    executor: auto
    max_wall_minutes: 30
    max_usd: 1
    requires:
      disk_gib: 8
      ram_gib: 8
      net_down_mbps: 100
"#;

    fn write_temp(contents: &str) -> NamedTempFile {
        let mut file = NamedTempFile::new().expect("create temp file");
        file.write_all(contents.as_bytes())
            .expect("write temp file");
        file
    }

    fn verification_json() -> String {
        serde_json::to_string(&VerificationReport {
            reconstruction: larql_factory::ReconstructionResult {
                layers_sampled: 1,
                max_abs_diff: 0.0,
                min_cosine: 1.0,
            },
            logit_match: larql_factory::LogitMatchResult {
                top1_agreement: 1.0,
                bits_per_char_drift: 0.0,
            },
            verified_from_hub: true,
        })
        .unwrap()
    }

    fn manifest_json() -> String {
        serde_json::to_string(&larql_vindex_spec::test_fixtures::sample_manifest()).unwrap()
    }

    #[test]
    fn run_render_succeeds_with_no_slices() {
        let recipe = write_temp(RECIPE_YAML);
        let manifest = write_temp(&manifest_json());
        let verification = write_temp(&verification_json());

        let args = RenderArgs {
            recipe: recipe.path().to_path_buf(),
            manifest: manifest.path().to_path_buf(),
            verification: verification.path().to_path_buf(),
            slices: None,
        };
        assert!(run_render(args).is_ok());
    }

    #[test]
    fn run_render_succeeds_with_slices() {
        let recipe = write_temp(RECIPE_YAML);
        let manifest = write_temp(&manifest_json());
        let verification = write_temp(&verification_json());
        let slices = write_temp(
            &serde_json::to_string(&[SliceSummary {
                preset: "full".into(),
                size_bytes: 1024,
            }])
            .unwrap(),
        );

        let args = RenderArgs {
            recipe: recipe.path().to_path_buf(),
            manifest: manifest.path().to_path_buf(),
            verification: verification.path().to_path_buf(),
            slices: Some(slices.path().to_path_buf()),
        };
        assert!(run_render(args).is_ok());
    }

    #[test]
    fn run_render_errors_on_missing_recipe() {
        let manifest = write_temp(&manifest_json());
        let verification = write_temp(&verification_json());
        let args = RenderArgs {
            recipe: PathBuf::from("/nonexistent/recipe.yaml"),
            manifest: manifest.path().to_path_buf(),
            verification: verification.path().to_path_buf(),
            slices: None,
        };
        let err = run_render(args).unwrap_err();
        assert!(err.to_string().contains("reading"));
    }

    #[test]
    fn run_render_errors_on_malformed_manifest() {
        let recipe = write_temp(RECIPE_YAML);
        let manifest = write_temp("not valid json");
        let verification = write_temp(&verification_json());
        let args = RenderArgs {
            recipe: recipe.path().to_path_buf(),
            manifest: manifest.path().to_path_buf(),
            verification: verification.path().to_path_buf(),
            slices: None,
        };
        let err = run_render(args).unwrap_err();
        assert!(err.to_string().contains("parsing"));
    }

    #[test]
    fn run_dispatches_render_variant() {
        let recipe = write_temp(RECIPE_YAML);
        let manifest = write_temp(&manifest_json());
        let verification = write_temp(&verification_json());
        let args = RenderArgs {
            recipe: recipe.path().to_path_buf(),
            manifest: manifest.path().to_path_buf(),
            verification: verification.path().to_path_buf(),
            slices: None,
        };
        assert!(run(CardCommand::Render(args)).is_ok());
    }
}
