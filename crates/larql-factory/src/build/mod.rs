//! `larql recipe build` — the Vindex Factory build-stage driver
//! (docs/vindex-factory.md §7).
//!
//! Scope, decided after tracing the actual reusable tooling (none of
//! the spec's original reuse assumptions held as written — see each
//! stage module's doc comment for the specific correction): this
//! module runs PREFLIGHT through RELEASE for real, orchestrating
//! existing `larql` subcommands as subprocesses via [`runner`]. MIRROR
//! (R2) and REGISTER (chuk-experiments-server) are not implemented
//! here — nothing in this codebase talks to either today, and the
//! spec's own text assumes they're owned by the rig's worker
//! infrastructure, not the `larql` binary itself; [`record::BuildRecord`]
//! is the structured hand-off point for an external wrapper to do both,
//! the same way `dec0-loopback.sh` already wraps `dec-bench`'s JSON
//! output. Reconstruction-fidelity and logit-match numeric checks
//! (§8.1) are also not implemented — building them correctly needs
//! per-architecture tensor-naming knowledge this session has no way to
//! validate against real model weights, so VERIFY here covers checksum
//! integrity only (reusing the existing `larql verify` command) rather
//! than a confident-but-unvalidated numeric check.

mod record;
// runner.rs (std::process::Command, no core/alloc equivalent) and
// stages/ (subprocess orchestration built on runner::CommandRunner)
// are native-only throughout -- confirmed via grep that nothing outside
// build/ references either, so unlike estimate/'s partial split, the
// whole pair excludes cleanly. record.rs is pure data (Class A alloc
// fix only) and stays available on wasm32.
#[cfg(not(target_arch = "wasm32"))]
mod runner;
#[cfg(not(target_arch = "wasm32"))]
mod stages;
#[cfg(test)]
mod tests;

pub use record::{BuildRecord, BuildStatus, OutputRecord, Stage};
#[cfg(not(target_arch = "wasm32"))]
pub use runner::{CommandOutput, CommandRunner, SubprocessRunner};

#[cfg(not(target_arch = "wasm32"))]
use std::path::{Path, PathBuf};

#[cfg(not(target_arch = "wasm32"))]
use crate::Recipe;

#[cfg(not(target_arch = "wasm32"))]
const HF_CACHE_SUBDIR: &str = "hf-cache";
#[cfg(not(target_arch = "wasm32"))]
const FULL_OUTPUT_SUBDIR: &str = "full.vindex";

/// Run every stage for `recipe`, using `scratch_dir` as the build's
/// working directory (created if missing) and `runner` to execute each
/// `larql` subcommand. Always returns a [`BuildRecord`] — a stage
/// failure is encoded in [`BuildRecord::status`], not a Rust `Err`, so
/// a caller always has a JSON-printable result either way.
#[cfg(not(target_arch = "wasm32"))]
pub fn run(runner: &dyn CommandRunner, recipe: &Recipe, scratch_dir: &Path) -> BuildRecord {
    let build_id = crate::build_id(recipe);
    let mut output_records: Vec<OutputRecord> = recipe
        .spec
        .outputs
        .iter()
        .map(|o| OutputRecord::new(&o.preset))
        .collect();

    macro_rules! fail {
        ($stage:expr, $message:expr) => {
            return BuildRecord {
                build_id,
                recipe_name: recipe.metadata.name.clone(),
                outputs: output_records,
                status: BuildStatus::Failed {
                    stage: $stage,
                    message: $message,
                },
            }
        };
    }

    if let Err(e) = stages::preflight::check_scratch_dir_writable(scratch_dir) {
        fail!(Stage::Preflight, e);
    }

    let hf_cache_dir = scratch_dir.join(HF_CACHE_SUBDIR);
    let full_dir = scratch_dir.join(FULL_OUTPUT_SUBDIR);

    if let Err(e) = stages::fetch::run(runner, recipe, &hf_cache_dir) {
        fail!(Stage::Fetch, e);
    }
    if let Err(e) = stages::extract::run(runner, recipe, &full_dir, &hf_cache_dir) {
        fail!(Stage::Extract, e);
    }

    // SLICE: resolve each output's local directory, slicing non-full
    // presets from the just-extracted full vindex.
    let mut output_dirs: Vec<PathBuf> = Vec::with_capacity(output_records.len());
    for output in &recipe.spec.outputs {
        let dir = if stages::slice::needs_slicing(output) {
            let dst = scratch_dir.join(format!("{}.vindex", output.preset));
            if let Err(e) = stages::slice::run(runner, &full_dir, &dst, output) {
                fail!(Stage::Slice, e);
            }
            dst
        } else {
            full_dir.clone()
        };
        output_dirs.push(dir);
    }

    // MANIFEST: measure what each output produced.
    for (i, dir) in output_dirs.iter().enumerate() {
        match stages::manifest::summarise(&output_records[i].preset, dir) {
            Ok(summary) => output_records[i].size_bytes = Some(summary.size_bytes),
            Err(e) => fail!(Stage::Manifest, e),
        }
    }

    // VERIFY: checksum integrity (see stages::verify's doc comment for
    // what this deliberately doesn't cover).
    for dir in &output_dirs {
        if let Err(e) = stages::verify::run(runner, dir) {
            fail!(Stage::Verify, e);
        }
    }

    // PUBLISH: private first — RELEASE flips visibility once every
    // output has published and verified.
    for (i, dir) in output_dirs.iter().enumerate() {
        match stages::publish::run(
            runner,
            dir,
            &recipe.metadata,
            &recipe.spec.publish,
            &recipe.spec.outputs[i],
        ) {
            Ok(repo) => output_records[i].repo = Some(repo),
            Err(e) => fail!(Stage::Publish, e),
        }
    }

    // RELEASE: flip every published repo to public.
    for i in 0..output_records.len() {
        let repo = output_records[i]
            .repo
            .clone()
            .expect("PUBLISH set every output's repo before RELEASE runs");
        if let Err(e) = stages::release::run(runner, &repo, &recipe.spec.publish.hub.repo_type) {
            fail!(Stage::Release, e);
        }
        output_records[i].released = true;
    }

    BuildRecord {
        build_id,
        recipe_name: recipe.metadata.name.clone(),
        outputs: output_records,
        status: BuildStatus::Passed,
    }
}
