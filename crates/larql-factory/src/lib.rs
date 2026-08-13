//! Vindex Factory driver — recipe schema, `build_id` canonicaliser, and
//! structural validator. See `docs/vindex-factory.md` §3.1: this crate
//! is the single implementation both the GitHub Action and the rig
//! worker call (as `larql recipe validate` / `larql recipe build-id`),
//! so there's nothing to keep in sync between them.
//!
//! Scope: §4's recipe schema, §5's `build_id`, the structural half of
//! §6.1's PR-check gate, §6.1 step 4's size/cost estimator, §9's card
//! generator, and §7's PREFLIGHT→RELEASE build driver (`run_build`) —
//! MIRROR and REGISTER are external-harness concerns, and VERIFY here
//! is checksum integrity only; see `build`'s module doc for why.

#![deny(missing_docs)]
// See crates/larql-core/src/lib.rs for the pattern-2 rationale. Like
// larql-models, this crate's real modules (build/ spawns subprocesses;
// estimate/ makes HTTP calls) are heavily native. build/runner.rs +
// build/stages/ are wholesale wasm32-excluded (std::process::Command,
// no core/alloc equivalent -- confirmed via grep no portable code
// depends on them); build/record.rs is pure data and stays available.
// estimate/http.rs and the two reqwest-calling fns in estimate/mod.rs
// are excluded the same way; estimate's other submodules (bytes/cost/
// dims/executor/preset_weights) are pure computation and stay
// available, same shape as larql-models' detect/ surgical split.
#![cfg_attr(target_arch = "wasm32", no_std)]
#[cfg(target_arch = "wasm32")]
#[macro_use]
extern crate alloc;

mod build;
mod build_id;
mod capabilities;
mod card;
mod constants;
mod estimate;
mod hex;
mod prelude;
mod recipe;
#[cfg(test)]
mod test_support;
mod validate;

#[cfg(not(target_arch = "wasm32"))]
pub use build::{run as run_build, CommandOutput, CommandRunner, SubprocessRunner};
pub use build::{BuildRecord, BuildStatus, OutputRecord, Stage};
pub use build_id::build_id;
pub use capabilities::{
    manifest as capabilities_manifest, ArchitectureCapability, CapabilityManifest,
};
// card::render needs serde_yaml transitively (via body::render_recipe),
// native-only -- see card/body/mod.rs. Recipe::from_yaml below is the
// same shape.
#[cfg(not(target_arch = "wasm32"))]
pub use card::render as render_card;
pub use card::{
    revision_tag, CardInputs, LogitMatchResult, ReconstructionResult, SliceSummary,
    VerificationReport,
};
#[cfg(not(target_arch = "wasm32"))]
pub use estimate::{estimate as estimate_size, HttpError as EstimateError};
pub use estimate::{ExecutorClass, ModelDims, OutputEstimate, SizeEstimate};
pub use recipe::{
    Budget, BudgetRequires, Extractor, HubPublish, LogitMatch, Metadata, MirrorPublish, OutputSpec,
    Publish, Recipe, Reconstruction, Source, Spec, Verify, API_VERSION, KIND,
};
pub use validate::{validate, RecipeError};
