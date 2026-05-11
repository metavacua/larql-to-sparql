//! Build pipeline — extract model weights into vindex format.

pub mod build;
pub mod build_from_vectors;
pub mod build_helpers;
pub mod callbacks;
pub mod checkpoint;
pub mod constants;
pub mod metadata;
pub mod moe_svd;
pub mod stage_labels;
pub mod streaming;

pub use build::build_vindex;
pub use build_from_vectors::build_vindex_from_vectors;
pub use callbacks::{IndexBuildCallbacks, SilentBuildCallbacks};
pub use checkpoint::{Checkpoint, ExtractPhase, CHECKPOINT_FILE};
pub use metadata::{snapshot_hf_metadata, SNAPSHOT_FILES};
pub use streaming::build_vindex_streaming;
