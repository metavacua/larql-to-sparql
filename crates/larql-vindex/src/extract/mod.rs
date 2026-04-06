//! Build pipeline — extract model weights into vindex format.

pub mod build;
pub mod build_from_vectors;
pub mod callbacks;
pub mod streaming;

pub use build::build_vindex;
pub use build::build_vindex_resume;
pub use build_from_vectors::build_vindex_from_vectors;
pub use streaming::build_vindex_streaming;
pub use callbacks::{IndexBuildCallbacks, SilentBuildCallbacks};
