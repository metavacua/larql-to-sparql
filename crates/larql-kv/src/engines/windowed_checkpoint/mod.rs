pub mod checkpoint_store;
// `index: &VectorIndex` unconditional.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod dispatch;
// `EngineStats` (pure data) stays portable; `WindowedCheckpointEngine`
// (`impl KvEngine`, upstream-gated) is internally gated -- see engine.rs.
pub mod engine;
pub mod extend;
pub mod token_archive;

pub use checkpoint_store::CheckpointStore;
pub use engine::EngineStats;
#[cfg(not(target_arch = "wasm32"))]
pub use engine::WindowedCheckpointEngine;
pub use extend::empty_prior;
#[cfg(not(target_arch = "wasm32"))]
pub use extend::{
    rs_extend_from_checkpoint, rs_extend_from_checkpoint_backend, rs_extend_from_checkpoint_quant,
    ExtendOutput,
};
pub use token_archive::TokenArchive;
