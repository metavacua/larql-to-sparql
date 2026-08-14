//! MOSS-TTS-Realtime auxiliary weights: the audio side of the checkpoint.
//!
//! The backbone (a stock Qwen3 under `language_model.`) is described by
//! [`crate::architectures::moss_tts_realtime`] and loads through the
//! normal paths. Everything else — the per-codebook input embedding
//! tables, the depth ("local") transformer, and its per-codebook output
//! heads — loads here, from the original safetensors directory.
//!
//! Nothing in this module hardcodes the released checkpoint's shape: every
//! count and id derives from `config.json` (`rvq`, `audio_vocab_size`,
//! `audio_pad_token`, `local_config`), so a synthetic miniature exercises
//! the same code paths as the 2.3B release.
//!
//! [`coverage`] is the step-1 gate of `docs/tts-funnel.md`: every tensor
//! in the checkpoint is consumed, or skipped for a stated reason, or the
//! load is wrong — there is no silent remainder (the GPT-OSS lesson,
//! AGENTS.md "a tensor the extractor doesn't name is silently dropped").

mod adapter;
mod config;
mod coverage;
mod keys;
mod loader;
mod weights;

pub use adapter::{depth_transformer_model, DepthTransformerModel};
pub use config::{LocalTransformerConfig, MossTtsRealtimeConfig};
pub use coverage::{classify_checkpoint_keys, CheckpointCoverage, JustifiedSkip};
pub use loader::load_moss_tts_aux_from_safetensors;
pub use weights::{LocalLayerWeights, LocalTransformerWeights, MossTtsRealtimeAuxWeights};

#[cfg(test)]
mod tests;
