//! Speech-model auxiliary weights — the non-backbone parts of a
//! speech-generating checkpoint (audio embedding tables, depth
//! transformers, per-codebook output heads).
//!
//! These side-load from the original safetensors directory, mirroring how
//! the vision tower loads outside the vindex
//! ([`crate::encoders::vision_tower`]): the vindex carries LM weights
//! only, and everything else lives alongside `config.json`. See
//! `docs/tts-funnel.md` for the funnel this module serves.

pub mod moss_tts_realtime;
