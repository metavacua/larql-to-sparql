//! Vindex integration — WalkFfn for inference.
//!
//! The build pipeline, weight IO, clustering, and format handling
//! now live in `larql-vindex`. This module provides only WalkFfn
//! (the FFN backend that uses vindex KNN for feature selection).

mod walk_ffn;

pub use walk_ffn::WalkFfn;
