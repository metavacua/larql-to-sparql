//! Build callbacks and utilities for vindex construction.
//!
//! The full build pipeline (EXTRACT) lives in `larql-inference` because it
//! needs the clustering module. This module provides the callback trait and
//! shared utilities that both the build pipeline and weight writer use.

/// Callbacks for index build progress.
pub trait IndexBuildCallbacks {
    fn on_stage(&mut self, _stage: &str) {}
    fn on_layer_start(&mut self, _component: &str, _layer: usize, _total: usize) {}
    fn on_feature_progress(&mut self, _component: &str, _layer: usize, _done: usize, _total: usize) {}
    fn on_layer_done(&mut self, _component: &str, _layer: usize, _elapsed_ms: f64) {}
    fn on_stage_done(&mut self, _stage: &str, _elapsed_ms: f64) {}
}

pub struct SilentBuildCallbacks;
impl IndexBuildCallbacks for SilentBuildCallbacks {}
