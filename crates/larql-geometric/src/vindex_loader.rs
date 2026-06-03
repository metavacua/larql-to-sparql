// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Ian Douglas Lawrence Norman McLean
use std::path::Path;

/// Weights for a single transformer layer's attention projections.
pub struct LayerAttnWeights {
    pub q_proj: Vec<f32>,
    pub k_proj: Vec<f32>,
    pub v_proj: Vec<f32>,
    pub o_proj: Vec<f32>,
}

/// Load attention projection weights for `layer` from a larql vindex at `path`.
///
/// Not yet implemented — returns `Err` until the vindex reader is wired up.
pub fn load_layer_attn(path: &Path, layer: usize) -> Result<LayerAttnWeights, String> {
    Err(format!(
        "load_layer_attn: vindex loader not yet implemented (path={}, layer={})",
        path.display(),
        layer,
    ))
}
