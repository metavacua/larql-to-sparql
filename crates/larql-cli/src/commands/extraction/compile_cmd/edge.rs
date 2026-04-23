//! The compile-into-weights primitive.
//!
//! Writes one (gate, up, down) triple at `slot` so the FFN fires on `trigger`
//! and contributes `write` (scaled) to the residual. Reference norms from the
//! original slot are preserved so the new edge sits in the same magnitude
//! regime as the trained slots; this is what makes the contribution land
//! cleanly without blowing out the residual.
//!
//! Convention from `experiments/07_wasm_compute/WASM_GATE_ARCHITECTURE.md` §3.1.2:
//!
//! ```text
//! gate[slot, :]  ← trigger̂ × g_norm × gate_scale
//! up[slot,   :]  ← trigger̂ × u_norm
//! down[:, slot]  ← write × (d_norm / ‖write‖) × alpha_mul
//! ```
//!
//! `trigger` and `write` are normalised internally; pass any non-zero
//! direction. `gate_scale` typically 30.0 (fires gate strongly); `alpha_mul`
//! typically 1.0 for residual-tag writes, 10.0 for token-embedding writes
//! routed through the LM head.
//!
//! This primitive is the lowest level of the COMPILE verb — `larql compile`
//! (CLI) calls it directly, and `COMPILE CURRENT INTO MODEL` (LQL) will
//! eventually call it through the executor. Lives here rather than in its
//! own crate because it has a single call site inside one crate; when a
//! second consumer (TinyModel, larql-lql executor) needs it, extract then.

use std::collections::HashMap;

use ndarray::ArcArray2;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum EdgeError {
    #[error("tensor not found: {0}")]
    MissingTensor(String),
    #[error("trigger has zero norm")]
    ZeroTrigger,
    #[error("write has zero norm")]
    ZeroWrite,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct EdgeStats {
    pub g_norm: f32,
    pub u_norm: f32,
    pub d_norm: f32,
    pub alpha: f32,
}

#[allow(clippy::too_many_arguments)]
pub fn install_edge(
    tensors: &mut HashMap<String, ArcArray2<f32>>,
    gate_key: &str,
    up_key: &str,
    down_key: &str,
    slot: usize,
    trigger: &[f32],
    write: &[f32],
    gate_scale: f32,
    alpha_mul: f32,
) -> Result<EdgeStats, EdgeError> {
    let trigger_norm = vec_norm(trigger);
    let write_norm = vec_norm(write);
    if trigger_norm < 1e-8 {
        return Err(EdgeError::ZeroTrigger);
    }
    if write_norm < 1e-8 {
        return Err(EdgeError::ZeroWrite);
    }

    let g_norm = row_norm(
        tensors
            .get(gate_key)
            .ok_or_else(|| EdgeError::MissingTensor(gate_key.into()))?,
        slot,
    );
    let u_norm = row_norm(
        tensors
            .get(up_key)
            .ok_or_else(|| EdgeError::MissingTensor(up_key.into()))?,
        slot,
    );
    let d_norm = col_norm(
        tensors
            .get(down_key)
            .ok_or_else(|| EdgeError::MissingTensor(down_key.into()))?,
        slot,
    );

    let g_scale = g_norm * gate_scale / trigger_norm;
    let u_scale = u_norm / trigger_norm;
    let alpha = (d_norm / write_norm) * alpha_mul;

    {
        let gt = tensors.get_mut(gate_key).unwrap();
        let hidden = gt.shape()[1];
        for j in 0..hidden.min(trigger.len()) {
            gt[[slot, j]] = trigger[j] * g_scale;
        }
    }
    {
        let ut = tensors.get_mut(up_key).unwrap();
        let hidden = ut.shape()[1];
        for j in 0..hidden.min(trigger.len()) {
            ut[[slot, j]] = trigger[j] * u_scale;
        }
    }
    {
        let dt = tensors.get_mut(down_key).unwrap();
        let hidden = dt.shape()[0];
        for j in 0..hidden.min(write.len()) {
            dt[[j, slot]] = write[j] * alpha;
        }
    }

    Ok(EdgeStats { g_norm, u_norm, d_norm, alpha })
}

fn vec_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn row_norm(tensor: &ArcArray2<f32>, row: usize) -> f32 {
    let r = tensor.row(row);
    r.dot(&r).sqrt()
}

fn col_norm(tensor: &ArcArray2<f32>, col: usize) -> f32 {
    let c = tensor.column(col);
    c.dot(&c).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn fresh_layer(ffn_dim: usize, hidden: usize) -> HashMap<String, ArcArray2<f32>> {
        let mut gate = Array2::<f32>::zeros((ffn_dim, hidden));
        let mut up = Array2::<f32>::zeros((ffn_dim, hidden));
        let mut down = Array2::<f32>::zeros((hidden, ffn_dim));
        for j in 0..hidden {
            gate[[0, j]] = 0.1;
            up[[0, j]] = 0.1;
            down[[j, 0]] = 0.1;
        }
        let mut h = HashMap::new();
        h.insert("gate".into(), gate.into_shared());
        h.insert("up".into(), up.into_shared());
        h.insert("down".into(), down.into_shared());
        h
    }

    #[test]
    fn install_writes_into_slot_zero() {
        let mut t = fresh_layer(4, 8);
        let trigger = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let write = vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        let stats = install_edge(&mut t, "gate", "up", "down", 0, &trigger, &write, 30.0, 1.0).unwrap();

        let gate = t.get("gate").unwrap();
        let expected = stats.g_norm * 30.0;
        assert!((gate[[0, 0]] - expected).abs() < 1e-5);
    }

    #[test]
    fn zero_trigger_rejected() {
        let mut t = fresh_layer(4, 8);
        let trigger = vec![0.0; 8];
        let write = vec![1.0; 8];
        let err = install_edge(&mut t, "gate", "up", "down", 0, &trigger, &write, 30.0, 1.0)
            .unwrap_err();
        assert!(matches!(err, EdgeError::ZeroTrigger));
    }

    #[test]
    fn missing_tensor_reports_key() {
        let mut t = fresh_layer(4, 8);
        let trigger = vec![1.0; 8];
        let write = vec![1.0; 8];
        let err = install_edge(&mut t, "missing_gate", "up", "down", 0, &trigger, &write, 30.0, 1.0)
            .unwrap_err();
        assert!(matches!(err, EdgeError::MissingTensor(k) if k == "missing_gate"));
    }

    #[test]
    fn magnitude_preservation_invariant() {
        let mut t = fresh_layer(4, 8);
        for &scale in &[0.1_f32, 1.0, 100.0] {
            let trigger: Vec<f32> = (0..8).map(|i| (i as f32 + 1.0) * scale).collect();
            let write = vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
            let stats = install_edge(&mut t, "gate", "up", "down", 0, &trigger, &write, 30.0, 1.0).unwrap();
            let gate = t.get("gate").unwrap();
            let gate_row_norm = (0..8).map(|j| gate[[0, j]].powi(2)).sum::<f32>().sqrt();
            let expected = stats.g_norm * 30.0;
            let rel_err = (gate_row_norm - expected).abs() / expected.max(1e-8);
            assert!(rel_err < 1e-5, "scale={scale}: rel_err={rel_err}");
        }
    }

    #[test]
    fn write_down_alpha_matches_stats() {
        let mut t = fresh_layer(4, 8);
        let trigger = vec![1.0; 8];
        let write = vec![0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let stats = install_edge(&mut t, "gate", "up", "down", 0, &trigger, &write, 30.0, 1.0).unwrap();
        let down = t.get("down").unwrap();
        for j in 0..8 {
            let expected = write[j] * stats.alpha;
            assert!((down[[j, 0]] - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn shorter_trigger_does_not_panic() {
        let mut t = fresh_layer(4, 8);
        let trigger = vec![1.0, 0.0, 0.0];
        let write = vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        install_edge(&mut t, "gate", "up", "down", 0, &trigger, &write, 30.0, 1.0).unwrap();
        let gate = t.get("gate").unwrap();
        assert!((gate[[0, 4]] - 0.1).abs() < 1e-5);
    }

    #[test]
    fn alpha_mul_scales_write_linearly() {
        let mut t = fresh_layer(4, 8);
        let trigger = vec![1.0; 8];
        let write = vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let s1 = install_edge(&mut t, "gate", "up", "down", 0, &trigger, &write, 30.0, 1.0).unwrap();
        let mut t2 = fresh_layer(4, 8);
        let s2 = install_edge(&mut t2, "gate", "up", "down", 0, &trigger, &write, 30.0, 5.0).unwrap();
        assert!((s2.alpha / s1.alpha - 5.0).abs() < 1e-5);
    }
}
