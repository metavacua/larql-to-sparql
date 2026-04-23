//! Cascade trie probe — PCA-32 + Logistic Regression classifier trained on
//! last-position hidden states at a fixed transformer layer (~15% depth).
//!
//! Loaded from a JSON file exported by `experiments/export_trie_probe.py`.
//! The exported file contains PCA components and LR weights; inference is
//! pure arithmetic — no Python dependency at runtime.

use std::path::Path;

use serde::Deserialize;

// ── Serialised probe format ───────────────────────────────────────────────────

#[derive(Deserialize)]
struct ProbeFile {
    layer: usize,
    hidden_size: usize,
    n_components: usize,
    routes: Vec<String>,
    pca_mean: Vec<f64>,
    pca_components: Vec<Vec<f64>>,   // [n_components, hidden_size]
    lr_coef: Vec<Vec<f64>>,          // [n_classes, n_components]
    lr_intercept: Vec<f64>,          // [n_classes]
    lr_classes: Vec<String>,         // route name per LR class index
}

// ── Public API ────────────────────────────────────────────────────────────────

/// A loaded cascade trie probe.
///
/// Call `classify(hidden)` with the last-position hidden state (as `f32` slice,
/// length = hidden_size) from transformer layer `self.layer()`.
pub struct CascadeTrie {
    pub layer: usize,
    hidden_size: usize,
    routes: Vec<String>,
    /// PCA: subtract mean, then multiply by components.
    pca_mean: Vec<f32>,
    /// [n_components, hidden_size] row-major.
    pca_components: Vec<f32>,
    n_components: usize,
    /// LR: [n_classes, n_components] row-major.
    lr_coef: Vec<f32>,
    lr_intercept: Vec<f32>,
    lr_classes: Vec<String>,
}

impl CascadeTrie {
    /// Load from a JSON file exported by `export_trie_probe.py`.
    pub fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let text = std::fs::read_to_string(path)?;
        let p: ProbeFile = serde_json::from_str(&text)?;

        // Flatten 2D vecs to row-major 1D for BLAS-free dot products.
        let pca_components: Vec<f32> = p.pca_components
            .into_iter()
            .flatten()
            .map(|v| v as f32)
            .collect();
        let lr_coef: Vec<f32> = p.lr_coef
            .into_iter()
            .flatten()
            .map(|v| v as f32)
            .collect();

        Ok(Self {
            layer: p.layer,
            hidden_size: p.hidden_size,
            routes: p.routes,
            pca_mean: p.pca_mean.into_iter().map(|v| v as f32).collect(),
            pca_components,
            n_components: p.n_components,
            lr_coef,
            lr_intercept: p.lr_intercept.into_iter().map(|v| v as f32).collect(),
            lr_classes: p.lr_classes,
        })
    }

    /// Classify a hidden state vector → route label (e.g. `"arithmetic"`).
    ///
    /// `hidden` must have length == `self.hidden_size`.
    /// Returns `"unknown"` if the slice length doesn't match.
    pub fn classify<'a>(&'a self, hidden: &[f32]) -> &'a str {
        if hidden.len() != self.hidden_size {
            return "unknown";
        }

        // ── PCA projection ──
        // z[k] = dot(hidden - mean, components[k])
        let mut z = vec![0.0f32; self.n_components];
        for (k, z_k) in z.iter_mut().enumerate() {
            let row = &self.pca_components[k * self.hidden_size..(k + 1) * self.hidden_size];
            let mut dot = 0.0f32;
            for i in 0..self.hidden_size {
                dot += (hidden[i] - self.pca_mean[i]) * row[i];
            }
            *z_k = dot;
        }

        // ── LR decision: argmax of (coef @ z + intercept) ──
        let n_classes = self.lr_classes.len();
        let mut best_idx = 0usize;
        let mut best_score = f32::NEG_INFINITY;
        for c in 0..n_classes {
            let row = &self.lr_coef[c * self.n_components..(c + 1) * self.n_components];
            let score: f32 = row.iter().zip(z.iter()).map(|(w, x)| w * x).sum::<f32>()
                + self.lr_intercept[c];
            if score > best_score {
                best_score = score;
                best_idx = c;
            }
        }

        &self.lr_classes[best_idx]
    }

    /// All route labels the probe was trained on.
    pub fn routes(&self) -> &[String] {
        &self.routes
    }
}
