use serde::{Deserialize, Serialize};

/// Wave/Particle/Wavelet activation regime for a transformer layer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Regime {
    /// Dense activations — many weak gates fire (e.g., early syntax layers).
    Wave,
    /// Sparse activations — few strong gates fire (e.g., MoE knowledge layers).
    Particle,
    /// Mixed — multi-resolution, some wave structure, some particle selectivity.
    Wavelet,
}

/// Per-layer canonical metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerCanonicalInfo {
    pub layer: usize,
    pub regime: Regime,
    /// Number of features that pass the on-shell filter (top 15% by c_score).
    pub on_shell_count: usize,
    /// Total features at this layer.
    pub total_features: usize,
    /// Fraction of features with c_score > 0.1 (activation density proxy).
    pub mean_density: f32,
}

/// Root canonical metadata written to `canonical_meta.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanonicalMeta {
    /// Format version; increment when the schema changes.
    pub version: u32,
    pub model: String,
    pub family: String,
    pub num_layers: usize,
    pub hidden_size: usize,
    /// Number of embedding rows sampled to estimate G.
    pub covariance_sample_size: usize,
    /// embed_scale used when computing G.
    pub embed_scale: f32,
    /// Cholesky factor L packed row-major lower-triangle:
    /// indices (i,j) with j<=i stored as L[i*(i+1)/2 + j].
    /// Length = hidden_size * (hidden_size + 1) / 2. Values are f64.
    pub cholesky_l_packed: Vec<f64>,
    /// Per-layer info.
    pub layers: Vec<LayerCanonicalInfo>,
}

impl CanonicalMeta {
    /// Unpack the lower-triangular Cholesky factor into a dense d×d matrix.
    pub fn unpack_cholesky_l(&self) -> ndarray::Array2<f64> {
        let d = self.hidden_size;
        let mut l = ndarray::Array2::<f64>::zeros((d, d));
        for i in 0..d {
            for j in 0..=i {
                l[[i, j]] = self.cholesky_l_packed[i * (i + 1) / 2 + j];
            }
        }
        l
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regime_serialises_as_snake_case() {
        assert_eq!(serde_json::to_string(&Regime::Wave).unwrap(), "\"wave\"");
        assert_eq!(serde_json::to_string(&Regime::Particle).unwrap(), "\"particle\"");
        assert_eq!(serde_json::to_string(&Regime::Wavelet).unwrap(), "\"wavelet\"");
    }

    #[test]
    fn canonical_meta_round_trips_through_json() {
        let meta = CanonicalMeta {
            version: 1,
            model: "test/model".into(),
            family: "llama".into(),
            num_layers: 2,
            hidden_size: 4,
            covariance_sample_size: 32,
            embed_scale: 1.0,
            // 4×4 lower triangle: 10 values
            cholesky_l_packed: vec![1.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0],
            layers: vec![
                LayerCanonicalInfo {
                    layer: 0, regime: Regime::Wave,
                    on_shell_count: 1, total_features: 4, mean_density: 0.75,
                },
                LayerCanonicalInfo {
                    layer: 1, regime: Regime::Particle,
                    on_shell_count: 1, total_features: 4, mean_density: 0.02,
                },
            ],
        };
        let json = serde_json::to_string_pretty(&meta).unwrap();
        let back: CanonicalMeta = serde_json::from_str(&json).unwrap();
        assert_eq!(back.family, "llama");
        assert_eq!(back.layers[0].regime, Regime::Wave);
        assert_eq!(back.layers[1].regime, Regime::Particle);
        assert_eq!(back.cholesky_l_packed.len(), 10);
    }

    #[test]
    fn unpack_cholesky_l_recovers_diagonal() {
        // Packed lower-triangle of 3×3: [L00, L10, L11, L20, L21, L22]
        let meta = CanonicalMeta {
            version: 1, model: "x".into(), family: "y".into(),
            num_layers: 1, hidden_size: 3,
            covariance_sample_size: 8, embed_scale: 1.0,
            cholesky_l_packed: vec![2.0, 1.0, 3.0, 4.0, 5.0, 6.0],
            layers: vec![],
        };
        let l = meta.unpack_cholesky_l();
        assert_eq!(l[[0, 0]], 2.0);
        assert_eq!(l[[1, 0]], 1.0);
        assert_eq!(l[[1, 1]], 3.0);
        assert_eq!(l[[2, 0]], 4.0);
        assert_eq!(l[[2, 1]], 5.0);
        assert_eq!(l[[2, 2]], 6.0);
        assert_eq!(l[[0, 1]], 0.0);
        assert_eq!(l[[0, 2]], 0.0);
    }
}
