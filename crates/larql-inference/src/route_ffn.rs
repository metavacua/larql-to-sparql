//! Route-based FFN — replaces gate computation with a routing table lookup.
//!
//! Instead of computing `gate_weight @ hidden_state` to find which features fire,
//! this backend looks up pre-recorded feature activations from a routing table.
//! The table is built by `extract-routes`: run forward passes, record which features
//! fire at each layer for each template pattern.
//!
//! At inference time:
//! 1. Match the input to a routing table entry (by relation pattern)
//! 2. For each layer, use the pre-recorded feature indices and activations
//! 3. Compute only `silu(activation) * (up_row @ x)` for those features
//! 4. Project through down vectors → output
//!
//! This eliminates the gate matmul entirely — the most expensive part of FFN.

use std::collections::HashMap;
use std::path::Path;

use ndarray::Array2;
use serde::Deserialize;

use crate::ffn::{sigmoid, FfnBackend};
use crate::model::ModelWeights;

// ── Route table structures ──

#[derive(Deserialize)]
struct RouteTableJson {
    routes: Vec<RouteEntryJson>,
}

#[derive(Deserialize)]
struct RouteEntryJson {
    relation: String,
    entity: String,
    features: Vec<FeatureHitJson>,
}

#[derive(Deserialize)]
struct FeatureHitJson {
    layer: usize,
    feature: usize,
    activation: f32,
}

/// Pre-loaded routing table: for each (relation, entity), the features that fire per layer.
type RouteMap = HashMap<(String, String), HashMap<usize, Vec<(usize, f32)>>>;

pub struct RouteTable {
    /// (relation, entity) -> layer -> [(feature_index, activation)]
    routes: RouteMap,
}

impl RouteTable {
    /// Load a routing table from the JSON file produced by `extract-routes`.
    pub fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let data = std::fs::read_to_string(path)?;
        let table: RouteTableJson = serde_json::from_str(&data)?;

        let mut routes: RouteMap = HashMap::new();

        for entry in &table.routes {
            let key = (entry.relation.clone(), entry.entity.clone());
            let layer_map = routes.entry(key).or_default();

            for hit in &entry.features {
                layer_map
                    .entry(hit.layer)
                    .or_default()
                    .push((hit.feature, hit.activation));
            }
        }

        // Sort each layer's features by activation magnitude (descending)
        for layer_map in routes.values_mut() {
            for feats in layer_map.values_mut() {
                feats.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
            }
        }

        Ok(Self { routes })
    }

    /// Get features for a specific (relation, entity) at a given layer.
    pub fn get_features(
        &self,
        relation: &str,
        entity: &str,
        layer: usize,
    ) -> Option<&[(usize, f32)]> {
        self.routes
            .get(&(relation.to_string(), entity.to_string()))
            .and_then(|m| m.get(&layer))
            .map(|v| v.as_slice())
    }

    /// Aggregate features across all entities for a relation at a given layer.
    /// Returns the union of all features, with averaged activations.
    pub fn get_pattern_features(
        &self,
        relation: &str,
        layer: usize,
        top_k: usize,
    ) -> Vec<(usize, f32)> {
        let mut accum: HashMap<usize, (f32, usize)> = HashMap::new();

        for ((rel, _entity), layer_map) in &self.routes {
            if rel != relation {
                continue;
            }
            if let Some(feats) = layer_map.get(&layer) {
                for &(feat, act) in feats {
                    let entry = accum.entry(feat).or_insert((0.0, 0));
                    entry.0 += act.abs();
                    entry.1 += 1;
                }
            }
        }

        let mut result: Vec<(usize, f32)> = accum
            .into_iter()
            .map(|(feat, (sum, count))| (feat, sum / count as f32))
            .collect();
        result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        result.truncate(top_k);
        result
    }

    pub fn num_routes(&self) -> usize {
        self.routes.len()
    }

    pub fn relations(&self) -> Vec<String> {
        let mut rels: Vec<String> = self
            .routes
            .keys()
            .map(|(r, _)| r.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        rels.sort();
        rels
    }
}

// ── Route FFN backend ──

/// FFN backend that uses pre-recorded gate activations directly.
/// Fast but inaccurate — the hidden state at inference differs from extraction.
pub struct RouteFfn<'a> {
    pub weights: &'a ModelWeights,
    pub route_table: &'a RouteTable,
    pub relation: String,
    pub entity: String,
    pub top_k: usize,
}

impl<'a> FfnBackend for RouteFfn<'a> {
    fn forward(&self, layer: usize, x: &Array2<f32>) -> Array2<f32> {
        let arch = &*self.weights.arch;
        let w_up = self.weights.tensors.get(&arch.ffn_up_key(layer)).unwrap();
        let w_down = self.weights.tensors.get(&arch.ffn_down_key(layer)).unwrap();

        let features = self
            .route_table
            .get_features(&self.relation, &self.entity, layer);

        match features {
            Some(feats) if !feats.is_empty() => {
                let k = feats.len().min(self.top_k);
                route_ffn_forward_prerecorded(x, w_up, w_down, &feats[..k])
            }
            _ => Array2::zeros(x.raw_dim()),
        }
    }

    fn forward_with_activation(
        &self,
        layer: usize,
        x: &Array2<f32>,
    ) -> (Array2<f32>, Array2<f32>) {
        let out = self.forward(layer, x);
        let intermediate = self
            .weights
            .tensors
            .get(&self.weights.arch.ffn_gate_key(layer))
            .map(|g| g.shape()[0])
            .unwrap_or(0);
        (out, Array2::zeros((x.shape()[0], intermediate)))
    }

    fn name(&self) -> &str {
        "route"
    }
}

/// FFN backend that uses the routing table for feature SELECTION only,
/// then computes actual gate activations for those features against the
/// real hidden state. Best of both worlds:
/// - Route table eliminates the full gate matmul (selects which features matter)
/// - Actual gate @ hidden for selected features (accurate activations)
pub struct RouteGuidedFfn<'a> {
    pub weights: &'a ModelWeights,
    pub route_table: &'a RouteTable,
    pub relation: String,
    pub entity: String,
    pub top_k: usize,
}

impl<'a> FfnBackend for RouteGuidedFfn<'a> {
    fn forward(&self, layer: usize, x: &Array2<f32>) -> Array2<f32> {
        let arch = &*self.weights.arch;
        let w_gate = self.weights.tensors.get(&arch.ffn_gate_key(layer)).unwrap();
        let w_up = self.weights.tensors.get(&arch.ffn_up_key(layer)).unwrap();
        let w_down = self.weights.tensors.get(&arch.ffn_down_key(layer)).unwrap();

        let features = self
            .route_table
            .get_features(&self.relation, &self.entity, layer);

        match features {
            Some(feats) if !feats.is_empty() => {
                let k = feats.len().min(self.top_k);
                // Extract just the feature indices — we'll compute actual activations
                let feature_indices: Vec<usize> = feats[..k].iter().map(|&(idx, _)| idx).collect();
                route_ffn_forward_guided(x, w_gate, w_up, w_down, &feature_indices)
            }
            _ => Array2::zeros(x.raw_dim()),
        }
    }

    fn forward_with_activation(
        &self,
        layer: usize,
        x: &Array2<f32>,
    ) -> (Array2<f32>, Array2<f32>) {
        let out = self.forward(layer, x);
        let intermediate = self
            .weights
            .tensors
            .get(&self.weights.arch.ffn_gate_key(layer))
            .map(|g| g.shape()[0])
            .unwrap_or(0);
        (out, Array2::zeros((x.shape()[0], intermediate)))
    }

    fn name(&self) -> &str {
        "route-guided"
    }
}

/// Pre-recorded activation variant: uses stored gate values (fast, less accurate).
fn route_ffn_forward_prerecorded(
    x: &Array2<f32>,
    w_up: &ndarray::ArrayBase<impl ndarray::Data<Elem = f32>, ndarray::Ix2>,
    w_down: &ndarray::ArrayBase<impl ndarray::Data<Elem = f32>, ndarray::Ix2>,
    features: &[(usize, f32)],
) -> Array2<f32> {
    let seq_len = x.shape()[0];
    let hidden = x.shape()[1];
    let mut out = Array2::<f32>::zeros((seq_len, hidden));

    for s in 0..seq_len {
        let x_row = x.row(s);

        for &(feat_idx, gate_act) in features {
            let silu_gate = gate_act * sigmoid(gate_act);
            let up_row = w_up.row(feat_idx);
            let up_val: f32 = up_row.iter().zip(x_row.iter()).map(|(a, b)| a * b).sum();
            let activation = silu_gate * up_val;

            if activation.abs() < 1e-8 {
                continue;
            }

            for j in 0..hidden {
                out[[s, j]] += activation * w_down[[j, feat_idx]];
            }
        }
    }

    out
}

/// Route-guided variant: uses route table for feature SELECTION,
/// then computes actual gate @ hidden for those features.
/// Eliminates the full gate matmul but keeps accurate activations.
fn route_ffn_forward_guided(
    x: &Array2<f32>,
    w_gate: &ndarray::ArrayBase<impl ndarray::Data<Elem = f32>, ndarray::Ix2>,    // (intermediate, hidden)
    w_up: &ndarray::ArrayBase<impl ndarray::Data<Elem = f32>, ndarray::Ix2>,      // (intermediate, hidden)
    w_down: &ndarray::ArrayBase<impl ndarray::Data<Elem = f32>, ndarray::Ix2>,    // (hidden, intermediate)
    feature_indices: &[usize],
) -> Array2<f32> {
    let seq_len = x.shape()[0];
    let hidden = x.shape()[1];
    let mut out = Array2::<f32>::zeros((seq_len, hidden));

    for s in 0..seq_len {
        let x_row = x.row(s);

        for &feat_idx in feature_indices {
            // Compute ACTUAL gate activation: gate_row @ x
            let gate_row = w_gate.row(feat_idx);
            let gate_val: f32 = gate_row.iter().zip(x_row.iter()).map(|(a, b)| a * b).sum();

            // SiLU on the actual gate activation
            let silu_gate = gate_val * sigmoid(gate_val);

            // up_proj: up_row @ x
            let up_row = w_up.row(feat_idx);
            let up_val: f32 = up_row.iter().zip(x_row.iter()).map(|(a, b)| a * b).sum();

            let activation = silu_gate * up_val;

            if activation.abs() < 1e-8 {
                continue;
            }

            // down projection: accumulate into output
            for j in 0..hidden {
                out[[s, j]] += activation * w_down[[j, feat_idx]];
            }
        }
    }

    out
}
