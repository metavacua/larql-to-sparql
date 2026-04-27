//! Core trace types.

use serde::{Deserialize, Serialize};
use crate::attention::AttentionWeights;
use crate::model::ModelWeights;

/// A single waypoint in the residual stream.
#[derive(Clone)]
pub struct TraceNode {
    /// Layer index (-1 = embedding).
    pub layer: i32,
    /// Token position in the sequence.
    pub position: usize,
    /// Residual vector after this layer. Shape: (hidden_size,).
    pub residual: Vec<f32>,
    /// What attention added at this layer. Zero for embedding layer.
    pub attn_delta: Vec<f32>,
    /// What FFN added at this layer. Zero for embedding layer.
    pub ffn_delta: Vec<f32>,
}

/// Per-layer summary across all captured positions.
#[derive(Clone, Serialize, Deserialize)]
pub struct LayerSummary {
    pub layer: i32,
    pub residual_norm: f32,
    pub attn_delta_norm: f32,
    pub ffn_delta_norm: f32,
    pub top1_token: String,
    pub top1_prob: f32,
}

/// One waypoint in an answer trajectory.
#[derive(Clone, Serialize, Deserialize)]
pub struct AnswerWaypoint {
    pub layer: i32,
    pub rank: u32,
    pub prob: f32,
    pub attn_logit: f32,
    pub ffn_logit: f32,
    pub residual_norm: f32,
}

/// Complete in-memory inference trace.
pub struct ResidualTrace {
    pub prompt: String,
    pub tokens: Vec<String>,
    pub token_ids: Vec<u32>,
    pub n_layers: usize,
    pub hidden_size: usize,
    pub nodes: Vec<TraceNode>,
    pub attention: Vec<(usize, AttentionWeights)>,
}

impl ResidualTrace {
    pub fn node(&self, layer: i32, position: usize) -> Option<&TraceNode> {
        self.nodes.iter().find(|n| n.layer == layer && n.position == position)
    }

    pub fn last_node(&self, layer: i32) -> Option<&TraceNode> {
        let last_pos = self.tokens.len().saturating_sub(1);
        self.node(layer, last_pos)
    }

    pub fn layer_nodes(&self, layer: i32) -> Vec<&TraceNode> {
        self.nodes.iter().filter(|n| n.layer == layer).collect()
    }

    pub fn position_trajectory(&self, position: usize) -> Vec<&TraceNode> {
        let mut traj: Vec<&TraceNode> = self.nodes.iter()
            .filter(|n| n.position == position)
            .collect();
        traj.sort_by_key(|n| n.layer);
        traj
    }

    pub fn vocab_project(&self, weights: &ModelWeights, vec: &[f32]) -> Vec<f32> {
        super::vocab::project_to_logits(weights, vec)
    }

    pub fn top_k(
        &self, weights: &ModelWeights, tokenizer: &tokenizers::Tokenizer,
        layer: i32, position: usize, k: usize,
    ) -> Vec<(String, f32)> {
        let node = match self.node(layer, position) {
            Some(n) => n,
            None => return vec![],
        };
        let logits = super::vocab::project_to_logits(weights, &node.residual);
        super::vocab::top_k_from_logits(&logits, tokenizer, k)
    }

    pub fn answer_trajectory(
        &self, weights: &ModelWeights, answer_token_id: u32,
    ) -> Vec<AnswerWaypoint> {
        let last_pos = self.tokens.len().saturating_sub(1);
        let mut traj = Vec::new();

        for layer in -1..self.n_layers as i32 {
            let node = match self.node(layer, last_pos) {
                Some(n) => n,
                None => continue,
            };
            let logits = super::vocab::project_to_logits(weights, &node.residual);
            let probs = super::vocab::softmax(&logits);
            let prob = probs[answer_token_id as usize];
            let rank = probs.iter().filter(|&&p| p > prob).count() as u32 + 1;

            let attn_logit = if node.attn_delta.iter().any(|&x| x != 0.0) {
                super::vocab::project_to_logits(weights, &node.attn_delta)[answer_token_id as usize]
            } else { 0.0 };
            let ffn_logit = if node.ffn_delta.iter().any(|&x| x != 0.0) {
                super::vocab::project_to_logits(weights, &node.ffn_delta)[answer_token_id as usize]
            } else { 0.0 };

            traj.push(AnswerWaypoint {
                layer, rank, prob, attn_logit, ffn_logit,
                residual_norm: super::vocab::vec_norm(&node.residual),
            });
        }
        traj
    }

    pub fn layer_summaries(
        &self, weights: &ModelWeights, tokenizer: &tokenizers::Tokenizer,
    ) -> Vec<LayerSummary> {
        let last_pos = self.tokens.len().saturating_sub(1);
        let mut summaries = Vec::new();
        for layer in -1..self.n_layers as i32 {
            let node = match self.node(layer, last_pos) {
                Some(n) => n,
                None => continue,
            };
            let logits = super::vocab::project_to_logits(weights, &node.residual);
            let top = super::vocab::top_k_from_logits(&logits, tokenizer, 1);
            let (tok, prob) = top.first().map(|(t, p)| (t.clone(), *p)).unwrap_or_default();
            summaries.push(LayerSummary {
                layer,
                residual_norm: super::vocab::vec_norm(&node.residual),
                attn_delta_norm: super::vocab::vec_norm(&node.attn_delta),
                ffn_delta_norm: super::vocab::vec_norm(&node.ffn_delta),
                top1_token: tok, top1_prob: prob,
            });
        }
        summaries
    }
}
