//! An independent oracle for fixture A.
//!
//! Deliberately shares nothing with the runtime but the weight values. No
//! `BoundTensor`, no `kernels`, no `expert::forward` — its softmax, its top-k
//! and its activation are written out again here.
//!
//! That duplication is the point. An oracle built from the runtime's own
//! primitives agrees with the runtime by construction and tests nothing; the
//! only failure it can catch is one in the glue. This one computes the layer
//! from its definition, so agreement is evidence.

use super::direct_moe::{
    down_at, gate_at, router_at, up_at, HIDDEN, INTERMEDIATE, POPULATION, TOP_K,
};

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

/// Every checkpoint, computed from the layer's definition.
pub struct Oracle {
    pub router_scores: Vec<f32>,
    /// Selected `(expert_id, final_weight)`, in selection order.
    pub selected: Vec<(u32, f32)>,
    /// Unweighted output per selected expert, in selection order.
    pub expert_outputs: Vec<(u32, Vec<f32>)>,
    pub reduced: Vec<f32>,
}

pub fn oracle(input: &[f32]) -> Oracle {
    let router_scores = route(input);
    let selected = select(&router_scores);
    let expert_outputs: Vec<(u32, Vec<f32>)> = selected
        .iter()
        .map(|&(id, _)| (id, expert(id as usize, input)))
        .collect();

    let mut reduced = vec![0.0f32; HIDDEN];
    for (&(_, weight), (_, out)) in selected.iter().zip(&expert_outputs) {
        for (acc, &v) in reduced.iter_mut().zip(out) {
            *acc += weight * v;
        }
    }

    Oracle {
        router_scores,
        selected,
        expert_outputs,
        reduced,
    }
}

/// Router logits, softmaxed over the whole population.
fn route(input: &[f32]) -> Vec<f32> {
    let mut logits: Vec<f32> = (0..POPULATION)
        .map(|e| (0..HIDDEN).map(|h| router_at(e, h) * input[h]).sum())
        .collect();
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for x in &mut logits {
        *x = (*x - max).exp();
        sum += *x;
    }
    for x in &mut logits {
        *x /= sum;
    }
    logits
}

/// Top-k by score, ties to the lower id, renormalised to sum to one.
fn select(scores: &[f32]) -> Vec<(u32, f32)> {
    let mut ranked: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
    ranked.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(core::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    ranked.truncate(TOP_K);
    let total: f32 = ranked.iter().map(|(_, w)| *w).sum();
    ranked
        .into_iter()
        .map(|(i, w)| (i as u32, w / total))
        .collect()
}

/// `down · ( silu(gate · x) ⊙ (up · x) )`, straight from the definition.
fn expert(expert: usize, input: &[f32]) -> Vec<f32> {
    let gated: Vec<f32> = (0..INTERMEDIATE)
        .map(|unit| {
            let g: f32 = (0..HIDDEN)
                .map(|h| gate_at(expert, unit, h) * input[h])
                .sum();
            let u: f32 = (0..HIDDEN).map(|h| up_at(expert, unit, h) * input[h]).sum();
            silu(g) * u
        })
        .collect();
    (0..HIDDEN)
        .map(|h| {
            (0..INTERMEDIATE)
                .map(|unit| down_at(expert, h, unit) * gated[unit])
                .sum()
        })
        .collect()
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}
