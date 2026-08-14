//! Instrumented boundaries inside one MoE execution.
//!
//! A final-vector equality assertion tells you an execution is wrong and
//! nothing else. These checkpoints localise it: a mismatch at `router_scores`
//! is scoring or the router operand, at `selection` it is top-k or a weight
//! policy, at `expert_output` it is region interpretation or activation, at
//! `reduced` it is the combine.
//!
//! # Why a sink rather than a returned struct
//!
//! The traced and untraced paths must be the *same* arithmetic — a fixture
//! that exercises a separate instrumented implementation proves that
//! implementation and not the one Gemma will run. So there is one generic
//! execution, and the sink is a type parameter: [`NoTrace`] has empty methods
//! and compiles away, leaving the token path with no allocation and no
//! branch, while [`CollectedTrace`] records.

use super::router::SelectedExpert;

/// Receives internal checkpoints during execution.
///
/// Every method defaults to doing nothing, so the production sink is a single
/// empty impl and adding a checkpoint never breaks an existing one.
pub trait TraceSink {
    /// Bank input, after any routed-input transform. Equals the residual for a
    /// direct MoE.
    fn routed_input(&mut self, _values: &[f32]) {}
    /// Router probabilities over the whole population, before selection.
    fn router_scores(&mut self, _values: &[f32]) {}
    /// How much room the selection had: `score[k-1] - score[k]`, or `None`
    /// when there is no boundary. Zero means an exact tie decided it.
    fn selection_margin(&mut self, _margin: Option<f32>) {}
    /// Selected experts in selection order, with final and raw weights.
    fn selection(&mut self, _selected: &[SelectedExpert]) {}
    /// One expert's output, before its routing weight is applied.
    fn expert_output(&mut self, _expert_id: u32, _values: &[f32]) {}
    /// The bank's combined output, before any routed-output transform.
    fn reduced(&mut self, _values: &[f32]) {}
    /// What the operation contributes to the residual stream.
    fn residual_delta(&mut self, _values: &[f32]) {}
}

/// The production sink: records nothing, costs nothing.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoTrace;

impl TraceSink for NoTrace {}

/// One expert's unweighted output, kept in selection order.
#[derive(Debug, Clone, PartialEq)]
pub struct ExpertOutput {
    pub expert_id: u32,
    pub values: Vec<f32>,
}

/// Records every checkpoint. For fixtures, differential tests and diagnosis.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CollectedTrace {
    pub routed_input: Vec<f32>,
    pub router_scores: Vec<f32>,
    /// `score[k-1] - score[k]`. `Some(0.0)` means an exact tie sat on the
    /// top-k boundary and the ordering policy, not the scores, chose.
    pub selection_margin: Option<f32>,
    pub selection: Vec<SelectedExpert>,
    pub expert_outputs: Vec<ExpertOutput>,
    pub reduced: Vec<f32>,
    pub residual_delta: Vec<f32>,
}

impl CollectedTrace {
    /// Selected expert ids, in selection order.
    pub fn selected_ids(&self) -> Vec<u32> {
        self.selection.iter().map(|s| s.expert_id).collect()
    }

    /// Final routing weights, in selection order.
    pub fn gate_weights(&self) -> Vec<f32> {
        self.selection.iter().map(|s| s.weight).collect()
    }

    /// Whether an exact tie sat on the top-k boundary.
    ///
    /// The triage question when two implementations disagree about which
    /// experts ran: true means they met a tie and resolved it differently,
    /// false means they disagreed about the scores themselves.
    pub fn selection_was_decided_by_a_tie(&self) -> bool {
        self.selection_margin == Some(0.0)
    }

    /// One expert's recorded output, by id.
    pub fn expert_output(&self, expert_id: u32) -> Option<&[f32]> {
        self.expert_outputs
            .iter()
            .find(|o| o.expert_id == expert_id)
            .map(|o| o.values.as_slice())
    }
}

impl TraceSink for CollectedTrace {
    fn routed_input(&mut self, values: &[f32]) {
        self.routed_input = values.to_vec();
    }

    fn router_scores(&mut self, values: &[f32]) {
        self.router_scores = values.to_vec();
    }

    fn selection_margin(&mut self, margin: Option<f32>) {
        self.selection_margin = margin;
    }

    fn selection(&mut self, selected: &[SelectedExpert]) {
        self.selection = selected.to_vec();
    }

    fn expert_output(&mut self, expert_id: u32, values: &[f32]) {
        self.expert_outputs.push(ExpertOutput {
            expert_id,
            values: values.to_vec(),
        });
    }

    fn reduced(&mut self, values: &[f32]) {
        self.reduced = values.to_vec();
    }

    fn residual_delta(&mut self, values: &[f32]) {
        self.residual_delta = values.to_vec();
    }
}
