//! Observation points on the canonical decode step (LQL-2 TRACE).
//!
//! There is exactly one semantic execution path; TRACE and any other
//! observer **subscribe to it** — nothing re-enacts the plan to emit
//! events. [`DecodeSession::step_observed`] fires these events at the
//! step's existing operation boundaries and computes exactly what
//! [`step`] computes: the parity gate demands the observed and
//! unobserved paths stay bit-identical, so an observer can never
//! change arithmetic or execution order.
//!
//! Deliberately coarse at this rung: layer and sublayer boundaries and
//! the head's logits — structure, not tensors. Finer taps (operand
//! reads, attention state, residual values) are later detail levels
//! and must arrive the same way: more events on the one executor,
//! never a second traversal.
//!
//! [`DecodeSession::step_observed`]: super::decode::DecodeSession::step_observed
//! [`step`]: super::decode::DecodeSession::step

/// One decode step's observation events, in execution order.
#[derive(Debug, Clone, PartialEq)]
pub enum StepEvent {
    /// The token was embedded at this absolute position.
    Embedded { position: usize },
    /// A layer's attention sublayer completed (residual add included).
    AttentionDone { layer: usize },
    /// A layer's FFN sublayer completed (residual add and any layer
    /// scale included) — the layer boundary.
    FfnDone { layer: usize },
    /// The output head priced the vocabulary for this position.
    Logits { vocab: usize },
}

/// Where in a layer an activation was taken.
///
/// Two sites, because two suffice: everything else is derivable from them
/// offline. `q/k/v` read the attention input; `gate/up` read the FFN
/// input; and `down`'s input is `act(gate(x)) * up(x)`, which a screen can
/// reconstruct from the FFN input and those two operands rather than
/// needing its own tap.
///
/// `o_proj` is the exception and is *not* covered: its input is the
/// attention core's output, which never surfaces at this boundary. A
/// consumer must exclude `o_proj` rather than approximate it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputSite {
    /// Normalised residual entering attention — input to q, k and v.
    Attention,
    /// Normalised residual entering the FFN — input to gate and up.
    Ffn,
    /// The FFN sublayer's own output, before any post-norm or residual
    /// scaling.
    ///
    /// Not an input, and present for one reason: it is the control that
    /// proves `down_proj`'s reconstructed input is the executor's. A
    /// screen that reconstructs `act(gate(x)) ⊙ up(x)` can check itself by
    /// multiplying through `down_proj` and comparing here — so the
    /// reconstruction is verified rather than believed.
    FfnOutput,
}

/// A subscriber to the canonical step's observation points.
pub trait StepObserver {
    fn event(&mut self, event: StepEvent);

    /// Observe an operand input's values. Separate from [`event`] so the
    /// values are borrowed rather than cloned into an event: capturing
    /// second moments needs to read the vector, not own it.
    ///
    /// [`event`]: Self::event
    fn operand_input(&mut self, _layer: usize, _site: InputSite, _values: &[f32]) {}
}

/// The default subscriber: observes nothing. [`DecodeSession::step`]
/// is `step_observed` with this observer, so the unobserved path is
/// the observed path by construction.
///
/// [`DecodeSession::step`]: super::decode::DecodeSession::step
pub struct NoopObserver;

impl StepObserver for NoopObserver {
    fn event(&mut self, _event: StepEvent) {}
}

/// Convenience subscriber: records every event, for tests and for
/// consumers that render after the step completes.
#[derive(Default)]
pub struct RecordingObserver {
    pub events: Vec<StepEvent>,
}

impl StepObserver for RecordingObserver {
    fn event(&mut self, event: StepEvent) {
        self.events.push(event);
    }
}
