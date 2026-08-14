//! Walk-layer gates.
//!
//! EXP-25 is encoded as graph fixtures in `exp25`. `walk_gates` holds
//! W0–W8, `enforce_gates` Phase B, `heterogeneous_gates` B10,
//! `exclusion_oracle` B1/B6, `cost_gates` C0–C10, `replay_gates` R0–R5,
//! and `composition_gates` what may not yet be claimed.

mod composition_gates;
mod cost_gates;
mod enforce_gates;
mod exclusion_oracle;
mod exp25;
mod heterogeneous_gates;
mod rendering;
mod replay_gates;
mod validation_refusals;
mod walk_gates;
