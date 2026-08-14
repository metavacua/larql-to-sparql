//! Integration gates for the semantic promotion engine (spec §22).
//!
//! Per-module unit gates live beside the code they cover. These are the
//! cross-module ones: G0 parity, the FFN-dispatch gate, the lifecycle
//! refusals, and the EXP-25 qualification matrix expressed as policy.

mod engine_delegation;
mod fixtures;
mod lifecycle;
mod parity;
mod position_seam;
mod qualification_gate;
mod row_position_storage;
mod window_policy_gates;
mod window_policy_refusals;
