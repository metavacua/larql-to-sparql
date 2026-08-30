//! Where the model's expert-routing decision becomes observable.
//!
//! `LARQL_MOE_ROUTE_TRACE=<path>` makes every routed MoE selection append a
//! JSON line. The sink itself lives in
//! [`ffn::expert_weight::trace`](crate::ffn::expert_weight::trace); this module
//! is the **attachment point** for the served tier, and it exists because of a
//! measurement failure worth stating plainly.
//!
//! ## The failure this module fixes
//!
//! The trace was originally attached inside one weight-materialisation
//! implementation — the f32 reference `ExpertWeightFfn`. That path has `layer`
//! in scope and pins against the HF reference, so its records are trustworthy.
//! But it is **not the path a served model runs.** GPT-OSS decoding from a
//! vindex routes through [`moe_route_from_router_input`](crate::cpu::ops::moe::
//! moe_route_from_router_input) and never touches `ExpertWeightFfn`, so
//! `LARQL_MOE_ROUTE_TRACE` produced an empty file on a model that was visibly
//! generating tokens. A research instrument silently observed nothing.
//!
//! There are in fact **two independent route decision implementations** in this
//! workspace — `router::select` on the reference tier and
//! `moe_route_from_router_input` on the served tier — and the diagnostic was
//! wired to one of them. Any future executor that adds a third would bypass it
//! again. The rule the design now follows:
//!
//! > **Attach diagnostics to the semantic decision boundary, not to a
//! > backend-specific implementation of it.**
//!
//! ## Why a scoped layer rather than a parameter
//!
//! The served route function is handed a router input and a weight struct,
//! neither of which carries a layer index, and `MoeLayerWeights` is built by
//! struct literal in dozens of places. Threading a new required field through
//! all of them to serve a diagnostic would be the tail wagging the dog.
//!
//! Instead the per-layer driver loop — the only place that genuinely knows
//! which layer is executing — installs a [`LayerScope`] for the duration of
//! that layer. The route function reads it.
//!
//! ## Refusal, not attribution by guess
//!
//! If no scope is active the observation is **refused and counted**, never
//! recorded against a guessed layer. The predecessor of this design recovered a
//! layer index from a global counter `% 30`, which produces a plausible trace
//! from an unsound attribution — the worst possible outcome for a measurement.
//! [`refused()`] makes an unattributed path visible instead of invisible, so a
//! silently-bypassing executor shows up as a non-zero count rather than as a
//! short file nobody questions.

use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::ffn::expert_weight::trace;

thread_local! {
    /// Layer currently executing on this thread, if a driver loop said so.
    static CURRENT_LAYER: Cell<Option<usize>> = const { Cell::new(None) };
}

/// Observations dropped because no layer scope was active.
static REFUSED: AtomicU64 = AtomicU64::new(0);

/// Routing observations that were refused for want of a layer attribution.
///
/// Non-zero after a decode means some executor reached the route boundary
/// outside any [`LayerScope`] — the trace is incomplete and the gap is that
/// executor, not the model.
pub fn refused() -> u64 {
    REFUSED.load(Ordering::Relaxed)
}

#[cfg(test)]
fn reset_refused() {
    REFUSED.store(0, Ordering::Relaxed);
}

/// Marks the executing layer for as long as it is held.
///
/// Install one per layer in the driver loop. Nested scopes restore the outer
/// value on drop, so a nested call cannot leave a stale attribution behind for
/// whatever runs next on this thread.
pub struct LayerScope {
    previous: Option<usize>,
}

impl LayerScope {
    pub fn new(layer: usize) -> Self {
        let previous = CURRENT_LAYER.with(|c| c.replace(Some(layer)));
        Self { previous }
    }
}

impl Drop for LayerScope {
    fn drop(&mut self) {
        CURRENT_LAYER.with(|c| c.set(self.previous));
    }
}

/// The layer this thread is executing, if any.
pub fn current_layer() -> Option<usize> {
    CURRENT_LAYER.with(|c| c.get())
}

/// Record one token's expert selection at the served route boundary.
///
/// No-op when tracing is off. Refused and counted when tracing is on but no
/// layer scope is active — see the module header.
pub fn observe(experts: &[usize]) {
    // `buffer` returns `None` when the sink is closed, which is the cheap
    // check that keeps an untraced decode free of this whole path.
    if trace::buffer(1).is_none() {
        return;
    }
    let Some(layer) = current_layer() else {
        // A silent incomplete trace is the exact failure this module exists to
        // prevent, so the first refusal says so once, loudly, on stderr.
        if REFUSED.fetch_add(1, Ordering::Relaxed) == 0 {
            // Naming the bypassing executor is the whole job here, so under
            // RUST_BACKTRACE the warning carries the call site that reached
            // the route boundary unscoped.
            if std::env::var_os("RUST_BACKTRACE").is_some() {
                eprintln!(
                    "[{}] unscoped route site:\n{}",
                    crate::options::ENV_MOE_ROUTE_TRACE,
                    std::backtrace::Backtrace::force_capture()
                );
            }
            eprintln!(
                "[{}] routing observed outside any LayerScope — this executor \
                 does not install one, so its selections are REFUSED rather \
                 than attributed to a guessed layer. The trace will be \
                 incomplete; install a LayerScope in that path's per-layer loop.",
                crate::options::ENV_MOE_ROUTE_TRACE
            );
        }
        return;
    };
    trace::record(layer, Some(vec![experts.to_vec()]));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_scope_sets_and_restores_the_layer() {
        assert_eq!(current_layer(), None);
        {
            let _outer = LayerScope::new(7);
            assert_eq!(current_layer(), Some(7));
            {
                let _inner = LayerScope::new(19);
                assert_eq!(current_layer(), Some(19));
            }
            assert_eq!(
                current_layer(),
                Some(7),
                "a nested scope must restore its parent, not clear the layer"
            );
        }
        assert_eq!(current_layer(), None, "the scope must not leak past drop");
    }

    #[test]
    fn scopes_do_not_leak_across_threads() {
        let _outer = LayerScope::new(3);
        let seen = std::thread::spawn(current_layer).join().unwrap();
        assert_eq!(seen, None, "a worker thread inherits no attribution");
        assert_eq!(current_layer(), Some(3));
    }

    #[test]
    fn observing_without_a_scope_is_refused_rather_than_attributed() {
        // The whole point: no scope means no record, not a record against a
        // guessed layer. Only meaningful when a sink exists; when tracing is
        // off the call returns before the refusal counter, which is correct —
        // an untraced run has nothing to be incomplete about.
        reset_refused();
        assert_eq!(current_layer(), None);
        observe(&[1, 2, 3]);
        let refusals = refused();
        if trace::buffer(1).is_some() {
            assert_eq!(refusals, 1, "tracing on + no scope must count a refusal");
        } else {
            assert_eq!(refusals, 0, "tracing off must not count refusals");
        }
    }

    #[test]
    fn observing_is_a_no_op_when_tracing_is_off() {
        // The untraced path must not even reach the layer lookup, so that a
        // production decode pays nothing for the instrument's existence.
        reset_refused();
        let _scope = LayerScope::new(0);
        observe(&[4, 5]);
        if trace::buffer(1).is_none() {
            assert_eq!(refused(), 0);
        }
    }
}
