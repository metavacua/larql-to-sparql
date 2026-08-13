//! [`ValidatedFfnLayerPolicy`] — a policy that has been validated
//! against a known model's layer count.

use super::backend_kind::FfnBackendKind;
use super::policy::FfnLayerPolicy;
use super::routing::RoutingPredicate;

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

/// A policy that has been validated against a known model's layer
/// count.
///
/// Distinct from [`FfnLayerPolicy`] at the type level so the type
/// system enforces "validate before build" — the only way to obtain
/// one is via [`FfnLayerPolicy::validate_for`], which performs the
/// layer-coverage and out-of-range checks. The struct is `pub` so
/// callers can name it in `Result` arms and function signatures, but
/// its fields and constructor are *not* `pub`. The non-public
/// constructor is the load-bearing mechanism — without it, the
/// newtype is documentation, not enforcement.
///
/// Validated policies are cheaply cloneable (the underlying
/// `FfnLayerPolicy` is `Clone`) and bindable to multiple models with
/// the same layer count. To validate against a different layer count,
/// extract the policy via [`Self::into_policy`] and re-validate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatedFfnLayerPolicy {
    policy: FfnLayerPolicy,
    num_layers: usize,
}

impl ValidatedFfnLayerPolicy {
    /// Construct without re-validating. **Internal only** — exposed
    /// `pub(super)` so [`FfnLayerPolicy::validate_for`] (in the
    /// sibling `policy` module) can build instances after running
    /// the validation logic. External callers must go through
    /// `validate_for`.
    pub(super) fn new_unchecked(policy: FfnLayerPolicy, num_layers: usize) -> Self {
        Self { policy, num_layers }
    }

    /// The layer count this policy was validated against.
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// Read-only access to the underlying policy. Useful for
    /// serializing the parsed shape into accuracy/bench JSON.
    pub fn policy(&self) -> &FfnLayerPolicy {
        &self.policy
    }

    /// Consume the validation handle and recover the underlying
    /// policy. Useful when the caller wants to re-validate against a
    /// different layer count.
    pub fn into_policy(self) -> FfnLayerPolicy {
        self.policy
    }

    /// Expand the policy's bindings into a per-layer kind list of
    /// length `num_layers`. Each entry is the [`FfnBackendKind`] that
    /// applies at that layer (following the bindings' declaration
    /// order; `Otherwise` catches anything not bound earlier).
    ///
    /// Public for callers that want to inspect the per-layer plan
    /// without constructing live backends — useful for the accuracy
    /// JSON's per-layer `ffn_backend` column when it lands.
    pub fn expand_to_layers(&self) -> Vec<&FfnBackendKind> {
        let mut out: Vec<Option<&FfnBackendKind>> = vec![None; self.num_layers];
        let mut otherwise_kind: Option<&FfnBackendKind> = None;
        for (predicate, kind) in self.policy.bindings() {
            match predicate {
                RoutingPredicate::All => {
                    // All-binding wins over un-filled slots only;
                    // explicit Layers bindings later in the spec
                    // already-filled slots stay. Validation guarantees
                    // there's no conflict at parse time.
                    for slot in out.iter_mut() {
                        if slot.is_none() {
                            *slot = Some(kind);
                        }
                    }
                }
                RoutingPredicate::Layers(ranges) => {
                    for r in ranges {
                        for layer in r.clone() {
                            if layer < self.num_layers {
                                out[layer] = Some(kind);
                            }
                        }
                    }
                }
                RoutingPredicate::Otherwise => {
                    otherwise_kind = Some(kind);
                }
            }
        }
        out.into_iter()
            .map(|slot| {
                slot.or(otherwise_kind)
                    .expect("ValidatedFfnLayerPolicy invariant: every layer is bound")
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── validate_for Ok cases ───────────────────────────────────────────────
    //
    // Err cases live in policy.rs's tests (which exercises the validation
    // logic). These confirm the validated newtype gets built with the
    // expected num_layers, and the constructor invariants hold.

    #[test]
    fn validate_for_uniform_all_covers_any_layer_count() {
        let p = FfnLayerPolicy::from_spec("dense").unwrap();
        let v34 = p.validate_for(34).expect("dense @ 34 should validate");
        assert_eq!(v34.num_layers(), 34);
        let p1 = FfnLayerPolicy::from_spec("dense").unwrap();
        assert!(p1.validate_for(1).is_ok());
    }

    #[test]
    fn validate_for_explicit_partition_covers_correctly() {
        let p = FfnLayerPolicy::from_spec("{walk:k=100}@layers=14-27;{dense}@layers=0-13,28-33")
            .unwrap();
        assert!(p.validate_for(34).is_ok());
    }

    #[test]
    fn validate_for_otherwise_suffices_for_coverage() {
        let p = FfnLayerPolicy::from_spec("{walk}@layers=0-13;{dense}@otherwise").unwrap();
        assert!(p.validate_for(34).is_ok());
    }

    // ── ValidatedFfnLayerPolicy methods ─────────────────────────────────────

    #[test]
    fn records_num_layers_and_preserves_policy() {
        let p = FfnLayerPolicy::from_spec("{dense}@all").unwrap();
        let p_clone = p.clone();
        let v = p.validate_for(8).unwrap();
        assert_eq!(v.num_layers(), 8);
        assert_eq!(v.policy(), &p_clone);
    }

    #[test]
    fn into_policy_recovers_original() {
        let p = FfnLayerPolicy::from_spec("{walk:k=100}@all").unwrap();
        let p_clone = p.clone();
        let v = p.validate_for(34).unwrap();
        let recovered = v.into_policy();
        assert_eq!(recovered, p_clone);
    }

    // ── expand_to_layers ────────────────────────────────────────────────────

    #[test]
    fn expand_to_layers_uniform_dense() {
        let v = FfnLayerPolicy::from_spec("dense")
            .unwrap()
            .validate_for(4)
            .unwrap();
        let kinds = v.expand_to_layers();
        assert_eq!(kinds.len(), 4);
        for k in &kinds {
            assert_eq!(**k, FfnBackendKind::Dense);
        }
    }

    #[test]
    fn expand_to_layers_hybrid_map() {
        // Gemma-3-4B-style hybrid: walk:k=100 at L14-27, dense
        // elsewhere. Per-layer expansion: 14 denses, 14 walks, 6 denses.
        let v = FfnLayerPolicy::from_spec("{walk:k=100}@layers=14-27;{dense}@otherwise")
            .unwrap()
            .validate_for(34)
            .unwrap();
        let kinds = v.expand_to_layers();
        assert_eq!(kinds.len(), 34);
        for (i, k) in kinds.iter().enumerate() {
            match (i, *k) {
                (0..=13, FfnBackendKind::Dense) => {}
                (14..=27, FfnBackendKind::Walk { k: Some(100) }) => {}
                (28..=33, FfnBackendKind::Dense) => {}
                (i, kind) => panic!("layer {i} got unexpected kind {kind:?}"),
            }
        }
    }

    #[test]
    fn expand_to_layers_explicit_partition() {
        let v = FfnLayerPolicy::from_spec("{walk:k=50}@layers=0-15;{dense}@layers=16-33")
            .unwrap()
            .validate_for(34)
            .unwrap();
        let kinds = v.expand_to_layers();
        for (i, k) in kinds.iter().enumerate() {
            if i < 16 {
                assert_eq!(**k, FfnBackendKind::Walk { k: Some(50) });
            } else {
                assert_eq!(**k, FfnBackendKind::Dense);
            }
        }
    }
}
