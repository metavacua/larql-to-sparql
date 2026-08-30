//! Lifecycle-mutation invariant (§7 of `docs/runtime-lifecycle-design.md`):
//! dynamic model lifecycle is scoped to single-model topology only.
//!
//! `RouterTopology` freezes, at boot, which axum `Router` variant
//! `bootstrap::serve` actually built. That's a structural fact about
//! the running process — the route table it describes cannot change
//! shape without rebuilding the whole `Router`, which nothing in this
//! codebase does. `ModelSet`'s live count (via `AppState::is_multi_model`)
//! answers a different question — "how many models are bound right
//! now" — and once lifecycle mutation exists, that number can change
//! while `router_topology` never does. Conflating the two is exactly
//! the bug this module exists to make impossible: a multi-model boot
//! must refuse mutation outright, and a single-model boot must never
//! be allowed to grow past one binding, because axum was never given
//! a second route table to grow into.
//!
//! This module also carries [`LifecycleState`] — the single-slot
//! state flag `POST`/`DELETE /v1/runtime/model` (`routes/runtime_lifecycle.rs`)
//! transition through — and the pure decision functions
//! ([`decide_load`], [`decide_unload`]) behind those handlers. Keeping
//! the *decision* pure and separate from the handler's actual I/O
//! (spawn_blocking a load, poll a drain) means every state-transition
//! case is a synchronous unit test, not an integration test that has
//! to stand up a real container.

/// The router topology `bootstrap::serve` built, frozen at
/// `AppState` construction and never recomputed. See the module doc
/// for why this must not be derived from the live model count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouterTopology {
    /// `routes::single_model_router` — the route table this process
    /// was actually given. Dynamic lifecycle mutation is only ever
    /// permitted here, and only while the bound count stays 0 or 1.
    SingleModel,
    /// `routes::multi_model_router` — a route table sized for the
    /// boot-time model count. No lifecycle mutation is supported
    /// against it; growing or shrinking the bound set would require a
    /// router this process was never built with.
    MultiModel,
}

impl RouterTopology {
    /// The topology `bootstrap::serve` picks for `total_models` models
    /// at boot — the same threshold `AppState::is_multi_model` uses,
    /// applied once, before any mutation could exist. `bootstrap::serve`
    /// calls this directly so the router it builds and the topology
    /// `AppState` freezes can never disagree with each other.
    pub fn for_boot_count(total_models: usize) -> Self {
        if total_models > 1 {
            RouterTopology::MultiModel
        } else {
            RouterTopology::SingleModel
        }
    }
}

/// Why a proposed lifecycle mutation (load/unload/swap) was refused.
/// Both variants are permanent for the life of the process — neither
/// is a transient "try again" condition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LifecycleError {
    /// This process booted multi-model — `bootstrap::serve` built
    /// `multi_model_router` for a fixed, boot-time set of
    /// `/v1/{id}/...` routes. No lifecycle mutation is attempted
    /// against it at all, regardless of what the mutation would do to
    /// the count.
    StaticMultiModelTopology,
    /// This process booted single-model, but the proposed mutation
    /// would leave more than one model bound — the
    /// `single_model_router` route table has nowhere for a second
    /// model to live.
    DynamicMultiModelUnsupported,
}

impl std::fmt::Display for LifecycleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LifecycleError::StaticMultiModelTopology => write!(
                f,
                "this server booted with a static multi-model router topology; \
                 dynamic model lifecycle mutation is not supported"
            ),
            LifecycleError::DynamicMultiModelUnsupported => write!(
                f,
                "this mutation would leave more than one model bound, which the \
                 single-model router topology this server booted with cannot serve"
            ),
        }
    }
}

impl std::error::Error for LifecycleError {}

/// The single-slot model lifecycle state (0↔1 topology only — see
/// [`RouterTopology`]). One `AppState.lifecycle: Mutex<LifecycleState>`
/// guards this flag for the *whole* load/unload sequence, not just the
/// `ModelSet` mutation at the end of it: `Loading`/`Unloading` are set
/// before the actual (possibly slow) work starts, so a concurrent
/// second call sees them immediately and rejects outright rather than
/// racing the first to completion. See
/// `docs/runtime-lifecycle-design.md` §3.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LifecycleState {
    /// Nothing bound.
    Idle,
    /// A load is in flight.
    Loading,
    /// One model bound. `path` is the on-disk container it was loaded
    /// from — comparing it against a new `POST`'s path is what makes
    /// "load the same model again" idempotent rather than a conflict.
    /// Compared as plain `PathBuf` equality (no canonicalization) —
    /// a client that names the identical path string gets the
    /// idempotent case; a client that names an equivalent-but-spelled-
    /// differently path does not. Documented limitation, not a bug:
    /// keeping this exact-match is what keeps it a pure, filesystem-free
    /// comparison.
    Ready {
        model_id: String,
        path: std::path::PathBuf,
    },
    /// An unload is draining.
    Unloading { model_id: String },
}

/// What a `POST /v1/runtime/model` naming `requested_path` should do,
/// given the current state — pure, so every branch below is a
/// synchronous unit test rather than an integration test standing up
/// a real container.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoadDecision {
    /// Transition to `Loading` and actually attempt the load.
    Proceed,
    /// The exact model already requested is already bound — a no-op
    /// success, not an error.
    AlreadyBound,
    /// Refuse outright; the message names why.
    Refuse(String),
}

/// The decision behind `POST /v1/runtime/model`'s single-flighting.
pub fn decide_load(state: &LifecycleState, requested_path: &std::path::Path) -> LoadDecision {
    match state {
        LifecycleState::Loading => {
            LoadDecision::Refuse("a load is already in progress".to_string())
        }
        LifecycleState::Unloading { .. } => {
            LoadDecision::Refuse("an unload is already in progress".to_string())
        }
        LifecycleState::Ready { path, .. } if path == requested_path => LoadDecision::AlreadyBound,
        LifecycleState::Ready { model_id, .. } => LoadDecision::Refuse(format!(
            "model '{model_id}' is already bound; DELETE /v1/runtime/model first \
             (this endpoint does not support atomic replacement)"
        )),
        LifecycleState::Idle => LoadDecision::Proceed,
    }
}

/// What a `DELETE /v1/runtime/model` should do, given the current
/// state — the unload counterpart of [`LoadDecision`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnloadDecision {
    /// Transition to `Unloading` and actually attempt the unload,
    /// against this specific bound model id.
    Proceed { model_id: String },
    /// Nothing bound — unloading is a no-op success.
    AlreadyIdle,
    /// Refuse outright; the message names why.
    Refuse(String),
}

/// The decision behind `DELETE /v1/runtime/model`'s single-flighting.
pub fn decide_unload(state: &LifecycleState) -> UnloadDecision {
    match state {
        LifecycleState::Idle => UnloadDecision::AlreadyIdle,
        LifecycleState::Loading => {
            UnloadDecision::Refuse("a load is in progress; cannot unload yet".to_string())
        }
        LifecycleState::Unloading { .. } => {
            UnloadDecision::Refuse("an unload is already in progress".to_string())
        }
        LifecycleState::Ready { model_id, .. } => UnloadDecision::Proceed {
            model_id: model_id.clone(),
        },
    }
}

/// The pure decision behind [`AppState::validate_lifecycle_mutation`],
/// split out so it's testable without constructing a real `AppState`.
fn check_topology_invariant(
    topology: RouterTopology,
    proposed_count: usize,
) -> Result<(), LifecycleError> {
    if topology == RouterTopology::MultiModel {
        return Err(LifecycleError::StaticMultiModelTopology);
    }
    if proposed_count > 1 {
        return Err(LifecycleError::DynamicMultiModelUnsupported);
    }
    Ok(())
}

impl crate::state::AppState {
    /// Would a lifecycle mutation that leaves `proposed_count` models
    /// bound be allowed? A `MultiModel` boot refuses every mutation
    /// outright; a `SingleModel` boot allows one so long as the bound
    /// count stays 0 or 1. Callers pass the count the mutation would
    /// produce (e.g. "load while idle" proposes 1; "unload the bound
    /// model" proposes 0) — this only ever judges the destination
    /// state, never the mechanics of getting there.
    pub fn validate_lifecycle_mutation(&self, proposed_count: usize) -> Result<(), LifecycleError> {
        check_topology_invariant(self.router_topology, proposed_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn for_boot_count_matches_is_multi_models_own_threshold() {
        assert_eq!(
            RouterTopology::for_boot_count(0),
            RouterTopology::SingleModel
        );
        assert_eq!(
            RouterTopology::for_boot_count(1),
            RouterTopology::SingleModel
        );
        assert_eq!(
            RouterTopology::for_boot_count(2),
            RouterTopology::MultiModel
        );
        assert_eq!(
            RouterTopology::for_boot_count(5),
            RouterTopology::MultiModel
        );
    }

    #[test]
    fn multi_model_topology_refuses_every_mutation() {
        // Even a mutation that would leave the count unchanged, or
        // drop it to 0 or 1, is refused — there's no router to serve
        // any post-mutation shape, so the proposed count never even
        // gets consulted.
        for proposed in [0, 1, 2, 3] {
            assert_eq!(
                check_topology_invariant(RouterTopology::MultiModel, proposed),
                Err(LifecycleError::StaticMultiModelTopology),
                "proposed count {proposed} must still be refused under MultiModel topology"
            );
        }
    }

    #[test]
    fn single_model_topology_allows_zero_or_one() {
        assert_eq!(
            check_topology_invariant(RouterTopology::SingleModel, 0),
            Ok(())
        );
        assert_eq!(
            check_topology_invariant(RouterTopology::SingleModel, 1),
            Ok(())
        );
    }

    #[test]
    fn single_model_topology_refuses_growing_past_one() {
        assert_eq!(
            check_topology_invariant(RouterTopology::SingleModel, 2),
            Err(LifecycleError::DynamicMultiModelUnsupported)
        );
        assert_eq!(
            check_topology_invariant(RouterTopology::SingleModel, 7),
            Err(LifecycleError::DynamicMultiModelUnsupported)
        );
    }

    #[test]
    fn lifecycle_error_display_names_the_reason() {
        assert!(format!("{}", LifecycleError::StaticMultiModelTopology).contains("static"));
        assert!(
            format!("{}", LifecycleError::DynamicMultiModelUnsupported).contains("single-model")
        );
    }

    #[test]
    fn lifecycle_error_is_a_std_error() {
        fn assert_error<E: std::error::Error>() {}
        assert_error::<LifecycleError>();
    }

    // ── decide_load ─────────────────────────────────────────────────

    fn ready(model_id: &str, path: &str) -> LifecycleState {
        LifecycleState::Ready {
            model_id: model_id.to_string(),
            path: std::path::PathBuf::from(path),
        }
    }

    #[test]
    fn decide_load_proceeds_from_idle() {
        assert_eq!(
            decide_load(&LifecycleState::Idle, std::path::Path::new("/a")),
            LoadDecision::Proceed
        );
    }

    #[test]
    fn decide_load_refuses_while_already_loading() {
        assert!(matches!(
            decide_load(&LifecycleState::Loading, std::path::Path::new("/a")),
            LoadDecision::Refuse(_)
        ));
    }

    #[test]
    fn decide_load_refuses_while_unloading() {
        let state = LifecycleState::Unloading {
            model_id: "m".to_string(),
        };
        assert!(matches!(
            decide_load(&state, std::path::Path::new("/a")),
            LoadDecision::Refuse(_)
        ));
    }

    #[test]
    fn decide_load_is_idempotent_for_the_exact_same_path() {
        let state = ready("m", "/a");
        assert_eq!(
            decide_load(&state, std::path::Path::new("/a")),
            LoadDecision::AlreadyBound
        );
    }

    #[test]
    fn decide_load_refuses_a_different_model_without_replacement() {
        let state = ready("m", "/a");
        let decision = decide_load(&state, std::path::Path::new("/b"));
        match decision {
            LoadDecision::Refuse(msg) => {
                assert!(
                    msg.contains('m'),
                    "should name the currently bound model: {msg}"
                );
                assert!(
                    msg.contains("DELETE"),
                    "should point at the unload step: {msg}"
                );
            }
            other => panic!("expected Refuse, got {other:?}"),
        }
    }

    // ── decide_unload ───────────────────────────────────────────────

    #[test]
    fn decide_unload_is_idempotent_when_already_idle() {
        assert_eq!(
            decide_unload(&LifecycleState::Idle),
            UnloadDecision::AlreadyIdle
        );
    }

    #[test]
    fn decide_unload_proceeds_against_the_bound_model_id() {
        let state = ready("m", "/a");
        assert_eq!(
            decide_unload(&state),
            UnloadDecision::Proceed {
                model_id: "m".to_string()
            }
        );
    }

    #[test]
    fn decide_unload_refuses_while_loading() {
        assert!(matches!(
            decide_unload(&LifecycleState::Loading),
            UnloadDecision::Refuse(_)
        ));
    }

    #[test]
    fn decide_unload_refuses_while_already_unloading() {
        let state = LifecycleState::Unloading {
            model_id: "m".to_string(),
        };
        assert!(matches!(decide_unload(&state), UnloadDecision::Refuse(_)));
    }
}
