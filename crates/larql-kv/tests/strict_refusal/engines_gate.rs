//! The gate, swept across every engine rather than every entry point.
//!
//! [`crate::gate`] fixes the engine (`StandardEngine`) and varies the entry
//! point; this fixes the operation and varies the engine. Both axes matter
//! because a refusal has to survive two different rings — the dispatch
//! helpers and `larql_kv::engines::layer_ffn_or_moe` — and either can be
//! correct while the other erases.

use crate::engines::{drive, Coverage, Op, ALL, APOLLO};
use larql_execution::RefusalKind;
use larql_inference::ffn::MoeFfn;
use larql_inference::test_utils::make_test_gemma4_moe_weights;

use crate::routes::{ExecutingRoute, RefusingRoute};

/// Strict + a refusing route → `Err`, carrying the refusal's own kind, in
/// every engine that dispatches experts, on both operations.
///
/// This is the closing condition for "the other five engines". Before it,
/// `layer_ffn_or_moe` logged the refusal to stderr and returned the dense
/// half, so a strict route through markov-rs, markov-rs-codec,
/// turbo-quant, unlimited-context or boundary-per-layer produced a
/// complete-looking hidden state whose experts never ran — and the engine
/// had no way to know.
#[test]
fn strict_refusal_terminates_in_every_expert_routing_engine() {
    let weights = make_test_gemma4_moe_weights();
    assert!(
        weights.arch.is_hybrid_moe(),
        "fixture must take the MoE branch or this test proves nothing"
    );

    let mut checked = 0usize;
    let mut engines = 0usize;
    for under_test in ALL.iter().filter(|e| e.coverage == Coverage::RoutesExperts) {
        engines += 1;
        for op in Op::ALL {
            for kind in RefusalKind::ALL {
                let route = RefusingRoute::new(kind);
                let ffn = MoeFfn::strict(&weights, &route);
                let label = under_test.label;

                let err = match drive(under_test, op, &weights, &ffn) {
                    Ok(hidden) => panic!(
                        "{label}/{op:?} with a {kind} refusal returned Ok({:?}) — a strict \
                         route produced output for a layer whose experts never ran",
                        hidden.shape()
                    ),
                    Err(e) => e,
                };
                assert_eq!(
                    err.refusal_kind(),
                    Some(kind),
                    "{label}/{op:?}: the refusal's own classification must survive to the \
                     engine — flattening it here sends someone to repair the wrong thing; \
                     got {err:?}"
                );
                checked += 1;
            }
        }
    }
    assert_eq!(
        checked,
        engines * Op::ALL.len() * RefusalKind::ALL.len(),
        "coverage shrank — no silent caps"
    );
    assert!(
        engines >= 7,
        "expert-routing engine coverage shrank ({engines} < 7)"
    );
}

/// The same engines serve normally when the route executes.
///
/// Without this the gate above is satisfiable by an engine that refuses
/// everything, which is a different bug wearing the same result.
#[test]
fn an_executing_route_serves_in_every_expert_routing_engine() {
    let weights = make_test_gemma4_moe_weights();
    for under_test in ALL.iter().filter(|e| e.coverage == Coverage::RoutesExperts) {
        for op in Op::ALL {
            let ffn = MoeFfn::strict(&weights, &ExecutingRoute);
            let hidden = drive(under_test, op, &weights, &ffn).unwrap_or_else(|e| {
                panic!(
                    "{}/{op:?}: an executing route must serve: {e:?}",
                    under_test.label
                )
            });
            assert_eq!(hidden.shape(), &[1, weights.hidden_size]);
            assert!(
                ffn.all_experts_executed() && ffn.refusal().is_none(),
                "{}/{op:?}: a clean run must record nothing",
                under_test.label
            );
        }
    }
}

/// An engine with no expert seam refuses the **architecture**, before any
/// route is consulted.
///
/// This is the correction for the worse failure. Such an engine used to run
/// the dense half of every layer and return an apparently valid answer — a
/// different model wearing the same answer shape, which nothing downstream
/// could detect. `Unsupported` is the honest classification: the operands are
/// present and well-formed, this executor has no bound kernel for them, and
/// the response is to pick another.
///
/// Asserted with a route that *executes*, so the refusal cannot be coming
/// from the route. It is the engine declining the model.
#[test]
fn an_engine_with_no_expert_seam_refuses_the_architecture_itself() {
    let weights = make_test_gemma4_moe_weights();
    let no_seam: Vec<_> = ALL
        .iter()
        .filter(|e| e.coverage == Coverage::NoExpertSeam)
        .chain(std::iter::once(&APOLLO))
        .collect();

    let mut checked = 0usize;
    for under_test in &no_seam {
        let label = under_test.label;
        for op in Op::ALL {
            // An *executing* route, so the refusal cannot be coming from the
            // route — it is the engine declining the model.
            let ffn = MoeFfn::strict(&weights, &ExecutingRoute);
            let err = match drive(under_test, op, &weights, &ffn) {
                Ok(hidden) => panic!(
                    "{label}/{op:?} returned Ok({:?}) — a forward with no expert seam \
                     answered for a model whose weights declare routed experts",
                    hidden.shape()
                ),
                Err(e) => e,
            };
            assert_eq!(
                err.refusal_kind(),
                Some(RefusalKind::Unsupported),
                "{label}/{op:?}: the operands are fine and this executor cannot serve \
                 them, which is exactly Unsupported; got {err:?}"
            );
            assert!(
                err.engine_state_is_retryable(),
                "{label}/{op:?}: refusing before any forward work mutates nothing"
            );
            assert!(
                ffn.refusal().is_none(),
                "{label}/{op:?}: the route was never consulted, so nothing may be recorded"
            );
            checked += 1;
        }
    }
    assert_eq!(no_seam.len(), 1, "no-expert-seam population changed");
    assert_eq!(checked, no_seam.len() * Op::ALL.len(), "coverage shrank");
}
