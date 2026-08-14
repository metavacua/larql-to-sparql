//! Semantic representability plan over architecture inventories (V3-G1/G2).
//!
//! Consumes the G0 inventory (`larql inspect-hf`) for one or more artifacts
//! treated as a model system, and answers: **can the VINDEX3 schema
//! faithfully describe this system — and if not, exactly why not?**
//!
//! Since G2, "representable" has one definition: **the system-graph builder
//! placed it** ([`super::graph::build_from_inventories`]). Objects the
//! builder placed are representable with their graph ids as proof; groups
//! it could not place, and interfaces it could not resolve, come back as
//! blocking findings. There is no separate capability table to drift out of
//! sync with the schema.
//!
//! The other finding sources are unchanged from G1:
//!
//! - `mismatched` — declared-vs-resolved value comparison (`consumed` is
//!   never trusted; values are compared);
//! - `unrepresented` unconsumed config keys, graded by semantic class,
//!   where a key nobody has judged (`unknown`) blocks.
//!
//! The verdict is fail-closed and the exit gate is mechanical:
//! `blocking == 0` before a single weight byte is converted.

pub mod compare;
pub mod report;
pub mod semantics;

#[cfg(test)]
mod tests;
/// Glimmer-shaped inventory fixtures, shared with the graph tests.
#[cfg(test)]
pub mod tests_support;

use larql_models::inventory::{ArchitectureInventory, KeyStatus};

use super::graph::{build_from_inventories, BuiltGraph, ComponentRole};

pub use report::{
    ArtifactPlan, Finding, FindingCategory, InterfacePlan, PlanSummary, SemanticClass, SystemPlan,
    PLAN_SCHEMA,
};

/// Build the system plan over one or more inventories.
///
/// `named` pairs one display name per inventory (the CLI passes directory
/// stems).
pub fn plan_system(named: &[(String, ArchitectureInventory)]) -> SystemPlan {
    let built = build_from_inventories(named);

    let artifacts: Vec<ArtifactPlan> = named
        .iter()
        .map(|(name, inventory)| plan_artifact(name, inventory, &built))
        .collect();

    let interfaces: Vec<InterfacePlan> = built
        .graph
        .edges
        .iter()
        .map(|edge| InterfacePlan {
            producer_component: edge.producer_component.clone(),
            producer_layers: edge.producer_layers.clone(),
            consumer_component: edge.consumer_component.clone(),
            consumer_object: edge.consumer_object.clone(),
            block_size: edge.block_size,
        })
        .collect();

    let mut summary = PlanSummary::default();
    for finding in artifacts.iter().flat_map(|a| &a.findings) {
        match finding.category {
            FindingCategory::Representable => summary.representable += 1,
            FindingCategory::Mismatched => summary.mismatched += 1,
            FindingCategory::Unrepresented => summary.unrepresented += 1,
            FindingCategory::Interface => summary.interfaces += 1,
        }
        if finding.blocks() {
            summary.blocking += 1;
        }
    }
    summary.interfaces += interfaces.len();
    let admissible = summary.blocking == 0;

    SystemPlan {
        schema: PLAN_SCHEMA,
        artifacts,
        interfaces,
        admissible,
        summary,
        graph: built.graph,
    }
}

/// Plan one artifact: value comparison, unconsumed-key grading, and the
/// graph builder's verdict on its tensors, topology and interfaces.
fn plan_artifact(
    name: &str,
    inventory: &ArchitectureInventory,
    built: &BuiltGraph,
) -> ArtifactPlan {
    let mut findings = compare::compare(inventory);
    findings.extend(unconsumed_key_findings(inventory));
    findings.extend(placed_object_findings(name, built));
    findings.extend(unplaced_group_findings(name, built));
    findings.extend(attention_policy_findings(name, built));
    findings.extend(execution_surface_findings(name, built));
    findings.extend(unresolved_interface_findings(name, built));
    ArtifactPlan {
        name: name.to_string(),
        model_type: inventory.identity.model_type.clone(),
        findings,
    }
}

/// Every unconsumed config key, graded by the semantics registry. A key the
/// registry has never seen grades `Unknown` and blocks — unjudged is not
/// admissible.
fn unconsumed_key_findings(inventory: &ArchitectureInventory) -> Vec<Finding> {
    inventory
        .config_keys
        .iter()
        .filter(|fact| fact.status == KeyStatus::Unconsumed)
        .map(|fact| {
            let class = semantics::classify_key(semantics::leaf_of(&fact.path));
            Finding {
                category: FindingCategory::Unrepresented,
                class,
                component: semantics::component_of(&fact.path),
                subject: fact.path.clone(),
                declared: Some(fact.value.clone()),
                resolved: None,
                detail: "declared by the checkpoint, read by nothing in any registered parser"
                    .to_string(),
            }
        })
        .collect()
}

/// One representable finding per logical object this artifact's tensors
/// bind into — the graph id is the proof of a home.
fn placed_object_findings(artifact: &str, built: &BuiltGraph) -> Vec<Finding> {
    built
        .graph
        .objects
        .iter()
        .filter(|object| {
            object
                .source_bindings
                .iter()
                .any(|b| b.artifact == artifact)
        })
        .map(|object| {
            let bytes: u64 = object
                .source_bindings
                .iter()
                .filter(|b| b.artifact == artifact)
                .map(|b| b.bytes)
                .sum();
            let encodings: Vec<&str> = object
                .representations
                .iter()
                .map(|r| r.encoding.as_str())
                .collect();
            Finding {
                category: FindingCategory::Representable,
                class: SemanticClass::TensorSemantic,
                component: object.component.clone(),
                subject: object.id.clone(),
                declared: None,
                resolved: None,
                detail: format!(
                    "placed as `{}` ({} bytes from this artifact; encodings: {})",
                    object.kind.name(),
                    bytes,
                    encodings.join(", "),
                ),
            }
        })
        .collect()
}

/// Blocking finding per tensor group the builder could not place.
fn unplaced_group_findings(artifact: &str, built: &BuiltGraph) -> Vec<Finding> {
    built
        .unplaced
        .iter()
        .filter(|u| u.artifact == artifact)
        .map(|u| Finding {
            category: FindingCategory::Unrepresented,
            class: SemanticClass::Unknown,
            component: String::new(),
            subject: u.prefix.clone(),
            declared: None,
            resolved: None,
            detail: u.reason.clone(),
        })
        .collect()
}

/// The attention policy of each component this artifact sourced: recorded
/// per layer in the graph (span, window, position incl. NoPE), so a hybrid
/// interleave is representable — and directly consumable by KV planning.
fn attention_policy_findings(artifact: &str, built: &BuiltGraph) -> Vec<Finding> {
    built
        .graph
        .components
        .iter()
        .filter(|c| c.source_artifact == artifact && c.role != ComponentRole::Perception)
        .filter_map(|component| {
            let table = component.attention.as_ref()?;
            let sliding = table
                .iter()
                .filter(|l| matches!(l.span, super::graph::policy::AttentionSpan::Sliding))
                .count();
            let nope = table
                .iter()
                .filter(|l| l.position == larql_models::config::PositionPolicy::None)
                .count();
            Some(Finding {
                category: FindingCategory::Representable,
                class: SemanticClass::ExecutionSemantic,
                component: component.id.clone(),
                subject: "attention_policy".to_string(),
                declared: None,
                resolved: None,
                detail: format!(
                    "per-layer policy recorded on component `{}`: {} sliding / {} full, \
                     {} NoPE layer(s)",
                    component.id,
                    sliding,
                    table.len() - sliding,
                    nope,
                ),
            })
        })
        .collect()
}

/// Execution-surface verdict per component this artifact sourced: a
/// representable finding when the surface is complete, a blocking one
/// itemising the missing source facts when it is not (V3-G5a). An
/// executor with a partial surface would have to default, which G5
/// forbids — so incompleteness refuses conversion up front.
fn execution_surface_findings(artifact: &str, built: &BuiltGraph) -> Vec<Finding> {
    let mut findings: Vec<Finding> = built
        .graph
        .components
        .iter()
        .filter(|c| c.source_artifact == artifact && c.execution.is_some())
        .map(|component| Finding {
            category: FindingCategory::Representable,
            class: SemanticClass::ExecutionSemantic,
            component: component.id.clone(),
            subject: format!("{}.execution_surface", component.id),
            declared: None,
            resolved: None,
            detail: format!(
                "execution surface complete (attention, ffn, norm{})",
                if component
                    .execution
                    .as_ref()
                    .is_some_and(|s| s.head.is_some())
                {
                    ", head"
                } else {
                    ""
                }
            ),
        })
        .collect();
    findings.extend(
        built
            .incomplete_surfaces
            .iter()
            .filter(|s| s.artifact == artifact)
            .map(|s| Finding {
                category: FindingCategory::Unrepresented,
                class: SemanticClass::ExecutionSemantic,
                component: s.component.clone(),
                subject: format!("{}.execution_surface", s.component),
                declared: None,
                resolved: None,
                detail: format!(
                    "execution surface incomplete — missing: {}",
                    s.missing.join(", ")
                ),
            }),
    );
    findings
}

/// Blocking finding per interface the builder could not resolve.
fn unresolved_interface_findings(artifact: &str, built: &BuiltGraph) -> Vec<Finding> {
    built
        .unresolved_interfaces
        .iter()
        .filter(|u| u.artifact == artifact)
        .map(|u| Finding {
            category: FindingCategory::Interface,
            class: SemanticClass::InterfaceSemantic,
            component: String::new(),
            subject: "hidden_state_interface".to_string(),
            declared: None,
            resolved: None,
            detail: u.reason.clone(),
        })
        .collect()
}
