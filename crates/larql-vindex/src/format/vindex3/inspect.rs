//! `larql vindex3 inspect` — reconstruct a system from its container alone.
//!
//! This is the G3 gate: everything reported here comes from `index.json`,
//! `system_graph.json` and segment headers. No transformers config, no HF
//! filenames, no safetensors headers, no architecture registry — once
//! encoded, the source checkpoint is not an authority, and this module
//! proves it by never touching one.

use std::path::Path;

use larql_models::config::PositionPolicy;
use serde::Serialize;

use super::encode::segment::read_segment_header;
use super::graph::policy::AttentionSpan;
use super::graph::SystemGraph;
use super::index::Vindex3Index;
use crate::error::VindexError;
use crate::format::filenames::INDEX_JSON;

/// A structural problem found while inspecting a container.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum InspectionDefect {
    /// Graph and directory disagree about what exists.
    DirectoryIncoherent(String),
    /// A segment file disagrees with its directory entry.
    SegmentIncoherent(String),
    /// Payload hash did not match the directory (verify mode only).
    PayloadCorrupt(String),
}

/// One component, as reconstructed purely from the container.
#[derive(Debug, Clone, Serialize)]
pub struct ComponentSummary {
    pub id: String,
    pub role: String,
    pub num_layers: usize,
    pub hidden_size: usize,
    /// (sliding, full) layer counts when the policy table exists.
    pub sliding_layers: Option<usize>,
    pub full_layers: Option<usize>,
    /// Layers with no positional encoding.
    pub nope_layers: Option<usize>,
    pub window: Option<usize>,
}

/// The full reconstruction.
#[derive(Debug, Clone, Serialize)]
pub struct SystemInspection {
    pub components: Vec<ComponentSummary>,
    pub graph: SystemGraph,
    pub index: Vindex3Index,
    pub defects: Vec<InspectionDefect>,
}

impl SystemInspection {
    pub fn is_coherent(&self) -> bool {
        self.defects.is_empty()
    }

    /// Execution completeness of the persisted graph (the V3-G5a gate):
    /// every component with executable objects carries the surface those
    /// operations read — answered from the container alone.
    pub fn execution_completeness(&self) -> Vec<super::graph::CompletenessDefect> {
        super::graph::execution_completeness(&self.graph)
    }
}

/// Inspect a container. `verify_payloads` additionally re-hashes every
/// segment file and compares against the directory — slower, but detects
/// byte corruption with no source access.
pub fn inspect_container(
    root: &Path,
    verify_payloads: bool,
) -> Result<SystemInspection, VindexError> {
    let index: Vindex3Index = serde_json::from_str(
        &std::fs::read_to_string(root.join(INDEX_JSON))
            .map_err(|e| VindexError::Parse(format!("read {INDEX_JSON}: {e}")))?,
    )
    .map_err(|e| VindexError::Parse(format!("parse {INDEX_JSON}: {e}")))?;

    let graph_name = index.system_graph.as_deref().ok_or_else(|| {
        VindexError::Parse(
            "container records no system graph — nothing to reconstruct \
             (a routed-MoE container opens via `larql show`, not here)"
                .into(),
        )
    })?;
    let graph: SystemGraph = serde_json::from_str(
        &std::fs::read_to_string(root.join(graph_name))
            .map_err(|e| VindexError::Parse(format!("read {graph_name}: {e}")))?,
    )
    .map_err(|e| VindexError::Parse(format!("parse {graph_name}: {e}")))?;

    let mut defects: Vec<InspectionDefect> = graph
        .validate()
        .into_iter()
        .map(|d| InspectionDefect::DirectoryIncoherent(format!("graph defect: {d:?}")))
        .collect();

    // Directory ↔ graph coherence: every object materialised, every entry
    // grounded in an object.
    for object in &graph.objects {
        if !index
            .representations
            .values()
            .any(|e| e.object == object.id)
        {
            defects.push(InspectionDefect::DirectoryIncoherent(format!(
                "object `{}` has no representation in the directory",
                object.id
            )));
        }
    }
    for (id, entry) in &index.representations {
        if !graph.objects.iter().any(|o| o.id == entry.object) {
            defects.push(InspectionDefect::DirectoryIncoherent(format!(
                "directory entry `{id}` references unknown object `{}`",
                entry.object
            )));
        }
        // Segment header agreement.
        match read_segment_header(&root.join(&entry.segment)) {
            Ok((header, payload_start)) => {
                if header.representation != *id {
                    defects.push(InspectionDefect::SegmentIncoherent(format!(
                        "`{}` says it materialises `{}`, directory says `{id}`",
                        entry.segment, header.representation
                    )));
                }
                if header.tensors.len() != entry.tensor_count {
                    defects.push(InspectionDefect::SegmentIncoherent(format!(
                        "`{}` carries {} tensors, directory says {}",
                        entry.segment,
                        header.tensors.len(),
                        entry.tensor_count
                    )));
                }
                let file_len = std::fs::metadata(root.join(&entry.segment))
                    .map(|m| m.len())
                    .unwrap_or(0);
                if file_len != payload_start + entry.payload_bytes {
                    defects.push(InspectionDefect::SegmentIncoherent(format!(
                        "`{}` is {file_len} bytes, expected {} header + {} payload",
                        entry.segment, payload_start, entry.payload_bytes
                    )));
                }
                if verify_payloads {
                    verify_segment_hash(root, entry, &mut defects)?;
                }
            }
            Err(e) => defects.push(InspectionDefect::SegmentIncoherent(format!(
                "`{}`: {e}",
                entry.segment
            ))),
        }
    }

    let components = graph
        .components
        .iter()
        .map(|c| {
            let table = c.attention.as_ref();
            let sliding = table.map(|t| {
                t.iter()
                    .filter(|l| matches!(l.span, AttentionSpan::Sliding))
                    .count()
            });
            ComponentSummary {
                id: c.id.clone(),
                role: format!("{:?}", c.role).to_lowercase(),
                num_layers: c.num_layers,
                hidden_size: c.hidden_size,
                sliding_layers: sliding,
                full_layers: table.map(|t| t.len()).zip(sliding).map(|(n, s)| n - s),
                nope_layers: table.map(|t| {
                    t.iter()
                        .filter(|l| l.position == PositionPolicy::None)
                        .count()
                }),
                window: table.and_then(|t| t.iter().find_map(|l| l.window)),
            }
        })
        .collect();

    Ok(SystemInspection {
        components,
        graph,
        index,
        defects,
    })
}

/// Re-hash one segment file and compare against the directory entry.
fn verify_segment_hash(
    root: &Path,
    entry: &super::index::RepresentationEntry,
    defects: &mut Vec<InspectionDefect>,
) -> Result<(), VindexError> {
    let actual = crate::format::checksums::sha256_file(&root.join(&entry.segment))?;
    if actual != entry.segment_sha256 {
        defects.push(InspectionDefect::PayloadCorrupt(format!(
            "`{}` hashes {actual}, directory says {}",
            entry.segment, entry.segment_sha256
        )));
    }
    Ok(())
}
