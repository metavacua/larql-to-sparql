//! The routed-bank override: expert bytes sourced from a VINDEX3 container.
//!
//! # What this closes
//!
//! Every VINDEX3 execution result before this one bound its operands out of a
//! **VINDEX2** file. `moe_bound` says so plainly, and the container ladder
//! (c8/c9) proved only that the same bytes could be *written* to a VINDEX3
//! container — not that anything would ever *read* one to compute with. That
//! left a gap nothing on the execution path could cross: `Vindex3Container`
//! appeared nowhere in this crate.
//!
//! ```text
//! before   VINDEX2 index ──> MoeLayerWeights ──> bound executor
//! now      VINDEX3 container ──> expert regions ─┘
//!                                (everything else still VINDEX2)
//! ```
//!
//! # Exactly one operand source is replaced
//!
//! The spine — tokenizer, config, embeddings, attention, norms, routers,
//! shared/dense FFN, LM head — is read from the VINDEX2 model, unchanged and
//! byte-for-byte. Only classes 4 and 5 (routed gate/up and routed down, spec
//! §4) come from the container. That is what makes a comparison against a
//! plain VINDEX2 run a statement about *the routed bytes* and nothing else: if
//! this route substituted anything further, a divergence would have somewhere
//! else to hide.
//!
//! This is also the K3 deployment topology in miniature — a small resident
//! spine plus huge paged routed banks — but it is **not** artifact identity. A
//! composed run is two directories, and a VINDEX3 model that still needs a
//! VINDEX2 directory to run is not yet a model. Container completeness (the
//! `control/`, `dense/`, `shared/` and sidecar classes of spec §5) is a
//! separate job and this type does not pretend to stand in for it.
//!
//! # Public API only, deliberately
//!
//! Regions are reached through `Vindex3Container -> segment -> region_bytes`
//! and never by parsing LYRW descriptors here. The binary layout is mid-change
//! (24 B → 28 B bank descriptors with an explicit `group_width`); a backend
//! that read the descriptor itself would have to be rewritten alongside it,
//! and would have been a second implementation of the reader in the meantime.
//!
//! # No fallback, ever
//!
//! If a layer, an expert or a role is missing, this refuses. It does not quietly
//! serve the VINDEX2 bytes that are sitting right there in the same process —
//! doing so would make "the model ran from VINDEX3" unfalsifiable, which is the
//! only claim the route exists to support.

use std::path::Path;

use larql_models::ModelWeights;
use ndarray::Array2;

use larql_compute::pipeline_layer::build_moe_weights;
use larql_compute::MoeLayerWeights;
use larql_vindex::format::lyrw2::region_role::RegionRole;
use larql_vindex::format::vindex3::import::routed_storage_key;
use larql_vindex::format::vindex3::Vindex3Container;

use super::moe_backend::{MoeBackendError, MoeExpertBackend};
use super::moe_bound::BoundMoeBackend;

/// Route name for diagnostics. Never branched on.
const ROUTE_NAME: &str = "moe-vindex3-container";

/// Bank ordinal of the routed bank within a segment. c8/c9 write one bank per
/// segment file, and spec §6 (draft-3) makes that a MUST.
const ROUTED_BANK: u16 = 0;

/// Expert bytes served from a VINDEX3 container, everything else from VINDEX2.
pub struct ContainerRoutedBackend {
    container: Vindex3Container,
    inner: BoundMoeBackend,
}

/// Why a container could not be composed with a loaded model.
///
/// Every variant names both sides. A composition refusal that said only
/// "incompatible" would send the reader to the wrong one of two artifacts.
#[derive(Debug)]
pub enum CompositionError {
    Open(String),
    /// The container describes a different model than the spine does.
    Identity {
        spine: String,
        container: String,
    },
    /// A routed layer the model needs is not in the container.
    MissingLayer {
        layer: usize,
    },
    /// The container's segment for a layer cannot be resolved or read.
    UnreadableLayer {
        layer: usize,
        why: String,
    },
    /// A shape the container declares disagrees with the model's.
    Shape {
        layer: usize,
        what: &'static str,
        spine: u64,
        container: u64,
    },
}

impl std::fmt::Display for CompositionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Open(why) => write!(f, "cannot open the routed container: {why}"),
            Self::Identity { spine, container } => write!(
                f,
                "the container describes `{container}` but the model is `{spine}` — \
                 composing them would serve one model's experts inside another"
            ),
            Self::MissingLayer { layer } => write!(
                f,
                "layer {layer} routes to experts but the container has no bank for \
                 it; `--routed-from` never falls back to the VINDEX2 bytes, because \
                 a run that silently did could not support the claim it exists for"
            ),
            Self::UnreadableLayer { layer, why } => {
                write!(f, "layer {layer}'s bank is unreadable: {why}")
            }
            Self::Shape {
                layer,
                what,
                spine,
                container,
            } => write!(
                f,
                "layer {layer}: the model expects {what} {spine}, the container \
                 declares {container}"
            ),
        }
    }
}

impl std::error::Error for CompositionError {}

impl ContainerRoutedBackend {
    /// Open `root` and check it can serve every routed layer `weights` needs.
    ///
    /// All validation happens here — before a prompt is encoded, before a
    /// token is generated. A composition that failed on layer 17 of the first
    /// forward pass would have already printed part of an answer produced by a
    /// model half of which was not the one asked for.
    pub fn open(
        root: &Path,
        weights: &ModelWeights,
        production: bool,
    ) -> Result<Self, CompositionError> {
        let container =
            Vindex3Container::open(root).map_err(|e| CompositionError::Open(e.to_string()))?;

        let backend = Self {
            container,
            inner: if production {
                BoundMoeBackend::production()
            } else {
                BoundMoeBackend::reference()
            },
        };
        backend.check_composable(weights)?;
        Ok(backend)
    }

    /// Refuse any container that cannot serve this model's routed layers.
    ///
    /// Coverage is checked against the **model**, not the container: a
    /// container with thirty banks still cannot serve a model with thirty-one
    /// routed layers, and asking the container what it has would let the
    /// missing one pass unnoticed.
    fn check_composable(&self, weights: &ModelWeights) -> Result<(), CompositionError> {
        let arch = &*weights.arch;
        let spine_model = weights.arch.family().to_string();
        let container_model = self.container.index().family.clone();
        if spine_model != container_model {
            return Err(CompositionError::Identity {
                spine: spine_model,
                container: container_model,
            });
        }

        for layer in 0..weights.num_layers {
            let Some(moe) = build_moe_weights(weights, arch, layer) else {
                continue; // A dense layer routes nowhere; the container owes it nothing.
            };
            let declared = self
                .container
                .layer(layer as u32)
                .ok_or(CompositionError::MissingLayer { layer })?;

            let experts = declared.routed_bank.experts as usize;
            if experts != moe.num_experts {
                return Err(CompositionError::Shape {
                    layer,
                    what: "experts",
                    spine: moe.num_experts as u64,
                    container: experts as u64,
                });
            }
            if let Some(dims) = declared.routed_bank.expert_dims.as_ref() {
                if dims.input as usize != weights.hidden_size {
                    return Err(CompositionError::Shape {
                        layer,
                        what: "hidden size",
                        spine: weights.hidden_size as u64,
                        container: dims.input as u64,
                    });
                }
                if dims.intermediate as usize != moe.intermediate_size {
                    return Err(CompositionError::Shape {
                        layer,
                        what: "semantic intermediate width",
                        spine: moe.intermediate_size as u64,
                        container: dims.intermediate as u64,
                    });
                }
            }

            // Resolve and read the bank now rather than at first token: a key
            // that resolves to a missing file is a composition failure, and
            // discovering it mid-generation would be one too late.
            let expected = expert_region_sizes(&moe);
            self.check_layer_regions(layer, experts, expected)?;
        }
        Ok(())
    }

    /// Every expert's two regions must be present and the size the model expects.
    fn check_layer_regions(
        &self,
        layer: usize,
        experts: usize,
        expected: (usize, usize),
    ) -> Result<(), CompositionError> {
        let reader = self
            .container
            .segment(&routed_storage_key(layer as u32))
            .map_err(|e| CompositionError::UnreadableLayer {
                layer,
                why: e.to_string(),
            })?;
        let (want_gate_up, want_down) = expected;

        for expert in 0..experts as u32 {
            for (role, want) in [
                (RegionRole::GateUpFused, want_gate_up),
                (RegionRole::Down, want_down),
            ] {
                let got = reader
                    .region_bytes(ROUTED_BANK, expert, role)
                    .map_err(|e| CompositionError::UnreadableLayer {
                        layer,
                        why: format!("expert {expert} {role:?}: {e}"),
                    })?
                    .ok_or_else(|| CompositionError::UnreadableLayer {
                        layer,
                        why: format!("expert {expert} has no {role:?} region"),
                    })?;
                if got.len() != want {
                    return Err(CompositionError::Shape {
                        layer,
                        what: "routed region bytes",
                        spine: want as u64,
                        container: got.len() as u64,
                    });
                }
            }
        }
        Ok(())
    }

    /// One line describing what this run actually composed.
    pub fn describe(&self, spine: &Path) -> String {
        format!(
            "composed run: VINDEX2 spine {} + VINDEX3 routed banks {}",
            spine.display(),
            self.container.root().display()
        )
    }
}

/// The byte length each of a layer's two region kinds must have.
///
/// Taken from the incumbent's own slices rather than recomputed from shapes:
/// the model already holds the authoritative lengths, and deriving them here
/// would be a second implementation of the layout to disagree with.
fn expert_region_sizes(moe: &MoeLayerWeights<'_>) -> (usize, usize) {
    (
        moe.experts_gate_up.first().map_or(0, |s| s.len()),
        moe.experts_down.first().map_or(0, |s| s.len()),
    )
}

impl MoeExpertBackend for ContainerRoutedBackend {
    fn forward_moe_seq(
        &self,
        weights: &ModelWeights,
        layer: usize,
        h: &Array2<f32>,
        norm_offset: f32,
        eps: f32,
    ) -> Result<Array2<f32>, MoeBackendError> {
        let arch = &*weights.arch;
        let Some(mut moe) = build_moe_weights(weights, arch, layer) else {
            // No experts to route into. Zeros, matching the in-process path —
            // erroring here would change the model rather than the route.
            return Ok(Array2::zeros((h.nrows(), h.ncols())));
        };

        // Replace *only* the routed operands. Every other field of `moe` —
        // router projection and scales, the norms, the policy flags — stays as
        // the VINDEX2 model produced it.
        let reader = self
            .container
            .segment(&routed_storage_key(layer as u32))
            .map_err(|e| MoeBackendError::Container(format!("layer {layer} bank: {e}")))?;

        let mut gate_up = Vec::with_capacity(moe.num_experts);
        let mut down = Vec::with_capacity(moe.num_experts);
        for expert in 0..moe.num_experts as u32 {
            for (role, sink) in [
                (RegionRole::GateUpFused, &mut gate_up),
                (RegionRole::Down, &mut down),
            ] {
                let bytes = reader
                    .region_bytes(ROUTED_BANK, expert, role)
                    .map_err(|e| {
                        MoeBackendError::Container(format!(
                            "layer {layer} expert {expert} {role:?}: {e}"
                        ))
                    })?
                    .ok_or_else(|| {
                        MoeBackendError::Container(format!(
                            "layer {layer} expert {expert} has no {role:?} region"
                        ))
                    })?;
                sink.push(bytes);
            }
        }
        moe.experts_gate_up = gate_up;
        moe.experts_down = down;

        Ok(self.inner.run_layer(layer, &moe, h, norm_offset, eps)?)
    }

    fn name(&self) -> &'static str {
        ROUTE_NAME
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::make_test_gemma4_moe_weights;
    use larql_vindex::format::lyrw2::region_format::RegionFormat;
    use larql_vindex::format::vindex3::{ContainerBuilder, MoeLayerSource};
    use tempfile::TempDir;

    /// Build a container from the fixture model's own expert bytes.
    ///
    /// `skip` omits that layer, so a test can produce a container that is
    /// well-formed but cannot serve the model it is composed with — the
    /// distinction the coverage check exists to make.
    fn container_for(weights: &larql_models::ModelWeights, skip: Option<usize>) -> TempDir {
        let dir = TempDir::new().unwrap();
        let mut builder = ContainerBuilder::create(dir.path()).unwrap();
        let arch = &*weights.arch;
        let mut wrote = false;
        for layer in 0..weights.num_layers {
            if Some(layer) == skip {
                continue;
            }
            let Some(moe) = build_moe_weights(weights, arch, layer) else {
                continue;
            };
            let source = MoeLayerSource {
                layer: layer as u32,
                experts_gate_up: moe.experts_gate_up.clone(),
                experts_down: moe.experts_down.clone(),
                format: RegionFormat::Q4K,
                hidden_size: weights.hidden_size as u32,
                gate_up_stored_intermediate: moe.intermediate_size as u32,
                down_stored_intermediate: moe.inter_padded() as u32,
                semantic_intermediate: moe.intermediate_size as u32,
                top_k: moe.top_k as u32,
            };
            if builder.add_moe_layer(&source).is_ok() {
                wrote = true;
            }
        }
        assert!(wrote, "fixture produced no routed layers to import");
        builder
            .finish(
                weights.arch.family(),
                weights.arch.family(),
                weights.hidden_size,
                weights.num_layers,
            )
            .unwrap();
        dir
    }

    #[test]
    fn a_container_missing_a_routed_layer_is_refused_before_generation() {
        // Coverage is checked against the model, so an omitted layer must be
        // caught at open — not at the token where that layer first runs, by
        // which point part of an answer has already been produced.
        let weights = make_test_gemma4_moe_weights();
        let full = container_for(&weights, None);
        ContainerRoutedBackend::open(full.path(), &weights, true)
            .expect("a complete container must compose");

        let missing = container_for(&weights, Some(0));
        let err = ContainerRoutedBackend::open(missing.path(), &weights, true)
            .err()
            .expect("a container missing a routed layer must be refused");
        let msg = err.to_string();
        assert!(
            msg.contains("no bank for it") || msg.contains("layer 0"),
            "the refusal must name the missing layer, got: {msg}"
        );
    }

    #[test]
    fn a_container_describing_another_model_is_refused() {
        let weights = make_test_gemma4_moe_weights();
        let dir = container_for(&weights, None);

        // Rewrite the family in place: same banks, different identity.
        let index_path = dir.path().join("index.json");
        let text = std::fs::read_to_string(&index_path).unwrap();
        let mut json: serde_json::Value = serde_json::from_str(&text).unwrap();
        json["family"] = serde_json::Value::String("not-this-model".into());
        std::fs::write(&index_path, serde_json::to_string_pretty(&json).unwrap()).unwrap();

        let err = ContainerRoutedBackend::open(dir.path(), &weights, true)
            .err()
            .expect("composing a container for another model must be refused");
        assert!(
            err.to_string().contains("not-this-model"),
            "the refusal must name both sides, got: {err}"
        );
    }

    #[test]
    fn a_composed_route_reports_both_artifacts() {
        // Provenance is not decoration: a composed run is two directories and
        // a result recorded against only one of them is unreproducible.
        let weights = make_test_gemma4_moe_weights();
        let dir = container_for(&weights, None);
        let backend = ContainerRoutedBackend::open(dir.path(), &weights, true).unwrap();
        let line = backend.describe(std::path::Path::new("/models/spine.vindex"));
        assert!(line.contains("/models/spine.vindex"), "{line}");
        assert!(line.contains(&dir.path().display().to_string()), "{line}");
    }
}
