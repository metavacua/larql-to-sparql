//! Choose an extraction target, then execute it.
//!
//! ```text
//! orchestrator        chooses the target      ← here
//! writers             execute one target each, and are leaves
//! checkpoint adapter  exposes a source's bytes and facts
//! container           records what it was told
//! ```
//!
//! The rule this file exists to keep: **writers do not choose container
//! generations.** Teaching `write_model_weights_kquant_with_opts` to notice
//! a native-capable source and branch would put VINDEX2/VINDEX3 policy
//! inside the writer that only knows how to transcode, and every later
//! representation question would accumulate in the same place.
//!
//! # Scope: expert banks, not the whole model
//!
//! This decides how the **routed expert banks** are stored. The spine —
//! tokenizer, config, embeddings, attention, norms, routers, shared/dense
//! FFN, LM head — is the legacy k-quant writer's job under either target,
//! and this function neither writes nor replaces it. That mirrors how a
//! composed run is actually consumed today: `ContainerRoutedBackend` takes
//! its spine from a VINDEX2 model and only its routed experts from a
//! VINDEX3 container.
//!
//! Saying so plainly matters, because "native extraction" could easily be
//! read as "a VINDEX3 model", and it is not one yet. Container
//! completeness is a separate job.

use std::path::Path;

use larql_models::ExpertFormat;

use crate::format::vindex3::ContainerBuilder;
use crate::format::weights::write_f32::WeightSource;
use crate::VindexError;

use super::native_moe::NativeMoeLayer;
use super::target::{admit, ExtractionRequest, ExtractionTarget, SourceCapabilities};

/// What the orchestrator did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExpertBankOutcome {
    /// Nothing extra was written: the legacy k-quant writer emits expert
    /// banks inline as part of the model weights, so this target needs no
    /// second artifact.
    LegacyInline,
    /// A VINDEX3 container was written, holding `layers` routed banks in
    /// the checkpoint's own representation.
    NativeContainer { layers: usize },
}

impl ExpertBankOutcome {
    pub fn target(&self) -> ExtractionTarget {
        match self {
            Self::LegacyInline => ExtractionTarget::LegacyKQuant,
            Self::NativeContainer { .. } => ExtractionTarget::NativeV3,
        }
    }
}

/// What this orchestrator can natively extract from `source`.
///
/// **Reports this orchestrator's reach, not the format's ambition.** Only
/// packed-MXFP4 checkpoints have a checkpoint adapter today, so anything
/// else answers `region_format: None` and a `Native` request is refused by
/// name. That is deliberately narrower than VINDEX3 itself — the container
/// happily holds BF16 and k-quant banks, and the Gemma import examples
/// write them — but those read from an existing VINDEX2 store rather than
/// from a checkpoint, which is a different source entirely.
///
/// Capabilities are probed from the **first layer that answers**, not from
/// the architecture alone: whether a source can hand over raw bytes is a
/// property of the source, and the layout is a property of the artifact.
pub fn inspect_capabilities(source: &dyn WeightSource) -> Result<SourceCapabilities, VindexError> {
    let arch = source.arch();
    let format = arch.expert_format();
    if format != ExpertFormat::PackedMxfp4 {
        // Honest about reach: no adapter, so no native route, and the
        // refusal will say the encoding has no region format.
        return Ok(SourceCapabilities::from_expert_format(format, None));
    }
    for layer in 0..source.num_layers() as u32 {
        // A dense layer in a hybrid stack yields `None`; keep looking
        // rather than concluding the model has no native banks.
        if let Some(native) = NativeMoeLayer::read(source, arch, layer)? {
            return Ok(native.capabilities());
        }
    }
    // Declares packed-MXFP4 experts, but no layer produced any. Not a
    // native source, and not an error — the legacy route still applies.
    Ok(SourceCapabilities::from_expert_format(format, None))
}

/// Decide and execute the expert-bank route.
///
/// `request` is the caller's intent; the target is what admission decided.
/// A `Native` request that the source cannot satisfy **errors** — it is
/// never quietly downgraded, because a caller comparing native bytes
/// against the Q6_K control would otherwise be handed the control twice.
pub fn extract_expert_banks(
    source: &dyn WeightSource,
    dest: &Path,
    request: ExtractionRequest,
    model_name: &str,
) -> Result<ExpertBankOutcome, VindexError> {
    let caps = inspect_capabilities(source)?;
    match admit(request, &caps)? {
        ExtractionTarget::LegacyKQuant => Ok(ExpertBankOutcome::LegacyInline),
        ExtractionTarget::NativeV3 => write_native_container(source, dest, model_name),
    }
}

/// Stream every routed layer's native bytes into a VINDEX3 container.
///
/// One layer is held at a time — `ContainerBuilder` writes each to its
/// final path and forgets it — so peak memory is one layer's experts
/// regardless of model size.
fn write_native_container(
    source: &dyn WeightSource,
    dest: &Path,
    model_name: &str,
) -> Result<ExpertBankOutcome, VindexError> {
    let arch = source.arch();
    let num_layers = source.num_layers();
    let mut builder = ContainerBuilder::create(dest)?;
    let mut layers = 0usize;

    for layer in 0..num_layers as u32 {
        // Dense layers in a hybrid stack are skipped, not refused: the
        // spine is a legitimate part of the architecture and this route
        // is about the routed banks.
        let Some(native) = NativeMoeLayer::read(source, arch, layer)? else {
            continue;
        };
        builder.add_moe_layer(&native.as_source())?;
        layers += 1;
    }

    // No `layers == 0` guard here on purpose: `ContainerBuilder::finish`
    // already refuses to write an index for an empty container, and a
    // second statement of that invariant is one that can drift out of
    // step with the first.
    builder.finish(
        model_name,
        arch.family(),
        arch.config().hidden_size,
        num_layers,
    )?;
    Ok(ExpertBankOutcome::NativeContainer { layers })
}

#[cfg(test)]
#[path = "orchestrate_tests.rs"]
mod tests;
