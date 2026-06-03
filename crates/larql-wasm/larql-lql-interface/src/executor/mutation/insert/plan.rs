//! Phase 1 of `INSERT INTO EDGES` (Compose mode): resolve the install
//! plan — which layers to write to, the target-token embedding for the
//! down vector, and whether the vindex carries model weights (which
//! determines whether later phases can run the canonical-prompt forward
//! pass for residual capture).

use crate::error::LqlError;
use crate::executor::Session;

/// Everything `exec_insert` needs to know about a planned compose-mode
/// install after reading the vindex config and embeddings. Small enough
/// to pass by reference to each subsequent phase.
pub(super) struct InstallPlan {
    /// Layer(s) to install the slot at. Always single-element in the
    /// current pipeline — `SPAN_HALF_LO/HI` are both 0 — but the type
    /// is a `Vec` so a future multi-layer install can drop in without
    /// a signature change.
    pub layers: Vec<usize>,
    /// Model hidden size — width of gate / up / down vectors.
    pub hidden: usize,
    /// Target-token embedding × `embed_scale`. First subtoken of
    /// `" {target}"` only; multi-token targets use only the first
    /// subtoken's embedding so the down vector unembeds cleanly (see
    /// insert/mod.rs for the full rationale).
    pub target_embed: Vec<f32>,
    /// First subtoken id for `" {target}"` — what the slot unembeds to.
    pub target_id: u32,
    /// True iff the vindex carries model weights. Gates residual capture
    /// (Phase 1b), balance, and cross-fact regression checks (Phase 3).
    pub use_constellation: bool,
}

impl Session {
    /// Read the vindex config + tokenizer + embeddings and build the
    /// `InstallPlan`. Pure config-side work: no forward passes, no
    /// residual capture, no mutation of `self`. Phase 1b
    /// (`capture_install_residuals`) does the expensive model work.
    pub(super) fn plan_install(
        &self,
        target: &str,
        layer_hint: Option<u32>,
    ) -> Result<InstallPlan, LqlError> {
        // Single-layer install — matches the Python reference exactly.
        // Earlier drafts used an 8-layer span (L20-L27) which is a
        // leftover from pre-install_compiled_slot work. With the
        // current strong-gate install (×30 scale), spreading the
        // payload across 8 layers lets the slot fire on any prompt
        // with even weak cosine alignment and hijacks unrelated
        // prompts (0/10 retrieval + 4/4 bleed on the 10-fact
        // constellation, previous run). One layer keeps the
        // signal-to-noise ratio the Python reference validated.
        const SPAN_HALF_LO: usize = 0;
        const SPAN_HALF_HI: usize = 0;

        let (path, config, _patched) = self.require_vindex()?;

        let bands = config
            .layer_bands
            .clone()
            .or_else(|| larql_vindex::LayerBands::for_family(&config.family, config.num_layers))
            .unwrap_or(larql_vindex::LayerBands {
                syntax: (0, config.num_layers.saturating_sub(1)),
                knowledge: (0, config.num_layers.saturating_sub(1)),
                output: (0, config.num_layers.saturating_sub(1)),
            });

        let layers = if let Some(l) = layer_hint {
            // `AT LAYER N` pins the install to a single layer.
            // Earlier versions treated this as a span centre and
            // installed across 8 layers; with the install_compiled_slot
            // install (×30 gate scale) that produced strong
            // cross-prompt hijack. See SPAN_HALF_LO/HI above.
            let center = l as usize;
            let max_layer = config.num_layers.saturating_sub(1);
            let lo = center.saturating_sub(SPAN_HALF_LO);
            let hi = (center + SPAN_HALF_HI).min(max_layer);
            (lo..=hi).collect::<Vec<usize>>()
        } else {
            // Default: the second-to-last layer of the knowledge
            // band — matches the Python reference's L26 choice on
            // Gemma 4B (`experiments/14_vindex_compilation` uses
            // INSTALL_LAYER = 26 which is knowledge.1 − 1). This
            // is where semantic retrieval has stabilised but the
            // residual hasn't yet been committed to output
            // formatting. One layer only.
            let layer = bands
                .knowledge
                .1
                .saturating_sub(1)
                .min(config.num_layers.saturating_sub(1));
            vec![layer]
        };

        let (embed, embed_scale) = larql_vindex::load_vindex_embeddings(path)
            .map_err(|e| LqlError::exec("failed to load embeddings", e))?;
        let tokenizer = larql_vindex::load_vindex_tokenizer(path)
            .map_err(|e| LqlError::exec("failed to load tokenizer", e))?;

        let hidden = embed.shape()[1];

        // Target embedding for down vector.
        //
        // We use ONLY the first token of `" " + target` (leading
        // space forces subword merging under BPE/SentencePiece).
        // Averaging across multi-token targets produces a blended
        // embedding that at unembed returns tail subtokens instead
        // of the target's first token — e.g. for "Canberra"
        // tokenised as [Can, berra] the averaged down vector
        // pushes the logits toward "berra" when we want "Can"
        // (which merges with "berra" in the continuation, still
        // producing "Canberra"). Matches Python
        // `install_compiled_slot` semantics in
        // `experiments/14_vindex_compilation`.
        let spaced_target = format!(" {target}");
        let target_encoding = tokenizer
            .encode(spaced_target.as_str(), false)
            .map_err(|e| LqlError::exec("tokenize error", e))?;
        let all_target_ids: Vec<u32> = target_encoding.get_ids().to_vec();
        let target_id = all_target_ids.first().copied().unwrap_or(0);

        let mut target_embed = vec![0.0f32; hidden];
        let row = embed.row(target_id as usize);
        for j in 0..hidden {
            target_embed[j] = row[j] * embed_scale;
        }

        Ok(InstallPlan {
            layers,
            hidden,
            target_embed,
            target_id,
            use_constellation: config.has_model_weights,
        })
    }
}
