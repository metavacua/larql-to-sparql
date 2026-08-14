/// Bytes per super-block of the EXPERIMENTAL 160-byte "pre-baked"
/// layout that `cpu::ops::q4_common::quantize_q4_kf` produces.
///
/// **No Metal kernel reads this layout.** Every live `Q4_KF`-tagged
/// kernel (`q4kf_qkv_proj`, `q4kf_proj`, `q4kf_ffn_gate_up`) hardcodes
/// the standard 144-byte GGUF Q4_K block and differs from the `Q4_K`
/// kernels only in its llama.cpp-exact inner loop. The capability audit
/// (F15) found this constant feeding `packed_block_layout`, so any
/// caller sizing a Q4_KF buffer through it disagreed with the shaders'
/// row stride by 16 bytes per super-block. `packed_block_layout` now
/// answers 144; this constant remains only for the CPU-side pre-baked
/// experiment and its tests.
pub const Q4_KF_BLOCK_BYTES: usize = 160;

/// Quantization format for a weight tensor.
/// Names match GGUF conventions (Q4_K, Q6_K, etc.).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[allow(non_camel_case_types)]
pub enum QuantFormat {
    Q4_0,  // 18 bytes per 32 values (one f16 scale)
    Q4_K,  // 144 bytes per 256 values (GGUF-canonical, Ollama-compatible)
    Q4_KF, // 144-byte GGUF Q4_K bytes decoded by the llama.cpp-exact kernels
    Q6_K,  // 210 bytes per 256 values (6-bit with sub-block scales)
    Q8_0,  // int8 values + separate f32 scales
    BF16,  // raw bfloat16 (2 bytes per value, no quantization scales)
    F16,   // raw float16  (2 bytes per value)
    F32,   // raw float32  (4 bytes per value)
    /// BitNet 1.58-bit ternary (GGML I2_S, type 36): 4 trits/byte packed
    /// row-major (`cols/4` bytes per row) plus a separate per-channel f32
    /// scale array. Unlike the block-quant formats, the weight is NOT a
    /// flat `&[u8]` block stream — it is carried by
    /// [`crate::cpu::ops::ternary_matvec::BitLinearWeight`] (bytes + scales)
    /// and served by the dedicated `ternary_matvec` dispatch, not the
    /// block-quant `quant_matvec` path (which has no per-channel-scale input).
    I2S,
}

impl QuantFormat {
    /// Packed block geometry as `(elements_per_block, bytes_per_block)`.
    ///
    /// This is the compute-side mirror of the GGML layout constants used by
    /// the quantizers. Callers that need byte offsets should ask the format
    /// instead of spelling `256 * 144` or `32 * 18` locally.
    pub fn packed_block_layout(self) -> Option<(usize, usize)> {
        use larql_models::quant::ggml;

        match self {
            Self::Q4_0 => Some((ggml::Q4_0_BLOCK_ELEMS, ggml::Q4_0_BLOCK_BYTES)),
            Self::Q4_K => Some((ggml::Q4_K_BLOCK_ELEMS, ggml::Q4_K_BLOCK_BYTES)),
            // Q4_KF is a KERNEL-ROUTE tag over standard 144-byte GGUF
            // Q4_K bytes, not a distinct storage layout: all three live
            // Q4_KF shaders hardcode 144 (audit F15). The 160-byte
            // pre-baked layout (`Q4_KF_BLOCK_BYTES`) has no kernel
            // consumer; answering 160 here mis-sized every buffer
            // derived through this method by 16 bytes per super-block.
            Self::Q4_KF => Some((ggml::Q4_K_BLOCK_ELEMS, ggml::Q4_K_BLOCK_BYTES)),
            Self::Q6_K => Some((ggml::Q6_K_BLOCK_ELEMS, ggml::Q6_K_BLOCK_BYTES)),
            _ => None,
        }
    }

    /// Byte length for a packed row-major matrix with `rows * cols` values.
    ///
    /// Current interleaved FFN fallback stores each matrix contiguously, so
    /// this intentionally preserves the historical flat packing calculation.
    /// Manifest-aware paths should prefer recorded offsets and lengths.
    pub fn packed_matrix_bytes(self, rows: usize, cols: usize) -> Option<usize> {
        let elems = rows.checked_mul(cols)?;
        let (block_elems, block_bytes) = self.packed_block_layout()?;
        Some(elems.div_ceil(block_elems) * block_bytes)
    }

    /// Whether this format uses the GGUF k-quant 256-element super-block
    /// layout that flows through the dedicated Q4_K / Q4_KF / Q6_K matvec
    /// dispatchers (vs the legacy block-32 Q4_0 / Q8_0 path). Used to gate
    /// the "skip Q8 quantize" fast path in `residual_norm` and FFN routing.
    ///
    /// Adding a future k-quant format (e.g. Q5_K) extends this one method,
    /// not the ~10 OR-chains it currently replaces. Roadmap #7
    /// (`FormatRoute` enum) is the fuller version of this idea; this helper
    /// is the contained step that addresses the user-visible code-duplication
    /// cost without rippling through 49 files.
    pub fn is_kquant_family(self) -> bool {
        matches!(self, Self::Q4_K | Self::Q4_KF | Self::Q6_K)
    }

    /// Whether this format uses the llama.cpp-exact "Q4_KF" pre-baked
    /// half-scale fast path (`q4kf_proj` shader). Distinct from the
    /// canonical `Q4_K` GGUF layout used by Ollama extracts.
    pub fn is_q4kf(self) -> bool {
        matches!(self, Self::Q4_KF)
    }

    /// Whether this format uses the legacy block-32 Q8 dispatch path
    /// (`q4_matvec` / `q8_matvec` against pre-quantised Q8 input). The
    /// inverse of [`Self::is_kquant_family`] for the dense matvec dispatch
    /// (the float-input `BF16` / `F16` / `F32` branches don't run on
    /// these dispatchers, so `is_legacy_q8` covers exactly the rest).
    pub fn is_legacy_q8(self) -> bool {
        matches!(self, Self::Q4_0 | Self::Q8_0)
    }

    /// Parse a GGUF-convention registry tag (`"Q4_K"`, `"Q6_K"`, …) into a
    /// `QuantFormat`. The canonical inverse of the names the extractor and
    /// weight manifests record; `None` for any tag with no compute mapping.
    ///
    /// This is the contained version of Roadmap #7's `from_registry_tag`:
    /// it lets the string-keyed matvec dispatchers (`q4k_q8k_matvec_parallel`,
    /// `kquant_forward::cached`) ask the format for its packed layout instead
    /// of re-spelling `(cols/256)*144` locally, without changing their `&str`
    /// call-site signatures.
    pub fn from_registry_tag(tag: &str) -> Option<Self> {
        Some(match tag {
            "Q4_0" => Self::Q4_0,
            "Q4_K" => Self::Q4_K,
            "Q4_KF" => Self::Q4_KF,
            "Q6_K" => Self::Q6_K,
            "Q8_0" => Self::Q8_0,
            "BF16" => Self::BF16,
            "F16" => Self::F16,
            "F32" => Self::F32,
            // BitNet ternary (GGML type 36). The vindex bitnet sidecar tags
            // its I2_S weight stream with this so the registry recognises it.
            "I2_S" => Self::I2S,
            _ => return None,
        })
    }

    /// Inverse of [`Self::from_registry_tag`] — the canonical registry tag
    /// string for this format. `from_registry_tag(f.registry_tag()) == Some(f)`
    /// for every variant. Used by writers that record the per-tensor format
    /// tag into the weight manifest / index.
    pub fn registry_tag(self) -> &'static str {
        match self {
            Self::Q4_0 => "Q4_0",
            Self::Q4_K => "Q4_K",
            Self::Q4_KF => "Q4_KF",
            Self::Q6_K => "Q6_K",
            Self::Q8_0 => "Q8_0",
            Self::BF16 => "BF16",
            Self::F16 => "F16",
            Self::F32 => "F32",
            Self::I2S => "I2_S",
        }
    }

    /// Whether this format is BitNet ternary (I2_S). Served by the dedicated
    /// `ternary_matvec` path with a [`crate::cpu::ops::ternary_matvec::BitLinearWeight`],
    /// never the block-quant `quant_matvec` dispatch.
    pub fn is_ternary(self) -> bool {
        matches!(self, Self::I2S)
    }

    /// Where this format keeps its dequantisation scales.
    ///
    /// Exhaustive by construction: a new format must answer this, rather
    /// than inheriting a default that happens to be right for the formats
    /// that existed when it was added.
    pub fn scale_storage(self) -> ScaleStorage {
        match self {
            // Block-packed: the scale rides inside each block.
            Self::Q4_0 | Self::Q4_K | Self::Q4_KF | Self::Q6_K => ScaleStorage::Inline,
            // "int8 values + separate f32 scales", per this enum's own doc.
            Self::Q8_0 => ScaleStorage::External(ExternalScaleKind::PerBlockF32),
            // Ternary carries a separate per-channel f32 array.
            Self::I2S => ScaleStorage::External(ExternalScaleKind::PerChannelF32),
            Self::BF16 | Self::F16 | Self::F32 => ScaleStorage::None,
        }
    }
}

/// How a format stores its dequantisation scales.
///
/// Distinct from [`QuantFormat::packed_block_layout`] on purpose. That
/// describes *representation geometry*; this describes the *auxiliary
/// storage contract*. They correlate perfectly today — every block-packed
/// format carries its scales inline — but they are not the same property,
/// and a future packed format with external scales would make
/// `packed_block_layout().is_some()` the wrong discriminator. Keeping them
/// apart is the point; conflating them is how the caller ended up
/// reconstructing the format's own rules.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScaleStorage {
    /// Scales live inside the packed blocks. No external array exists —
    /// not "an empty one".
    Inline,
    /// Scales live in a separate f32 array the caller must supply.
    External(ExternalScaleKind),
    /// Unquantised — there are no scales at all.
    None,
}

/// Shape of an external scale array, for formats that have one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalScaleKind {
    /// One f32 per 32-element block (Q8_0).
    PerBlockF32,
    /// One f32 per output channel (I2_S / BitNet ternary).
    PerChannelF32,
}

/// Auxiliary material a caller supplies alongside the packed bytes.
///
/// Deliberately cannot express *how* scales are stored — only whether the
/// caller is handing over an external array. The format owns the rest, so
/// there is no second description of the same truth to keep in sync.
#[derive(Clone, Copy, Debug, Default)]
pub enum QuantAux<'a> {
    /// No external scale array — either the format packs them inline or it
    /// is unquantised. Which of those it is, is the format's business.
    #[default]
    None,
    /// An external scale array, required by `Q8_0` and `I2S`.
    ExternalScales(&'a [f32]),
}

/// A quantized weight matrix — raw bytes with format tag.
///
/// Construct via [`QuantWeight::new`]; the fields are private so a caller
/// cannot assert an auxiliary-storage arrangement the format disagrees
/// with. Two states this previously permitted, both of which occurred in
/// this repository:
///
/// - `Q4_K` with an external scale buffer. Q4_K packs scales inline, so
///   the buffer was fabricated — bound as a zero-length resource at 24
///   sites, which destroys the distinction between "no scales are
///   required" and "scales are required and there are none".
/// - `Q4_0` with an external scale array (a test fixture did exactly
///   this, and passed). Q4_0's 18-byte block *is* an f16 scale plus 16
///   bytes of nibbles.
#[derive(Clone, Copy)]
pub struct QuantWeight<'a> {
    pub data: &'a [u8],
    /// Private on purpose, like `aux`: the pair is one validated fact,
    /// and either field writable alone lets a caller desynchronise them
    /// by plain assignment after `new` has checked the combination once.
    /// Read via [`QuantWeight::format`]; change via
    /// [`QuantWeight::with_format`], which re-checks against the aux the
    /// weight already carries.
    format: QuantFormat,
    aux: QuantAux<'a>,
}

/// The one validation of the (format, aux) pair. Both construction paths
/// funnel through here so there is exactly one statement of the contract.
///
/// # Panics
/// If `aux` disagrees with `format.scale_storage()`.
fn check_aux_matches_format(format: QuantFormat, aux: QuantAux<'_>) {
    match (format.scale_storage(), aux) {
        (ScaleStorage::External(_), QuantAux::ExternalScales(_))
        | (ScaleStorage::Inline, QuantAux::None)
        | (ScaleStorage::None, QuantAux::None) => {}
        (ScaleStorage::External(kind), QuantAux::None) => {
            panic!("{format:?} stores scales externally ({kind:?}) but none were supplied")
        }
        (ScaleStorage::Inline, QuantAux::ExternalScales(_)) => panic!(
            "{format:?} packs its scales inline; an external scale array is not a                  thing it has"
        ),
        (ScaleStorage::None, QuantAux::ExternalScales(_)) => {
            panic!("{format:?} is unquantised and has no scales")
        }
    }
}

impl<'a> QuantWeight<'a> {
    /// Build a weight, checking the auxiliary material against what the
    /// format actually requires.
    ///
    /// # Panics
    /// If `aux` disagrees with `format.scale_storage()`. This is a
    /// programming error in the same class as an unknown format tag, not
    /// a runtime condition — the alternative is a kernel reading a
    /// fabricated buffer, which is how this became worth enforcing.
    pub fn new(format: QuantFormat, data: &'a [u8], aux: QuantAux<'a>) -> Self {
        check_aux_matches_format(format, aux);
        Self { data, format, aux }
    }

    /// The weight's quantization format.
    pub fn format(&self) -> QuantFormat {
        self.format
    }

    /// The same weight reinterpreted under a different format tag,
    /// re-checked against the aux it already carries.
    ///
    /// This exists for test fixtures that build packed bytes once and
    /// retag them to steer dispatch. It can only move within the same
    /// auxiliary-storage class — e.g. Q4_K → Q4_KF (both inline), or
    /// Q8_0 → I2S (both external). Crossing classes panics, which is the
    /// hole this closes: `w.format = Q8_0` on an inline-format weight
    /// used to fabricate a Q8 weight with no scale source.
    ///
    /// # Panics
    /// If the new format's scale storage disagrees with the existing aux.
    pub fn with_format(self, format: QuantFormat) -> Self {
        check_aux_matches_format(format, self.aux);
        Self { format, ..self }
    }

    /// The external scale array, when the format has one.
    ///
    /// Returns `None` for inline and unquantised formats — and callers
    /// must bind no scale resource in that case rather than fabricating an
    /// empty one.
    pub fn external_scales(&self) -> Option<&'a [f32]> {
        match self.aux {
            crate::QuantAux::ExternalScales(s) => Some(s),
            crate::QuantAux::None => None,
        }
    }

    /// The column count this weight's rows are actually stored at.
    ///
    /// Writers pad each row to the format's block boundary
    /// (`pad_rows_to_block`), so a model whose inner dim is not a block
    /// multiple stores wider rows than its logical width — GPT-OSS's
    /// hidden 2880 lands as 3072-wide Q4_K rows. Kernels must consume
    /// the STORED width; running them at the logical width truncates
    /// the superblock count and desynchronises the row stride from row 1
    /// onward. The byte count is the authority — deriving from the
    /// logical width re-creates the assumption the padding removes
    /// (same contract as [`super::moe::stored_gate_up_cols`]).
    ///
    /// Falls back to `fallback_cols` when the bytes don't divide
    /// cleanly (legacy strides, synthetic fixtures) or would derive
    /// NARROWER than the logical width — a narrower answer means these
    /// bytes are not a padded row store of this matrix.
    pub fn stored_cols(&self, rows: usize, fallback_cols: usize) -> usize {
        if rows == 0 || !self.data.len().is_multiple_of(rows) {
            return fallback_cols;
        }
        let row_bytes = self.data.len() / rows;
        let cols = match self.format.packed_block_layout() {
            Some((block_elems, block_bytes)) => {
                if !row_bytes.is_multiple_of(block_bytes) {
                    return fallback_cols;
                }
                row_bytes / block_bytes * block_elems
            }
            None => match self.format {
                QuantFormat::BF16 | QuantFormat::F16 => row_bytes / 2,
                QuantFormat::F32 => row_bytes / 4,
                _ => return fallback_cols,
            },
        };
        if cols >= fallback_cols {
            cols
        } else {
            fallback_cols
        }
    }
}

impl Default for QuantWeight<'_> {
    fn default() -> Self {
        // Q4_0 is inline, so the empty default needs no aux.
        Self {
            data: &[],
            format: QuantFormat::Q4_0,
            aux: crate::QuantAux::None,
        }
    }
}

#[cfg(test)]
mod scale_storage_tests {
    use super::*;

    /// Every format answers, and the answer matches the enum's own
    /// documentation of how it stores scales.
    #[test]
    fn scale_storage_is_exhaustive_and_matches_the_documented_layout() {
        use ExternalScaleKind::*;
        use ScaleStorage::*;
        let table = [
            (QuantFormat::Q4_0, Inline),
            (QuantFormat::Q4_K, Inline),
            (QuantFormat::Q4_KF, Inline),
            (QuantFormat::Q6_K, Inline),
            (QuantFormat::Q8_0, External(PerBlockF32)),
            (QuantFormat::I2S, External(PerChannelF32)),
            (QuantFormat::BF16, None),
            (QuantFormat::F16, None),
            (QuantFormat::F32, None),
        ];
        for (f, expected) in table {
            assert_eq!(f.scale_storage(), expected, "{f:?}");
        }
    }

    /// Q4_KF's packed layout is the standard 144-byte GGUF Q4_K block —
    /// the tag selects the llama.cpp-exact kernels, not a storage
    /// format. The 160-byte pre-baked layout (`Q4_KF_BLOCK_BYTES`) has
    /// no kernel consumer; answering it here mis-sized every derived
    /// buffer by 16 bytes per super-block (capability audit F15).
    #[test]
    fn q4_kf_packed_layout_is_gguf_q4_k() {
        use larql_models::quant::ggml;
        assert_eq!(
            QuantFormat::Q4_KF.packed_block_layout(),
            Some((ggml::Q4_K_BLOCK_ELEMS, ggml::Q4_K_BLOCK_BYTES)),
        );
        assert_eq!(
            QuantFormat::Q4_KF.packed_block_layout(),
            QuantFormat::Q4_K.packed_block_layout(),
        );
    }

    /// Inline formats are exactly the block-packed ones *today*. Asserted
    /// as an observation, not used as the discriminator — representation
    /// geometry and auxiliary storage are different properties that
    /// merely correlate, and a future packed format with external scales
    /// must break this test rather than silently misbehave.
    #[test]
    fn inline_and_block_packed_coincide_today() {
        for f in [
            QuantFormat::Q4_0,
            QuantFormat::Q4_K,
            QuantFormat::Q4_KF,
            QuantFormat::Q6_K,
        ] {
            assert!(f.packed_block_layout().is_some());
            assert_eq!(f.scale_storage(), ScaleStorage::Inline);
        }
    }

    // ── the Phase A specification, as a matrix ──────────────────────
    //
    //                     no aux      external scales
    //   Q4_0 / Q4_K         ok             panic
    //   Q4_KF / Q6_K        ok             panic
    //   Q8_0 / I2S        panic              ok
    //   F16 / F32 / BF16    ok             panic

    #[test]
    fn inline_formats_accept_no_aux() {
        for f in [
            QuantFormat::Q4_0,
            QuantFormat::Q4_K,
            QuantFormat::Q4_KF,
            QuantFormat::Q6_K,
        ] {
            let w = QuantWeight::new(f, &[0u8; 4], QuantAux::None);
            assert!(w.external_scales().is_none(), "{f:?}");
        }
    }

    #[test]
    fn unquantised_formats_accept_no_aux() {
        for f in [QuantFormat::BF16, QuantFormat::F16, QuantFormat::F32] {
            let w = QuantWeight::new(f, &[0u8; 4], QuantAux::None);
            assert!(w.external_scales().is_none(), "{f:?}");
        }
    }

    #[test]
    fn external_formats_accept_and_expose_their_scales() {
        let s = [1.0f32, 2.0];
        for f in [QuantFormat::Q8_0, QuantFormat::I2S] {
            let w = QuantWeight::new(f, &[0u8; 4], QuantAux::ExternalScales(&s));
            assert_eq!(w.external_scales(), Some(&s[..]), "{f:?}");
        }
    }

    /// The state that produced the dead O-projection fixture and the 24
    /// fabricated buffers: an inline format carrying external scales.
    #[test]
    #[should_panic(expected = "packs its scales inline")]
    fn q4k_cannot_carry_external_scales() {
        let s = [1.0f32];
        let _ = QuantWeight::new(QuantFormat::Q4_K, &[0u8; 4], QuantAux::ExternalScales(&s));
    }

    /// Q4_0's 18-byte block *is* an f16 scale plus 16 bytes of nibbles.
    /// A test fixture in this repository supplied external scales for it
    /// and passed; that is now unrepresentable.
    #[test]
    #[should_panic(expected = "packs its scales inline")]
    fn q4_0_cannot_carry_external_scales() {
        let s = [1.0f32];
        let _ = QuantWeight::new(QuantFormat::Q4_0, &[0u8; 4], QuantAux::ExternalScales(&s));
    }

    #[test]
    #[should_panic(expected = "unquantised")]
    fn unquantised_formats_cannot_carry_scales() {
        let s = [1.0f32];
        let _ = QuantWeight::new(QuantFormat::F32, &[0u8; 4], QuantAux::ExternalScales(&s));
    }

    /// The other half: a format that genuinely needs scales cannot exist
    /// without them. Previously `scales: None` on a Q8_0 weight was a
    /// silently-constructible state.
    #[test]
    #[should_panic(expected = "stores scales externally")]
    fn q8_0_cannot_exist_without_scales() {
        let _ = QuantWeight::new(QuantFormat::Q8_0, &[0u8; 4], QuantAux::None);
    }

    #[test]
    #[should_panic(expected = "stores scales externally")]
    fn i2s_cannot_exist_without_scales() {
        let _ = QuantWeight::new(QuantFormat::I2S, &[0u8; 4], QuantAux::None);
    }

    /// The default is a valid state, not merely a zeroed one.
    #[test]
    fn default_weight_is_internally_consistent() {
        let w = QuantWeight::default();
        assert_eq!(w.format().scale_storage(), ScaleStorage::Inline);
        assert!(w.external_scales().is_none());
    }

    // ── with_format: retagging within an aux class is fine; crossing
    //    classes is the desynchronisation hole and must panic ─────────

    #[test]
    fn with_format_allows_retagging_within_the_inline_class() {
        let w = QuantWeight::new(QuantFormat::Q4_K, &[0u8; 4], QuantAux::None);
        let w = w.with_format(QuantFormat::Q4_KF);
        assert_eq!(w.format(), QuantFormat::Q4_KF);
        assert!(w.external_scales().is_none());
    }

    #[test]
    fn with_format_allows_retagging_within_the_external_class() {
        let s = [1.0f32, 2.0];
        let w = QuantWeight::new(QuantFormat::Q8_0, &[0u8; 4], QuantAux::ExternalScales(&s));
        let w = w.with_format(QuantFormat::I2S);
        assert_eq!(w.format(), QuantFormat::I2S);
        assert_eq!(w.external_scales(), Some(&s[..]));
    }

    /// The exact shape of the hole this closes: a weight built inline
    /// (no scales) retagged to a format whose kernels read an external
    /// scale buffer that does not exist.
    #[test]
    #[should_panic(expected = "stores scales externally")]
    fn with_format_refuses_inline_to_external_retag() {
        let w = QuantWeight::new(QuantFormat::Q4_K, &[0u8; 4], QuantAux::None);
        let _ = w.with_format(QuantFormat::Q8_0);
    }

    #[test]
    #[should_panic(expected = "packs its scales inline")]
    fn with_format_refuses_external_to_inline_retag() {
        let s = [1.0f32];
        let w = QuantWeight::new(QuantFormat::Q8_0, &[0u8; 4], QuantAux::ExternalScales(&s));
        let _ = w.with_format(QuantFormat::Q4_K);
    }
}

#[cfg(test)]
mod stored_cols_tests {
    use super::*;
    use crate::cpu::ops::q4_common::{quantize_q4_k, quantize_q6_k};

    /// Quantise `rows` rows of `logical` values each, zero-padded to the
    /// next super-block boundary — the writer's `pad_rows_to_block` shape.
    fn padded_rows_q4k(rows: usize, logical: usize) -> Vec<u8> {
        let (block, _) = QuantFormat::Q4_K.packed_block_layout().unwrap();
        let padded = logical.div_ceil(block) * block;
        let mut data = vec![0.0f32; rows * padded];
        for (i, v) in data.iter_mut().enumerate() {
            if i % padded < logical {
                *v = ((i % 31) as f32) - 15.0;
            }
        }
        quantize_q4_k(&data)
    }

    /// The padded-store case the derivation exists for: rows written at
    /// the next block boundary (GPT-OSS's hidden 2880 → 3072 shape, here
    /// at test scale) answer the STORED width, not the logical one.
    #[test]
    fn padded_q4k_rows_answer_the_stored_width() {
        let (block, _) = QuantFormat::Q4_K.packed_block_layout().unwrap();
        let (rows, logical) = (4usize, block + block / 4); // 320 → stored 512
        let stored = logical.div_ceil(block) * block;
        let bytes = padded_rows_q4k(rows, logical);
        let w = QuantWeight::new(QuantFormat::Q4_K, &bytes, QuantAux::None);
        assert_eq!(w.stored_cols(rows, logical), stored);
    }

    #[test]
    fn padded_q6k_rows_answer_the_stored_width() {
        let (block, _) = QuantFormat::Q6_K.packed_block_layout().unwrap();
        let (rows, logical) = (3usize, block / 2); // 128 → stored 256
        let stored = logical.div_ceil(block) * block;
        let mut data = vec![0.0f32; rows * stored];
        for (i, v) in data.iter_mut().enumerate() {
            if i % stored < logical {
                *v = (i % 17) as f32;
            }
        }
        let bytes = quantize_q6_k(&data);
        let w = QuantWeight::new(QuantFormat::Q6_K, &bytes, QuantAux::None);
        assert_eq!(w.stored_cols(rows, logical), stored);
    }

    /// Block-aligned rows are their own stored width — the derivation is
    /// the identity on every model that never needed padding.
    #[test]
    fn aligned_rows_are_a_no_op() {
        let (block, _) = QuantFormat::Q4_K.packed_block_layout().unwrap();
        let rows = 2usize;
        let bytes = padded_rows_q4k(rows, block);
        let w = QuantWeight::new(QuantFormat::Q4_K, &bytes, QuantAux::None);
        assert_eq!(w.stored_cols(rows, block), block);
    }

    /// Bytes that don't divide by the row count aren't a row store of
    /// this matrix — answer the caller's fallback, never a guess.
    #[test]
    fn indivisible_bytes_fall_back() {
        let bytes = vec![0u8; 1001];
        let w = QuantWeight::new(QuantFormat::Q4_K, &bytes, QuantAux::None);
        assert_eq!(w.stored_cols(2, 96), 96);
    }

    /// Zero rows can't define a row width.
    #[test]
    fn zero_rows_fall_back() {
        let w = QuantWeight::new(QuantFormat::Q4_K, &[], QuantAux::None);
        assert_eq!(w.stored_cols(0, 64), 64);
    }

    /// A derivation NARROWER than the logical width means these bytes are
    /// not a padded row store — the padding contract can only widen.
    #[test]
    fn narrower_than_logical_falls_back() {
        let (block, _) = QuantFormat::Q4_K.packed_block_layout().unwrap();
        let bytes = padded_rows_q4k(2, block); // stored width = 1 block
        let w = QuantWeight::new(QuantFormat::Q4_K, &bytes, QuantAux::None);
        assert_eq!(w.stored_cols(2, 3 * block), 3 * block);
    }

    /// Unquantised rows derive from their element size.
    #[test]
    fn f32_rows_derive_from_element_size() {
        let bytes = vec![0u8; 2 * 96 * 4];
        let w = QuantWeight::new(QuantFormat::F32, &bytes, QuantAux::None);
        assert_eq!(w.stored_cols(2, 80), 96);
    }
}
