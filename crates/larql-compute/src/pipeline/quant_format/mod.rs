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
    /// 176 bytes per 256 values (5-bit with sub-block scales).
    ///
    /// Carried because it is the cheapest **exact** container for MXFP4
    /// weights: reconstructing the fp4 alphabet on an affine grid needs
    /// 25 levels, Q5_K has 32, and Q4_K's 16 is why the transcode cannot
    /// simply drop to Q4_K. At 5.5 bpw against Q6_K's 6.5625 it is the
    /// lossless fallback if the native MXFP4 path does not pay off.
    ///
    /// **No Metal kernel exists yet** — the format is expressible and
    /// routes fail loudly rather than silently falling through to Q4_K.
    Q5_K,
    Q6_K, // 210 bytes per 256 values (6-bit with sub-block scales)
    Q8_0, // int8 values + separate f32 scales
    BF16, // raw bfloat16 (2 bytes per value, no quantization scales)
    F16,  // raw float16  (2 bytes per value)
    F32,  // raw float32  (4 bytes per value)
    /// BitNet 1.58-bit ternary (GGML I2_S, type 36): 4 trits/byte packed
    /// row-major (`cols/4` bytes per row) plus a separate per-channel f32
    /// scale array. Unlike the block-quant formats, the weight is NOT a
    /// flat `&[u8]` block stream — it is carried by
    /// [`crate::cpu::ops::ternary_matvec::BitLinearWeight`] (bytes + scales)
    /// and served by the dedicated `ternary_matvec` dispatch, not the
    /// block-quant `quant_matvec` path (which has no per-channel-scale input).
    I2S,
    /// MXFP4 (OCP microscaling FP4), as GPT-OSS ships it: a 4-bit LUT index
    /// per weight plus **one e8m0 exponent byte per 32-element group**, held
    /// in a separate stream. 4.25 bpw all-in.
    ///
    /// The first format here that is block-packed *and* carries external
    /// scales — the combination [`ScaleStorage`]'s doc anticipated. Its
    /// scales are `u8` exponents, not `f32`: widening them to f32 would
    /// take the format from 4.25 to 5.0 bpw and throw away a third of the
    /// bandwidth win that is the entire reason to serve MXFP4 natively.
    /// That is why [`ExternalScaleKind::PerGroupE8M0`] exists rather than
    /// reusing `PerBlockF32`.
    ///
    /// Not a member of [`Self::is_kquant_family`]: that names the GGUF
    /// 256-element super-block family, and MXFP4's group is 32.
    MXFP4,
}

impl QuantFormat {
    /// Packed block geometry as `(elements_per_block, bytes_per_block)`.
    ///
    /// This is the compute-side mirror of the GGML layout constants used by
    /// the quantizers. Callers that need byte offsets should ask the format
    /// instead of spelling `256 * 144` or `32 * 18` locally.
    pub fn packed_block_layout(self) -> Option<(usize, usize)> {
        use larql_models::quant::{ggml, mxfp4};

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
            Self::Q5_K => Some((ggml::Q5_K_BLOCK_ELEMS, ggml::Q5_K_BLOCK_BYTES)),
            Self::Q6_K => Some((ggml::Q6_K_BLOCK_ELEMS, ggml::Q6_K_BLOCK_BYTES)),
            // The PACKED stream only — 32 weights in 16 nibble-bytes. The
            // e8m0 scale stream is external and deliberately not counted
            // here, so `packed_matrix_bytes` answers payload bytes and a
            // caller sizing the scale buffer must ask `scale_storage`.
            // Answering `None` instead would have been the conservative
            // choice, but it also forfeits `stored_cols`' padded-row
            // derivation, which GPT-OSS's hidden 2880 → 3072 shape needs.
            Self::MXFP4 => Some((mxfp4::MXFP4_GROUP_ELEMS, mxfp4::MXFP4_GROUP_BYTES)),
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
        matches!(self, Self::Q4_K | Self::Q4_KF | Self::Q5_K | Self::Q6_K)
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
            "Q5_K" => Self::Q5_K,
            "Q6_K" => Self::Q6_K,
            "Q8_0" => Self::Q8_0,
            "BF16" => Self::BF16,
            "F16" => Self::F16,
            "F32" => Self::F32,
            // BitNet ternary (GGML type 36). The vindex bitnet sidecar tags
            // its I2_S weight stream with this so the registry recognises it.
            "I2_S" => Self::I2S,
            // Spelled as the OCP/GPT-OSS checkpoints spell it, and matched
            // by `LayerWeightFormat::MXFP4`'s registry tag so a natively
            // stored expert bank survives the writer → loader round trip.
            "MXFP4" => Self::MXFP4,
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
            Self::Q5_K => "Q5_K",
            Self::Q6_K => "Q6_K",
            Self::Q8_0 => "Q8_0",
            Self::BF16 => "BF16",
            Self::F16 => "F16",
            Self::F32 => "F32",
            Self::I2S => "I2_S",
            Self::MXFP4 => "MXFP4",
        }
    }

    /// Whether this format is MXFP4, served by the dedicated
    /// `mxfp4_grouped_experts` / `mxfp4_matvec` kernels against a separate
    /// e8m0 exponent stream. The MXFP4 sibling of [`Self::is_ternary`] —
    /// both name "block-quant `quant_matvec` cannot serve this".
    pub fn is_mxfp4(self) -> bool {
        matches!(self, Self::MXFP4)
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
            Self::Q4_0 | Self::Q4_K | Self::Q4_KF | Self::Q5_K | Self::Q6_K => ScaleStorage::Inline,
            // "int8 values + separate f32 scales", per this enum's own doc.
            Self::Q8_0 => ScaleStorage::External(ExternalScaleKind::PerBlockF32),
            // Ternary carries a separate per-channel f32 array.
            Self::I2S => ScaleStorage::External(ExternalScaleKind::PerChannelF32),
            // One e8m0 exponent byte per 32-weight group, external.
            Self::MXFP4 => ScaleStorage::External(ExternalScaleKind::PerGroupE8M0),
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
    /// One **e8m0 exponent byte** per 32-weight group (MXFP4).
    ///
    /// Byte-valued, not f32, and that is load-bearing rather than a detail:
    /// at one byte per 32 weights the scale stream costs 0.25 bpw, where an
    /// f32 array would cost 1.0 and take MXFP4 from 4.25 bpw to 5.0 —
    /// forfeiting a third of the reason to serve it natively. Decode is
    /// `larql_models::quant::mxfp4::e8m0_to_f32`.
    PerGroupE8M0,
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
    /// An external f32 scale array, required by `Q8_0` and `I2S`.
    ExternalScales(&'a [f32]),
    /// An external **e8m0 exponent byte** stream, required by `MXFP4`.
    ///
    /// A separate variant rather than an f32 array the caller pre-decodes,
    /// so the 0.25-bpw scale stream reaches the kernel in its stored width.
    ExternalE8M0(&'a [u8]),
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
    use ExternalScaleKind::*;
    match (format.scale_storage(), aux) {
        // The external kinds are matched against the aux *width*, not just
        // against "is external". Before MXFP4 every external format was
        // f32-scaled, so `External(_)` paired with any array was sound;
        // it no longer is, and the loose arm would have let an e8m0 byte
        // stream bind where a kernel reads f32 (and the reverse).
        (ScaleStorage::External(PerBlockF32 | PerChannelF32), QuantAux::ExternalScales(_))
        | (ScaleStorage::External(PerGroupE8M0), QuantAux::ExternalE8M0(_))
        | (ScaleStorage::Inline, QuantAux::None)
        | (ScaleStorage::None, QuantAux::None) => {}
        (ScaleStorage::External(kind), QuantAux::None) => {
            panic!("{format:?} stores scales externally ({kind:?}) but none were supplied")
        }
        (ScaleStorage::External(kind), _) => panic!(
            "{format:?} stores scales as {kind:?}; the supplied aux is a different width"
        ),
        (ScaleStorage::Inline, _) => panic!(
            "{format:?} packs its scales inline; an external scale array is not a                  thing it has"
        ),
        (ScaleStorage::None, _) => {
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
            crate::QuantAux::None | crate::QuantAux::ExternalE8M0(_) => None,
        }
    }

    /// The external e8m0 exponent stream, when the format has one (MXFP4).
    ///
    /// Deliberately a separate accessor from [`Self::external_scales`]
    /// rather than a decode-on-read: a caller that wants f32 scales for an
    /// MXFP4 weight is asking the wrong question, and silently handing back
    /// a converted array is how the stored width gets lost on the way to a
    /// kernel that reads bytes.
    pub fn external_e8m0(&self) -> Option<&'a [u8]> {
        match self.aux {
            crate::QuantAux::ExternalE8M0(s) => Some(s),
            crate::QuantAux::None | crate::QuantAux::ExternalScales(_) => None,
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
mod tests;
