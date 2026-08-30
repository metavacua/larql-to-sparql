//! The FFN step entry: picks the per-format chain for a layer.
//!
//! Split out of `encode_ffn.rs`; [`super`] holds the shared types.

#[allow(unused_imports)]
use super::*;
use crate::MetalBackend;
use larql_compute::FullPipelineLayer;
use metal::ComputeCommandEncoderRef;

impl MetalBackend {
    /// Encode the full FFN block (gate / up / activation / down) into
    /// the encoder. The path is selected per-operand from the layer's
    /// own weight formats; the function returns the same `down_out`
    /// buffer the caller passed in via `bufs`. No commit/flush — the
    /// caller owns encoder lifecycle.
    ///
    /// Routing rules (capability audit F3). The previous branch keyed
    /// on `gate.format().is_kquant_family()` alone, which decoded a
    /// Q6_K gate with the Q4_K kernels (210-byte blocks read at a
    /// 144-byte stride) and a Q8_0 gate with the Q4_0 kernels — silent
    /// corruption in both cases — and never consulted `up.format()` at
    /// all. Now:
    /// - a gated layer's gate and up must share a format (no fused
    ///   kernel decodes a mixed pair; refuse rather than reinterpret);
    /// - a non-gated layer routes on `up`, its actual first matvec —
    ///   `gate` may be a default-constructed placeholder;
    /// - Q6_K runs the separated per-format chain (`q6k_matvec` per
    ///   projection), mirroring the prefill path;
    /// - formats with no decode FFN kernel refuse loudly.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_ffn_step(
        &self,
        enc: &ComputeCommandEncoderRef,
        layer: &FullPipelineLayer,
        bufs: FfnBufs<'_>,
        dims: FfnDims,
    ) {
        let FfnDims {
            hidden,
            inter,
            inter_padded,
        } = dims;
        let inter_val = inter as u32;
        let inter_padded_val = inter_padded as u32;
        let hidden_val = hidden as u32;

        use larql_compute::QuantFormat;
        let route_fmt = validate_ffn_formats(layer);

        match route_fmt {
            QuantFormat::Q4_KF => {
                self.encode_q4kf_ffn(enc, layer, &bufs, hidden, inter, hidden_val, inter_val);
            }
            QuantFormat::Q4_K => {
                self.encode_q4k_ffn(
                    enc,
                    layer,
                    &bufs,
                    hidden,
                    inter,
                    inter_padded,
                    hidden_val,
                    inter_val,
                    inter_padded_val,
                );
            }
            QuantFormat::Q6_K => {
                self.encode_q6k_ffn(enc, layer, &bufs, hidden, inter, inter_val);
            }
            QuantFormat::Q4_0 => {
                self.encode_q4_0_ffn(enc, layer, &bufs, hidden, inter, hidden_val, inter_val);
            }
            // validate_ffn_formats admits only the four arms above.
            other => unreachable!("validate_ffn_formats admitted {other:?}"),
        }
    }

    // ── Q6_K (separated per-format chain) ────────────────────────────────────
}
