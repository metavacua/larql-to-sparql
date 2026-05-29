//! `MetalBackend`'s `ComputeBackend`-family trait implementations.
//!
//! One file per sub-trait — mirrors the `backend/` split. The umbrella
//! `ComputeBackend` impl (`name`, `device_info`, `supports`) lives
//! here; sub-trait impls are in their own files.

mod decode;
mod matmul;
mod quant_matvec;

use super::*;
use crate::backend::{Capability, ComputeBackend};

impl ComputeBackend for MetalBackend {
    fn name(&self) -> &str {
        "metal (GPU)"
    }

    fn device_info(&self) -> String {
        format!("Metal GPU, FLOP threshold: {}", self.flop_threshold())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn supports(&self, cap: Capability) -> bool {
        // Metal accelerates everything in the menu.
        matches!(
            cap,
            Capability::F32Gemv
                | Capability::F16Gemv
                | Capability::QuantMatVec
                | Capability::Q4VecMat
                | Capability::Q4PairBatch
                | Capability::FullPipelineQ4
                | Capability::MultiLayerQ4Ffn
                | Capability::DecodeToken
                | Capability::DecodeMoe
                | Capability::DecodeQ4KMoe
                | Capability::DecodeProfile
                | Capability::PrefillQ4
                | Capability::HeterogeneousAttention
        )
    }
}
