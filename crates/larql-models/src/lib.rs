// See crates/larql-core/src/lib.rs for the pattern-2 (own-crate-missing-
// no_std) rationale. Unlike larql-core, this crate is filesystem/mmap-
// heavy throughout (loading/, detect/, connectors/, encoders/, speech/,
// weights/ all touch std::fs/std::path/memmap2) -- rather than guess
// which whole modules need excluding without a compiler in the loop
// (cross-module references, e.g. weights::ModelWeights used from
// otherwise-portable modules, are easy to get wrong by inspection alone),
// this round applies only the confirmed-safe crate-level attribute and
// lets the next real CI round show exactly what's needed next.
#![cfg_attr(target_arch = "wasm32", no_std)]
#[cfg(target_arch = "wasm32")]
#[macro_use]
extern crate alloc;

// connectors/, encoders/, loading/, speech/ are excluded wholesale on
// wasm32: all four are fundamentally file/mmap-loading pipelines
// (std::fs, std::path, memmap2/safetensors throughout -- confirmed via
// CI), with no cross-reference from any portable module -- confirmed by
// grep, EXCEPT this exact check first wrongly included detect/ in the
// same bucket (its ModelError type and detect_from_json* functions are
// pure logic, used from quant/ggml/*.rs, which the first pass of this
// grep didn't cover). detect/ is therefore NOT wholesale-excluded --
// see its own module for the surgical field/function-level split
// instead, same shape as weights.rs's Mmap field.
pub mod architectures;
mod collections;
// `ModelWeights::vectors`/`tensors`/etc. and `DequantScratch` are `pub` and
// resolve (on wasm32) to this crate's own `HashMap<K, V, BuildHasherDefault
// <FnvHasher>>` -- a public field can't have a type built from a private
// component (rustc reports "type is private" once a downstream crate's
// code needs to resolve a trait bound through it, e.g. calling `.get()`),
// so the collections module's own items must be reachable even though the
// module itself stays private (its internal layout isn't API).
pub use collections::{FnvHasher, HashMap, HashSet};
pub mod config;
#[cfg(not(target_arch = "wasm32"))]
pub mod connectors;
pub mod defaults;
pub mod detect;
#[cfg(not(target_arch = "wasm32"))]
pub mod encoders;
#[cfg(not(target_arch = "wasm32"))]
pub mod loading;
pub mod multimodal;
mod prelude;
pub mod quant;
#[cfg(not(target_arch = "wasm32"))]
pub mod speech;
pub(crate) mod tensor_keys;
#[cfg(any(test, feature = "test-utils"))]
pub mod test_fixtures;
pub mod validation;
pub mod vectors;
pub mod weights;

pub use config::{
    Activation, ExpertFormat, ExpertGatePolicy, ExpertRoutingPolicy, FfnType, Llama3RopeScaling,
    ModelArchitecture, ModelConfig, MoeRouterKind, NormType, QkNormScope, RopeScaling,
    YarnRopeScaling, LAYER_TYPE_FULL_ATTENTION, LAYER_TYPE_SLIDING_ATTENTION, ROPE_TYPE_DEFAULT,
    ROPE_TYPE_LINEAR, ROPE_TYPE_LLAMA3, ROPE_TYPE_YARN,
};
// detect_from_json/detect_from_json_validated/ModelError are pure logic
// (dispatch on an already-parsed JSON value) -- portable. Only
// detect_architecture/detect_architecture_validated take &std::path::Path
// and read config.json from disk, so only those two are native-only.
#[cfg(not(target_arch = "wasm32"))]
pub use detect::{detect_architecture, detect_architecture_validated};
pub use detect::{detect_from_json, detect_from_json_validated, ModelError};
pub use multimodal::{
    Connector as MmConnector, ModalEncoder, ModalInput, Modality, MultiModalProtocol,
    PlaceholderProtocol, PrecomputedScaling, TokenBudget,
};
pub use validation::{ConfigValidationError, ConfigValidationResult};

pub use architectures::deepseek::DeepSeekArch;
pub use architectures::gemma2::Gemma2Arch;
pub use architectures::gemma3::Gemma3Arch;
pub use architectures::gemma4::Gemma4Arch;
pub use architectures::generic::GenericArch;
pub use architectures::gpt2::Gpt2Arch;
pub use architectures::gpt_oss::GptOssArch;
pub use architectures::granite::GraniteArch;
pub use architectures::llama::LlamaArch;
pub use architectures::mistral::MistralArch;
pub use architectures::mixtral::MixtralArch;
pub use architectures::moss_tts_realtime::MossTtsRealtimeArch;
pub use architectures::qwen::QwenArch;
pub use architectures::starcoder2::StarCoder2Arch;
pub use architectures::tinymodel::TinyModelArch;

pub use vectors::{
    TopKEntry, VectorFileHeader, VectorRecord, ALL_COMPONENTS, COMPONENT_ATTN_OV,
    COMPONENT_ATTN_QK, COMPONENT_EMBEDDINGS, COMPONENT_FFN_DOWN, COMPONENT_FFN_GATE,
    COMPONENT_FFN_UP,
};
pub use weights::{DequantScratch, ModelWeights, WeightArray, WeightsView};

#[cfg(not(target_arch = "wasm32"))]
pub use loading::{
    is_ffn_tensor, load_gguf, load_gguf_validated, load_model_dir, load_model_dir_filtered,
    load_model_dir_filtered_validated, load_model_dir_validated, load_model_dir_walk_only,
    load_model_dir_walk_only_validated, resolve_model_path, I2S_SCALE_SUFFIX,
};
