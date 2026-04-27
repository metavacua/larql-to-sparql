pub mod architectures;
pub mod config;
pub mod detect;
pub mod loading;
pub mod quant;
pub mod vectors;
pub mod weights;

pub use config::{Activation, ExpertFormat, FfnType, ModelArchitecture, ModelConfig, NormType, RopeScaling};
pub use detect::{detect_architecture, detect_from_json, ModelError};

pub use architectures::deepseek::DeepSeekArch;
pub use architectures::gemma2::Gemma2Arch;
pub use architectures::gemma3::Gemma3Arch;
pub use architectures::gemma4::Gemma4Arch;
pub use architectures::generic::GenericArch;
pub use architectures::gpt_oss::GptOssArch;
pub use architectures::granite::GraniteArch;
pub use architectures::llama::LlamaArch;
pub use architectures::mistral::MistralArch;
pub use architectures::mixtral::MixtralArch;
pub use architectures::qwen::QwenArch;
pub use architectures::starcoder2::StarCoder2Arch;
pub use architectures::tinymodel::TinyModelArch;

pub use vectors::{
    TopKEntry, VectorFileHeader, VectorRecord, ALL_COMPONENTS, COMPONENT_ATTN_OV,
    COMPONENT_ATTN_QK, COMPONENT_EMBEDDINGS, COMPONENT_FFN_DOWN, COMPONENT_FFN_GATE,
    COMPONENT_FFN_UP,
};
pub use weights::{ModelWeights, WeightArray};

pub use loading::{
    is_ffn_tensor, load_gguf, load_model_dir, load_model_dir_filtered,
    load_model_dir_walk_only, resolve_model_path,
};
