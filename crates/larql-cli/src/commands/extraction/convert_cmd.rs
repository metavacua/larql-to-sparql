use std::path::PathBuf;

use clap::{Args, Subcommand};

#[derive(Args)]
pub struct ConvertArgs {
    #[command(subcommand)]
    command: ConvertCommand,
}

#[derive(Subcommand)]
enum ConvertCommand {
    /// Convert a GGUF model to a vindex.
    GgufToVindex {
        /// Path to the .gguf file.
        input: PathBuf,

        /// Output vindex directory.
        #[arg(short, long)]
        output: PathBuf,

        /// Extract level: browse (default), inference, all.
        #[arg(long, default_value = "browse")]
        level: String,

        /// Store in f16 (half precision).
        #[arg(long)]
        f16: bool,
    },

    /// Convert a safetensors model to a vindex (alias for extract-index).
    SafetensorsToVindex {
        /// Path to the model directory.
        input: PathBuf,

        /// Output vindex directory.
        #[arg(short, long)]
        output: PathBuf,

        /// Extract level: browse (default), inference, all.
        #[arg(long, default_value = "browse")]
        level: String,

        /// Store in f16.
        #[arg(long)]
        f16: bool,
    },

    /// Show GGUF file metadata and tensor info.
    GgufInfo {
        /// Path to the .gguf file.
        input: PathBuf,
    },
}

pub fn run(args: ConvertArgs) -> Result<(), Box<dyn std::error::Error>> {
    match args.command {
        ConvertCommand::GgufToVindex { input, output, level, f16 } => {
            run_gguf_to_vindex(&input, &output, &level, f16)
        }
        ConvertCommand::SafetensorsToVindex { input, output, level, f16 } => {
            run_safetensors_to_vindex(&input, &output, &level, f16)
        }
        ConvertCommand::GgufInfo { input } => {
            run_gguf_info(&input)
        }
    }
}

fn run_gguf_to_vindex(
    input: &std::path::Path,
    output: &std::path::Path,
    level: &str,
    use_f16: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("Loading GGUF: {}", input.display());

    let gguf = larql_models::loading::gguf::GgufFile::open(input)?;

    // Show metadata summary
    if let Some(name) = gguf.metadata.get("general.name") {
        eprintln!("  Model: {:?}", name);
    }
    if let Some(arch) = gguf.metadata.get("general.architecture") {
        eprintln!("  Architecture: {:?}", arch);
    }

    eprintln!("  Loading and dequantizing tensors...");
    let weights = larql_models::load_gguf(input)?;

    eprintln!(
        "  {} layers, hidden_size={}, intermediate_size={}, vocab_size={}",
        weights.num_layers, weights.hidden_size, weights.intermediate_size, weights.vocab_size
    );

    let extract_level = match level {
        "inference" => larql_vindex::ExtractLevel::Inference,
        "all" => larql_vindex::ExtractLevel::All,
        _ => larql_vindex::ExtractLevel::Browse,
    };

    let dtype = if use_f16 {
        larql_vindex::StorageDtype::F16
    } else {
        larql_vindex::StorageDtype::F32
    };

    let model_name = gguf.metadata.get("general.name")
        .and_then(|v| v.as_str())
        .unwrap_or("gguf-model")
        .to_string();

    // Find tokenizer — check same directory as GGUF file
    let tokenizer = input.parent()
        .and_then(|dir| {
            let tok_path = dir.join("tokenizer.json");
            if tok_path.exists() {
                larql_vindex::tokenizers::Tokenizer::from_file(&tok_path).ok()
            } else {
                None
            }
        });

    let tokenizer_ref = tokenizer.as_ref().ok_or(
        "tokenizer.json not found next to GGUF file. Place it in the same directory."
    )?;

    eprintln!("\nExtracting to {}", output.display());

    let mut callbacks = SilentCallbacks;
    larql_vindex::build_vindex(
        &weights,
        tokenizer_ref,
        &model_name,
        output,
        10,
        extract_level,
        dtype,
        &mut callbacks,
    )?;

    eprintln!("Done: {}", output.display());
    Ok(())
}

fn run_safetensors_to_vindex(
    input: &std::path::Path,
    output: &std::path::Path,
    level: &str,
    use_f16: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    // This is essentially extract-index
    eprintln!("Loading safetensors: {}", input.display());
    let weights = larql_models::load_model_dir(input)?;
    let tokenizer = larql_vindex::load_vindex_tokenizer(input)
        .or_else(|_| {
            // Try to load from the model directory
            let tok_path = input.join("tokenizer.json");
            larql_vindex::tokenizers::Tokenizer::from_file(&tok_path)
                .map_err(|e| larql_vindex::VindexError::Parse(e.to_string()))
        })?;

    let extract_level = match level {
        "inference" => larql_vindex::ExtractLevel::Inference,
        "all" => larql_vindex::ExtractLevel::All,
        _ => larql_vindex::ExtractLevel::Browse,
    };

    let dtype = if use_f16 {
        larql_vindex::StorageDtype::F16
    } else {
        larql_vindex::StorageDtype::F32
    };

    let model_name = input.file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "model".into());

    eprintln!("Extracting to {}", output.display());

    let mut callbacks = SilentCallbacks;
    larql_vindex::build_vindex(
        &weights,
        &tokenizer,
        &model_name,
        output,
        10,
        extract_level,
        dtype,
        &mut callbacks,
    )?;

    eprintln!("Done: {}", output.display());
    Ok(())
}

fn run_gguf_info(input: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    let gguf = larql_models::loading::gguf::GgufFile::open(input)?;

    println!("GGUF: {}", input.display());
    println!();

    // Print metadata
    println!("Metadata ({} keys):", gguf.metadata.len());
    let mut keys: Vec<&String> = gguf.metadata.keys().collect();
    keys.sort();
    for key in &keys {
        let val = &gguf.metadata[*key];
        match val {
            larql_models::loading::gguf::GgufValue::String(s) => {
                if s.len() > 80 {
                    println!("  {}: \"{}...\"", key, &s[..80]);
                } else {
                    println!("  {}: \"{}\"", key, s);
                }
            }
            larql_models::loading::gguf::GgufValue::Array(arr) => {
                println!("  {}: [{} elements]", key, arr.len());
            }
            other => println!("  {}: {:?}", key, other),
        }
    }

    println!();

    // Print synthesised config
    let config = gguf.to_config_json();
    println!("Detected config:");
    println!("  {}", serde_json::to_string_pretty(&config)?);

    Ok(())
}

struct SilentCallbacks;
impl larql_vindex::IndexBuildCallbacks for SilentCallbacks {}
