//! Single-edge compilation: one prompt + one answer → one compiled edge.
//!
//! Captures the residual at the target layer for the prompt, looks up the
//! answer token's embedding, installs an edge that fires only on this prompt
//! and pushes the answer token through the LM head. CLI-driven; contrasts
//! with patch mode (vindex-driven, many edges).

use std::collections::HashMap;

use ndarray::ArcArray2;

use super::edge::install_edge;
use super::detect::detect_ffn_pattern;
use super::save::{copy_model_config, merge_for_save, write_safetensors};
use super::CompileArgs;

pub fn run(args: CompileArgs) -> Result<(), Box<dyn std::error::Error>> {
    let prompt = args.prompt.as_ref().unwrap();
    let answer = args.answer.as_ref().unwrap();

    eprintln!("LARQL AOT Compiler — single mode");
    eprintln!("  base:   {}", args.base.display());
    eprintln!("  prompt: {}...", &prompt[..prompt.len().min(60)]);
    eprintln!("  answer: {}", answer);
    eprintln!("  layer:  {}", args.layer);
    eprintln!("  slot:   {}", args.slot);
    eprintln!("  output: {}", args.output.display());

    eprintln!("\nLoading model...");
    let weights = larql_models::loading::load_model_dir(&args.base)?;
    let config = weights.arch.config();
    eprintln!("  {} layers, dim={}", config.num_layers, config.hidden_size);

    let tokenizer_path = args.base.join("tokenizer.json");
    if !tokenizer_path.exists() {
        return Err(format!(
            "tokenizer.json not found in {}",
            args.base.display()
        )
        .into());
    }
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| format!("tokenizer: {}", e))?;

    let encoding = tokenizer
        .encode(prompt.as_str(), true)
        .map_err(|e| format!("tokenize: {}", e))?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();
    eprintln!("  prompt tokens: {}", token_ids.len());

    eprintln!("\nCapturing L{} residual...", args.layer);
    let residuals = larql_inference::forward::capture_residuals(
        &weights,
        &token_ids,
        &[args.layer],
    );
    let (_, residual) = residuals
        .into_iter()
        .find(|(l, _)| *l == args.layer)
        .ok_or("failed to capture residual")?;

    let trigger_norm: f32 = residual.iter().map(|x| x * x).sum::<f32>().sqrt();
    eprintln!("  trigger norm: {:.2}", trigger_norm);

    let ans_encoding = tokenizer
        .encode(answer.as_str(), false)
        .map_err(|e| format!("tokenize answer: {}", e))?;
    let ans_ids = ans_encoding.get_ids();
    if ans_ids.is_empty() {
        return Err("answer tokenizes to empty".into());
    }
    let ans_token = ans_ids[0];
    eprintln!(
        "  answer token: {} → {:?}",
        ans_token,
        tokenizer.decode(&[ans_token], false).unwrap_or_default()
    );

    let hidden = config.hidden_size;
    let write: Vec<f32> = (0..hidden)
        .map(|j| weights.embed[[ans_token as usize, j]])
        .collect();

    let gate_pattern = detect_ffn_pattern(&weights.tensors, "gate");
    let up_pattern = detect_ffn_pattern(&weights.tensors, "up");
    let down_pattern = detect_ffn_pattern(&weights.tensors, "down");

    let gate_key = gate_pattern.replace("{}", &args.layer.to_string());
    let up_key = up_pattern.replace("{}", &args.layer.to_string());
    let down_key = down_pattern.replace("{}", &args.layer.to_string());

    let mut modified: HashMap<String, ArcArray2<f32>> = HashMap::new();
    for key in [&gate_key, &up_key, &down_key] {
        let original = weights
            .tensors
            .get(key)
            .ok_or_else(|| format!("tensor not found: {}", key))?;
        modified.insert(key.clone(), original.to_owned().into());
    }

    eprintln!("\nInstalling edge...");
    let stats = install_edge(
        &mut modified,
        &gate_key,
        &up_key,
        &down_key,
        args.slot,
        &residual,
        &write,
        args.gate_scale,
        args.alpha,
    )?;
    eprintln!(
        "  gate_scale={}, alpha={:.3}",
        args.gate_scale, stats.alpha
    );
    eprintln!("  installed at L{} slot {}", args.layer, args.slot);

    eprintln!("\nSaving compiled model...");
    std::fs::create_dir_all(&args.output)?;
    let merged = merge_for_save(&weights, modified);
    let output_file = args.output.join("model.safetensors");
    write_safetensors(&merged.tensors, &merged.vectors, &output_file)?;

    let file_size = std::fs::metadata(&output_file)?.len();
    eprintln!(
        "  saved: {} ({:.1} GB, {} tensors, {} vectors)",
        output_file.display(),
        file_size as f64 / 1e9,
        merged.tensors.len(),
        merged.vectors.len(),
    );

    copy_model_config(&args.base, &args.output);

    eprintln!("\nDone.");
    eprintln!(
        "  larql compile --base {} --prompt \"...\" --answer \"{}\" → {}",
        args.base.display(),
        answer,
        args.output.display()
    );
    Ok(())
}
