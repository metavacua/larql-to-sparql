//! Apply the base model's HuggingFace-style chat template to a prompt.
//!
//! Every HF chat model ships a Jinja template in `tokenizer_config.json`
//! under the `chat_template` key (plus `bos_token` / `eos_token` for
//! substitution). Served deployments (Ollama, vLLM, HF generate) wrap
//! user messages with this template before inference, so to install a
//! compiled edge whose trigger matches the served residual, we have to
//! apply the same wrap here.
//!
//! This helper avoids hardcoding any model-specific template — it reads
//! whatever the base model ships.

use std::path::Path;

use minijinja::{context, Environment, Value};
use serde_json::Value as JsonValue;

/// Load the base model's chat template and render it over a single
/// user message with `add_generation_prompt=true`. Returns the wrapped
/// string ready to tokenize.
pub fn render_user_prompt(
    base_dir: &Path,
    user_prompt: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let cfg_path = base_dir.join("tokenizer_config.json");
    if !cfg_path.exists() {
        return Err(format!(
            "tokenizer_config.json not found in {} — cannot apply chat template",
            base_dir.display()
        )
        .into());
    }
    let cfg_text = std::fs::read_to_string(&cfg_path)?;
    let cfg: JsonValue = serde_json::from_str(&cfg_text)?;

    let template = cfg
        .get("chat_template")
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            "tokenizer_config.json has no `chat_template` string; pass --no-chat-template to use the raw prompt".to_string()
        })?
        .to_string();

    let bos_token = extract_token(&cfg, "bos_token");
    let eos_token = extract_token(&cfg, "eos_token");
    let pad_token = extract_token(&cfg, "pad_token");

    let mut env = Environment::new();
    // `raise_exception` is a convention some HF templates use for error paths.
    env.add_function("raise_exception", |msg: String| -> Result<Value, minijinja::Error> {
        Err(minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, msg))
    });
    env.add_template("chat", &template)?;
    let tmpl = env.get_template("chat")?;

    let messages = vec![context! {
        role => "user",
        content => user_prompt,
    }];

    let rendered = tmpl.render(context! {
        messages => messages,
        add_generation_prompt => true,
        bos_token => bos_token,
        eos_token => eos_token,
        pad_token => pad_token,
    })?;

    Ok(rendered)
}

fn extract_token(cfg: &JsonValue, key: &str) -> String {
    match cfg.get(key) {
        Some(JsonValue::String(s)) => s.clone(),
        // Tokenizer config sometimes stores tokens as objects: {"content": "<bos>", ...}
        Some(JsonValue::Object(o)) => o
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        _ => String::new(),
    }
}
