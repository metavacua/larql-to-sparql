/// LLM-mediated expert dispatch test.
///
/// Feeds natural language prompts through the real model (gemma-3-4b-it),
/// asks it to emit `{"op": "...", "args": {...}}` JSON, then dispatches
/// through ExpertRegistry exactly as the Option 3 test does — but now
/// the (op, args) pair comes from the model, not hardwired constants.
///
/// Requires:
///   - LARQL_MODEL env var pointing to a model path or HuggingFace ID
///     (defaults to "google/gemma-3-4b-it")
///   - larql-experts pre-built for wasm32-wasip1
///
/// Skip behaviour: any missing pre-condition prints a message and returns
/// cleanly — `cargo test` reports the test as passed (skipped).
use std::path::PathBuf;

use larql_inference::{
    encode_prompt, forward::generate_cached, InferenceModel, WeightFfn,
};
use larql_inference::experts::ExpertRegistry;
use serde_json::{json, Value};

// ── Infrastructure ────────────────────────────────────────────────────────────

fn model_id() -> String {
    std::env::var("LARQL_MODEL").unwrap_or_else(|_| "google/gemma-3-4b-it".to_string())
}

fn wasm_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../larql-experts/target/wasm32-wasip1/release")
}

/// Wrap prompt in Gemma instruction-tuning chat template.
fn chat(prompt: &str) -> String {
    format!(
        "<start_of_turn>user\n{prompt}\n<end_of_turn>\n<start_of_turn>model\n"
    )
}

/// Extract the first JSON object `{...}` from a model response string.
fn extract_json(text: &str) -> Option<Value> {
    let start = text.find('{')?;
    // Walk forward matching braces to find the closing `}`
    let mut depth = 0usize;
    let mut end = None;
    for (i, ch) in text[start..].char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    end = Some(start + i + 1);
                    break;
                }
            }
            _ => {}
        }
    }
    let json_str = &text[start..end?];
    serde_json::from_str(json_str).ok()
}

// ── Cases ─────────────────────────────────────────────────────────────────────

struct LlmCase {
    prompt: &'static str,
    expected_op: &'static str,
    expected: LlmExpected,
}

enum LlmExpected {
    Exact(Value),
    Approx(f64, f64),
    Field(&'static str, Value),
}

fn cases() -> Vec<LlmCase> {
    vec![
        LlmCase {
            prompt: "What is the GCD of 144 and 60?",
            expected_op: "gcd",
            expected: LlmExpected::Exact(json!(12)),
        },
        LlmCase {
            prompt: "Is 97 a prime number?",
            expected_op: "is_prime",
            expected: LlmExpected::Exact(json!(true)),
        },
        LlmCase {
            prompt: "What is 10 factorial?",
            expected_op: "factorial",
            expected: LlmExpected::Exact(json!(3628800)),
        },
        LlmCase {
            prompt: "Write 2024 as a Roman numeral.",
            expected_op: "to_roman",
            expected: LlmExpected::Exact(json!("MMXXIV")),
        },
        LlmCase {
            prompt: "Is 2024 a leap year?",
            expected_op: "is_leap_year",
            expected: LlmExpected::Exact(json!(true)),
        },
        LlmCase {
            prompt: "How many days are in February 2026?",
            expected_op: "days_in_month",
            expected: LlmExpected::Exact(json!(28)),
        },
        LlmCase {
            prompt: "Reverse the string \"hello world\".",
            expected_op: "reverse",
            expected: LlmExpected::Exact(json!("dlrow olleh")),
        },
        LlmCase {
            prompt: "Is \"racecar\" a palindrome?",
            expected_op: "is_palindrome",
            expected: LlmExpected::Exact(json!(true)),
        },
    ]
}

/// System prompt — kept minimal to reduce prefill cost on CPU.
const SYSTEM: &str = r#"Respond with ONLY a JSON object {"op":"...","args":{...}}.
ops: gcd{"a","b"}, is_prime{"n"}, factorial{"n"}, to_roman{"n"}, is_leap_year{"year"}, days_in_month{"year","month"}, reverse{"s"}, is_palindrome{"s"}
No extra text."#;

// ── Single test function ──────────────────────────────────────────────────────

#[test]
fn llm_dispatch_pipeline() {
    // ── Pre-conditions ──
    if !wasm_dir().exists() {
        eprintln!("skip: wasm dir missing — run `cargo build --target wasm32-wasip1 --release` in larql-experts");
        return;
    }

    let mid = model_id();
    let model = match InferenceModel::load(&mid) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("skip: could not load model {mid:?}: {e}");
            return;
        }
    };
    eprintln!("model: {mid}  ({} layers)", model.num_layers());

    let mut reg = ExpertRegistry::load_dir(&wasm_dir()).expect("load_dir");
    let ffn = WeightFfn { weights: model.weights() };

    let mut passed = 0usize;
    let mut failed = 0usize;

    for case in cases() {
        // Build prompt: system context + question
        let full_prompt = format!("{SYSTEM}\n\nQuestion: {}", case.prompt);
        let wrapped = chat(&full_prompt);

        let ids = match encode_prompt(model.tokenizer(), &*model.weights().arch, &wrapped) {
            Ok(v) => v,
            Err(e) => { eprintln!("  FAIL tokenize: {e}"); failed += 1; continue; }
        };

        // Generate — 128 tokens is plenty for a short JSON object
        let mut output = String::new();
        generate_cached(
            model.weights(),
            model.tokenizer(),
            &ffn,
            &ids,
            128,
            |_id, tok| output.push_str(tok),
        );

        eprintln!("\n  prompt:  {}", case.prompt);
        eprintln!("  raw out: {output:?}");

        // Parse JSON from output
        let parsed = match extract_json(&output) {
            Some(v) => v,
            None => {
                eprintln!("  FAIL: no JSON object in output");
                failed += 1;
                continue;
            }
        };

        let op = match parsed.get("op").and_then(|v| v.as_str()) {
            Some(s) => s.to_string(),
            None => {
                eprintln!("  FAIL: JSON has no 'op' key: {parsed}");
                failed += 1;
                continue;
            }
        };
        let args = parsed.get("args").cloned().unwrap_or(json!({}));

        eprintln!("  op={op}  args={args}");

        // Soft-check op name (warn but don't fail on wrong op — extraction accuracy
        // is the metric, dispatch correctness is secondary here)
        if op != case.expected_op {
            eprintln!("  WARN: expected op={}, got op={op}", case.expected_op);
        }

        // Dispatch
        let result = match reg.call(&op, &args) {
            Some(r) => r,
            None => {
                eprintln!("  FAIL: registry returned None for op={op} args={args}");
                failed += 1;
                continue;
            }
        };
        let got = result.value;

        // Assert result
        let ok = match &case.expected {
            LlmExpected::Exact(exp) => {
                if got == *exp { true } else {
                    eprintln!("  FAIL: got {got}, expected {exp}");
                    false
                }
            }
            LlmExpected::Approx(exp, tol) => {
                let f = got.as_f64().unwrap_or(f64::NAN);
                if (f - exp).abs() <= *tol { true } else {
                    eprintln!("  FAIL: got {f}, expected {exp} ± {tol}");
                    false
                }
            }
            LlmExpected::Field(key, exp) => {
                let field = got.get(key).unwrap_or(&Value::Null);
                if field == exp { true } else {
                    eprintln!("  FAIL: field '{key}': got {field}, expected {exp}");
                    false
                }
            }
        };

        if ok {
            eprintln!("  ok  [{op}] {}", case.prompt);
            passed += 1;
        } else {
            failed += 1;
        }
    }

    let total = passed + failed;
    eprintln!("\n{passed}/{total} llm dispatch cases passed");
    assert_eq!(failed, 0, "{failed}/{total} cases failed");
}
