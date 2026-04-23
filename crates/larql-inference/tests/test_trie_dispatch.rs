/// Trie-routed constrained dispatch test.
///
/// Pipeline:
///   1. Capture hidden state at L5 with `forward_to_layer` (partial prefill,
///      only 6/34 layers — cheap).
///   2. `CascadeTrie::classify` runs PCA-32 + LR to get a route label
///      ("arithmetic", "date", "code", "factual", "logical").
///   3. `generate_cached_constrained` runs the full decode loop; the mask
///      closure restricts the op-name vocabulary to ops belonging to the
///      predicted route only.
///   4. Parsed (op, args) dispatch through `ExpertRegistry`.
///
/// This is the wired Option 3+probe path: the trie replaces the hardwired
/// route labels from test_constrained_dispatch.rs.
use std::collections::HashSet;
use std::path::PathBuf;

use larql_inference::{
    encode_prompt,
    forward::{generate_cached_constrained, forward_to_layer},
    trie::CascadeTrie,
    InferenceModel, WeightFfn,
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

/// Derive the probe filename for a given model ID using the same slug rule as
/// export_trie_probe.py: replace "/" with "--".
fn probe_path_for_model(mid: &str) -> PathBuf {
    // Allow explicit override.
    if let Ok(p) = std::env::var("LARQL_PROBE_PATH") {
        return PathBuf::from(p);
    }
    let slug = mid.replace('/', "--");
    let filename = format!("cascade_trie_{slug}_probe.json");
    // CARGO_MANIFEST_DIR = .../chris-source/larql/crates/larql-inference
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../lazarus-play/experiments")
        .join(filename)
}

/// Wrap a prompt in the correct chat template for the given model.
fn chat_for_model(mid: &str, prompt: &str) -> String {
    if mid.contains("Mistral") || mid.contains("mistral") {
        // Mistral instruction format
        format!("[INST] {prompt} [/INST]")
    } else if mid.contains("Llama") || mid.contains("llama") {
        // Llama-3 instruction format
        format!(
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    } else {
        // Gemma (default)
        format!("<start_of_turn>user\n{prompt}\n<end_of_turn>\n<start_of_turn>model\n")
    }
}

/// Find the first valid JSON object that contains an "op" field.
/// Tries each `{...}` block in order, normalising fullwidth punctuation first.
fn extract_json(text: &str) -> Option<Value> {
    // Normalise fullwidth punctuation first, then fix Mistral's missing comma
    // before "args": `"VALUE"args":` → `"VALUE","args":`.
    // Two-step: protect the already-correct `,"args":` form, then fix any
    // remaining bare `"args":` (which shares its leading " with the value's
    // closing "), then restore.
    let normalised = text
        .replace('，', ",")
        .replace('：', ":")
        .replace(",\"args\":", "\x01")        // protect correct form
        .replace("\"args\":", "\",\"args\":") // fix missing comma, keep closing "
        .replace('\x01', ",\"args\":");
    let text = &*normalised;
    let mut search_from = 0;
    while let Some(rel) = text[search_from..].find('{') {
        let start = search_from + rel;
        let mut depth = 0usize;
        let mut end = None;
        for (i, ch) in text[start..].char_indices() {
            match ch { '{' => depth += 1, '}' => { depth -= 1; if depth == 0 { end = Some(start + i + 1); break; } } _ => {} }
        }
        let end = match end { Some(e) => e, None => break };
        if let Ok(v) = serde_json::from_str::<Value>(&text[start..end]) {
            if v.get("op").is_some() { return Some(v); }
        }
        search_from = start + 1;
    }
    None
}

// ── Route → ops mapping ───────────────────────────────────────────────────────

/// Static mapping from route label to the expert IDs that handle it.
/// The mask uses this to whittle down the 126-op vocabulary to ~10-30 ops.
fn ops_for_route<'a>(route: &str, reg: &'a ExpertRegistry) -> Vec<&'a str> {
    let expert_ids: &[&str] = match route {
        "arithmetic" => &["arithmetic", "statistics", "geometry", "trig", "finance"],
        "date"       => &["date"],
        // Include "arithmetic" in code route: Roman numeral / base conversion ops
        // are semantically format-like and some models (Mistral) route them here.
        "code"       => &["string_ops", "hash", "sql", "arithmetic", "statistics", "geometry", "trig", "finance"],
        "factual"    => &["unit", "element", "http_status", "luhn", "isbn"],
        "logical"    => &["logic"],
        _            => return reg.ops().into_iter().collect(), // unknown → unconstrained
    };
    reg.list()
        .into_iter()
        .filter(|m| expert_ids.contains(&m.id.as_str()))
        .flat_map(|m| m.ops.iter().map(|s| s.as_str()))
        .collect()
}

// ── Op-name mask ──────────────────────────────────────────────────────────────

struct RouteOpMask<'a> {
    allowed_ops: Vec<&'a str>,
    tokenizer: tokenizers::Tokenizer,
    op_token_cache: Option<Vec<u32>>,
    generated_text: String,
}

impl<'a> RouteOpMask<'a> {
    fn new(allowed_ops: Vec<&'a str>, tokenizer: tokenizers::Tokenizer) -> Self {
        Self { allowed_ops, tokenizer, op_token_cache: None, generated_text: String::new() }
    }

    fn op_tokens(&mut self) -> &[u32] {
        if self.op_token_cache.is_none() {
            let valid_chars: HashSet<char> = self.allowed_ops.iter()
                .flat_map(|op| op.chars())
                .chain(std::iter::once('"'))
                .collect();
            let vocab_size = self.tokenizer.get_vocab_size(false);
            let ids: Vec<u32> = (0..vocab_size as u32)
                .filter(|&id| {
                    self.tokenizer.decode(&[id], false)
                        .map(|s| !s.is_empty() && (s == "\"" || s.chars().all(|c| valid_chars.contains(&c))))
                        .unwrap_or(false)
                })
                .collect();
            self.op_token_cache = Some(ids);
        }
        self.op_token_cache.as_ref().unwrap()
    }

    fn apply(&mut self, generated_ids: &[u32], logits: &mut Vec<f32>) {
        self.generated_text = self.tokenizer.decode(generated_ids, true).unwrap_or_default();

        // Detect if we're inside the op-name field.
        let in_op_name = if let Some(pos) = self.generated_text.find("{\"op\":\"") {
            let after = &self.generated_text[pos + 7..];
            !after.contains('"') // not yet closed
        } else {
            false
        };

        if !in_op_name { return; }

        let so_far = {
            let pos = self.generated_text.find("{\"op\":\"").unwrap();
            self.generated_text[pos + 7..].to_string()
        };

        let _ = self.op_tokens();
        let candidate_ids: Vec<u32> = self.op_token_cache.as_ref().unwrap().clone();
        let allowed_ops: Vec<&str> = self.allowed_ops.clone();
        let tokenizer = &self.tokenizer;

        let valid_next: HashSet<u32> = candidate_ids.iter().copied()
            .filter(|&id| {
                let s = tokenizer.decode(&[id], false).unwrap_or_default();
                if s == "\"" {
                    allowed_ops.iter().any(|op| *op == so_far.as_str())
                } else if !s.is_empty() {
                    let candidate = format!("{so_far}{s}");
                    allowed_ops.iter().any(|op| op.starts_with(candidate.as_str()))
                } else { false }
            })
            .collect();

        if !valid_next.is_empty() {
            for (i, v) in logits.iter_mut().enumerate() {
                if !valid_next.contains(&(i as u32)) { *v = f32::NEG_INFINITY; }
            }
        }
    }
}

// ── Cases ─────────────────────────────────────────────────────────────────────

struct Case {
    prompt: &'static str,
    expected_route: &'static str,
    expected_op: &'static str,
    expected_result: Value,
}

fn cases() -> Vec<Case> {
    vec![
        Case { prompt: "What is the GCD of 144 and 60?",  expected_route: "arithmetic", expected_op: "gcd",          expected_result: json!(12) },
        Case { prompt: "Is 97 a prime number?",            expected_route: "arithmetic", expected_op: "is_prime",     expected_result: json!(true) },
        Case { prompt: "What is 10 factorial?",            expected_route: "arithmetic", expected_op: "factorial",    expected_result: json!(3628800) },
        Case { prompt: "Write 2024 as a Roman numeral.",   expected_route: "arithmetic", expected_op: "to_roman",     expected_result: json!("MMXXIV") },
        Case { prompt: "Is 2024 a leap year?",             expected_route: "date",       expected_op: "is_leap_year", expected_result: json!(true) },
        Case { prompt: "How many days are in February 2026?", expected_route: "date",    expected_op: "days_in_month",expected_result: json!(28) },
        Case { prompt: "Reverse the string \"helloworld\".", expected_route: "code",    expected_op: "reverse",      expected_result: json!("dlrowolleh") },
        Case { prompt: "Is \"racecar\" a palindrome?",     expected_route: "code",       expected_op: "is_palindrome",expected_result: json!(true) },
    ]
}

/// Terse system prompt for Gemma (works well with its instruction tuning).
const SYSTEM_GEMMA: &str = r#"Respond with ONLY a JSON object {"op":"...","args":{...}}.
ops: gcd{"a","b"}, is_prime{"n"}, factorial{"n"}, to_roman{"n"}, is_leap_year{"year"}, days_in_month{"year","month"}, reverse{"s"}, is_palindrome{"s"}
No extra text."#;

/// Explicit system prompt for Mistral — includes worked examples and
/// spells out that args must be a JSON object with named keys.
/// String values must preserve spaces exactly as given.
const SYSTEM_MISTRAL: &str = r#"Answer with ONLY a JSON object, nothing else.
Format: {"op":"OPERATION","args":{"KEY":VALUE}}
Preserve all characters exactly — including spaces inside strings.
Number example: {"op":"gcd","args":{"a":144,"b":60}}
String example for 'reverse the string "hello world"': {"op":"reverse","args":{"s":"hello world"}}
Available ops and argument names:
gcd(a,b)  is_prime(n)  factorial(n)  to_roman(n)
is_leap_year(year)  days_in_month(year,month)
reverse(s)  is_palindrome(s)"#;

/// System prompt for Llama-3 (similar structure to Gemma but with examples).
const SYSTEM_LLAMA: &str = r#"Output ONLY a JSON object {"op":"...","args":{...}}, no extra text.
Number example: {"op":"gcd","args":{"a":144,"b":60}}
String example: {"op":"reverse","args":{"s":"hello world"}}
ops: gcd(a,b), is_prime(n), factorial(n), to_roman(n), is_leap_year(year), days_in_month(year,month), reverse(s), is_palindrome(s)"#;

fn system_for_model(mid: &str) -> &'static str {
    if mid.contains("Mistral") || mid.contains("mistral") { SYSTEM_MISTRAL }
    else if mid.contains("Llama") || mid.contains("llama") { SYSTEM_LLAMA }
    else { SYSTEM_GEMMA }
}

// ── Test ──────────────────────────────────────────────────────────────────────

#[test]
fn trie_dispatch_pipeline() {
    if !wasm_dir().exists() {
        eprintln!("skip: wasm dir missing");
        return;
    }

    let mid = model_id();
    let pp = probe_path_for_model(&mid);
    if !pp.exists() {
        eprintln!("skip: probe missing for {mid} — run experiments/export_trie_probe.py --model {mid}");
        eprintln!("  expected: {}", pp.display());
        return;
    }

    let model = match InferenceModel::load(&mid) {
        Ok(m) => m,
        Err(e) => { eprintln!("skip: {e}"); return; }
    };
    eprintln!("model: {mid}  ({} layers)", model.num_layers());

    let trie = CascadeTrie::load(&pp).expect("load probe");
    eprintln!("probe: L{}  routes: {:?}", trie.layer, trie.routes());

    let mut reg = ExpertRegistry::load_dir(&wasm_dir()).expect("load_dir");
    let ffn = WeightFfn { weights: model.weights() };

    let mut passed = 0usize;
    let mut failed = 0usize;

    for case in cases() {
        let system = system_for_model(&mid);
        let full_prompt = format!("{system}\n\nQuestion: {}", case.prompt);
        let wrapped = chat_for_model(&mid, &full_prompt);

        // Full wrapped prompt for generation.
        let ids_gen = match encode_prompt(model.tokenizer(), &*model.weights().arch, &wrapped) {
            Ok(v) => v,
            Err(e) => { eprintln!("  FAIL tokenize: {e}"); failed += 1; continue; }
        };
        // Bare question (no system prompt, no chat template) for the L5 probe.
        // The probe was trained on plain question-format prompts so it needs
        // the same distribution at inference time.
        let ids_probe = match encode_prompt(model.tokenizer(), &*model.weights().arch, case.prompt) {
            Ok(v) => v,
            Err(e) => { eprintln!("  FAIL tokenize probe: {e}"); failed += 1; continue; }
        };

        // ── Step 1: L5 probe (partial prefill on bare question, 6 layers only) ──
        let h5 = forward_to_layer(model.weights(), &ids_probe, trie.layer);
        // Last-position hidden state
        let last_row = h5.row(h5.shape()[0] - 1);
        let hidden: Vec<f32> = last_row.to_vec();
        let route = trie.classify(&hidden).to_string();

        // ── Step 2: narrow op vocabulary to this route ──
        let allowed_ops = ops_for_route(&route, &reg);
        eprintln!("\n  prompt:   {}", case.prompt);
        eprintln!("  route:    {route}{}  ({} ops)",
            if route == case.expected_route { "" } else { " ← WRONG" },
            allowed_ops.len());

        // ── Step 3: grammar-constrained generation ──
        let mut mask = RouteOpMask::new(allowed_ops, model.tokenizer().clone());
        let mut output = String::new();
        generate_cached_constrained(
            model.weights(),
            model.tokenizer(),
            &ffn,
            &ids_gen,
            128,
            |gen_ids, logits| mask.apply(gen_ids, logits),
            |_id, tok| output.push_str(tok),
        );
        eprintln!("  raw out: {output:?}");

        let parsed = match extract_json(&output) {
            Some(v) => v,
            None => { eprintln!("  FAIL: no JSON"); failed += 1; continue; }
        };
        let op = match parsed.get("op").and_then(|v| v.as_str()) {
            Some(s) => s.to_string(),
            None => { eprintln!("  FAIL: no op"); failed += 1; continue; }
        };
        let args = parsed.get("args").cloned().unwrap_or(json!({}));
        eprintln!("  op={op}{}  args={args}",
            if op == case.expected_op { "" } else { " ← WRONG OP" });

        if op != case.expected_op {
            eprintln!("  FAIL: expected op={}", case.expected_op);
            failed += 1;
            continue;
        }

        // ── Step 4: dispatch ──
        match reg.call(&op, &args) {
            Some(r) if r.value == case.expected_result => {
                eprintln!("  ok  [{route}/{op}] {} → {}", case.prompt, r.value);
                passed += 1;
            }
            Some(r) => { eprintln!("  FAIL: got {}, expected {}", r.value, case.expected_result); failed += 1; }
            None    => { eprintln!("  FAIL: registry None  op={op} args={args}"); failed += 1; }
        }
    }

    let total = passed + failed;
    eprintln!("\n{passed}/{total} trie dispatch cases passed");
    assert_eq!(failed, 0, "{failed} cases failed");
}
