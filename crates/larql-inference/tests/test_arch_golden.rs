//! Architecture regression guard: one "golden-token" case per architecture.
//!
//! The goal is to catch regressions when we change a cross-cutting path (MoE,
//! norm application, QK-norm, RoPE, KV-sharing, quantization) that silently
//! breaks an architecture we don't routinely test by hand. Each case loads a
//! real vindex out of `~/.cache/larql/local/` (or the workspace `output/`
//! directory), runs a few-token Metal decode for a fixed prompt, and asserts
//! that the first generated token contains a known-good substring.
//!
//! **Skip behaviour.** If the vindex for an architecture is missing from the
//! cache, the corresponding test prints a skip note and returns Ok — CI stays
//! green and the test is a no-op. Set `LARQL_ARCH_STRICT=1` to turn a missing
//! vindex into a hard failure (useful locally to catch "I forgot to pull").
//!
//! **Env knobs.**
//!   - `LARQL_ARCH_STRICT=1` — require every case's vindex to be present.
//!   - `LARQL_ARCH_PROMPT=<text>` — override the shared prompt (default:
//!     `"The capital of France is"`).
//!   - `LARQL_ARCH_TOKENS=<n>` — override the generated-token budget (default 3).
//!
//! **Why not `#[ignore]`?** `cargo test` runs these by default so anyone who
//! breaks an arch in an edit-test loop notices immediately. Skipped cases
//! aren't failures; skipped cases are the common path on CI that doesn't
//! cache 40 GB of weights.

use std::path::{Path, PathBuf};

use larql_compute::{cpu_backend, default_backend, ComputeBackend};
use larql_inference::encode_prompt;
use larql_inference::layer_graph::{generate as gen, CachedLayerGraph};
use larql_vindex::{
    load_model_weights_q4k, load_vindex_tokenizer, QuantFormat, SilentLoadCallbacks, VectorIndex,
};

/// Which backend flavour to exercise. GPU uses the platform default (Metal
/// on macOS, falls back to CPU elsewhere); CPU uses the pure-Rust backend
/// so we can assert the compute paths stay in lockstep.
#[derive(Clone, Copy)]
enum BackendKind { Gpu, Cpu }

impl BackendKind {
    fn name(&self) -> &'static str {
        match self { Self::Gpu => "gpu", Self::Cpu => "cpu" }
    }
    fn backend(&self) -> Box<dyn ComputeBackend> { match self {
        Self::Gpu => default_backend(),
        Self::Cpu => cpu_backend(),
    }}
}

/// One architecture we want to guard against regressions.
///
/// `vindex_name` is the cache shorthand (matches the directory in
/// `~/.cache/larql/local/<name>.vindex/`). `expected_substring` is a
/// substring the generated text must contain to pass — substring rather
/// than exact-match because tokenisers differ on leading-space, and the
/// first token of "The capital of France is" might be `" Paris"`,
/// `"Paris"`, `"\u{2581}Paris"`, etc.
struct ArchCase {
    arch_family: &'static str,
    vindex_name: &'static str,
    expected_substring: &'static str,
    /// Marks architectures where the CPU backend deliberately does not
    /// implement the forward path yet (hybrid MoE at time of writing).
    /// The CPU test skips cleanly rather than firing a false-positive
    /// regression. Flip to `false` once CPU parity ships.
    cpu_unimplemented: bool,
}

/// The default golden suite. Add a row per supported architecture; the test
/// below walks this list.
///
/// The expected substring is whatever the base model actually continues
/// with — we're guarding against "did we break this arch?" not "is this
/// model factually correct?". Instruct-tuned Gemmas do answer "Paris";
/// Llama 2 base rambles into "a city of contrasts"; Mistral base gets it.
const CASES: &[ArchCase] = &[
    ArchCase {
        arch_family: "gemma3", vindex_name: "gemma3-4b-q4k-v2",
        expected_substring: "Paris", cpu_unimplemented: false,
    },
    ArchCase {
        arch_family: "gemma4-dense", vindex_name: "gemma4-31b-q4k",
        expected_substring: "Paris", cpu_unimplemented: false,
    },
    // Hybrid-MoE: GPU path currently regresses (outer-norm combine still
    // wrong on the CPU-side MoE accumulation — tracked as task #12). CPU
    // backend has no MoE forward implementation at all; flag it so the
    // test skips cleanly rather than asserting against a silently-dense
    // run that would look "right enough" to mask regressions.
    ArchCase {
        arch_family: "gemma4-moe", vindex_name: "gemma-4-26B-A4B-it",
        expected_substring: "Paris", cpu_unimplemented: true,
    },
    // Llama 2 base isn't instruct-tuned — "a city of contrasts" is its
    // actual continuation. Anchor on "city" rather than "Paris".
    ArchCase {
        arch_family: "llama2", vindex_name: "llama2-7b-q4k",
        expected_substring: "city", cpu_unimplemented: false,
    },
    ArchCase {
        arch_family: "mistral", vindex_name: "mistral-7b-v0.1-q4k",
        expected_substring: "Paris", cpu_unimplemented: false,
    },
];

/// Resolve a vindex directory from the local cache or workspace output dir.
/// Returns `None` if nothing matched — caller decides whether that's a skip
/// or a hard error based on `LARQL_ARCH_STRICT`.
fn find_vindex(name: &str) -> Option<PathBuf> {
    let filename = format!("{name}.vindex");

    // Absolute-override env var.
    if let Ok(env_path) = std::env::var(format!("LARQL_VINDEX_{}", name.to_uppercase().replace('-', "_"))) {
        let p = PathBuf::from(env_path);
        if p.is_dir() { return Some(p); }
    }

    // Known external location used by the 26B A4B test weights.
    let chris_models = PathBuf::from("/Users/christopherhay/chris-models").join(&filename);
    if chris_models.is_dir() { return Some(chris_models); }

    let home = std::env::var("HOME").ok()?;
    let candidates = [
        PathBuf::from(&home).join(".cache/larql/local").join(&filename),
        PathBuf::from("output").join(&filename),
    ];
    candidates.into_iter().find(|p| p.is_dir())
}

/// Load a vindex and run the first few tokens through the requested backend.
/// Returns the concatenated generated text (token surface forms joined in order).
fn run_case(
    vindex_path: &Path,
    prompt: &str,
    max_tokens: usize,
    backend_kind: BackendKind,
) -> Result<String, String> {
    let mut cb = SilentLoadCallbacks;

    let cfg = larql_vindex::load_vindex_config(vindex_path)
        .map_err(|e| format!("load_vindex_config: {e}"))?;
    if cfg.quant != QuantFormat::Q4k {
        return Err(format!("only Q4k vindexes are supported by this suite (got {:?})", cfg.quant));
    }

    let weights = load_model_weights_q4k(vindex_path, &mut cb)
        .map_err(|e| format!("load_model_weights_q4k: {e}"))?;
    let tokenizer = load_vindex_tokenizer(vindex_path)
        .map_err(|e| format!("load_vindex_tokenizer: {e}"))?;
    let mut q4_index = VectorIndex::load_vindex(vindex_path, &mut cb)
        .map_err(|e| format!("VectorIndex::load_vindex: {e}"))?;
    q4_index.load_attn_q4k(vindex_path).map_err(|e| format!("load_attn_q4k: {e}"))?;
    q4_index.load_interleaved_q4k(vindex_path).map_err(|e| format!("load_interleaved_q4k: {e}"))?;
    let _ = q4_index.load_lm_head_q4(vindex_path);

    let prompt_ids = encode_prompt(&tokenizer, &*weights.arch, prompt)
        .map_err(|e| format!("encode_prompt: {e}"))?;

    let backend = backend_kind.backend();
    let cached = CachedLayerGraph::from_residuals(Vec::new());
    let num_layers = weights.num_layers;

    let result = gen(
        &weights,
        &tokenizer,
        &prompt_ids,
        max_tokens,
        &q4_index,
        &*backend,
        &cached,
        0..num_layers,
    );
    Ok(result.tokens.iter().map(|(t, _)| t.as_str()).collect())
}

fn strict_mode() -> bool {
    matches!(std::env::var("LARQL_ARCH_STRICT").ok().as_deref(), Some("1") | Some("true"))
}

fn prompt() -> String {
    std::env::var("LARQL_ARCH_PROMPT").unwrap_or_else(|_| "The capital of France is".to_string())
}

fn max_tokens() -> usize {
    std::env::var("LARQL_ARCH_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3)
}

/// Exercise one case on one backend. Asserts on success/failure; calls
/// `eprintln!` + returns for a skip so the test stays green when the vindex
/// isn't on disk.
fn exercise_case(case: &ArchCase, backend_kind: BackendKind) {
    // Gate tests whose CPU forward isn't implemented yet (MoE). Skip rather
    // than falsely pass against a silently-dense CPU fallback.
    if matches!(backend_kind, BackendKind::Cpu) && case.cpu_unimplemented {
        eprintln!(
            "[{}/{}] skip: CPU forward is not implemented for this architecture yet",
            case.arch_family, backend_kind.name(),
        );
        return;
    }

    let Some(vindex_path) = find_vindex(case.vindex_name) else {
        if strict_mode() {
            panic!(
                "[{}/{}] vindex `{}` not found in cache (LARQL_ARCH_STRICT=1)",
                case.arch_family, backend_kind.name(), case.vindex_name,
            );
        }
        eprintln!(
            "[{}/{}] skip: vindex `{}` not found in ~/.cache/larql/local/ or output/ — \
             set LARQL_ARCH_STRICT=1 to fail instead.",
            case.arch_family, backend_kind.name(), case.vindex_name,
        );
        return;
    };
    eprintln!("[{}/{}] vindex: {}", case.arch_family, backend_kind.name(), vindex_path.display());

    let prompt = prompt();
    let max = max_tokens();

    let out = run_case(&vindex_path, &prompt, max, backend_kind).unwrap_or_else(|e| {
        panic!("[{}/{}] run_case failed: {e}", case.arch_family, backend_kind.name())
    });

    eprintln!("[{}/{}] prompt={prompt:?} generated={out:?}",
        case.arch_family, backend_kind.name());
    assert!(
        out.to_lowercase().contains(&case.expected_substring.to_lowercase()),
        "[{}/{}] generated text {out:?} does not contain expected substring {:?}",
        case.arch_family, backend_kind.name(), case.expected_substring,
    );
}

// ── One #[test] per (architecture × backend) ──────────────────────────────
//
// Kept as individual functions (rather than a table-driven loop) so a single
// regression surfaces as one clearly-named failing test, not a buried
// "assertion failed at index 2". GPU uses `default_backend()` (Metal on
// macOS); CPU uses `cpu_backend()`. Both paths must stay in lockstep — a
// change that breaks one is a bug even if the other still passes.

#[test] fn arch_gemma3_4b_gpu()         { exercise_case(&CASES[0], BackendKind::Gpu); }
#[test] fn arch_gemma3_4b_cpu()         { exercise_case(&CASES[0], BackendKind::Cpu); }
#[test] fn arch_gemma4_31b_dense_gpu()  { exercise_case(&CASES[1], BackendKind::Gpu); }
#[test] fn arch_gemma4_31b_dense_cpu()  { exercise_case(&CASES[1], BackendKind::Cpu); }
#[test] fn arch_gemma4_26b_a4b_moe_gpu(){ exercise_case(&CASES[2], BackendKind::Gpu); }
#[test] fn arch_gemma4_26b_a4b_moe_cpu(){ exercise_case(&CASES[2], BackendKind::Cpu); }
#[test] fn arch_llama2_7b_gpu()         { exercise_case(&CASES[3], BackendKind::Gpu); }
#[test] fn arch_llama2_7b_cpu()         { exercise_case(&CASES[3], BackendKind::Cpu); }
#[test] fn arch_mistral_7b_gpu()        { exercise_case(&CASES[4], BackendKind::Gpu); }
#[test] fn arch_mistral_7b_cpu()        { exercise_case(&CASES[4], BackendKind::Cpu); }
