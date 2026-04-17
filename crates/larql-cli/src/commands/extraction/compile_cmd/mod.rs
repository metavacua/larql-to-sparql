//! `larql compile` — AOT compilation of vindex patches or single facts to
//! standard safetensors checkpoints. Output runs in any inference engine
//! without LARQL.
//!
//! Two modes:
//! - **Single** (`--prompt` + `--answer`): one compiled edge from a prompt's
//!   residual at `--layer`, writing the answer token. CLI-driven; used for
//!   the pi/Gauss demos and any prompt→answer pair.
//! - **Patch** (`--vindex`): replays Insert ops from .vlp patch files into
//!   the model's FFN slots. Vindex-driven; many edges per run.
//!
//! The install primitive in [`edge::install_edge`] mirrors the convention
//! described in `experiments/07_wasm_compute/WASM_GATE_ARCHITECTURE.md` §3.1.2.

use std::path::PathBuf;

use clap::Args;

mod chat;
mod detect;
mod edge;
mod patch;
mod save;
mod single;

#[derive(Args)]
pub struct CompileArgs {
    /// Path to the base model (directory with safetensors, or HF model ID).
    #[arg(long)]
    pub base: PathBuf,

    /// Path to the vindex (with patches to compile). Not needed for fact mode.
    #[arg(long)]
    pub vindex: Option<PathBuf>,

    /// Output directory for the compiled model safetensors.
    #[arg(short, long)]
    pub output: PathBuf,

    /// Gate scale for compiled edges (default: 1.0).
    /// Previous default 30.0 saturated silu on every question prompt and
    /// leaked the edge into unrelated queries; 1.0 keeps natural usage
    /// clean on Gemma 3 4B. See experiments/07_wasm_compute/RESULTS.md.
    #[arg(long, default_value = "1.0")]
    pub gate_scale: f32,

    /// Alpha multiplier for initial write magnitude (default: 0.3).
    /// The balancer (single mode) refines this after install by scaling
    /// the down vector up/down until the target-token probability lands
    /// in [--floor, --ceiling].
    #[arg(long, default_value = "0.3")]
    pub alpha: f32,

    // ── Balancer options (single mode only) ─────────────────────
    /// Minimum probability the target token must reach before the
    /// balancer stops scaling up the down vector.
    #[arg(long, default_value = "0.40")]
    pub floor: f64,

    /// Maximum probability the target token may reach before the
    /// balancer starts scaling down. Too-confident installs over-ride
    /// context and regress unrelated prompts.
    #[arg(long, default_value = "0.85")]
    pub ceiling: f64,

    /// Maximum balancer iterations. Default 0 — the balancer is opt-in
    /// because `larql_inference::forward::predict` is systematically
    /// "peakier" than HF transformers' forward pass on the same weights,
    /// so scaling the down vector to reach [floor, ceiling] in Rust's
    /// simulation over-dampens the edge relative to deployed inference.
    /// Leaving this at 0 installs at --alpha / --gate-scale and trusts
    /// the caller's pre-tuned defaults (the paraphrase-sweep sweet spot:
    /// g=1.0, α=0.3). Set --max-iters >0 only if you have reason to
    /// believe Rust's predict tracks HF for your model.
    #[arg(long, default_value = "0")]
    pub max_iters: u32,

    /// Skip applying the base model's `tokenizer_config.json::chat_template`
    /// to the prompt before tokenising. By default the template is loaded
    /// from the base model and rendered (so the trigger residual captured
    /// here matches what a served/chat-wrapped deployment will produce).
    /// Only set this for raw-prompt experiments.
    #[arg(long, default_value = "false")]
    pub no_chat_template: bool,

    // ── Fact compilation mode ─────────────────────────────────
    /// Prompt text whose residual becomes the trigger direction.
    #[arg(long)]
    pub prompt: Option<String>,

    /// Correct answer token to compile into the weights.
    #[arg(long)]
    pub answer: Option<String>,

    /// Layer to install the compiled edge at (default: 30).
    #[arg(long, default_value = "30")]
    pub layer: usize,

    /// FFN slot to install the compiled edge at (default: 9000).
    #[arg(long, default_value = "9000")]
    pub slot: usize,
}

pub fn run(args: CompileArgs) -> Result<(), Box<dyn std::error::Error>> {
    if args.prompt.is_some() && args.answer.is_some() {
        return single::run(args);
    }
    if args.vindex.is_none() {
        return Err("either --vindex or --prompt + --answer required".into());
    }
    patch::run(args)
}
