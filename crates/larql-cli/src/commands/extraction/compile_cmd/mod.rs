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

    /// Gate scale for compiled edges (default: 30.0).
    #[arg(long, default_value = "30.0")]
    pub gate_scale: f32,

    /// Alpha multiplier for write magnitude (default: 10.0).
    #[arg(long, default_value = "10.0")]
    pub alpha: f32,

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
