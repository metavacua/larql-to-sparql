use std::path::PathBuf;

use clap::Args;

#[derive(Args)]
pub struct InduceProgramArgs {
    /// Self-contained Q4K vindex directory.
    #[arg(long)]
    pub index: PathBuf,

    /// JSONL prompt file.
    #[arg(long)]
    pub prompts: PathBuf,

    /// Output directory.
    #[arg(long)]
    pub out: PathBuf,

    /// Target head as layer:head, e.g. 0:6.
    #[arg(long)]
    pub head: String,

    /// PQ group to induce a program for.
    #[arg(long, default_value_t = 0)]
    pub group: usize,

    /// PQ base config as k:groups:bits, e.g. 192:48:4.
    #[arg(long, default_value = "192:48:4")]
    pub base_config: String,

    /// Maximum prompts per stratum. 0 = unlimited.
    #[arg(long, default_value_t = 0)]
    pub max_per_stratum: usize,

    /// Train/eval split modulus.
    #[arg(long, default_value_t = 1)]
    pub eval_mod: usize,

    /// Offset for --eval-mod split.
    #[arg(long, default_value_t = 0)]
    pub eval_offset: usize,

    /// W_O-visible singular-value cutoff.
    #[arg(long, default_value_t = 1e-6)]
    pub sigma_rel_cutoff: f64,

    /// Lloyd iterations per PQ group.
    #[arg(long, default_value_t = 25)]
    pub pq_iters: usize,

    /// Use Metal GPU for forward passes (requires --features metal on macOS).
    /// ~10× faster than CPU; reduces CEGIS loop from ~2.5h to ~15 minutes.
    #[arg(long, default_value_t = false)]
    pub metal: bool,
}
