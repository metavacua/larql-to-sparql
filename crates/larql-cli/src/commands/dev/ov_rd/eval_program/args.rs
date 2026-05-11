use std::path::PathBuf;

use clap::Args;

#[derive(Args)]
pub struct EvalProgramArgs {
    #[arg(long)]
    pub index: PathBuf,

    #[arg(long)]
    pub program: PathBuf,

    #[arg(long)]
    pub prompts: PathBuf,

    #[arg(long)]
    pub out: PathBuf,

    /// Maximum prompts per stratum. 0 = unlimited.
    #[arg(long, default_value_t = 0)]
    pub max_per_stratum: usize,

    #[arg(long, default_value_t = 1)]
    pub eval_mod: usize,

    #[arg(long, default_value_t = 0)]
    pub eval_offset: usize,

    #[arg(long, default_value_t = 1e-6)]
    pub sigma_rel_cutoff: f64,

    #[arg(long, default_value_t = 25)]
    pub pq_iters: usize,

    /// Use Metal GPU for forward passes (requires --features metal on macOS).
    /// ~10× faster than CPU for the Mode D injection evaluations.
    #[arg(long, default_value_t = false)]
    pub metal: bool,
}
