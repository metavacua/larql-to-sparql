use std::path::PathBuf;

use clap::Args;

use super::program::Program;

#[derive(Args)]
pub(super) struct NormalizeProgramArgs {
    /// Input program JSON.
    #[arg(long)]
    program: PathBuf,

    /// Output path for the normalized program JSON.
    #[arg(long)]
    out: PathBuf,
}

pub(super) fn run_normalize_program(
    args: NormalizeProgramArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    let text = std::fs::read_to_string(&args.program)?;
    let mut program: Program = serde_json::from_str(&text)?;
    program.validate()?;
    program.normalize();

    let file = std::fs::File::create(&args.out)?;
    serde_json::to_writer_pretty(file, &program)?;

    for stage in &program.stages {
        let map = stage.effective_map.as_ref().map(|m| m.len()).unwrap_or(0);
        eprintln!("stage '{}': {} effective remaps", stage.name, map);
        if let Some(m) = &stage.effective_map {
            for (src, dst) in m {
                eprintln!("  {src} -> {dst}");
            }
        }
        for g in &stage.guards {
            eprintln!(
                "  guard code {}: preserves when {}",
                g.code, g.preserves_when
            );
        }
    }
    eprintln!("Wrote {}", args.out.display());
    Ok(())
}
