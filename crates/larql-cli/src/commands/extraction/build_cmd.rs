use std::path::PathBuf;

use clap::Args;

#[derive(Args)]
pub struct BuildArgs {
    /// Directory containing the Vindexfile (defaults to current directory).
    #[arg(default_value = ".")]
    dir: PathBuf,

    /// Build stage (e.g. dev, prod, edge). If omitted, uses top-level directives only.
    #[arg(long)]
    stage: Option<String>,

    /// Output directory for the built vindex (defaults to ./build/vindex/).
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Compile the built vindex to a model format after building.
    #[arg(long)]
    compile: Option<String>,
}

pub fn run(args: BuildArgs) -> Result<(), Box<dyn std::error::Error>> {
    let vindexfile_path = args.dir.join("Vindexfile");
    if !vindexfile_path.exists() {
        return Err(format!("Vindexfile not found: {}", vindexfile_path.display()).into());
    }

    eprintln!("Parsing Vindexfile: {}", vindexfile_path.display());

    let vf = larql_vindex::parse_vindexfile(&vindexfile_path)?;

    // Summary
    let stage_str = args.stage.as_deref().unwrap_or("(default)");
    let num_patches = vf.directives.iter().filter(|d| matches!(d, larql_vindex::VindexfileDirective::Patch(_))).count();
    let num_inserts = vf.directives.iter().filter(|d| matches!(d, larql_vindex::VindexfileDirective::Insert { .. })).count();
    let num_deletes = vf.directives.iter().filter(|d| matches!(d, larql_vindex::VindexfileDirective::Delete { .. })).count();
    eprintln!(
        "  Stage: {}, {} patches, {} inserts, {} deletes, {} stages defined",
        stage_str, num_patches, num_inserts, num_deletes, vf.stages.len(),
    );

    // Build
    eprintln!("\nBuilding...");
    let result = larql_vindex::build_from_vindexfile(
        &vf,
        args.stage.as_deref(),
        &args.dir,
    )?;

    // Print build history
    eprintln!("\nBuild history:");
    for (i, layer) in result.layers.iter().enumerate() {
        let mod_str = if layer.features_modified > 0 {
            format!(" ({} features)", layer.features_modified)
        } else {
            String::new()
        };
        eprintln!("  Layer {}: {}{}", i, layer.directive, mod_str);
    }

    // Save to output directory
    let output_dir = args.output.unwrap_or_else(|| args.dir.join("build").join("vindex"));
    std::fs::create_dir_all(&output_dir)?;

    eprintln!("\nSaving to {}...", output_dir.display());

    let layer_infos = result.index.save_gate_vectors(&output_dir)?;
    let dm_count = result.index.save_down_meta(&output_dir)?;

    let mut config = result.config;
    config.layers = layer_infos;
    config.checksums = larql_vindex::format::checksums::compute_checksums(&output_dir).ok();
    larql_vindex::VectorIndex::save_config(&config, &output_dir)?;

    eprintln!("  Features: {}", dm_count);

    // Total overrides
    let total_modified: usize = result.layers.iter().map(|l| l.features_modified).sum();
    eprintln!(
        "  Total: {} features modified from base",
        total_modified
    );

    if let Some(format) = args.compile {
        eprintln!("\nCompiling to {} format...", format);
        eprintln!("  (compile not yet implemented — built vindex saved at {})", output_dir.display());
    }

    eprintln!("\nDone. Usage:");
    eprintln!("  larql repl");
    eprintln!("  larql> USE \"{}\";", output_dir.display());

    Ok(())
}
