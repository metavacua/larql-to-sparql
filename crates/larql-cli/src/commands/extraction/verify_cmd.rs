use std::path::PathBuf;

use clap::Args;

#[derive(Args)]
pub struct VerifyArgs {
    /// Path to the .vindex directory to verify.
    vindex: PathBuf,
}

pub fn run(args: VerifyArgs) -> Result<(), Box<dyn std::error::Error>> {
    if !args.vindex.is_dir() {
        return Err(format!("not a directory: {}", args.vindex.display()).into());
    }

    // A VINDEX2 container is opaque blobs, so verifying it means checksums.
    // A VINDEX3 container declares its own structure, so it can answer the
    // stronger question — *will this bind?* — without executing anything.
    if larql_vindex::format::generation::detect_generation(&args.vindex)?
        == larql_vindex::format::generation::ContainerGeneration::V3
    {
        return verify_v3(&args.vindex);
    }

    let config = larql_vindex::load_vindex_config(&args.vindex)?;

    let stored = match &config.checksums {
        Some(c) if !c.is_empty() => c,
        _ => {
            eprintln!("No checksums in index.json. Run extract to generate them.");
            return Ok(());
        }
    };

    eprintln!(
        "Verifying: {} ({} files)",
        args.vindex.display(),
        stored.len()
    );

    let results = larql_vindex::format::checksums::verify_checksums(&args.vindex, stored)?;

    let mut all_ok = true;
    for (filename, ok) in &results {
        let path = args.vindex.join(filename);
        let size_str = std::fs::metadata(&path)
            .map(|m| {
                let mb = m.len() as f64 / (1024.0 * 1024.0);
                if mb > 1024.0 {
                    format!("{:.2} GB", mb / 1024.0)
                } else {
                    format!("{:.1} MB", mb)
                }
            })
            .unwrap_or_else(|_| "missing".into());

        if *ok {
            println!("  {} ... OK ({})", filename, size_str);
        } else {
            println!("  {} ... FAILED ({})", filename, size_str);
            all_ok = false;
        }
    }

    if all_ok {
        println!("\nAll {} files verified.", results.len());
    } else {
        println!("\nVerification FAILED. Some files have been modified or corrupted.");
        std::process::exit(1);
    }

    Ok(())
}

/// Structural verification of a VINDEX3 container.
///
/// Checks what a checksum sweep cannot: that the container's own declarations
/// are mutually consistent and complete enough to bind. Opening it already
/// established that `index.json` parses, the manifest parses and validates,
/// and every declared storage key resolves to a file; `verify` adds segment
/// parsing, programme satisfaction and per-entry region bounds.
///
/// Execution parity is deliberately excluded — it needs an input and a kernel,
/// and folding it in would make routine verification cost a forward pass.
fn verify_v3(path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("Verifying: {} (VINDEX3, structural)", path.display());
    let container = larql_vindex::format::vindex3::Vindex3Container::open(path)?;

    println!(
        "  index.json ......... OK (schema {})",
        container.index().version
    );
    println!(
        "  moe_manifest ....... OK ({} MoE layer(s))",
        container.manifest().layers.len()
    );
    println!(
        "  storage keys ....... OK ({} segment(s) resolved)",
        container.index().segments.len()
    );

    let defects = container.verify();
    if defects.is_empty() {
        println!("  structure .......... OK (bindable)");
        println!("\nAll checks passed.");
        return Ok(());
    }
    println!("  structure .......... {} defect(s)", defects.len());
    for d in &defects {
        println!("    - {d}");
    }
    Err(format!(
        "{} structural defect(s); container is not bindable",
        defects.len()
    )
    .into())
}
