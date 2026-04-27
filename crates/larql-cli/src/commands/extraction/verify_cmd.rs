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

    let config = larql_vindex::load_vindex_config(&args.vindex)?;

    let stored = match &config.checksums {
        Some(c) if !c.is_empty() => c,
        _ => {
            eprintln!("No checksums in index.json. Run extract to generate them.");
            return Ok(());
        }
    };

    eprintln!("Verifying: {} ({} files)", args.vindex.display(), stored.len());

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
