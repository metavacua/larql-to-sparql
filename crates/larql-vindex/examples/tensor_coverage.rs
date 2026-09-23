//! Audit a checkpoint's tensors against what its architecture can name.
//!
//! ```text
//! cargo run -p larql-vindex --example tensor_coverage -- <checkpoint-dir>
//! ```
//!
//! Reads the safetensors headers directly rather than loading the model, so
//! it sees the checkpoint as it is on disk — before any loader-side filtering
//! or renaming. That is the level the silent-drop bugs live at.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use larql_vindex::extract::coverage;

fn shard_paths(dir: &Path) -> std::io::Result<Vec<PathBuf>> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    v.sort();
    Ok(v)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::args()
        .nth(1)
        .ok_or("usage: tensor_coverage <checkpoint-dir>")?;
    let dir = PathBuf::from(dir);

    let arch = larql_models::detect_architecture(&dir)?;
    let num_layers = arch.config().num_layers;
    let prefixes = arch.key_prefixes_to_strip();

    // Enumerate every tensor name across every shard header.
    let mut names: BTreeSet<String> = BTreeSet::new();
    for path in shard_paths(&dir)? {
        let file = std::fs::File::open(&path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        let (_, meta) = safetensors::SafeTensors::read_metadata(&mmap)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        for name in meta.tensors().keys() {
            let mut n = name.as_str();
            for p in prefixes {
                if let Some(stripped) = n.strip_prefix(p) {
                    n = stripped;
                    break;
                }
            }
            names.insert(n.to_string());
        }
    }

    println!(
        "{}\n  architecture: {}  layers: {}  experts: {}",
        dir.display(),
        arch.family(),
        num_layers,
        arch.num_experts()
    );

    let report = coverage::audit(&*arch, num_layers, names.iter().map(|s| s.as_str()));
    println!("{}", report.summary());

    if !report.dropped.is_empty() {
        let mut by_rule: std::collections::BTreeMap<&str, usize> = Default::default();
        for (_, rule) in &report.dropped {
            *by_rule.entry(rule.name).or_default() += 1;
        }
        println!("dropped by rule:");
        for (rule, n) in by_rule {
            println!("  {rule}: {n}");
        }
    }

    std::process::exit(if report.is_clean() { 0 } else { 1 });
}
