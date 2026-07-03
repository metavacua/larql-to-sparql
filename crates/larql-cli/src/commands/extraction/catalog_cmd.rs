use clap::Parser;
use larql_knowledge::{build::assemble, relations::VALIDATED, fetch::fetch_pairs};

#[derive(Parser, Debug)]
pub struct CatalogArgs {
    /// Output path for catalog.json
    #[arg(short, long)]
    pub output: std::path::PathBuf,
    /// Max pairs to fetch per relation
    #[arg(long, default_value_t = 30)]
    pub limit: usize,
}

pub fn run(args: CatalogArgs) -> Result<(), Box<dyn std::error::Error>> {
    let mut entries = Vec::new();
    for spec in VALIDATED {
        let pairs = fetch_pairs(&spec.query(args.limit))?;
        eprintln!("{}: {} pairs", spec.name, pairs.len());
        entries.push((spec, pairs));
    }
    let total: usize = entries.iter().map(|(_, p)| p.len()).sum();
    let json = assemble(&entries);
    std::fs::write(&args.output, serde_json::to_string_pretty(&json)?)?;
    println!("wrote {} ({} relations, {} pairs) to {}",
        args.output.display(), entries.len(), total, args.output.display());
    Ok(())
}
