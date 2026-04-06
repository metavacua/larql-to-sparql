use std::path::PathBuf;

use clap::Args;

#[derive(Args)]
pub struct StatsArgs {
    /// Path to graph file (.larql.json or .larql.bin).
    graph: PathBuf,
}

pub fn run(args: StatsArgs) -> Result<(), Box<dyn std::error::Error>> {
    let graph = larql_core::load(&args.graph)?;
    let stats = graph.stats();

    println!("Graph: {}", args.graph.display());
    println!("  Entities:    {}", stats.entities);
    println!("  Edges:       {}", stats.edges);
    println!("  Relations:   {}", stats.relations);
    println!("  Components:  {}", stats.connected_components);
    println!("  Avg degree:  {:.2}", stats.avg_degree);
    println!("  Avg conf:    {:.4}", stats.avg_confidence);
    println!("  Sources:");
    for (source, count) in &stats.sources {
        println!("    {source:12} {count}");
    }

    Ok(())
}
