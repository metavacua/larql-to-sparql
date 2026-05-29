use std::path::PathBuf;

use clap::Args;

#[derive(Args)]
pub struct DescribeArgs {
    /// Path to graph file (.larql.json or .larql.bin).
    #[arg(short, long)]
    graph: PathBuf,

    /// Entity to describe.
    entity: String,
}

pub fn run(args: DescribeArgs) -> Result<(), Box<dyn std::error::Error>> {
    let graph = larql_core::load(&args.graph)?;
    let result = graph.describe(&args.entity);

    println!("{}", args.entity);

    if !result.outgoing.is_empty() {
        println!("  Outgoing:");
        for edge in &result.outgoing {
            println!(
                "    --{}--> {}  ({:.2})",
                edge.relation, edge.object, edge.confidence
            );
        }
    }

    if !result.incoming.is_empty() {
        println!("  Incoming:");
        for edge in &result.incoming {
            println!(
                "    {} --{}-->  ({:.2})",
                edge.subject, edge.relation, edge.confidence
            );
        }
    }

    if result.outgoing.is_empty() && result.incoming.is_empty() {
        println!("  (not found in graph)");
    }

    Ok(())
}
