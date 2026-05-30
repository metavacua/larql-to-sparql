use std::path::PathBuf;

use clap::Args;
#[allow(unused_imports)]
use alloc::{boxed::Box, string::{String, ToString}, vec::Vec, vec, format, borrow::{Cow, ToOwned}, rc::Rc, sync::Arc, collections::{BTreeMap, BTreeSet, VecDeque, BinaryHeap}};
#[cfg(target_arch = "wasm32")]
#[allow(unused_imports)]
use hashbrown::{HashMap, HashSet};
#[cfg(not(target_arch = "wasm32"))]
#[allow(unused_imports)]
use std::collections::{HashMap, HashSet};
#[derive(Args)]
pub struct StatsArgs {
    /// Path to graph file (.larql.json or .larql.bin).
    graph: PathBuf,
}

pub fn run(args: StatsArgs) -> Result<(), Box<dyn core::error::Error>> {
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
