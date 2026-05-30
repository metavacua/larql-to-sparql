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
#[cfg(target_arch = "wasm32")]
#[allow(unused_imports)]
use larql_wasm_math::FloatExt as _;
#[derive(Args)]
pub struct QueryArgs {
    /// Path to graph file (.larql.json or .larql.bin).
    #[arg(short, long)]
    graph: PathBuf,

    /// Entity to query.
    subject: String,

    /// Relation to filter (optional).
    relation: Option<String>,
}

pub fn run(args: QueryArgs) -> Result<(), Box<dyn std::error::Error>> {
    let graph = larql_core::load(&args.graph)?;
    let edges = graph.select(&args.subject, args.relation.as_deref());

    if edges.is_empty() {
        println!("No results for '{}'", args.subject);
        return Ok(());
    }

    for edge in edges {
        println!(
            "  {} --{}--> {}  ({:.2})",
            edge.subject, edge.relation, edge.object, edge.confidence
        );
    }

    Ok(())
}
