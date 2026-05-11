use std::path::PathBuf;

use clap::Args;
use larql_core::{filter_graph, FilterConfig, MetadataPredicate, SourceType};

const METADATA_LAYER: &str = "layer";
const METADATA_SELECTIVITY: &str = "selectivity";
const METADATA_C_IN: &str = "c_in";
const METADATA_C_OUT: &str = "c_out";

#[derive(Args)]
pub struct FilterArgs {
    /// Input graph file (.larql.json or .larql.bin).
    graph: PathBuf,

    /// Output graph file.
    #[arg(short, long)]
    output: PathBuf,

    /// Minimum confidence (0.0-1.0).
    #[arg(long)]
    min_confidence: Option<f64>,

    /// Maximum confidence (0.0-1.0).
    #[arg(long)]
    max_confidence: Option<f64>,

    /// Minimum layer (inclusive, from edge metadata).
    #[arg(long)]
    min_layer: Option<usize>,

    /// Maximum layer (inclusive, from edge metadata).
    #[arg(long)]
    max_layer: Option<usize>,

    /// Minimum selectivity (from edge metadata).
    #[arg(long)]
    min_selectivity: Option<f64>,

    /// Minimum c_in magnitude (from edge metadata).
    #[arg(long)]
    min_c_in: Option<f64>,

    /// Minimum c_out magnitude (from edge metadata).
    #[arg(long)]
    min_c_out: Option<f64>,

    /// Include only these relations (repeatable).
    #[arg(long = "relation", num_args = 1)]
    relations: Vec<String>,

    /// Exclude these relations (repeatable).
    #[arg(long = "exclude-relation", num_args = 1)]
    exclude_relations: Vec<String>,

    /// Include only these source types (repeatable: parametric, document, etc.).
    #[arg(long = "source", num_args = 1)]
    sources: Vec<String>,

    /// Subject must contain this substring.
    #[arg(long)]
    subject_contains: Option<String>,

    /// Object must contain this substring.
    #[arg(long)]
    object_contains: Option<String>,
}

fn parse_source(s: &str) -> Option<SourceType> {
    serde_json::from_value(serde_json::Value::String(s.to_string())).ok()
}

pub fn run(args: FilterArgs) -> Result<(), Box<dyn std::error::Error>> {
    let graph = larql_core::load(&args.graph)?;

    let mut metadata = Vec::new();
    if let Some(min_layer) = args.min_layer {
        metadata.push(MetadataPredicate::u64_min(
            METADATA_LAYER,
            u64::try_from(min_layer)?,
        ));
    }
    if let Some(max_layer) = args.max_layer {
        metadata.push(MetadataPredicate::u64_max(
            METADATA_LAYER,
            u64::try_from(max_layer)?,
        ));
    }
    if let Some(min_selectivity) = args.min_selectivity {
        metadata.push(MetadataPredicate::f64_min(
            METADATA_SELECTIVITY,
            min_selectivity,
        ));
    }
    if let Some(min_c_in) = args.min_c_in {
        metadata.push(MetadataPredicate::f64_min(METADATA_C_IN, min_c_in));
    }
    if let Some(min_c_out) = args.min_c_out {
        metadata.push(MetadataPredicate::f64_min(METADATA_C_OUT, min_c_out));
    }

    let config = FilterConfig {
        min_confidence: args.min_confidence,
        max_confidence: args.max_confidence,
        metadata,
        relations: if args.relations.is_empty() {
            None
        } else {
            Some(args.relations)
        },
        exclude_relations: if args.exclude_relations.is_empty() {
            None
        } else {
            Some(args.exclude_relations)
        },
        sources: if args.sources.is_empty() {
            None
        } else {
            let parsed: Vec<SourceType> = args
                .sources
                .iter()
                .filter_map(|s| parse_source(s))
                .collect();
            if parsed.is_empty() {
                None
            } else {
                Some(parsed)
            }
        },
        subject_contains: args.subject_contains,
        object_contains: args.object_contains,
    };

    let filtered = filter_graph(&graph, &config);
    let removed = graph.edge_count() - filtered.edge_count();

    larql_core::save(&filtered, &args.output)?;
    eprintln!(
        "Filtered: {} → {} edges ({} removed)",
        graph.edge_count(),
        filtered.edge_count(),
        removed
    );

    Ok(())
}
