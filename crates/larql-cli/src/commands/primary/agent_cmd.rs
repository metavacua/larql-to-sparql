use std::io::BufRead as _;
use std::path::PathBuf;

use clap::Args;
use larql_core::core::graph::Graph;
use larql_core::io::load_graph;

#[derive(Args)]
pub struct AgentArgs {
    /// Root directory of the codebase to extract JIT.
    #[arg(long, value_name = "DIR", conflicts_with = "vindex")]
    pub codebase: Option<PathBuf>,

    /// Path to an existing .larql.json graph file.
    #[arg(long, value_name = "PATH", conflicts_with = "codebase")]
    pub vindex: Option<PathBuf>,

    /// One-shot prompt to evaluate.
    #[arg(long, value_name = "TEXT", conflicts_with = "interactive")]
    pub prompt: Option<String>,

    /// Enter interactive REPL mode (reads prompts from stdin line by line).
    #[arg(long, conflicts_with = "prompt")]
    pub interactive: bool,
}

/// Process a single prompt against the graph.
///
/// Supported patterns (case-sensitive prefix match):
/// - `DESCRIBE <entity>`            — all edges involving `<entity>`
/// - `WHAT CALLS <entity>`          — incoming edges where relation contains "call"
/// - `WHAT DOES <entity> IMPLEMENT` — outgoing edges where relation == "implements"
/// - `WHAT DEPENDS ON <entity>`     — outgoing edges where relation == "depends_on"
/// - `WALK <entity>`                — multi-hop walk with no relation filter
/// - anything else                  — describe the first whitespace-separated word
///
/// Returns an empty string when no matching edges are found.
pub fn respond_to_prompt(graph: &Graph, prompt: &str) -> String {
    let trimmed = prompt.trim();

    if let Some(entity) = trimmed.strip_prefix("DESCRIBE ") {
        let entity = entity.trim();
        let r = graph.describe(entity);
        return format_describe(&r);
    }

    if let Some(entity) = trimmed.strip_prefix("WHAT CALLS ") {
        let entity = entity.trim();
        let edges: Vec<_> = graph
            .select_reverse(entity, None)
            .into_iter()
            .filter(|e| e.relation.contains("call"))
            .collect();
        if edges.is_empty() {
            return String::new();
        }
        return edges
            .iter()
            .map(|e| format!("{} --[{}]--> {}", e.subject, e.relation, e.object))
            .collect::<Vec<_>>()
            .join("\n");
    }

    if let Some(rest) = trimmed.strip_prefix("WHAT DOES ") {
        if let Some(entity) = rest.strip_suffix(" IMPLEMENT") {
            let entity = entity.trim();
            let edges: Vec<_> = graph
                .select(entity, Some("implements"))
                .into_iter()
                .collect();
            if edges.is_empty() {
                return String::new();
            }
            return edges
                .iter()
                .map(|e| format!("{} --[{}]--> {}", e.subject, e.relation, e.object))
                .collect::<Vec<_>>()
                .join("\n");
        }
    }

    if let Some(entity) = trimmed.strip_prefix("WHAT DEPENDS ON ") {
        let entity = entity.trim();
        let edges: Vec<_> = graph
            .select(entity, Some("depends_on"))
            .into_iter()
            .collect();
        if edges.is_empty() {
            return String::new();
        }
        return edges
            .iter()
            .map(|e| format!("{} --[{}]--> {}", e.subject, e.relation, e.object))
            .collect::<Vec<_>>()
            .join("\n");
    }

    if let Some(entity) = trimmed.strip_prefix("WALK ") {
        let entity = entity.trim();
        // walk with no relation filter: walk(entity, &[]) returns immediately
        // since there are no hops to follow; format as the entity itself.
        let result = graph.walk(entity, &[]);
        return match result {
            Some((final_entity, path)) => {
                if path.is_empty() {
                    format!("WALK {entity} → {final_entity}")
                } else {
                    let steps: Vec<_> = path
                        .iter()
                        .map(|e| format!("{} --[{}]--> {}", e.subject, e.relation, e.object))
                        .collect();
                    steps.join("\n")
                }
            }
            None => String::new(),
        };
    }

    // Fallback: describe the first word.
    let first = trimmed.split_whitespace().next().unwrap_or("");
    if first.is_empty() {
        return String::new();
    }
    let r = graph.describe(first);
    format_describe(&r)
}

fn format_describe(r: &larql_core::core::graph::DescribeResult) -> String {
    if r.outgoing.is_empty() && r.incoming.is_empty() {
        return String::new();
    }
    let mut parts = Vec::new();
    for e in &r.outgoing {
        parts.push(format!("{} --[{}]--> {}", e.subject, e.relation, e.object));
    }
    for e in &r.incoming {
        parts.push(format!("{} --[{}]--> {}", e.subject, e.relation, e.object));
    }
    parts.join("\n")
}

pub fn run(args: AgentArgs) -> Result<(), Box<dyn std::error::Error>> {
    // Validate mutually-exclusive source/mode pairs.
    if args.codebase.is_none() && args.vindex.is_none() {
        return Err("one of --codebase or --vindex is required".into());
    }
    if !args.interactive && args.prompt.is_none() {
        return Err("one of --prompt or --interactive is required".into());
    }

    // Load or extract the graph.
    let graph: Graph = if let Some(ref root) = args.codebase {
        let root = root.canonicalize()?;
        eprintln!("Extracting codebase from: {}", root.display());
        let g = larql_codebase::extract_codebase(&root).map_err(|e| format!("{e}"))?;
        eprintln!("  {} nodes, {} edges", g.node_count(), g.edge_count());
        g
    } else {
        let path = args.vindex.as_ref().unwrap();
        eprintln!("Loading graph: {}", path.display());
        load_graph(path).map_err(|e| format!("{e}"))?
    };

    if args.interactive {
        // REPL mode: read prompts from stdin line by line until EOF.
        let stdin = std::io::stdin();
        for line_result in stdin.lock().lines() {
            let line = line_result?;
            let response = respond_to_prompt(&graph, &line);
            if response.is_empty() {
                println!("(no results)");
            } else {
                println!("{response}");
            }
        }
    } else {
        // One-shot mode.
        let prompt = args.prompt.as_deref().unwrap_or("");
        let response = respond_to_prompt(&graph, prompt);
        if response.is_empty() {
            println!("(no results)");
        } else {
            println!("{response}");
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use larql_core::core::edge::Edge;

    #[test]
    fn describe_pattern_returns_nonempty_string() {
        let mut g = Graph::new();
        g.add_edge(Edge::new("module_a", "calls", "module_b"));
        g.add_edge(Edge::new("module_a", "depends_on", "lib_c"));
        let result = respond_to_prompt(&g, "DESCRIBE module_a");
        assert!(!result.is_empty(), "DESCRIBE should return non-empty output");
        assert!(
            result.contains("module_a"),
            "output should mention the entity"
        );
        assert!(
            result.contains("calls"),
            "output should include relation label"
        );
    }

    #[test]
    fn interactive_flag_is_parsed_correctly() {
        // AgentArgs derives Args, not Parser; wrap it to test parsing.
        #[derive(Parser)]
        struct Wrapper {
            #[command(flatten)]
            inner: AgentArgs,
        }
        let parsed =
            Wrapper::try_parse_from(["test", "--vindex", "some.json", "--interactive"]).unwrap();
        assert!(parsed.inner.interactive, "--interactive should be set");
        assert!(
            parsed.inner.prompt.is_none(),
            "--prompt should not be set when --interactive is used"
        );
        assert_eq!(
            parsed.inner.vindex.as_deref(),
            Some(std::path::Path::new("some.json"))
        );
    }

    #[test]
    fn what_calls_returns_matching_edges() {
        let mut g = Graph::new();
        g.add_edge(Edge::new("foo", "calls", "bar"));
        g.add_edge(Edge::new("baz", "depends_on", "bar"));
        let result = respond_to_prompt(&g, "WHAT CALLS bar");
        assert!(!result.is_empty());
        assert!(result.contains("foo"));
        assert!(result.contains("calls"));
    }

    #[test]
    fn what_depends_on_returns_outgoing_depends_edges() {
        let mut g = Graph::new();
        g.add_edge(Edge::new("foo", "depends_on", "bar"));
        g.add_edge(Edge::new("foo", "calls", "baz"));
        let result = respond_to_prompt(&g, "WHAT DEPENDS ON foo");
        assert!(!result.is_empty());
        assert!(result.contains("depends_on"));
        // should not include the calls edge
        assert!(!result.contains("baz"));
    }

    #[test]
    fn unknown_entity_returns_empty_string() {
        let g = Graph::new();
        let result = respond_to_prompt(&g, "DESCRIBE nonexistent_entity");
        assert_eq!(result, "");
    }
}
