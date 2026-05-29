use std::collections::{HashSet, VecDeque};

use crate::core::edge::Edge;
use crate::core::enums::SourceType;
use crate::core::graph::Graph;

use super::chain::chain_tokens;
use super::provider::ModelProvider;
use super::templates::TemplateRegistry;

pub const DEFAULT_MAX_DEPTH: u32 = 3;
pub const DEFAULT_MAX_ENTITIES: usize = 1000;
pub const DEFAULT_MIN_CONFIDENCE: f64 = 0.3;
pub const DEFAULT_MAX_CHAIN_TOKENS: usize = 5;
pub const DEFAULT_EDGE_SOURCE: SourceType = SourceType::Parametric;
pub const DEFAULT_MAX_ENTITY_WORDS: usize = 4;
pub const DEFAULT_SKIPPED_ENTITY_PREFIXES: &[&str] = &["the ", "a "];
pub const METADATA_FORWARD_PASSES: &str = "forward_passes";
pub const METADATA_MODEL: &str = "model";

/// BFS extraction configuration.
pub struct BfsConfig {
    pub max_depth: u32,
    pub max_entities: usize,
    pub min_confidence: f64,
    pub max_chain_tokens: usize,
    pub edge_source: SourceType,
    pub should_follow_entity: fn(&str) -> bool,
}

impl Default for BfsConfig {
    fn default() -> Self {
        Self {
            max_depth: DEFAULT_MAX_DEPTH,
            max_entities: DEFAULT_MAX_ENTITIES,
            min_confidence: DEFAULT_MIN_CONFIDENCE,
            max_chain_tokens: DEFAULT_MAX_CHAIN_TOKENS,
            edge_source: DEFAULT_EDGE_SOURCE,
            should_follow_entity: default_should_follow_entity,
        }
    }
}

/// Callbacks for progress reporting.
pub trait BfsCallbacks {
    fn on_entity(&mut self, _entity: &str, _depth: u32, _visited: usize, _queue: usize) {}
    fn on_edge(&mut self, _edge: &Edge, _depth: u32) {}
    fn on_checkpoint(&mut self, _graph: &Graph) {}
}

/// No-op callbacks for headless extraction.
pub struct SilentCallbacks;
impl BfsCallbacks for SilentCallbacks {}

/// Run BFS extraction from seed entities.
///
/// For each entity: probe all templates, cache results as edges,
/// queue valid entities discovered in answers. Checkpoint after
/// each entity.
pub fn extract_bfs(
    provider: &dyn ModelProvider,
    templates: &TemplateRegistry,
    seeds: &[String],
    config: &BfsConfig,
    graph: &mut Graph,
    callbacks: &mut dyn BfsCallbacks,
) -> BfsResult {
    let mut visited: HashSet<String> = HashSet::new();
    let mut queue: VecDeque<(String, u32)> = seeds.iter().map(|s| (s.clone(), 0u32)).collect();

    let mut total_passes: usize = 0;
    let mut edges_added: usize = 0;

    while let Some((entity, depth)) = queue.pop_front() {
        if visited.contains(&entity) || depth > config.max_depth {
            continue;
        }
        if visited.len() >= config.max_entities {
            break;
        }
        visited.insert(entity.clone());

        callbacks.on_entity(&entity, depth, visited.len(), queue.len());

        for template in templates.all() {
            let prompt = template.format(&entity);
            let max_tok = if template.multi_token {
                config.max_chain_tokens
            } else {
                1
            };

            let stop_tokens = if template.stop_tokens.is_empty() {
                None
            } else {
                Some(template.stop_tokens.as_slice())
            };

            let result = match chain_tokens(
                provider,
                &prompt,
                max_tok,
                config.min_confidence,
                stop_tokens,
            ) {
                Ok(r) => r,
                Err(_) => continue,
            };

            total_passes += result.num_passes;

            if !result.answer.is_empty() && result.avg_probability() >= config.min_confidence {
                let edge = Edge::new(&entity, &template.relation, &result.answer)
                    .with_confidence(result.avg_probability())
                    .with_source(config.edge_source.clone())
                    .with_metadata(
                        METADATA_FORWARD_PASSES,
                        serde_json::Value::from(result.num_passes as u64),
                    )
                    .with_metadata(
                        METADATA_MODEL,
                        serde_json::Value::from(provider.model_name()),
                    );

                if !graph.exists(&entity, &template.relation, &result.answer) {
                    graph.add_edge(edge.clone());
                    edges_added += 1;
                    callbacks.on_edge(&edge, depth);
                }

                // Queue valid entities for further exploration
                let obj = result.answer.trim().to_string();
                if !visited.contains(&obj)
                    && depth < config.max_depth
                    && (config.should_follow_entity)(&obj)
                {
                    queue.push_back((obj, depth + 1));
                }
            }
        }

        callbacks.on_checkpoint(graph);
    }

    BfsResult {
        entities_visited: visited.len(),
        edges_added,
        total_forward_passes: total_passes,
        queue_remaining: queue.len(),
    }
}

#[derive(Debug, Clone)]
pub struct BfsResult {
    pub entities_visited: usize,
    pub edges_added: usize,
    pub total_forward_passes: usize,
    pub queue_remaining: usize,
}

/// Default follow policy for BFS extraction.
///
/// This is configurable via `BfsConfig::should_follow_entity` because entity
/// syntax is language/domain policy, not graph-engine logic.
pub fn default_should_follow_entity(text: &str) -> bool {
    let clean = text.trim();
    if clean.is_empty() {
        return false;
    }

    let first = clean.chars().next().unwrap();
    if !first.is_uppercase() && !first.is_ascii_digit() {
        return false;
    }

    // Skip articles
    let lower = clean.to_lowercase();
    if DEFAULT_SKIPPED_ENTITY_PREFIXES
        .iter()
        .any(|prefix| lower.starts_with(prefix))
    {
        return false;
    }

    // Skip long phrases
    if clean.split_whitespace().count() > DEFAULT_MAX_ENTITY_WORDS {
        return false;
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_entity() {
        assert!(default_should_follow_entity("France"));
        assert!(default_should_follow_entity("Paris"));
        assert!(default_should_follow_entity("1756"));
        assert!(default_should_follow_entity("New York"));
        assert!(!default_should_follow_entity("the city"));
        assert!(!default_should_follow_entity(
            "a very long phrase that is not an entity"
        ));
        assert!(!default_should_follow_entity("lowercase"));
        assert!(!default_should_follow_entity(""));
    }
}
