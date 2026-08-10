//! BFS and DFS traversal with depth tracking and visit order.

#[cfg(target_arch = "wasm32")]
use crate::prelude::*;
#[cfg(target_arch = "wasm32")]
use alloc::collections::VecDeque;
#[cfg(not(target_arch = "wasm32"))]
use std::collections::VecDeque;
use crate::collections::{HashMap, HashSet};

use crate::core::edge::Edge;
use crate::core::graph::Graph;

/// Result of a graph traversal.
#[derive(Debug)]
pub struct TraversalResult {
    /// Entities visited in order.
    pub nodes: Vec<String>,
    /// Edges traversed in order.
    pub edges: Vec<Edge>,
    /// Depth of each visited node from the source.
    pub depths: HashMap<String, usize>,
    /// Maximum depth reached.
    pub max_depth: usize,
}

/// Breadth-first search from a source entity.
pub fn bfs(graph: &Graph, source: &str, max_depth: usize) -> TraversalResult {
    let mut visited: HashSet<String> = HashSet::default();
    let mut discovered: HashSet<String> = HashSet::default();
    let mut queue: VecDeque<(String, usize)> = VecDeque::new();
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    let mut depths = HashMap::default();
    let mut max_depth_reached = 0;

    queue.push_back((source.to_string(), 0));
    discovered.insert(source.to_string());

    while let Some((node, depth)) = queue.pop_front() {
        if visited.contains(&node) || depth > max_depth {
            continue;
        }
        visited.insert(node.clone());
        nodes.push(node.clone());
        depths.insert(node.clone(), depth);
        if depth > max_depth_reached {
            max_depth_reached = depth;
        }

        for edge in graph.select(&node, None) {
            if depth < max_depth && discovered.insert(edge.object.clone()) {
                edges.push(edge.clone());
                queue.push_back((edge.object.clone(), depth + 1));
            }
        }
    }

    TraversalResult {
        nodes,
        edges,
        depths,
        max_depth: max_depth_reached,
    }
}

/// Depth-first search from a source entity.
pub fn dfs(graph: &Graph, source: &str, max_depth: usize) -> TraversalResult {
    let mut visited: HashSet<String> = HashSet::default();
    let mut discovered: HashSet<String> = HashSet::default();
    let mut stack: Vec<(String, usize)> = vec![(source.to_string(), 0)];
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    let mut depths = HashMap::default();
    let mut max_depth_reached = 0;
    discovered.insert(source.to_string());

    while let Some((node, depth)) = stack.pop() {
        if visited.contains(&node) || depth > max_depth {
            continue;
        }
        visited.insert(node.clone());
        nodes.push(node.clone());
        depths.insert(node.clone(), depth);
        if depth > max_depth_reached {
            max_depth_reached = depth;
        }

        for edge in graph.select(&node, None) {
            if depth < max_depth && discovered.insert(edge.object.clone()) {
                edges.push(edge.clone());
                stack.push((edge.object.clone(), depth + 1));
            }
        }
    }

    TraversalResult {
        nodes,
        edges,
        depths,
        max_depth: max_depth_reached,
    }
}
