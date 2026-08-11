//! Connected components analysis.
//!
//! Basic component counting is on `Graph::stats()`. This module provides
//! richer analysis: enumerate components, find largest, check connectivity.

#[cfg(target_arch = "wasm32")]
use crate::prelude::*;
#[cfg(target_arch = "wasm32")]
use alloc::collections::VecDeque;
#[cfg(not(target_arch = "wasm32"))]
use std::collections::VecDeque;
use crate::collections::HashSet;
use crate::core::graph::Graph;

/// Enumerate all connected components as sets of node names.
/// Treats the graph as undirected (follows both adjacency and reverse edges).
pub fn connected_components(graph: &Graph) -> Vec<Vec<String>> {
    // Explicit `HashSet<String>` (not bare `::default()`) because
    // `crate::collections::HashSet` is a fixed-hasher single-param type
    // alias on wasm32 (no ambiguity) but re-exports std's real 2-param
    // generic type on native, where the hasher parameter needs a type
    // position to resolve its default (E0283 otherwise).
    let mut visited: HashSet<String> = HashSet::default();
    let mut components = Vec::new();

    let all_nodes = graph.list_entities();

    for node in all_nodes {
        if visited.contains(&node) {
            continue;
        }

        // BFS from this node
        let mut component = Vec::new();
        let mut queue = VecDeque::new();
        queue.push_back(node.clone());
        visited.insert(node.clone());

        while let Some(current) = queue.pop_front() {
            component.push(current.clone());

            // Forward edges
            for edge in graph.select(&current, None) {
                if !visited.contains(&edge.object) {
                    visited.insert(edge.object.clone());
                    queue.push_back(edge.object.clone());
                }
            }
            // Reverse edges
            for edge in graph.select_reverse(&current, None) {
                if !visited.contains(&edge.subject) {
                    visited.insert(edge.subject.clone());
                    queue.push_back(edge.subject.clone());
                }
            }
        }

        component.sort();
        components.push(component);
    }

    components.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
    components
}

/// Check if two nodes are in the same connected component.
pub fn are_connected(graph: &Graph, a: &str, b: &str) -> bool {
    let mut visited: HashSet<String> = HashSet::default();
    let mut queue = VecDeque::new();
    queue.push_back(a.to_string());
    visited.insert(a.to_string());

    while let Some(current) = queue.pop_front() {
        if current == b {
            return true;
        }
        for edge in graph.select(&current, None) {
            if !visited.contains(&edge.object) {
                visited.insert(edge.object.clone());
                queue.push_back(edge.object.clone());
            }
        }
        for edge in graph.select_reverse(&current, None) {
            if !visited.contains(&edge.subject) {
                visited.insert(edge.subject.clone());
                queue.push_back(edge.subject.clone());
            }
        }
    }
    false
}
