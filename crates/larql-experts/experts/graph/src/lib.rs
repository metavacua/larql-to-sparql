//! # Graph-algorithms expert
//!
//! Unweighted graph algorithms. Edges are `[from, to]` arrays. Set
//! `directed: true` to enable directed-graph variants (cycle detection,
//! topological sort).
//!
//! ## Ops
//!
//! - `most_central {edges} → {node: string, degree: int}` (highest-degree node)
//! - `has_cycle {edges, directed?: bool} → bool`
//! - `connected_components {edges} → int`
//! - `topological_sort {edges, directed?: bool} → [string] | null`
//!   (null when the graph contains a cycle)
//! - `is_bipartite {edges} → bool`
//! - `degrees {edges} → [{node, degree}]`

use expert_interface::{arg_bool, expert_exports, json, Value};

expert_exports!(
    id = "graph",
    tier = 1,
    description = "Graph algorithms: centrality, cycle detection, components, topological sort, bipartite",
    version = "0.2.0",
    ops = [
        "most_central",
        "has_cycle",
        "connected_components",
        "topological_sort",
        "is_bipartite",
        "degrees",
    ],
    dispatch = dispatch
);

fn dispatch(op: &str, args: &Value) -> Option<Value> {
    let directed = arg_bool(args, "directed").unwrap_or(false);
    let g = parse_graph(args.get("edges")?, directed)?;
    match op {
        "most_central" => {
            if g.nodes.is_empty() { return None; }
            let i = (0..g.nodes.len()).max_by_key(|&i| g.adj[i].len())?;
            Some(json!({"node": g.nodes[i], "degree": g.adj[i].len()}))
        }
        "has_cycle" => Some(json!(if directed {
            has_cycle_directed(&g)
        } else {
            has_cycle_undirected(&g)
        })),
        "connected_components" => Some(json!(count_components(&g))),
        "topological_sort" => match topo_sort(&g) {
            Some(order) => Some(json!(order)),
            None => Some(Value::Null),
        },
        "is_bipartite" => Some(json!(is_bipartite(&g))),
        "degrees" => {
            let degrees: Vec<Value> = g
                .nodes
                .iter()
                .enumerate()
                .map(|(i, n)| json!({"node": n, "degree": g.adj[i].len()}))
                .collect();
            Some(json!(degrees))
        }
        _ => None,
    }
}

struct Graph {
    nodes: Vec<String>,
    adj: Vec<Vec<usize>>,
    dadj: Vec<Vec<usize>>,
}

impl Graph {
    fn new() -> Self { Self { nodes: Vec::new(), adj: Vec::new(), dadj: Vec::new() } }
    fn node(&mut self, name: &str) -> usize {
        if let Some(p) = self.nodes.iter().position(|n| n == name) { return p; }
        self.nodes.push(name.to_string());
        self.adj.push(Vec::new());
        self.dadj.push(Vec::new());
        self.nodes.len() - 1
    }
}

fn parse_graph(v: &Value, directed: bool) -> Option<Graph> {
    let arr = v.as_array()?;
    let mut g = Graph::new();
    for e in arr {
        let row = e.as_array()?;
        if row.len() < 2 { return None; }
        let a = row[0].as_str()?;
        let b = row[1].as_str()?;
        let ai = g.node(a);
        let bi = g.node(b);
        if !g.adj[ai].contains(&bi) { g.adj[ai].push(bi); }
        if !g.adj[bi].contains(&ai) { g.adj[bi].push(ai); }
        if directed && !g.dadj[ai].contains(&bi) { g.dadj[ai].push(bi); }
    }
    Some(g)
}

fn has_cycle_undirected(g: &Graph) -> bool {
    let n = g.nodes.len();
    let mut visited = vec![false; n];
    fn dfs(g: &Graph, u: usize, parent: Option<usize>, visited: &mut Vec<bool>) -> bool {
        visited[u] = true;
        for &v in &g.adj[u] {
            if !visited[v] {
                if dfs(g, v, Some(u), visited) { return true; }
            } else if Some(v) != parent {
                return true;
            }
        }
        false
    }
    for start in 0..n {
        if !visited[start] && dfs(g, start, None, &mut visited) { return true; }
    }
    false
}

fn has_cycle_directed(g: &Graph) -> bool {
    let n = g.nodes.len();
    let mut state = vec![0u8; n];
    fn dfs(g: &Graph, u: usize, state: &mut Vec<u8>) -> bool {
        state[u] = 1;
        for &v in &g.dadj[u] {
            if state[v] == 1 { return true; }
            if state[v] == 0 && dfs(g, v, state) { return true; }
        }
        state[u] = 2;
        false
    }
    for start in 0..n {
        if state[start] == 0 && dfs(g, start, &mut state) { return true; }
    }
    false
}

fn count_components(g: &Graph) -> usize {
    let n = g.nodes.len();
    if n == 0 { return 0; }
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(p: &mut [usize], x: usize) -> usize {
        if p[x] != x { p[x] = find(p, p[x]); }
        p[x]
    }
    for u in 0..n {
        for &v in &g.adj[u] {
            let ru = find(&mut parent, u);
            let rv = find(&mut parent, v);
            if ru != rv { parent[ru] = rv; }
        }
    }
    let mut roots: Vec<usize> = (0..n).map(|i| find(&mut parent, i)).collect();
    roots.sort_unstable();
    roots.dedup();
    roots.len()
}

fn topo_sort(g: &Graph) -> Option<Vec<String>> {
    let n = g.nodes.len();
    let mut indeg = vec![0usize; n];
    for u in 0..n { for &v in &g.dadj[u] { indeg[v] += 1; } }
    let mut queue: std::collections::VecDeque<usize> =
        (0..n).filter(|&i| indeg[i] == 0).collect();
    let mut order = Vec::new();
    while let Some(u) = queue.pop_front() {
        order.push(g.nodes[u].clone());
        for &v in &g.dadj[u] {
            indeg[v] -= 1;
            if indeg[v] == 0 { queue.push_back(v); }
        }
    }
    if order.len() == n { Some(order) } else { None }
}

fn is_bipartite(g: &Graph) -> bool {
    let n = g.nodes.len();
    let mut color = vec![255u8; n];
    for start in 0..n {
        if color[start] != 255 { continue; }
        color[start] = 0;
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(start);
        while let Some(u) = queue.pop_front() {
            for &v in &g.adj[u] {
                if color[v] == 255 {
                    color[v] = 1 - color[u];
                    queue.push_back(v);
                } else if color[v] == color[u] {
                    return false;
                }
            }
        }
    }
    true
}
