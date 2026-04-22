//! # Dijkstra / shortest-path expert
//!
//! Weighted undirected graph algorithms. Edges are `[from, to, weight?]`
//! arrays; if `weight` is omitted it defaults to `1`. Node names are arbitrary
//! strings.
//!
//! ## Ops
//!
//! - `shortest_path {edges, from, to} → {distance: int, path: [string]} | null`
//!   (null when `to` is unreachable from `from`)
//! - `reachable {edges, from, to} → {reachable: bool, path: [string] | null}`
//! - `mst {edges} → {weight: int, edges: [[from, to, weight]]}` (Kruskal's)

use expert_interface::{arg_str, expert_exports, json, Value};

expert_exports!(
    id = "dijkstra",
    tier = 1,
    description = "Graph algorithms: shortest path, reachability, minimum spanning tree",
    version = "0.2.0",
    ops = ["shortest_path", "reachable", "mst"],
    dispatch = dispatch
);

fn dispatch(op: &str, args: &Value) -> Option<Value> {
    let g = parse_graph(args.get("edges")?)?;
    match op {
        "shortest_path" => {
            let from = g.idx(arg_str(args, "from")?)?;
            let to = g.idx(arg_str(args, "to")?)?;
            let (dist, prev) = dijkstra(&g, from);
            if dist[to] == INF { return Some(Value::Null); }
            let path: Vec<String> = reconstruct(&prev, to).iter().map(|&i| g.nodes[i].clone()).collect();
            Some(json!({"distance": dist[to], "path": path}))
        }
        "reachable" => {
            let from = g.idx(arg_str(args, "from")?)?;
            let to = g.idx(arg_str(args, "to")?)?;
            match bfs(&g, from, to) {
                Some(p) => {
                    let path: Vec<String> = p.iter().map(|&i| g.nodes[i].clone()).collect();
                    Some(json!({"reachable": true, "path": path}))
                }
                None => Some(json!({"reachable": false, "path": Value::Null})),
            }
        }
        "mst" => {
            let (weight, edges) = kruskal(&g);
            let mst: Vec<Value> = edges
                .iter()
                .map(|e| json!([g.nodes[e.from], g.nodes[e.to], e.weight]))
                .collect();
            Some(json!({"weight": weight, "edges": mst}))
        }
        _ => None,
    }
}

#[derive(Clone)]
struct Edge { from: usize, to: usize, weight: u64 }

struct Graph { nodes: Vec<String>, edges: Vec<Edge> }

impl Graph {
    fn new() -> Self { Self { nodes: Vec::new(), edges: Vec::new() } }
    fn node(&mut self, name: &str) -> usize {
        if let Some(p) = self.nodes.iter().position(|n| n == name) { return p; }
        self.nodes.push(name.to_string());
        self.nodes.len() - 1
    }
    fn idx(&self, name: &str) -> Option<usize> { self.nodes.iter().position(|n| n == name) }
}

fn parse_graph(v: &Value) -> Option<Graph> {
    let arr = v.as_array()?;
    let mut g = Graph::new();
    for e in arr {
        let row = e.as_array()?;
        if row.len() < 2 { return None; }
        let a = row[0].as_str()?;
        let b = row[1].as_str()?;
        let w = row.get(2).and_then(|v| v.as_u64()).unwrap_or(1);
        let ai = g.node(a);
        let bi = g.node(b);
        g.edges.push(Edge { from: ai, to: bi, weight: w });
        g.edges.push(Edge { from: bi, to: ai, weight: w });
    }
    Some(g)
}

const INF: u64 = u64::MAX / 2;

fn dijkstra(g: &Graph, start: usize) -> (Vec<u64>, Vec<Option<usize>>) {
    let n = g.nodes.len();
    let mut dist = vec![INF; n];
    let mut prev = vec![None; n];
    let mut visited = vec![false; n];
    dist[start] = 0;
    for _ in 0..n {
        let u = (0..n).filter(|&i| !visited[i]).min_by_key(|&i| dist[i]);
        let u = match u { Some(x) => x, None => break };
        if dist[u] == INF { break; }
        visited[u] = true;
        for e in &g.edges {
            if e.from == u && !visited[e.to] {
                let alt = dist[u].saturating_add(e.weight);
                if alt < dist[e.to] { dist[e.to] = alt; prev[e.to] = Some(u); }
            }
        }
    }
    (dist, prev)
}

fn reconstruct(prev: &[Option<usize>], target: usize) -> Vec<usize> {
    let mut path = Vec::new();
    let mut cur = target;
    loop {
        path.push(cur);
        match prev[cur] { Some(p) => cur = p, None => break }
    }
    path.reverse();
    path
}

fn bfs(g: &Graph, start: usize, end: usize) -> Option<Vec<usize>> {
    let n = g.nodes.len();
    let mut visited = vec![false; n];
    let mut prev = vec![None::<usize>; n];
    let mut queue = std::collections::VecDeque::new();
    queue.push_back(start);
    visited[start] = true;
    while let Some(u) = queue.pop_front() {
        if u == end { return Some(reconstruct(&prev, end)); }
        for e in &g.edges {
            if e.from == u && !visited[e.to] {
                visited[e.to] = true;
                prev[e.to] = Some(u);
                queue.push_back(e.to);
            }
        }
    }
    None
}

struct UnionFind { parent: Vec<usize>, rank: Vec<u8> }

impl UnionFind {
    fn new(n: usize) -> Self { Self { parent: (0..n).collect(), rank: vec![0; n] } }
    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x { self.parent[x] = self.find(self.parent[x]); }
        self.parent[x]
    }
    fn union(&mut self, x: usize, y: usize) -> bool {
        let rx = self.find(x); let ry = self.find(y);
        if rx == ry { return false; }
        match self.rank[rx].cmp(&self.rank[ry]) {
            std::cmp::Ordering::Less => self.parent[rx] = ry,
            std::cmp::Ordering::Greater => self.parent[ry] = rx,
            std::cmp::Ordering::Equal => { self.parent[ry] = rx; self.rank[rx] += 1; }
        }
        true
    }
}

fn kruskal(g: &Graph) -> (u64, Vec<Edge>) {
    let n = g.nodes.len();
    let mut unique: Vec<Edge> = g.edges.iter().filter(|e| e.from < e.to).cloned().collect();
    unique.sort_unstable_by_key(|e| e.weight);
    let mut uf = UnionFind::new(n);
    let mut weight = 0u64;
    let mut out = Vec::new();
    for e in unique {
        if uf.union(e.from, e.to) {
            weight += e.weight;
            out.push(e);
        }
    }
    (weight, out)
}
