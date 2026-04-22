//! # Conway's Game of Life expert
//!
//! Grids are 2-D rectangular arrays of `0` (dead) or `1` (live). Non-zero
//! input values are normalised to `1`. All rows must share the same width.
//! Boundary behaviour: cells outside the grid are dead.
//!
//! ## Ops
//!
//! - `step {grid: [[0|1]]} → [[0|1]]`
//! - `simulate {grid: [[0|1]], generations: int} → {grid, live: int, generations: int}`

use expert_interface::{arg_u64, expert_exports, json, Value};

expert_exports!(
    id = "conway",
    tier = 1,
    description = "Conway's Game of Life: single step, N-generation simulation",
    version = "0.2.0",
    ops = ["step", "simulate"],
    dispatch = dispatch
);

fn dispatch(op: &str, args: &Value) -> Option<Value> {
    let grid = parse_grid(args.get("grid")?)?;
    match op {
        "step" => Some(json!(step(&grid))),
        "simulate" => {
            let gens = arg_u64(args, "generations")? as u32;
            let mut g = grid;
            for _ in 0..gens { g = step(&g); }
            Some(json!({
                "grid": g,
                "live": count_live(&g),
                "generations": gens,
            }))
        }
        _ => None,
    }
}

type Grid = Vec<Vec<u8>>;

fn parse_grid(v: &Value) -> Option<Grid> {
    let arr = v.as_array()?;
    let mut grid = Vec::new();
    for row in arr {
        let r = row.as_array()?;
        let cells: Option<Vec<u8>> = r.iter().map(|c| c.as_u64().map(|n| if n != 0 { 1 } else { 0 })).collect();
        grid.push(cells?);
    }
    if grid.is_empty() { return None; }
    let cols = grid[0].len();
    if grid.iter().any(|r| r.len() != cols) { return None; }
    Some(grid)
}

fn count_live_neighbours(g: &Grid, r: usize, c: usize) -> u8 {
    let rows = g.len() as i32;
    let cols = g[0].len() as i32;
    let mut n = 0u8;
    for dr in -1..=1 {
        for dc in -1..=1 {
            if dr == 0 && dc == 0 { continue; }
            let nr = r as i32 + dr;
            let nc = c as i32 + dc;
            if nr >= 0 && nr < rows && nc >= 0 && nc < cols {
                n += g[nr as usize][nc as usize];
            }
        }
    }
    n
}

fn step(g: &Grid) -> Grid {
    let rows = g.len();
    let cols = g[0].len();
    let mut next = vec![vec![0u8; cols]; rows];
    for r in 0..rows {
        for c in 0..cols {
            let alive = g[r][c] == 1;
            let n = count_live_neighbours(g, r, c);
            next[r][c] = match (alive, n) {
                (true, 2) | (true, 3) => 1,
                (false, 3) => 1,
                _ => 0,
            };
        }
    }
    next
}

fn count_live(g: &Grid) -> usize {
    g.iter().flat_map(|r| r.iter()).filter(|&&c| c == 1).count()
}
