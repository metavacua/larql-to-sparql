//! # SQL expert
//!
//! In-memory evaluator for a small subset of SQL. A single `execute` call
//! accepts a script containing `CREATE TABLE` / `INSERT INTO` / `SELECT`
//! statements separated by `;`. State lives only for the duration of the
//! call — there is no cross-call table persistence.
//!
//! Supported in `SELECT`:
//!   - `COUNT(*)`, `COUNT(col)`, `SUM(col)`, `AVG(col)`, `MIN(col)`, `MAX(col)`
//!   - `*` or a comma-separated column list
//!   - `WHERE col OP value` with `=`, `!=`, `<>`, `<`, `<=`, `>`, `>=`
//!   - `ORDER BY col [ASC|DESC]`
//!   - `LIMIT n`
//!
//! ## Ops
//!
//! - `execute {sql: string} → Value`
//!
//! Return shape: aggregates and single-row/single-column selects return a
//! scalar; multi-column selects return an object per row; multi-row selects
//! return an array.

use expert_interface::{arg_str, expert_exports, json, Value};

expert_exports!(
    id = "sql",
    tier = 1,
    description = "In-memory SQL evaluator: CREATE TABLE, INSERT, SELECT with aggregates and WHERE",
    version = "0.2.0",
    ops = [
        ("execute", ["sql"]),
    ],
    dispatch = dispatch
);

fn dispatch(op: &str, args: &Value) -> Option<Value> {
    match op {
        "execute" => execute(arg_str(args, "sql")?),
        _ => None,
    }
}

fn execute(sql: &str) -> Option<Value> {
    let stmts = split_statements(sql);
    let mut tables: Vec<Table> = Vec::new();
    let mut last: Option<Value> = None;
    for stmt in &stmts {
        let upper = stmt.to_uppercase();
        let t = upper.trim_start();
        if t.starts_with("CREATE TABLE") { execute_create(stmt, &mut tables); }
        else if t.starts_with("INSERT INTO") { execute_insert(stmt, &mut tables); }
        else if t.starts_with("SELECT") { last = execute_select(stmt, &tables); }
    }
    last
}

#[derive(Clone)]
enum V { Num(f64), Str(String), Null }

impl V {
    fn as_f64(&self) -> Option<f64> {
        match self {
            V::Num(n) => Some(*n),
            V::Str(s) => s.parse::<f64>().ok(),
            V::Null => None,
        }
    }
    fn to_json(&self) -> Value {
        match self {
            V::Num(n) => {
                if (n - (*n as i64 as f64)).abs() < 1e-10 { json!(*n as i64) } else { json!(n) }
            }
            V::Str(s) => json!(s),
            V::Null => Value::Null,
        }
    }
}

struct Table {
    name: String,
    columns: Vec<String>,
    rows: Vec<Vec<V>>,
}

impl Table {
    fn new(name: &str, columns: Vec<String>) -> Self { Self { name: name.to_string(), columns, rows: Vec::new() } }
    fn col_idx(&self, name: &str) -> Option<usize> {
        let l = name.to_lowercase();
        self.columns.iter().position(|c| c.to_lowercase() == l)
    }
    fn insert_row(&mut self, values: Vec<V>) {
        if values.len() == self.columns.len() { self.rows.push(values); }
    }
}

fn split_statements(sql: &str) -> Vec<String> {
    sql.split(';').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect()
}

fn parse_value(s: &str) -> V {
    let s = s.trim();
    if s.starts_with('\'') && s.ends_with('\'') && s.len() >= 2 {
        return V::Str(s[1..s.len() - 1].to_string());
    }
    if let Ok(n) = s.parse::<f64>() { return V::Num(n); }
    if s.eq_ignore_ascii_case("null") { return V::Null; }
    V::Str(s.to_string())
}

fn split_csv(s: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut buf = String::new();
    let mut in_str = false;
    for c in s.chars() {
        if c == '\'' { in_str = !in_str; buf.push(c); }
        else if c == ',' && !in_str { parts.push(buf.trim().to_string()); buf.clear(); }
        else { buf.push(c); }
    }
    if !buf.trim().is_empty() { parts.push(buf.trim().to_string()); }
    parts
}

fn execute_create(stmt: &str, tables: &mut Vec<Table>) {
    let upper = stmt.to_uppercase();
    let after = match upper.find("TABLE ") { Some(p) => &stmt[p + 6..], None => return };
    let paren_start = match after.find('(') { Some(p) => p, None => return };
    let name = after[..paren_start].trim();
    let paren_end = match after.rfind(')') { Some(p) => p, None => return };
    let cols: Vec<String> = after[paren_start + 1..paren_end]
        .split(',')
        .map(|c| c.split_whitespace().next().unwrap_or("").to_string())
        .filter(|s| !s.is_empty())
        .collect();
    tables.retain(|t| t.name.to_lowercase() != name.to_lowercase());
    tables.push(Table::new(name, cols));
}

fn execute_insert(stmt: &str, tables: &mut [Table]) {
    let upper = stmt.to_uppercase();
    let after = match upper.find("INTO ") { Some(p) => &stmt[p + 5..], None => return };
    let tbl_end = after.find(|c: char| c.is_whitespace() || c == '(').unwrap_or(after.len());
    let tbl_name = after[..tbl_end].trim();
    let table = match tables.iter_mut().find(|t| t.name.to_lowercase() == tbl_name.to_lowercase()) {
        Some(t) => t, None => return,
    };
    let upper2 = after.to_uppercase();
    let values_pos = match upper2.find("VALUES") { Some(p) => p, None => return };
    let after_values = &after[values_pos + 6..];
    let start = match after_values.find('(') { Some(p) => p, None => return };
    let end = match after_values.rfind(')') { Some(p) => p, None => return };
    let values: Vec<V> = split_csv(&after_values[start + 1..end])
        .iter()
        .map(|s| parse_value(s))
        .collect();
    table.insert_row(values);
}

fn execute_select(stmt: &str, tables: &[Table]) -> Option<Value> {
    let upper = stmt.to_uppercase();
    let from_pos = upper.find(" FROM ")?;
    let after_from = &stmt[from_pos + 6..];
    let tbl_end = after_from.find(|c: char| c.is_whitespace() || c == ';').unwrap_or(after_from.len());
    let tbl_name = after_from[..tbl_end].trim();
    let table = tables.iter().find(|t| t.name.to_lowercase() == tbl_name.to_lowercase())?;
    let sel_clause = stmt[7..from_pos].trim();

    let where_filter = {
        let upper_after = after_from[tbl_end..].to_uppercase();
        if let Some(wp) = upper_after.find("WHERE ") {
            parse_where(&after_from[tbl_end + wp + 6..])
        } else { None }
    };

    let mut rows: Vec<&Vec<V>> = table
        .rows
        .iter()
        .filter(|row| match &where_filter {
            Some((col, op, val)) => match table.col_idx(col) {
                Some(ci) => apply_where(&row[ci], op, val),
                None => true,
            },
            None => true,
        })
        .collect();

    if let Some((col, asc)) = order_by(stmt) {
        if let Some(ci) = table.col_idx(&col) {
            rows.sort_by(|a, b| {
                let va = a[ci].as_f64();
                let vb = b[ci].as_f64();
                let ord = match (va, vb) {
                    (Some(x), Some(y)) => x.partial_cmp(&y).unwrap_or(std::cmp::Ordering::Equal),
                    _ => std::cmp::Ordering::Equal,
                };
                if asc { ord } else { ord.reverse() }
            });
        }
    }
    if let Some(lim) = limit(stmt) {
        rows.truncate(lim);
    }

    evaluate_select(sel_clause, table, &rows)
}

fn order_by(stmt: &str) -> Option<(String, bool)> {
    let upper = stmt.to_uppercase();
    let pos = upper.find("ORDER BY ")?;
    let after = &stmt[pos + 9..];
    let end = after.find([';', '\n']).unwrap_or(after.len());
    let s = after[..end].trim();
    let asc = !s.to_uppercase().contains("DESC");
    Some((s.split_whitespace().next().unwrap_or("").to_string(), asc))
}

fn limit(stmt: &str) -> Option<usize> {
    let upper = stmt.to_uppercase();
    let pos = upper.find("LIMIT ")?;
    stmt[pos + 6..].split_whitespace().next().and_then(|s| s.parse::<usize>().ok())
}

fn parse_where(s: &str) -> Option<(String, String, String)> {
    let upper = s.to_uppercase();
    let end = upper.find("ORDER BY").or_else(|| upper.find("LIMIT")).unwrap_or(s.len());
    let cond = s[..end].trim();
    for op in &[">=", "<=", "<>", "!=", "=", ">", "<"] {
        if let Some(pos) = cond.find(op) {
            let col = cond[..pos].trim().to_string();
            let val = cond[pos + op.len()..].trim().trim_matches('\'').to_string();
            return Some((col, op.to_string(), val));
        }
    }
    None
}

fn apply_where(cell: &V, op: &str, val: &str) -> bool {
    if let (Some(cn), Ok(vn)) = (cell.as_f64(), val.parse::<f64>()) {
        return match op {
            "=" => (cn - vn).abs() < 1e-12,
            "!=" | "<>" => (cn - vn).abs() >= 1e-12,
            ">" => cn > vn, ">=" => cn >= vn,
            "<" => cn < vn, "<=" => cn <= vn,
            _ => false,
        };
    }
    let cs = match cell { V::Str(s) => s.clone(), V::Num(n) => format!("{}", n), V::Null => String::new() };
    match op {
        "=" => cs.eq_ignore_ascii_case(val),
        "!=" | "<>" => !cs.eq_ignore_ascii_case(val),
        _ => false,
    }
}

fn evaluate_select(sel: &str, table: &Table, rows: &[&Vec<V>]) -> Option<Value> {
    let upper = sel.to_uppercase().trim().to_string();
    if upper == "COUNT(*)" || upper == "COUNT( * )" {
        return Some(json!(rows.len()));
    }
    for agg in &["COUNT", "SUM", "AVG", "MAX", "MIN"] {
        let pat = format!("{}(", agg);
        if let Some(pos) = upper.find(&pat) {
            let after = &sel[pos + agg.len() + 1..];
            let end = after.find(')').unwrap_or(after.len());
            let col_name = after[..end].trim().trim_matches('*');
            if col_name == "*" || *agg == "COUNT" {
                let ci = if col_name == "*" { None } else { table.col_idx(col_name) };
                let values: Vec<f64> = rows
                    .iter()
                    .filter_map(|r| if let Some(i) = ci { r[i].as_f64() } else { Some(1.0) })
                    .collect();
                return Some(agg_result(agg, &values));
            }
            if let Some(ci) = table.col_idx(col_name) {
                let values: Vec<f64> = rows.iter().filter_map(|r| r[ci].as_f64()).collect();
                return Some(agg_result(agg, &values));
            }
        }
    }
    if sel.trim() == "*" {
        let out: Vec<Value> = rows
            .iter()
            .map(|row| {
                let obj: expert_interface::serde_json::Map<String, Value> = table
                    .columns
                    .iter()
                    .zip(row.iter())
                    .map(|(c, v)| (c.clone(), v.to_json()))
                    .collect();
                Value::Object(obj)
            })
            .collect();
        return Some(json!(out));
    }
    let cols: Vec<&str> = sel.split(',').map(|s| s.trim()).collect();
    let idxs: Vec<usize> = cols.iter().filter_map(|c| table.col_idx(c)).collect();
    if idxs.is_empty() { return None; }
    if cols.len() == 1 {
        let ci = idxs[0];
        let vals: Vec<Value> = rows.iter().map(|r| r[ci].to_json()).collect();
        if vals.len() == 1 { return Some(vals.into_iter().next().unwrap()); }
        return Some(json!(vals));
    }
    let out: Vec<Value> = rows
        .iter()
        .map(|row| {
            let obj: expert_interface::serde_json::Map<String, Value> = cols
                .iter()
                .zip(idxs.iter())
                .map(|(c, &i)| (c.to_string(), row[i].to_json()))
                .collect();
            Value::Object(obj)
        })
        .collect();
    Some(json!(out))
}

fn agg_result(agg: &str, values: &[f64]) -> Value {
    if values.is_empty() {
        return if agg == "COUNT" { json!(0) } else { Value::Null };
    }
    let r = match agg {
        "COUNT" => values.len() as f64,
        "SUM" => values.iter().sum(),
        "AVG" => values.iter().sum::<f64>() / values.len() as f64,
        "MAX" => values.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        "MIN" => values.iter().cloned().fold(f64::INFINITY, f64::min),
        _ => return Value::Null,
    };
    if (r - (r as i64 as f64)).abs() < 1e-10 { json!(r as i64) } else { json!(r) }
}
