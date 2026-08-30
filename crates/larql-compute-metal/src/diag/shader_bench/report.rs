//! Result printing, baseline comparison, and JSON emission.
//!
//! Split out of `shader_bench.rs`; [`super`] composes the pieces.

#[allow(unused_imports)]
use super::*;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub(crate) fn print_results(results: &[BenchResult]) {
    println!(
        "{:<34} {:<14} {:>5} {:>5} {:>9} {:>9} {:>9} {:>9} {:>8} {:<16}",
        "Kernel",
        "Family",
        "rows",
        "thr",
        "iso_ms",
        "iso_sd",
        "bat_ms",
        "GB/s",
        "nonzero",
        "Sanity"
    );
    println!("{}", "-".repeat(130));
    for r in results.iter().filter(|r| r.status == "bench") {
        println!(
            "{:<34} {:<14} {:>5} {:>5} {:>9.4} {:>9.4} {:>9.4} {:>9.1} {:>8} {:<16}",
            r.name,
            r.family,
            r.rows_per_tg.unwrap_or_default(),
            r.threads_per_tg.unwrap_or_default(),
            r.isolated_ms.unwrap_or_default(),
            r.isolated_sd_ms.unwrap_or_default(),
            r.batched_ms.unwrap_or_default(),
            r.batched_gbs.unwrap_or_default(),
            r.output_nonzero.unwrap_or_default(),
            r.sanity,
        );
    }
    println!();
    println!("Use batched ms/GB/s for promotion decisions; isolated numbers include per-call command-buffer overhead.");
}

#[derive(Debug, Clone)]
pub(crate) struct BaselineResult {
    pub(crate) family: String,
    pub(crate) batched_ms: Option<f64>,
}

pub(crate) fn load_baseline(path: &PathBuf) -> Result<HashMap<String, BaselineResult>, String> {
    let src = std::fs::read_to_string(path).map_err(|e| format!("read compare json: {e}"))?;
    let mut out = HashMap::new();
    let mut rest = src.as_str();
    while let Some(start) = rest.find('{') {
        rest = &rest[start + 1..];
        let Some(end) = rest.find('}') else {
            break;
        };
        let obj = &rest[..end];
        rest = &rest[end + 1..];
        let Some(name) = json_field_string(obj, "name") else {
            continue;
        };
        let family = json_field_string(obj, "family").unwrap_or_default();
        let batched_ms = json_field_number(obj, "batched_ms");
        out.insert(name, BaselineResult { family, batched_ms });
    }
    if out.is_empty() {
        return Err(format!(
            "compare json `{}` did not contain shader bench results",
            path.display()
        ));
    }
    Ok(out)
}

pub(crate) fn print_compare(
    current: &[BenchResult],
    baseline: &HashMap<String, BaselineResult>,
    path: &Path,
    threshold_pct: f64,
) {
    println!();
    println!(
        "Comparison vs {} (batched_ms, threshold={threshold_pct:.1}%):",
        path.display()
    );
    println!(
        "{:<34} {:<14} {:>10} {:>10} {:>9} {:<10}",
        "Kernel", "Family", "base_ms", "cur_ms", "delta", "Verdict"
    );
    println!("{}", "-".repeat(94));

    let mut improved = 0usize;
    let mut flat = 0usize;
    let mut regressed = 0usize;
    let mut missing = 0usize;

    for r in current.iter().filter(|r| r.status == "bench") {
        let Some(cur_ms) = r.batched_ms else {
            continue;
        };
        let Some(base) = baseline.get(r.name) else {
            missing += 1;
            continue;
        };
        let Some(base_ms) = base.batched_ms else {
            missing += 1;
            continue;
        };
        if base_ms <= 0.0 {
            missing += 1;
            continue;
        }
        let delta = (cur_ms - base_ms) / base_ms * 100.0;
        let verdict = if delta > threshold_pct {
            regressed += 1;
            "regressed"
        } else if delta < -threshold_pct {
            improved += 1;
            "improved"
        } else {
            flat += 1;
            "flat"
        };
        let family = if base.family.is_empty() {
            r.family
        } else {
            base.family.as_str()
        };
        println!(
            "{:<34} {:<14} {:>10.4} {:>10.4} {:>8.1}% {:<10}",
            r.name, family, base_ms, cur_ms, delta, verdict
        );
    }

    println!("summary: improved={improved} flat={flat} regressed={regressed} missing={missing}");
}

pub(crate) fn json_field_string(obj: &str, key: &str) -> Option<String> {
    let pattern = format!("\"{key}\":\"");
    let start = obj.find(&pattern)? + pattern.len();
    let mut out = String::new();
    let mut escaped = false;
    for ch in obj[start..].chars() {
        if escaped {
            out.push(ch);
            escaped = false;
        } else if ch == '\\' {
            escaped = true;
        } else if ch == '"' {
            return Some(out);
        } else {
            out.push(ch);
        }
    }
    None
}

pub(crate) fn json_field_number(obj: &str, key: &str) -> Option<f64> {
    let pattern = format!("\"{key}\":");
    let start = obj.find(&pattern)? + pattern.len();
    let tail = obj[start..].trim_start();
    if tail.starts_with("null") {
        return None;
    }
    let len = tail
        .char_indices()
        .take_while(|(_, ch)| ch.is_ascii_digit() || matches!(ch, '-' | '+' | '.' | 'e' | 'E'))
        .map(|(idx, ch)| idx + ch.len_utf8())
        .last()?;
    tail[..len].parse::<f64>().ok()
}

pub(crate) fn to_json(results: &[BenchResult]) -> String {
    let mut s = String::from("[\n");
    for (i, r) in results.iter().enumerate() {
        if i > 0 {
            s.push_str(",\n");
        }
        s.push_str("  {");
        write!(s, "\"name\":\"{}\"", json_escape(r.name)).unwrap();
        write!(s, ",\"family\":\"{}\"", json_escape(r.family)).unwrap();
        write!(s, ",\"status\":\"{}\"", json_escape(r.status)).unwrap();
        write!(s, ",\"shape\":\"{}\"", json_escape(&r.shape)).unwrap();
        write!(s, ",\"rows_per_tg\":{}", opt_u64(r.rows_per_tg)).unwrap();
        write!(s, ",\"threads_per_tg\":{}", opt_u64(r.threads_per_tg)).unwrap();
        write!(s, ",\"bytes_per_call\":{}", r.bytes_per_call).unwrap();
        write!(s, ",\"isolated_ms\":{}", opt_f64(r.isolated_ms)).unwrap();
        write!(s, ",\"isolated_sd_ms\":{}", opt_f64(r.isolated_sd_ms)).unwrap();
        write!(s, ",\"batched_ms\":{}", opt_f64(r.batched_ms)).unwrap();
        write!(s, ",\"batched_gbs\":{}", opt_f64(r.batched_gbs)).unwrap();
        write!(s, ",\"output_nonzero\":{}", opt_usize(r.output_nonzero)).unwrap();
        write!(s, ",\"sanity\":\"{}\"", json_escape(r.sanity)).unwrap();
        write!(s, ",\"note\":\"{}\"", json_escape(r.note)).unwrap();
        s.push('}');
    }
    s.push_str("\n]\n");
    s
}

pub(crate) fn opt_u64(v: Option<u64>) -> String {
    v.map(|v| v.to_string()).unwrap_or_else(|| "null".into())
}

pub(crate) fn opt_usize(v: Option<usize>) -> String {
    v.map(|v| v.to_string()).unwrap_or_else(|| "null".into())
}

pub(crate) fn opt_f64(v: Option<f64>) -> String {
    v.map(|v| format!("{v:.6}"))
        .unwrap_or_else(|| "null".into())
}

pub(crate) fn json_escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}
