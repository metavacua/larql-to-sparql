//! Reading the routing trace written by the reference MoE backend.
//!
//! Format is one JSON object per line, emitted by
//! `larql-compute/src/ffn/expert_weight/trace.rs`. Each record is already one
//! contiguous run of positions — that is what `seq` guarantees — so records map
//! to [`Run`]s one-for-one and no regrouping is needed here. `seq` itself is
//! not deserialised: it exists to *establish* the run boundary, and once the
//! boundary is a record boundary it carries no further information.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use serde::Deserialize;

use super::stats::Run;

#[derive(Deserialize)]
struct Record {
    layer: usize,
    experts: Vec<Vec<usize>>,
}

/// A parsed trace, grouped by layer.
#[derive(Debug)]
pub struct Trace {
    /// Runs of positions, keyed by layer id. Ordered so reports read top-down.
    pub layers: BTreeMap<usize, Vec<Run>>,
}

impl Trace {
    /// Largest expert id seen, plus one.
    ///
    /// A lower bound on the model's expert count, not the count itself: an
    /// expert that never fires on this corpus is invisible here. The caller
    /// should prefer an explicit count and fall back to this.
    pub fn observed_num_experts(&self) -> usize {
        self.runs()
            .flatten()
            .flatten()
            .copied()
            .max()
            .map_or(0, |max| max + 1)
    }

    /// Largest selection size seen — the model's top-k.
    pub fn top_k(&self) -> usize {
        self.runs().flatten().map(Vec::len).max().unwrap_or(0)
    }

    pub fn positions(&self) -> usize {
        self.runs().map(Vec::len).sum()
    }

    pub fn num_runs(&self) -> usize {
        self.layers.values().map(Vec::len).sum()
    }

    fn runs(&self) -> impl Iterator<Item = &Run> {
        self.layers.values().flatten()
    }
}

/// Parse a JSONL trace, reporting the offending line number on bad input.
pub fn load(path: &Path) -> Result<Trace, Box<dyn std::error::Error>> {
    let mut layers: BTreeMap<usize, Vec<Run>> = BTreeMap::new();
    for (index, line) in BufReader::new(File::open(path)?).lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let record: Record = serde_json::from_str(&line)
            .map_err(|e| format!("{}:{}: {e}", path.display(), index + 1))?;
        layers.entry(record.layer).or_default().push(record.experts);
    }
    if layers.is_empty() {
        return Err(format!("{}: no routing records", path.display()).into());
    }
    Ok(Trace { layers })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_trace(body: &str) -> tempfile::NamedTempFile {
        let mut file = tempfile::NamedTempFile::new().expect("temp file");
        file.write_all(body.as_bytes()).expect("write trace");
        file.flush().expect("flush trace");
        file
    }

    const TWO_LAYERS: &str = concat!(
        r#"{"layer":0,"seq":0,"experts":[[1,2],[2,3]]}"#,
        "\n",
        r#"{"layer":0,"seq":1,"experts":[[0,1]]}"#,
        "\n",
        r#"{"layer":1,"seq":0,"experts":[[3,7]]}"#,
        "\n",
    );

    #[test]
    fn load_groups_records_by_layer() {
        let file = write_trace(TWO_LAYERS);
        let trace = load(file.path()).expect("parses");

        assert_eq!(trace.layers.len(), 2);
        assert_eq!(trace.layers[&0].len(), 2, "two runs at layer 0");
        assert_eq!(trace.layers[&1].len(), 1);
    }

    #[test]
    fn shape_accessors_summarise_the_trace() {
        let file = write_trace(TWO_LAYERS);
        let trace = load(file.path()).expect("parses");

        assert_eq!(trace.observed_num_experts(), 8, "max id 7, plus one");
        assert_eq!(trace.top_k(), 2);
        assert_eq!(trace.positions(), 4);
        assert_eq!(trace.num_runs(), 3);
    }

    /// Runs stay separate so a consumer never measures adjacency across the
    /// window boundary the run split exists to mark.
    #[test]
    fn records_are_not_concatenated_within_a_layer() {
        let file = write_trace(TWO_LAYERS);
        let trace = load(file.path()).expect("parses");
        let lengths: Vec<usize> = trace.layers[&0].iter().map(Vec::len).collect();
        assert_eq!(lengths, vec![2, 1]);
    }

    #[test]
    fn blank_lines_are_skipped() {
        let file = write_trace(&format!("\n{TWO_LAYERS}\n\n"));
        assert_eq!(load(file.path()).expect("parses").num_runs(), 3);
    }

    #[test]
    fn malformed_lines_report_their_line_number() {
        let file = write_trace("{\"layer\":0,\"experts\":[[1]]}\nnot json\n");
        let err = load(file.path()).expect_err("must reject").to_string();
        assert!(err.contains(":2:"), "expected line 2 in {err:?}");
    }

    #[test]
    fn an_empty_trace_is_an_error_not_an_empty_report() {
        let file = write_trace("\n\n");
        assert!(load(file.path()).is_err());
    }

    #[test]
    fn a_missing_file_is_an_error() {
        assert!(load(Path::new("/nonexistent/trace.jsonl")).is_err());
    }
}
