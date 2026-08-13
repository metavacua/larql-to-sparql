//! Opt-in per-token expert-routing trace for the reference MoE backend.
//!
//! Off unless [`options::ENV_MOE_ROUTE_TRACE`] names a writable path, in which
//! case every [`ExpertWeightFfn`](super::ExpertWeightFfn) MoE block appends one
//! JSON line per call:
//!
//! ```text
//! {"layer":0,"seq":0,"experts":[[3,17,22,9],[3,1,22,30], ...]}
//! ```
//!
//! `seq` counts calls *for that layer*, so records sharing `(layer, seq)` are
//! one contiguous run of positions. Adjacency is only meaningful inside a run:
//! the Shannon scorer advances a 512/256 window, so position 0 of one call does
//! not follow the last position of the previous one, and a consumer that
//! ignores `seq` would measure a spurious discontinuity at every window edge.
//!
//! **Why the trace lives on this path and not on `cpu_moe_forward`.** This is
//! the f32 reference tier, the one pinned to the GPT-OSS reference at < 1e-5
//! (`docs/k3-funnel.md` §4.7.5), so the selections it records are the
//! selections the reference would make. It is also handed `layer` together with
//! the whole `[tokens, hidden]` block, so both trace coordinates are already in
//! scope — the quantised path has neither, and its existing `LARQL_MOE_DEBUG`
//! recovers a layer index from a global counter mod 30, which is not an
//! attribution a measurement can rest on.

// This whole file's real machinery (TraceWriter, open_sink, sink,
// record_into) is native-only: it is a File/BufWriter-backed sink
// gated by an env var, and wasm32v1-none has neither std::fs nor
// std::env at all. TraceWriter is confirmed unused outside this file
// (grep), so it's wholesale native-only rather than surgically split.
// buffer()/record() -- the only two items anything else in the crate
// calls -- get wasm32 stubs matching the "tracing disabled" behavior
// that already happens natively whenever the env var is unset.
#[cfg(not(target_arch = "wasm32"))]
use std::collections::HashMap;
#[cfg(not(target_arch = "wasm32"))]
use std::fs::File;
#[cfg(not(target_arch = "wasm32"))]
use std::io::{BufWriter, Write};
#[cfg(not(target_arch = "wasm32"))]
use std::sync::{Mutex, OnceLock};

#[cfg(not(target_arch = "wasm32"))]
use crate::options;

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

/// Selected expert ids per token, for one MoE block call.
pub type BlockRouting = Vec<Vec<usize>>;

/// Appends `(layer, seq, experts)` records as JSON lines.
///
/// Generic over the sink so the record format and the per-layer `seq`
/// bookkeeping are testable without touching the filesystem or process env.
#[cfg(not(target_arch = "wasm32"))]
pub struct TraceWriter<W: Write> {
    out: W,
    seq: HashMap<usize, usize>,
}

#[cfg(not(target_arch = "wasm32"))]
impl<W: Write> TraceWriter<W> {
    pub fn new(out: W) -> Self {
        Self {
            out,
            seq: HashMap::new(),
        }
    }

    /// Next call index for `layer`, counting from 0. Layers count separately.
    fn next_seq(&mut self, layer: usize) -> usize {
        let slot = self.seq.entry(layer).or_insert(0);
        let seq = *slot;
        *slot += 1;
        seq
    }

    /// Append one MoE block's routing.
    ///
    /// The JSON is written by hand rather than through `serde_json`: every
    /// field is a non-negative integer, so there is nothing to escape and no
    /// encoder to get wrong, and `larql-compute` does not take a production
    /// dependency for the sake of a diagnostic. The tests parse the output back
    /// with a real JSON parser, so the format stays honest.
    ///
    /// Flushes per record. The reference tier spends seconds per layer, so the
    /// cost is invisible, and a trace from a run that was killed part-way stays
    /// analysable up to its last complete line.
    pub fn record(&mut self, layer: usize, experts: &BlockRouting) -> std::io::Result<()> {
        let seq = self.next_seq(layer);
        write!(self.out, "{{\"layer\":{layer},\"seq\":{seq},\"experts\":[")?;
        for (token, selection) in experts.iter().enumerate() {
            if token > 0 {
                self.out.write_all(b",")?;
            }
            self.out.write_all(b"[")?;
            for (rank, expert) in selection.iter().enumerate() {
                let separator = if rank > 0 { "," } else { "" };
                write!(self.out, "{separator}{expert}")?;
            }
            self.out.write_all(b"]")?;
        }
        writeln!(self.out, "]}}")?;
        self.out.flush()
    }
}

/// Open the JSONL sink named by `path`; `None` (with an explanation on
/// stderr) when the file cannot be created, so a bad path degrades to
/// tracing-off rather than a failed run.
///
/// Split from [`sink`] so both open arms are testable: the `OnceLock` there
/// resolves once per process, pinned to whatever environment the test binary
/// started with.
#[cfg(not(target_arch = "wasm32"))]
fn open_sink(path: Option<String>) -> Option<Mutex<TraceWriter<BufWriter<File>>>> {
    let path = path?;
    match File::create(&path) {
        Ok(file) => Some(Mutex::new(TraceWriter::new(BufWriter::new(file)))),
        Err(err) => {
            eprintln!(
                "[{}] cannot open {path}: {err} — routing trace disabled",
                options::ENV_MOE_ROUTE_TRACE
            );
            None
        }
    }
}

/// Process-wide sink, resolved once from the environment on first use.
#[cfg(not(target_arch = "wasm32"))]
fn sink() -> Option<&'static Mutex<TraceWriter<BufWriter<File>>>> {
    static SINK: OnceLock<Option<Mutex<TraceWriter<BufWriter<File>>>>> = OnceLock::new();
    SINK.get_or_init(|| open_sink(options::env_nonempty_value(options::ENV_MOE_ROUTE_TRACE)))
        .as_ref()
}

/// A routing buffer sized for `tokens`, or `None` when tracing is off.
///
/// Returning `None` is what keeps the untraced path allocation-free: the caller
/// holds an `Option` and never builds the per-token id vectors at all.
///
/// wasm32v1-none has neither std::fs nor std::env, so tracing can never
/// be turned on there -- `None` unconditionally, the same value the
/// native path returns whenever the env var is unset.
#[cfg(target_arch = "wasm32")]
pub fn buffer(_tokens: usize) -> Option<BlockRouting> {
    None
}

#[cfg(not(target_arch = "wasm32"))]
pub fn buffer(tokens: usize) -> Option<BlockRouting> {
    sink().map(|_| Vec::with_capacity(tokens))
}

/// Append `experts` for `layer`. No-op when tracing is off.
///
/// A diagnostic must not take down a scoring run, so I/O and lock failures are
/// reported and swallowed rather than propagated.
#[cfg(target_arch = "wasm32")]
pub fn record(_layer: usize, _experts: Option<BlockRouting>) {}

#[cfg(not(target_arch = "wasm32"))]
pub fn record(layer: usize, experts: Option<BlockRouting>) {
    let (Some(experts), Some(sink)) = (experts, sink()) else {
        return;
    };
    record_into(sink, layer, &experts);
}

/// Split from [`record`] for the same reason as [`open_sink`]: the global
/// sink cannot be configured in-process, and the failure arms deserve tests.
#[cfg(not(target_arch = "wasm32"))]
fn record_into<W: Write>(sink: &Mutex<TraceWriter<W>>, layer: usize, experts: &BlockRouting) {
    match sink.lock() {
        Ok(mut writer) => {
            if let Err(err) = writer.record(layer, experts) {
                eprintln!(
                    "[{}] write failed at layer {layer}: {err}",
                    options::ENV_MOE_ROUTE_TRACE
                );
            }
        }
        Err(_) => eprintln!(
            "[{}] sink poisoned; trace is incomplete",
            options::ENV_MOE_ROUTE_TRACE
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Parse a `TraceWriter`'s output back into one JSON value per line.
    fn lines(raw: &[u8]) -> Vec<serde_json::Value> {
        std::str::from_utf8(raw)
            .expect("trace output is utf-8")
            .lines()
            .map(|l| serde_json::from_str(l).expect("each line is a JSON object"))
            .collect()
    }

    #[test]
    fn record_writes_one_json_line_per_block() {
        let mut w = TraceWriter::new(Vec::new());
        w.record(3, &vec![vec![1, 2], vec![2, 7]]).unwrap();

        let out = lines(&w.out);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0]["layer"], 3);
        assert_eq!(out[0]["seq"], 0);
        assert_eq!(out[0]["experts"], serde_json::json!([[1, 2], [2, 7]]));
    }

    /// `seq` is what lets a consumer avoid measuring adjacency across a window
    /// boundary, so per-layer independence is load-bearing, not cosmetic.
    fn seq_of(value: &serde_json::Value) -> u64 {
        value["seq"].as_u64().expect("seq is a number")
    }

    #[test]
    fn seq_counts_per_layer_not_globally() {
        let mut w = TraceWriter::new(Vec::new());
        for _ in 0..2 {
            w.record(0, &vec![vec![1]]).unwrap();
            w.record(1, &vec![vec![2]]).unwrap();
        }

        let out = lines(&w.out);
        let seqs: Vec<(u64, u64)> = out
            .iter()
            .map(|v| (v["layer"].as_u64().unwrap(), seq_of(v)))
            .collect();
        assert_eq!(seqs, vec![(0, 0), (1, 0), (0, 1), (1, 1)]);
    }

    #[test]
    fn empty_block_still_records_and_advances_seq() {
        let mut w = TraceWriter::new(Vec::new());
        w.record(0, &vec![]).unwrap();
        w.record(0, &vec![vec![4]]).unwrap();

        let out = lines(&w.out);
        assert_eq!(out[0]["experts"], serde_json::json!([]));
        assert_eq!(seq_of(&out[1]), 1);
    }

    #[test]
    fn write_errors_surface_to_the_caller() {
        /// A sink that fails on first write, standing in for a full disk.
        struct Failing;
        impl Write for Failing {
            fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
                Err(std::io::Error::other("no space"))
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }

        let mut w = TraceWriter::new(Failing);
        assert!(w.record(0, &vec![vec![1]]).is_err());
    }

    /// With no path set, tracing stays off and the hot path allocates nothing.
    #[test]
    fn buffer_is_none_when_unconfigured() {
        assert!(super::buffer(8).is_none());
        super::record(0, None);
    }

    #[test]
    fn open_sink_absent_path_is_off() {
        assert!(open_sink(None).is_none());
    }

    #[test]
    fn open_sink_unwritable_path_degrades_to_off() {
        // A path under a file (not a directory) cannot be created anywhere.
        let bad = format!(
            "{}/not-a-dir/trace.jsonl",
            std::env::temp_dir()
                .join(format!("larql-trace-test-{}", std::process::id()))
                .join("plain-file")
                .display()
        );
        assert!(open_sink(Some(bad)).is_none());
    }

    #[test]
    fn open_sink_writable_path_round_trips_a_record() {
        let path =
            std::env::temp_dir().join(format!("larql-trace-test-{}.jsonl", std::process::id()));
        let sink = open_sink(Some(path.display().to_string())).expect("temp file is creatable");

        record_into(&sink, 5, &vec![vec![7, 2]]);
        drop(sink.into_inner().expect("unpoisoned")); // flush the BufWriter

        let raw = std::fs::read(&path).expect("trace file exists");
        let out = lines(&raw);
        assert_eq!(out[0]["layer"], 5);
        assert_eq!(out[0]["experts"], serde_json::json!([[7, 2]]));
        let _ = std::fs::remove_file(&path);
    }

    /// A diagnostic must not take down a scoring run: both failure arms
    /// (writer error, poisoned lock) report and return.
    #[test]
    fn record_into_swallows_writer_errors_and_poison() {
        struct Failing;
        impl Write for Failing {
            fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
                Err(std::io::Error::other("no space"))
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let failing = Mutex::new(TraceWriter::new(Failing));
        record_into(&failing, 0, &vec![vec![1]]);

        let poisoned = Mutex::new(TraceWriter::new(Vec::new()));
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = poisoned.lock().unwrap();
            panic!("poison the lock");
        }));
        record_into(&poisoned, 1, &vec![vec![2]]);
    }
}
