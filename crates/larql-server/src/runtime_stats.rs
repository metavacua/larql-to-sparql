//! Runtime performance/activity recorder backing `GET /v1/runtime`
//! ([`crate::routes::runtime`]).
//!
//! Design rule: the `/v1/runtime` handler never computes a performance
//! number itself — it only snapshots facts recorded here by the code
//! that actually ran a generation. [`GenerationTally`] sums the real
//! per-request timings/counts (`larql_inference::GenerateResult` for a
//! VINDEX2 model, the prefill/decode wall-clock split taken in
//! `crate::vindex3::generate_v3_request` for a VINDEX3 container);
//! [`GenerationTally::into_sample`] turns a tally into the one number
//! `/v1/runtime` reports. That keeps the handler itself a pure
//! snapshot — it cannot silently drift from what a client actually
//! experienced, because it never re-derives the number a different way.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use larql_inference::GenerateResult;

/// Milliseconds per second — named so the tok/s arithmetic below reads
/// as "per second", not a bare magic `1000.0`.
const MS_PER_SEC: f64 = 1000.0;

/// One completed generation's measured performance. Every field here
/// was computed from a real timer/token-count at the call site — this
/// struct is a carrier, not a place that invents numbers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GenerationSample {
    pub prefill_tokens_per_second: f64,
    pub decode_tokens_per_second: f64,
    pub latency_ms: f64,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
}

impl GenerationSample {
    /// `0.0` is the crate-wide "not measured" sentinel (the same
    /// convention `GenerateResult::decode_tok_s` already uses) — turn
    /// it into `None` for serialization so `/v1/runtime` reports
    /// `null` rather than a fake zero rate.
    pub fn prefill_tps_or_none(&self) -> Option<f64> {
        (self.prefill_tokens_per_second > 0.0).then_some(self.prefill_tokens_per_second)
    }

    pub fn decode_tps_or_none(&self) -> Option<f64> {
        (self.decode_tokens_per_second > 0.0).then_some(self.decode_tokens_per_second)
    }
}

/// Accumulates prompt/completion token counts and prefill/decode wall
/// time across one request's generation work. A batched
/// `/v1/completions` call folds several prompts into one tally before
/// recording; a single chat/completion call folds in exactly one.
#[derive(Debug, Default, Clone, Copy)]
pub struct GenerationTally {
    prompt_tokens: usize,
    completion_tokens: usize,
    prefill_ms: f64,
    decode_ms_total: f64,
    decode_steps: usize,
}

impl GenerationTally {
    pub fn new() -> Self {
        Self::default()
    }

    /// Fold in one VINDEX2 [`GenerateResult`] — `prefill_ms` and
    /// `decode_ms` are populated by the inference crate itself
    /// (`larql_inference::layer_graph::generate*`), not measured here.
    pub fn add_v2(&mut self, result: &GenerateResult, prompt_tokens: usize) {
        self.prompt_tokens += prompt_tokens;
        self.completion_tokens += result.tokens.len();
        self.prefill_ms += result.prefill_ms;
        self.decode_ms_total += result.decode_ms.iter().sum::<f64>();
        self.decode_steps += result.decode_ms.len();
    }

    /// Fold in one VINDEX3 generation. `prefill_ms`/`decode_ms_total`
    /// come from the wall-clock split taken around `prefill_into` /
    /// `continue_session` in `crate::vindex3::generate_v3_request` —
    /// the V3 driver itself carries no timing, so the server times the
    /// two calls it already makes rather than inventing a number.
    pub fn add_v3(
        &mut self,
        prompt_tokens: usize,
        completion_tokens: usize,
        prefill_ms: f64,
        decode_ms_total: f64,
    ) {
        self.prompt_tokens += prompt_tokens;
        self.completion_tokens += completion_tokens;
        self.prefill_ms += prefill_ms;
        self.decode_ms_total += decode_ms_total;
        self.decode_steps += completion_tokens;
    }

    /// Turn the accumulated totals into a reportable sample.
    /// `latency_ms` is the caller's own end-to-end wall clock for the
    /// request — it may exceed `prefill_ms + decode_ms_total` by
    /// request-handling overhead (tokenizing, response shaping, lock
    /// acquisition), which is expected and not an error.
    pub fn into_sample(self, latency_ms: f64) -> GenerationSample {
        let prefill_tokens_per_second = if self.prefill_ms > 0.0 && self.prompt_tokens > 0 {
            self.prompt_tokens as f64 * MS_PER_SEC / self.prefill_ms
        } else {
            0.0
        };
        let decode_tokens_per_second = if self.decode_ms_total > 0.0 && self.decode_steps > 0 {
            self.decode_steps as f64 * MS_PER_SEC / self.decode_ms_total
        } else {
            0.0
        };
        GenerationSample {
            prefill_tokens_per_second,
            decode_tokens_per_second,
            latency_ms,
            prompt_tokens: self.prompt_tokens,
            completion_tokens: self.completion_tokens,
        }
    }
}

/// RAII marker for one in-flight generation request. Held for the
/// lifetime of the actual generation work — including inside a
/// detached `spawn_blocking` closure that a streaming response hands
/// off to and which outlives the async handler that created the guard
/// — so `RuntimeRecorder::active_requests` reflects work genuinely in
/// flight, not just an async handler's own stack frame.
pub struct GenerationGuard {
    recorder: Arc<RuntimeRecorder>,
}

impl Drop for GenerationGuard {
    fn drop(&mut self) {
        self.recorder.active_requests.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Shared server-wide runtime recorder backing `GET /v1/runtime`. One
/// instance lives on `AppState` behind an `Arc` so generation handlers
/// can clone the recorder alone into a `spawn_blocking` closure,
/// independently of the rest of `AppState`.
pub struct RuntimeRecorder {
    last_sample: Mutex<Option<GenerationSample>>,
    active_requests: AtomicU32,
}

impl Default for RuntimeRecorder {
    fn default() -> Self {
        Self::new()
    }
}

impl RuntimeRecorder {
    pub fn new() -> Self {
        Self {
            last_sample: Mutex::new(None),
            active_requests: AtomicU32::new(0),
        }
    }

    /// Record the most recently completed generation. Overwrites any
    /// prior sample — `/v1/runtime` reports the latest request, not a
    /// history.
    pub fn record(&self, sample: GenerationSample) {
        let mut guard = self
            .last_sample
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        *guard = Some(sample);
    }

    /// The most recently recorded sample, if any generation has
    /// completed since the server started.
    pub fn last_sample(&self) -> Option<GenerationSample> {
        *self
            .last_sample
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Mark one generation as started. `active_requests` drops back
    /// down when the returned guard is dropped. Takes `self` by
    /// `Arc` (rather than `&self`) so the guard can hold its own
    /// clone and outlive the caller's scope — the shape a streaming
    /// handler needs to move the guard into its `spawn_blocking`
    /// closure.
    pub fn enter_generation(self: Arc<Self>) -> GenerationGuard {
        self.active_requests.fetch_add(1, Ordering::AcqRel);
        GenerationGuard { recorder: self }
    }

    pub fn active_requests(&self) -> u32 {
        self.active_requests.load(Ordering::Acquire)
    }
}

/// Current resident-set size of this process, in bytes — `getrusage`'s
/// `ru_maxrss`. That is a *peak*, not a live-instantaneous RSS; for a
/// model server whose big allocations happen once at load and don't
/// shrink, peak and current track each other closely in practice, and
/// a peak is still a real measurement rather than a fabricated one.
/// Returns `None` on a syscall failure or a non-Unix target, so
/// callers report `null` instead of a made-up number.
///
/// macOS reports `ru_maxrss` in bytes; every other POSIX target
/// (Linux included) reports it in kilobytes — a well-known libc
/// portability wart, not a bug in this function.
pub fn resident_bytes() -> Option<u64> {
    #[cfg(unix)]
    {
        // SAFETY: `usage` is read only after `getrusage` returns 0,
        // which is documented to fully populate the struct.
        let usage = unsafe {
            let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
            if libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) != 0 {
                return None;
            }
            usage.assume_init()
        };
        let raw = usage.ru_maxrss;
        if raw < 0 {
            return None;
        }
        let bytes_per_unit: u64 = if cfg!(target_os = "macos") { 1 } else { 1024 };
        Some(raw as u64 * bytes_per_unit)
    }
    #[cfg(not(unix))]
    {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn result_with(prefill_ms: f64, decode_ms: Vec<f64>, n_tokens: usize) -> GenerateResult {
        GenerateResult {
            tokens: vec![(String::new(), 1.0); n_tokens],
            prefill_ms,
            decode_ms,
            stage_timings: Default::default(),
            error: None,
        }
    }

    #[test]
    fn tally_add_v2_accumulates_across_multiple_results() {
        let mut tally = GenerationTally::new();
        tally.add_v2(&result_with(10.0, vec![5.0, 5.0], 2), 3);
        tally.add_v2(&result_with(20.0, vec![5.0], 1), 4);
        let sample = tally.into_sample(100.0);
        assert_eq!(sample.prompt_tokens, 7);
        assert_eq!(sample.completion_tokens, 3);
        // prefill: 7 prompt tokens over 30ms total = 7000/30 tok/s.
        assert!((sample.prefill_tokens_per_second - 7.0 * 1000.0 / 30.0).abs() < 1e-6);
        // decode: 3 steps over 15ms total = 200 tok/s.
        assert!((sample.decode_tokens_per_second - 200.0).abs() < 1e-6);
        assert_eq!(sample.latency_ms, 100.0);
    }

    #[test]
    fn tally_add_v3_matches_add_v2_shape() {
        let mut tally = GenerationTally::new();
        tally.add_v3(5, 2, 40.0, 20.0);
        let sample = tally.into_sample(80.0);
        assert_eq!(sample.prompt_tokens, 5);
        assert_eq!(sample.completion_tokens, 2);
        assert!((sample.prefill_tokens_per_second - 5.0 * 1000.0 / 40.0).abs() < 1e-6);
        assert!((sample.decode_tokens_per_second - 2.0 * 1000.0 / 20.0).abs() < 1e-6);
    }

    #[test]
    fn v2_and_v3_tallies_combine_in_one_sample() {
        // A hypothetical mixed accounting path isn't real yet, but the
        // accumulator must not assume it's fed exactly one add_* call —
        // batched /v1/completions already folds N prompts in.
        let mut tally = GenerationTally::new();
        tally.add_v2(&result_with(10.0, vec![10.0], 1), 2);
        tally.add_v3(2, 1, 10.0, 10.0);
        let sample = tally.into_sample(1.0);
        assert_eq!(sample.prompt_tokens, 4);
        assert_eq!(sample.completion_tokens, 2);
    }

    #[test]
    fn empty_tally_reports_zero_rates_not_nan() {
        let sample = GenerationTally::new().into_sample(0.0);
        assert_eq!(sample.prefill_tokens_per_second, 0.0);
        assert_eq!(sample.decode_tokens_per_second, 0.0);
        assert_eq!(sample.prefill_tps_or_none(), None);
        assert_eq!(sample.decode_tps_or_none(), None);
    }

    #[test]
    fn sample_or_none_hides_the_zero_sentinel_but_not_real_rates() {
        let zero = GenerationTally::new().into_sample(5.0);
        assert!(zero.prefill_tps_or_none().is_none());
        assert!(zero.decode_tps_or_none().is_none());

        let mut tally = GenerationTally::new();
        tally.add_v3(10, 5, 10.0, 10.0);
        let real = tally.into_sample(5.0);
        assert!(real.prefill_tps_or_none().is_some());
        assert!(real.decode_tps_or_none().is_some());
    }

    #[test]
    fn recorder_starts_with_no_sample_and_zero_active() {
        let r = RuntimeRecorder::new();
        assert!(r.last_sample().is_none());
        assert_eq!(r.active_requests(), 0);
    }

    #[test]
    fn record_overwrites_with_the_latest_sample() {
        let r = RuntimeRecorder::new();
        r.record(GenerationTally::new().into_sample(1.0));
        assert_eq!(r.last_sample().unwrap().latency_ms, 1.0);
        r.record(GenerationTally::new().into_sample(2.0));
        assert_eq!(r.last_sample().unwrap().latency_ms, 2.0);
    }

    #[test]
    fn enter_generation_increments_and_drop_decrements() {
        let r = Arc::new(RuntimeRecorder::new());
        assert_eq!(r.active_requests(), 0);
        let guard_a = Arc::clone(&r).enter_generation();
        assert_eq!(r.active_requests(), 1);
        let guard_b = Arc::clone(&r).enter_generation();
        assert_eq!(r.active_requests(), 2);
        drop(guard_a);
        assert_eq!(r.active_requests(), 1);
        drop(guard_b);
        assert_eq!(r.active_requests(), 0);
    }

    #[test]
    fn guard_survives_a_move_into_a_spawned_thread() {
        // The whole point of the `Arc<Self>` receiver: the guard must
        // be `Send + 'static` so a streaming handler can move it into
        // the `spawn_blocking` closure that outlives the async fn's
        // own stack frame.
        let r = Arc::new(RuntimeRecorder::new());
        let guard = Arc::clone(&r).enter_generation();
        let handle = std::thread::spawn(move || {
            let _g = guard;
            std::thread::sleep(std::time::Duration::from_millis(10));
        });
        assert_eq!(r.active_requests(), 1);
        handle.join().unwrap();
        assert_eq!(r.active_requests(), 0);
    }

    #[test]
    fn resident_bytes_reports_a_plausible_value_on_unix() {
        #[cfg(unix)]
        {
            let bytes = resident_bytes().expect("getrusage should succeed for this process");
            // A running test binary needs at least a few MB resident;
            // anything under 1 MB would mean the unit conversion is
            // wrong (e.g. treating macOS bytes as Linux kilobytes).
            assert!(
                bytes > 1_000_000,
                "implausibly small resident_bytes: {bytes}"
            );
        }
    }
}
