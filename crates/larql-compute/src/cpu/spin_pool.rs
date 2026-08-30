//! Spin-barrier thread pool for the decode hot path.
//!
//! Rayon puts idle workers to sleep between parallel sections (the right call
//! for batch throughput, the wrong one for a tight decode loop). A 26B-A4B
//! token runs ~200 small fork-join sections — attention Q/K/V/O, dense
//! gate_up/down, the expert fold, lm_head, per layer — and a `/usr/bin/sample`
//! profile attributed ~30% of decode thread-time to the resulting churn:
//! workers asleep in `wait_until_cold`, the driver blocked in
//! `in_worker_cold -> LockLatch::wait_and_reset -> __psynch_cvwait`, plus the
//! condvar wake latency paid on *every* section.
//!
//! This pool keeps workers HOT. They spin on an epoch counter and only
//! [`park`](std::thread::park_timeout) after a long idle gap, so a
//! `for_each_chunk` dispatched microseconds after the previous one finds them
//! already spinning — ready in ~ns, no condvar round-trip. The dispatcher
//! participates as the n-th worker; chunks are owned by static contiguous-block
//! assignment (participant `p` runs one unbroken run of `num/n` chunks), which
//! keeps the `completed == num_chunks` barrier sound across back-to-back
//! dispatches *and* keeps each owner's reads sequential — see `run_chunks` for
//! why the block/stride choice is worth 1.87× at lm_head-class shapes.
//! When a worker has to wait it backs off spin → yield → park, so it stays
//! cooperative under contention. Modeled on llama.cpp's persistent thread
//! pool + `ggml_barrier`.
//!
//! [`enabled`] gates whether callers route through here or stay on rayon. It is
//! **on by default** (idle workers park, so a quiet pool costs ~0 CPU);
//! `LARQL_SPIN_POOL=0` forces the rayon path. Either way the arithmetic is
//! identical — only *which threads run which chunks* differs.

use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;
use std::time::Duration;

thread_local! {
    /// True while this thread is executing a dispatched chunk body. Guards
    /// against reentrant `for_each_chunk` (a body that itself dispatches): the
    /// nested call runs serially inline rather than deadlocking on the pool.
    static IN_BODY: Cell<bool> = const { Cell::new(false) };
}

/// Adaptive-backoff thresholds (iterations of the wait loop) for a worker
/// waiting on the next dispatch. It escalates spin → yield → park:
///
/// - **spin** (`< SPIN_HOT`): `spin_loop()` for ~hundreds of µs. This is the
///   same pure-spin window that produced the measured decode win, so *active
///   decode behaviour is unchanged* — every inter-section gap within a token
///   stays in the spin phase, giving a ~ns wake.
/// - **yield** (`< YIELD_UNTIL`): a *brief* `yield_now()` bridge — one last
///   chance for an about-to-arrive dispatch before paying a park/unpark. This
///   used to be 128k iterations, which is not a bridge but a busy loop with a
///   syscall in it: `yield_now` returns immediately on an idle core, so the
///   phase burned as much CPU as spinning *plus* 128k `sched_yield` calls per
///   idle transition per worker.
/// - **park** (otherwise): deep idle between requests / runs. The dispatcher
///   unparks all workers on every dispatch (and `Drop` unparks for shutdown),
///   so the timeout is a pure liveness backstop and can be long. It used to be
///   **50µs**, which meant a "parked" worker woke 20k times/sec — measured
///   3.40 CPU-seconds per 5s of idle (68% of a core, ~856k involuntary context
///   switches, 11 workers, M3 Max) burned by an idle pool, forever, because
///   `global()` is process-lifetime.
///
/// Net: spin = the win during active decode; park = actually-zero CPU when the
/// decode loop is idle — which is what makes on-by-default safe.
const SPIN_HOT: u32 = 256_000;
const YIELD_UNTIL: u32 = SPIN_HOT + 64;
const PARK_BACKSTOP: Duration = Duration::from_secs(1);

/// Cross-thread dispatch state. The `epoch` release store wakes workers; the
/// `slot_seq` seqlock + `task_gen` stamp make the task fields safe to read,
/// because epoch-wakeup alone does not prove the slot still belongs to the
/// observed dispatch (see `slot_seq`).
struct Shared {
    /// Bumped once per `for_each_chunk`; workers wake when it changes.
    epoch: AtomicU64,
    /// Seqlock over the task fields (`data`/`tramp`/`num_chunks`/`completed`/
    /// `panicked`/`task_gen`): odd while the dispatcher rewrites them, bumped
    /// even when stable. Readers snapshot the fields between two equal even
    /// reads. Needed because a worker can observe the epoch bump for dispatch
    /// N yet reach the fields only after N's barrier passed (possible exactly
    /// when its block in N was empty, so the barrier didn't need it) and
    /// dispatch N+1 has begun overwriting the slot.
    slot_seq: AtomicU64,
    /// Epoch value the current slot contents belong to, written inside the
    /// seqlock. A participant runs the slot only when this matches the epoch
    /// it observed — otherwise its `completed` increments would count toward
    /// a dispatch it did not run, the barrier would release early, and the
    /// closure would drop while another participant still executes it.
    task_gen: AtomicU64,
    /// Chunks finished this dispatch; the barrier waits for `== num_chunks`.
    /// With static block ownership each chunk is run exactly once, so this
    /// reaching `num_chunks` proves every trampoline call has returned — no
    /// worker can still touch the (about-to-drop) closure.
    completed: AtomicUsize,
    /// Chunk count for the current dispatch.
    num_chunks: AtomicUsize,
    /// Type-erased `&F` for the current dispatch (valid until the barrier).
    data: AtomicPtr<()>,
    /// `fn(*const (), usize)` trampoline that recovers `&F` and calls it.
    tramp: AtomicUsize,
    /// Set on drop; workers observe it and exit.
    shutdown: AtomicBool,
    /// Set when any chunk this dispatch panicked — a cheap flag the dispatcher
    /// checks after the barrier without locking on the happy path.
    panicked: AtomicBool,
    /// The first chunk panic's payload. A panicking body still increments
    /// `completed` (so the barrier finishes instead of hanging on a dead
    /// worker), and the dispatcher `resume_unwind`s this afterward — so a panic
    /// propagates to the caller exactly like rayon, rather than killing a
    /// worker thread and live-locking every future dispatch.
    panic_payload: Mutex<Option<Box<dyn std::any::Any + Send + 'static>>>,
}

/// A persistent spin-barrier pool. Owns `n-1` worker threads; the thread that
/// calls [`for_each_chunk`] is the n-th participant.
pub struct SpinPool {
    shared: Arc<Shared>,
    workers: Vec<thread::JoinHandle<()>>,
    n_threads: usize,
    /// Serializes dispatchers. Uncontended (≈one atomic CAS) for the normal
    /// single-driver decode loop; serializes the rare concurrent dispatch
    /// (`bench --concurrent N`, multi-threaded test harness) so the shared
    /// epoch/cursor state stays consistent.
    dispatch_lock: Mutex<()>,
}

/// Recover `&F` from the type-erased data pointer and invoke it for `chunk`.
///
/// # Safety
/// `data` must point to the live `F` published for the current epoch (the
/// dispatcher keeps it on its stack until the completion barrier passes), and
/// `F: Sync` (multiple threads call it concurrently).
fn trampoline<F: Fn(usize) + Sync>(data: *const (), chunk: usize) {
    // SAFETY: see fn docs — `data` is `&F` published under the epoch fence and
    // outlives every call within the dispatch.
    let f = unsafe { &*(data as *const F) };
    f(chunk);
}

fn worker_loop(shared: Arc<Shared>, worker_id: usize, n_participants: usize) {
    let mut seen_epoch = 0u64;
    loop {
        // Wait for a new dispatch (spin first, park if idle persists).
        let mut spins = 0u32;
        let epoch = loop {
            let e = shared.epoch.load(Ordering::Acquire);
            if e != seen_epoch {
                break e;
            }
            if shared.shutdown.load(Ordering::Relaxed) {
                return;
            }
            spins += 1;
            if spins < SPIN_HOT {
                std::hint::spin_loop();
            } else if spins < YIELD_UNTIL {
                std::thread::yield_now();
            } else {
                thread::park_timeout(PARK_BACKSTOP);
            }
        };
        seen_epoch = epoch;
        if shared.shutdown.load(Ordering::Relaxed) {
            return;
        }
        // Pass the epoch this worker is answering; `run_chunks` refuses the
        // slot if it has already been re-published for a later dispatch (the
        // loop then re-observes the new epoch and runs it exactly once).
        run_chunks(&shared, worker_id, n_participants, epoch);
    }
}

/// Run this participant's statically-assigned chunks, as one **contiguous
/// block** of the chunk index space.
///
/// Static ownership — rather than a shared resettable cursor — plus the
/// `task_gen` guard is what makes `completed == num_chunks` a sound barrier
/// across back-to-back dispatches: no participant can re-claim a chunk the
/// next dispatch reset, and no participant can run (or count toward) a
/// dispatch other than the one whose epoch it observed, so once the count is
/// reached every trampoline call has returned and the closure is safe to
/// drop. *Contiguous* blocks are as static as the round-robin stride this
/// previously used (`c += n_participants`) — each chunk still has exactly one
/// owner — so that argument is untouched by the change from stride to block.
///
/// **Why contiguous rather than strided.** Chunks map to contiguous byte ranges
/// of the weight slab, so a strided owner walks the *whole* slab at a stride of
/// `n_participants × chunk_bytes` and the hardware sequential prefetcher never
/// engages. Measured on `benches/q4k_q8k_matvec.rs::bench_mt_production`
/// (M3 Max, AC, 2026-07-28): at the lm_head-class shape 65536×2816 (415 MB,
/// 2048 chunks × 50 KB, 811 KB stride) the strided pool ran **58.5 GiB/s vs
/// rayon's 109.6** — a 1.87× loss on a default-on path — while every
/// per-layer shape *won* by 45–113%, because those slabs (3–33 MB) are
/// SLC-resident and striding costs nothing there. Blocking keeps the
/// small-shape win (dispatch overhead is what that one is about) and gives each
/// owner one sequential run at large sizes.
///
/// Chunk cost is uniform for every current caller (equal row counts per chunk),
/// so the load-balancing advantage strided ownership would have under uneven
/// chunk cost does not apply; llama.cpp's pool partitions rows the same way.
fn run_chunks(shared: &Shared, participant_id: usize, n_participants: usize, expected_gen: u64) {
    // Seqlock-consistent snapshot of the task slot, refused unless it still
    // belongs to `expected_gen`. Guards the empty-block straggler: a
    // participant whose block in dispatch N was empty is not needed by N's
    // barrier, so N can finish and N+1 can rewrite the slot while this
    // participant sits between its epoch read and these loads. Running a
    // mismatched slot would execute N+1's chunks attributed to N — and then
    // again when the loop notices the epoch moved — over-counting `completed`
    // so N+1's barrier releases while other participants still execute the
    // (then dropped) closure.
    let seq_before = shared.slot_seq.load(Ordering::Acquire);
    if seq_before & 1 == 1 {
        return; // dispatcher mid-rewrite; caller re-observes the epoch
    }
    let num = shared.num_chunks.load(Ordering::Relaxed);
    let tramp_addr = shared.tramp.load(Ordering::Relaxed);
    let data = shared.data.load(Ordering::Relaxed) as *const ();
    let gen = shared.task_gen.load(Ordering::Relaxed);
    std::sync::atomic::fence(Ordering::Acquire);
    if shared.slot_seq.load(Ordering::Relaxed) != seq_before || gen != expected_gen {
        return; // torn snapshot, or the slot was re-published for a later dispatch
    }
    if tramp_addr == 0 || num == 0 || participant_id >= n_participants {
        return;
    }
    // SAFETY: `tramp_addr` is a `fn(*const (), usize)` stored by the dispatcher
    // before the epoch release; recovered here after the epoch acquire.
    let tramp: fn(*const (), usize) = unsafe { std::mem::transmute(tramp_addr) };
    // Contiguous block for this participant. The first `num % n` participants
    // take one extra chunk, so the assignment covers `0..num` exactly once with
    // a size spread of at most 1 — the property `completed == num_chunks`
    // relies on.
    let base = num / n_participants;
    let rem = num % n_participants;
    let start = participant_id * base + participant_id.min(rem);
    let end = start + base + usize::from(participant_id < rem);

    let mut c = start;
    while c < end {
        // `IN_BODY` makes a reentrant `for_each_chunk` (a body that dispatches)
        // fall back to serial instead of deadlocking. run_chunks is only
        // entered at top level, so the prior value is always false.
        IN_BODY.with(|b| b.set(true));
        // Catch a panicking body so we still `completed.fetch_add` below: a
        // worker that unwound out of the loop would never count its chunk and
        // the dispatcher would spin the barrier forever. The first payload is
        // kept and re-raised on the dispatcher.
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| tramp(data, c)));
        IN_BODY.with(|b| b.set(false));
        if let Err(payload) = r {
            if !shared.panicked.swap(true, Ordering::AcqRel) {
                *shared
                    .panic_payload
                    .lock()
                    .unwrap_or_else(|e| e.into_inner()) = Some(payload);
            }
        }
        shared.completed.fetch_add(1, Ordering::Release);
        c += 1;
    }
}

impl SpinPool {
    /// Build a pool with `n_threads` total participants (spawns `n_threads-1`
    /// persistent workers; the dispatcher is the n-th). `n_threads <= 1` makes
    /// [`for_each_chunk`] run inline with no workers.
    pub fn new(n_threads: usize) -> Self {
        let n_threads = n_threads.max(1);
        let shared = Arc::new(Shared {
            epoch: AtomicU64::new(0),
            slot_seq: AtomicU64::new(0),
            task_gen: AtomicU64::new(0),
            completed: AtomicUsize::new(0),
            num_chunks: AtomicUsize::new(0),
            data: AtomicPtr::new(std::ptr::null_mut()),
            tramp: AtomicUsize::new(0),
            shutdown: AtomicBool::new(false),
            panicked: AtomicBool::new(false),
            panic_payload: Mutex::new(None),
        });
        let workers = (1..n_threads)
            .map(|i| {
                let shared = Arc::clone(&shared);
                thread::Builder::new()
                    .name(format!("larql-spin-{i}"))
                    // Participant `i` of `n_threads`; the dispatcher is 0.
                    .spawn(move || worker_loop(shared, i, n_threads))
                    .expect("spawn spin-pool worker")
            })
            .collect();
        Self {
            shared,
            workers,
            n_threads,
            dispatch_lock: Mutex::new(()),
        }
    }

    /// Number of participating threads (workers + dispatcher).
    pub fn num_threads(&self) -> usize {
        self.n_threads
    }

    /// Run `body(chunk_idx)` for every `chunk_idx in 0..num_chunks`, across the
    /// pool, blocking until all chunks complete.
    ///
    /// `body` must only touch data disjoint per `chunk_idx` — exactly the
    /// contract of `slice::par_chunks_mut().enumerate().for_each()`, which this
    /// replaces. The calling thread participates, so this is *not* reentrant:
    /// `body` must not itself call `for_each_chunk` on the same pool.
    pub fn for_each_chunk<F: Fn(usize) + Sync>(&self, num_chunks: usize, body: F) {
        if num_chunks == 0 {
            return;
        }
        // No workers, or already inside a dispatched body (reentrant): run the
        // chunks serially on this thread. The reentrancy fallback also avoids
        // deadlocking against `dispatch_lock` if a body ever dispatches.
        if self.workers.is_empty() || IN_BODY.with(|b| b.get()) {
            for c in 0..num_chunks {
                body(c);
            }
            return;
        }
        // Serialize dispatchers so the shared epoch/cursor state is consistent;
        // uncontended in the single-driver decode loop.
        let _dispatch = self.dispatch_lock.lock().unwrap_or_else(|e| e.into_inner());
        let shared = &self.shared;
        // Publish the task inside the slot seqlock, then release it to workers
        // via the epoch bump. The seqlock + `task_gen` let a straggler from the
        // previous dispatch (empty block there, so the barrier didn't wait for
        // it) detect that the slot no longer belongs to the epoch it observed.
        let gen = shared.epoch.load(Ordering::Relaxed) + 1;
        shared.slot_seq.fetch_add(1, Ordering::AcqRel); // odd: slot unstable
        shared
            .data
            .store(&body as *const F as *mut (), Ordering::Relaxed);
        shared
            .tramp
            .store(trampoline::<F> as *const () as usize, Ordering::Relaxed);
        shared.num_chunks.store(num_chunks, Ordering::Relaxed);
        shared.completed.store(0, Ordering::Relaxed);
        shared.panicked.store(false, Ordering::Relaxed);
        shared.task_gen.store(gen, Ordering::Relaxed);
        shared.slot_seq.fetch_add(1, Ordering::Release); // even: slot stable
        shared.epoch.store(gen, Ordering::Release);

        // Wake any worker that parked during an idle gap so the barrier never
        // stalls ~park_timeout waiting on its assigned block. Unparking a
        // still-spinning worker just sets its token (harmless). During tight
        // back-to-back decode dispatches workers stay spinning and this is a
        // no-op fast path.
        for w in &self.workers {
            w.thread().unpark();
        }

        // The dispatcher participates as participant 0. Its expected
        // generation is the one it just published, so it always runs.
        run_chunks(shared, 0, self.n_threads, gen);

        // Completion barrier: wait until every chunk has finished. With static
        // block ownership, `completed == num_chunks` means every trampoline
        // call has returned (panics still count, see run_chunks), so it is safe
        // to let `body` drop as this returns.
        //
        // This backs off spin → yield, for the same reason `worker_loop` does.
        // It used to be an unbounded `spin_loop()`, which **livelocks under
        // oversubscription**: a dispatcher burns a whole core spinning on
        // `completed` while the worker that owes the last chunk cannot get
        // scheduled. With more runnable threads than cores — several concurrent
        // dispatchers, this pool's persistent workers, and a `cargo test`
        // harness running other tests alongside — every spinner holds its core
        // for a full quantum and forward progress stalls for *seconds*. That is
        // the `stress_concurrent_realistic_decode_shape_no_corruption` hang
        // (reproduced 1-in-12 at `--test-threads=8`, worse in a full workspace
        // run, worst under coverage instrumentation, and invisible standalone
        // because a quiet machine has cores to spare).
        //
        // The backoff `worker_loop` received when the pool went default-on was
        // only ever applied there; this barrier was left as a pure spin.
        //
        // It escalates to `yield_now` and **stays** there — it must never park.
        // Nothing unparks a dispatcher: workers signal completion through the
        // `completed` counter alone, so a parked dispatcher would sleep until
        // its own timeout with no one to wake it. Yielding is what cedes the
        // core to the worker we are waiting on.
        //
        // The fast path is unchanged: chunks complete in microseconds, so a
        // decode-shaped dispatch never leaves the spin phase, and the measured
        // +28% is preserved by construction.
        let mut spins = 0u32;
        while shared.completed.load(Ordering::Acquire) < num_chunks {
            spins += 1;
            if spins < SPIN_HOT {
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }

        // Re-raise the first chunk panic on this (the dispatching) thread, so a
        // panicking body propagates to the caller like a serial loop or rayon —
        // instead of being swallowed on a worker. Drop the dispatch guard first
        // so the pool stays usable after the unwind.
        if shared.panicked.load(Ordering::Acquire) {
            let payload = shared
                .panic_payload
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .take();
            drop(_dispatch);
            if let Some(payload) = payload {
                std::panic::resume_unwind(payload);
            }
        }
    }
}

impl Drop for SpinPool {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, Ordering::Relaxed);
        // Bump epoch so any spinning worker breaks out and re-checks shutdown,
        // and unpark so a parked worker doesn't sleep out its PARK_BACKSTOP
        // before noticing. The store/bump-then-unpark order matters: a worker
        // that checked the epoch just before the bump has its park token set
        // and returns immediately.
        self.shared.epoch.fetch_add(1, Ordering::Release);
        for w in &self.workers {
            w.thread().unpark();
        }
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

/// Number of performance ("P") cores, when the OS exposes a heterogeneous
/// topology. `None` on homogeneous machines and anywhere the query is
/// unavailable — callers then fall back to the plain thread count.
///
/// Apple silicon reports `hw.nperflevels > 1` with level 0 = performance and
/// level 1 = efficiency (M3 Max: 12 P + 4 E).
#[cfg(target_os = "macos")]
fn performance_cores() -> Option<usize> {
    fn sysctl_usize(name: &std::ffi::CStr) -> Option<usize> {
        let mut out: i32 = 0;
        let mut len = std::mem::size_of::<i32>();
        // SAFETY: `name` is a NUL-terminated C string; `out`/`len` are a
        // correctly sized i32 destination and its length, as sysctlbyname
        // requires. It writes at most `len` bytes and updates `len`.
        let rc = unsafe {
            libc::sysctlbyname(
                name.as_ptr(),
                (&mut out as *mut i32).cast(),
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        (rc == 0 && out > 0).then_some(out as usize)
    }

    // Only meaningful when there really are multiple performance levels.
    if sysctl_usize(c"hw.nperflevels")? < 2 {
        return None;
    }
    sysctl_usize(c"hw.perflevel0.logicalcpu")
}

#[cfg(not(target_os = "macos"))]
fn performance_cores() -> Option<usize> {
    None
}

/// Process-wide pool, built on first use.
///
/// Sized to the active rayon thread count, **capped at the performance-core
/// count on heterogeneous CPUs**.
///
/// The cap is load-bearing, not a tuning preference. This pool partitions
/// chunks *statically*, so an efficiency core is handed the same share as a
/// performance core and the completion barrier waits on the slowest
/// participant; rayon's work-stealing rebalances instead, which is why the
/// pathology is specific to this pool. Measured on M3 Max (12 P + 4 E), AC
/// power, `benches/q4k_q8k_matvec.rs::bench_mt_production` at 65536×2816 with
/// `--measurement-time 20`:
///
/// | participants | GiB/s |
/// |---|---|
/// | 16 (12 P + 4 E) | 33.0 |
/// | 12 | 124.0 |
/// | 8 | 126.7 |
///
/// End-to-end on `gemma4-26b-a4b-q4k` (`--cpu -n 50`): **10.6 tok/s at 16
/// participants vs 37.3 at 8** — a 3.5× collapse. Straggler cost scales with
/// the work per dispatch, so small per-layer matvecs still won (+45–113% vs
/// rayon) and only the lm_head-class shape fell over; that is why this hid.
///
/// It hid for a second reason worth recording: `configure_rayon_threads` is
/// called **only** from the CLI's bench path, where it picks 8 on Apple
/// silicon. `larql run` / `larql serve` configure nothing and inherit rayon's
/// default (`available_parallelism()` = 16). So the benchmark harness was
/// structurally unable to observe a bug that every non-bench path hit, and the
/// historical spin-pool numbers (+28%, ~35 tok/s) were all taken at 8.
///
/// Capping here rather than in the CLI fixes it for every consumer —
/// larql-server, embedders, tests — not just the one binary that happened to
/// set a thread count. Bandwidth-bound decode loses nothing: attainable read
/// bandwidth saturates at **two** threads (`examples/membw_probe.rs`), so the
/// E-cores were contributing ~nothing even before they became stragglers.
///
/// See `docs/diagnoses/memory-bandwidth-roofline.md`.
pub fn global() -> &'static SpinPool {
    static POOL: OnceLock<SpinPool> = OnceLock::new();
    POOL.get_or_init(|| {
        let rayon_threads = rayon::current_num_threads().max(1);
        // An explicit smaller thread count still wins — this only removes the
        // E-cores from an otherwise-unconstrained pool.
        let n = performance_cores().map_or(rayon_threads, |p| rayon_threads.min(p.max(1)));
        SpinPool::new(n)
    })
}

/// Whether the decode hot path routes parallel sections through the spin pool
/// instead of rayon. **On by default** — the spin-then-yield backoff makes it
/// safe on shared/contended machines — set `LARQL_SPIN_POOL=0` to force the
/// rayon path (e.g. for an A/B or a heavily oversubscribed host). Either path
/// is numerically identical; only *which threads run which chunks* differs.
pub fn enabled() -> bool {
    crate::options::spin_pool_enabled()
}

/// Drop-in for `out.par_chunks_mut(chunk).enumerate().for_each(|(ci, c)| body(ci, c))`
/// that routes through the spin pool when [`enabled`], else stays on rayon.
///
/// `body(chunk_idx, chunk)` receives each disjoint `chunk`-sized (last shorter)
/// slice of `out` and its index — identical semantics either way, so the
/// arithmetic is unchanged; only *which thread runs which chunk* differs.
pub fn par_chunks_mut<T, F>(out: &mut [T], chunk: usize, body: F)
where
    T: Send,
    F: Fn(usize, &mut [T]) + Sync + Send,
{
    if chunk == 0 || out.is_empty() {
        return;
    }
    if enabled() {
        let total = out.len();
        let n = total.div_ceil(chunk);
        let base = out.as_mut_ptr() as usize;
        global().for_each_chunk(n, |ci| {
            let start = ci * chunk;
            // `start < total` holds by construction for `ci < n` - but this
            // feeds a raw `.add(start)` below with no bounds check of its
            // own, so `saturating_sub` (not `-`) turns any violation of that
            // invariant into an inert zero-length chunk instead of a wrapped
            // `usize` producing a wild slice length (release builds have no
            // overflow-checks).
            let len = chunk.min(total.saturating_sub(start));
            if len == 0 {
                return;
            }
            // SAFETY: chunk index `ci` owns the disjoint range
            // `[start, start+len)` of `out`; no two chunks overlap, and the
            // dispatch barrier keeps `out` borrowed for the whole call.
            let s = unsafe { std::slice::from_raw_parts_mut((base as *mut T).add(start), len) };
            body(ci, s);
        });
    } else {
        use rayon::prelude::*;
        out.par_chunks_mut(chunk)
            .enumerate()
            .for_each(|(ci, c)| body(ci, c));
    }
}

/// Two-output sibling of [`par_chunks_mut`] for kernels that write `a` and `b`
/// at the same row index (e.g. the fused gate/up dual matvec). `a` and `b` must
/// have the same length; `body(chunk_idx, a_chunk, b_chunk)` gets the matching
/// disjoint slices.
pub fn par_chunks_mut2<T, F>(a: &mut [T], b: &mut [T], chunk: usize, body: F)
where
    T: Send,
    F: Fn(usize, &mut [T], &mut [T]) + Sync + Send,
{
    debug_assert_eq!(a.len(), b.len(), "par_chunks_mut2 needs equal-length a/b");
    if chunk == 0 || a.is_empty() {
        return;
    }
    if enabled() {
        let total = a.len();
        let n = total.div_ceil(chunk);
        let base_a = a.as_mut_ptr() as usize;
        let base_b = b.as_mut_ptr() as usize;
        global().for_each_chunk(n, |ci| {
            let start = ci * chunk;
            // See the matching comment in `par_chunks_mut` above.
            let len = chunk.min(total.saturating_sub(start));
            if len == 0 {
                return;
            }
            // SAFETY: disjoint per-chunk ranges of `a` and `b` (separate
            // buffers); barrier keeps both borrowed for the call.
            let sa = unsafe { std::slice::from_raw_parts_mut((base_a as *mut T).add(start), len) };
            let sb = unsafe { std::slice::from_raw_parts_mut((base_b as *mut T).add(start), len) };
            body(ci, sa, sb);
        });
    } else {
        use rayon::prelude::*;
        a.par_chunks_mut(chunk)
            .zip(b.par_chunks_mut(chunk))
            .enumerate()
            .for_each(|(ci, (ca, cb))| body(ci, ca, cb));
    }
}

#[cfg(test)]
mod tests;
