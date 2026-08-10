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

// The whole persistent-thread-pool machinery below (Shared, SpinPool,
// trampoline, worker_loop, run_chunks, the two impl blocks,
// performance_cores, global) is native-only: wasm32v1-none has no OS
// threads at all, so there is nothing to pool. par_chunks_mut/
// par_chunks_mut2 (the actual call sites everything else in this crate
// uses) get a wasm32-only sequential fallback further down instead --
// same call signature, no caller changes needed, and this file's own
// doc comments already establish sequential vs. parallel execution is
// numerically identical ("only which thread runs which chunk
// differs"). enabled() becomes an unconditional `false` on wasm32
// (there is no pool to route through). Two call sites elsewhere in
// this crate (attention/decode/gqa_step.rs, cpu/ops/moe/forward.rs)
// bypass this wrapper and call global()/enabled() directly; those need
// their own explicit wasm32 branch, fixed separately.
#[cfg(not(target_arch = "wasm32"))]
use std::cell::Cell;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, AtomicUsize, Ordering};
#[cfg(not(target_arch = "wasm32"))]
use std::sync::{Arc, Mutex, OnceLock};
#[cfg(not(target_arch = "wasm32"))]
use std::thread;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Duration;

#[cfg(not(target_arch = "wasm32"))]
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
#[cfg(not(target_arch = "wasm32"))]
const SPIN_HOT: u32 = 256_000;
#[cfg(not(target_arch = "wasm32"))]
const YIELD_UNTIL: u32 = SPIN_HOT + 64;
#[cfg(not(target_arch = "wasm32"))]
const PARK_BACKSTOP: Duration = Duration::from_secs(1);

/// Cross-thread dispatch state. The `epoch` release store wakes workers; the
/// `slot_seq` seqlock + `task_gen` stamp make the task fields safe to read,
/// because epoch-wakeup alone does not prove the slot still belongs to the
/// observed dispatch (see `slot_seq`).
#[cfg(not(target_arch = "wasm32"))]
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
#[cfg(not(target_arch = "wasm32"))]
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
#[cfg(not(target_arch = "wasm32"))]
fn trampoline<F: Fn(usize) + Sync>(data: *const (), chunk: usize) {
    // SAFETY: see fn docs — `data` is `&F` published under the epoch fence and
    // outlives every call within the dispatch.
    let f = unsafe { &*(data as *const F) };
    f(chunk);
}

#[cfg(not(target_arch = "wasm32"))]
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
#[cfg(not(target_arch = "wasm32"))]
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

#[cfg(not(target_arch = "wasm32"))]
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

#[cfg(not(target_arch = "wasm32"))]
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

// Only called from the now-native-only global() below; gated the same
// way to avoid an unused-function warning on wasm32 rather than
// leaving it as accidentally-portable dead code.
#[cfg(all(not(target_os = "macos"), not(target_arch = "wasm32")))]
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
#[cfg(not(target_arch = "wasm32"))]
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
///
/// Unconditionally `false` on wasm32: there is no pool to route through
/// (no OS threads at all), so every caller takes the sequential path in
/// [`par_chunks_mut`]/[`par_chunks_mut2`] below.
#[cfg(target_arch = "wasm32")]
pub fn enabled() -> bool {
    false
}

#[cfg(not(target_arch = "wasm32"))]
pub fn enabled() -> bool {
    crate::options::spin_pool_enabled()
}

/// Drop-in for `out.par_chunks_mut(chunk).enumerate().for_each(|(ci, c)| body(ci, c))`
/// that routes through the spin pool when [`enabled`], else stays on rayon.
///
/// `body(chunk_idx, chunk)` receives each disjoint `chunk`-sized (last shorter)
/// slice of `out` and its index — identical semantics either way, so the
/// arithmetic is unchanged; only *which thread runs which chunk* differs.
///
/// wasm32v1-none has no OS threads at all, so there is neither a spin
/// pool nor rayon to route through there -- this runs every chunk
/// sequentially on the calling "thread" instead, via the same safe
/// `slice::chunks_mut` the doc-comment example above describes. The
/// native raw-pointer/unsafe split-borrow trick exists specifically to
/// let *multiple threads* mutably alias disjoint parts of one slice at
/// once (something the borrow checker can't verify statically); with
/// no concurrency at all, a single safe mutable-borrow loop produces
/// the identical chunk/index assignment with no unsafe needed.
#[cfg(target_arch = "wasm32")]
pub fn par_chunks_mut<T, F>(out: &mut [T], chunk: usize, body: F)
where
    T: Send,
    F: Fn(usize, &mut [T]) + Sync + Send,
{
    if chunk == 0 || out.is_empty() {
        return;
    }
    for (ci, c) in out.chunks_mut(chunk).enumerate() {
        body(ci, c);
    }
}

#[cfg(not(target_arch = "wasm32"))]
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
///
/// See [`par_chunks_mut`]'s doc comment for the wasm32 sequential-fallback
/// rationale -- identical shape here, `.zip()`-ed over both slices.
#[cfg(target_arch = "wasm32")]
pub fn par_chunks_mut2<T, F>(a: &mut [T], b: &mut [T], chunk: usize, body: F)
where
    T: Send,
    F: Fn(usize, &mut [T], &mut [T]) + Sync + Send,
{
    debug_assert_eq!(a.len(), b.len(), "par_chunks_mut2 needs equal-length a/b");
    if chunk == 0 || a.is_empty() {
        return;
    }
    for (ci, (ca, cb)) in a.chunks_mut(chunk).zip(b.chunks_mut(chunk)).enumerate() {
        body(ci, ca, cb);
    }
}

#[cfg(not(target_arch = "wasm32"))]
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
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU32;

    // ── Wrapper scheduling parity ───────────────────────────────────────
    //
    // `par_chunks_mut`/`par_chunks_mut2` each branch on `enabled()`, so a CI
    // run with one ambient setting leaves the other arm dead — and CI sets
    // `LARQL_SPIN_POOL=0` (see .github/workflows/larql-compute.yml) because
    // a spin barrier has no free core to spin on there. These drive both
    // arms explicitly, so coverage and the equivalence claim both hold
    // regardless of the ambient flag.
    //
    // The override is thread-local rather than `std::env::set_var`, which
    // segfaults under parallel `getenv` (see `options::ENV_OVERRIDES`).

    /// Run `f` with the spin pool forced on, then forced off, returning
    /// both results for comparison.
    fn both_arms<T, F: Fn() -> T>(f: F) -> (T, T) {
        use crate::options::{clear_fast_path_overrides, set_fast_path_override, ENV_SPIN_POOL};
        set_fast_path_override(ENV_SPIN_POOL, true);
        let spin = f();
        set_fast_path_override(ENV_SPIN_POOL, false);
        let rayon = f();
        clear_fast_path_overrides();
        (spin, rayon)
    }

    #[test]
    fn par_chunks_mut_agrees_across_both_schedulers() {
        let run = || {
            let mut out = vec![0u64; 1000];
            par_chunks_mut(&mut out, 7, |ci, c| {
                for (i, slot) in c.iter_mut().enumerate() {
                    *slot = (ci as u64) * 1000 + i as u64;
                }
            });
            out
        };
        let (spin, rayon) = both_arms(run);
        assert_eq!(spin, rayon, "scheduler changed the result");
        // Ragged tail: 1000 is not a multiple of 7, so the last chunk is
        // short — the place an off-by-one would differ between arms.
        assert_eq!(spin.len(), 1000);
    }

    #[test]
    fn par_chunks_mut2_agrees_across_both_schedulers() {
        let run = || {
            let (mut a, mut b) = (vec![0u64; 517], vec![0u64; 517]);
            par_chunks_mut2(&mut a, &mut b, 32, |ci, ca, cb| {
                for (i, (x, y)) in ca.iter_mut().zip(cb.iter_mut()).enumerate() {
                    *x = (ci as u64) << 20 | i as u64;
                    *y = !*x;
                }
            });
            (a, b)
        };
        let (spin, rayon) = both_arms(run);
        assert_eq!(spin, rayon, "scheduler changed the result");
        // The two outputs must stay index-aligned, which is the whole
        // reason this wrapper exists.
        assert!(spin.0.iter().zip(&spin.1).all(|(x, y)| *y == !*x));
    }

    #[test]
    fn both_wrappers_no_op_on_empty_input_either_way() {
        let (spin, rayon) = both_arms(|| {
            let mut out: Vec<u64> = Vec::new();
            par_chunks_mut(&mut out, 8, |_, _| unreachable!("no chunks to run"));
            let (mut a, mut b): (Vec<u64>, Vec<u64>) = (Vec::new(), Vec::new());
            par_chunks_mut2(&mut a, &mut b, 8, |_, _, _| unreachable!("no chunks"));
            (out, a, b)
        });
        assert_eq!(spin, rayon);
    }

    #[test]
    fn a_zero_chunk_size_is_a_no_op_either_way() {
        let (spin, rayon) = both_arms(|| {
            let mut out = vec![9u64; 4];
            par_chunks_mut(&mut out, 0, |_, _| unreachable!("chunk==0 must not run"));
            out
        });
        assert_eq!(spin, rayon);
        assert_eq!(spin, vec![9u64; 4], "input must be untouched");
    }

    #[test]
    fn runs_every_chunk_exactly_once() {
        let pool = SpinPool::new(4);
        let hits: Vec<AtomicU32> = (0..1000).map(|_| AtomicU32::new(0)).collect();
        pool.for_each_chunk(hits.len(), |c| {
            hits[c].fetch_add(1, Ordering::Relaxed);
        });
        for (i, h) in hits.iter().enumerate() {
            assert_eq!(h.load(Ordering::Relaxed), 1, "chunk {i} ran != once");
        }
    }

    /// Each participant must get ONE unbroken run of chunk indices.
    ///
    /// `runs_every_chunk_exactly_once` pins the correctness invariant the
    /// barrier needs, and it passes under round-robin striding too — so it
    /// cannot catch a revert to `c += n_participants`. Contiguity is the
    /// *performance* invariant: chunks map to contiguous byte ranges of the
    /// weight slab, and a strided owner defeats the sequential prefetcher
    /// (measured 1.87× slower at the 415 MB lm_head-class shape). This test is
    /// what makes that regression loud instead of silent.
    ///
    /// Uneven counts are included deliberately: the `rem` participants take one
    /// extra chunk, which is where an off-by-one would overlap or gap two
    /// participants' ranges.
    #[test]
    fn each_participant_runs_a_contiguous_block() {
        for (n_threads, num_chunks) in [(4usize, 1000usize), (4, 1001), (4, 3), (8, 8), (3, 100)] {
            let pool = SpinPool::new(n_threads);
            let seen: std::sync::Mutex<Vec<(std::thread::ThreadId, usize)>> =
                std::sync::Mutex::new(Vec::new());
            pool.for_each_chunk(num_chunks, |c| {
                seen.lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .push((std::thread::current().id(), c));
            });

            let mut by_thread: std::collections::HashMap<std::thread::ThreadId, Vec<usize>> =
                std::collections::HashMap::new();
            for (tid, c) in seen.into_inner().unwrap_or_else(|e| e.into_inner()) {
                by_thread.entry(tid).or_default().push(c);
            }

            let mut covered = 0usize;
            for (tid, mut chunks) in by_thread {
                chunks.sort_unstable();
                let (first, last) = (chunks[0], chunks[chunks.len() - 1]);
                assert_eq!(
                    last - first + 1,
                    chunks.len(),
                    "participant {tid:?} got a non-contiguous block {chunks:?} \
                     (n_threads={n_threads}, num_chunks={num_chunks}) — this is the \
                     strided-ownership regression; see run_chunks"
                );
                covered += chunks.len();
            }
            assert_eq!(
                covered, num_chunks,
                "blocks must cover 0..{num_chunks} exactly (n_threads={n_threads})"
            );
        }
    }

    /// The global pool must never include efficiency cores.
    ///
    /// Static partitioning makes the barrier wait on the slowest participant,
    /// so one E-core in the pool cost 3.5× end-to-end on the 26B. This asserts
    /// the cap rather than the mechanism, because the mechanism only shows up
    /// under a large-shape benchmark on a heterogeneous box — the exact
    /// combination that let the regression ship unnoticed.
    #[test]
    fn global_pool_excludes_efficiency_cores() {
        let n = global().num_threads();
        assert!(n >= 1, "pool must have at least one participant");
        if let Some(p) = performance_cores() {
            assert!(
                n <= p,
                "global pool has {n} participants but only {p} performance cores — \
                 an efficiency core in a statically-partitioned pool stalls the barrier \
                 (see global() docs)"
            );
        }
        assert!(
            n <= rayon::current_num_threads().max(1),
            "pool must never exceed an explicitly configured thread count"
        );
    }

    #[test]
    fn disjoint_mut_writes_match_serial() {
        // The production pattern: each chunk writes its disjoint row range of a
        // shared output buffer via a raw pointer (caller guarantees disjoint).
        let pool = SpinPool::new(4);
        let rows = 517usize;
        let chunk = 32usize;
        let n_chunks = rows.div_ceil(chunk);
        let mut out = vec![0u64; rows];
        let ptr = out.as_mut_ptr() as usize;
        pool.for_each_chunk(n_chunks, |ci| {
            let start = ci * chunk;
            let end = (start + chunk).min(rows);
            for r in start..end {
                // SAFETY: chunks are disjoint row ranges of `out`.
                unsafe { *(ptr as *mut u64).add(r) = (r as u64) * 3 + 1 };
            }
        });
        for (r, v) in out.iter().enumerate() {
            assert_eq!(*v, (r as u64) * 3 + 1);
        }
    }

    #[test]
    fn parallel_sum_matches_serial() {
        let pool = SpinPool::new(8);
        let n = 100_000usize;
        let partials: Vec<AtomicU64> = (0..64).map(|_| AtomicU64::new(0)).collect();
        let chunk = n.div_ceil(64);
        pool.for_each_chunk(64, |ci| {
            let start = ci * chunk;
            let end = (start + chunk).min(n);
            let s: u64 = (start as u64..end as u64).sum();
            partials[ci].store(s, Ordering::Relaxed);
        });
        let got: u64 = partials.iter().map(|a| a.load(Ordering::Relaxed)).sum();
        let want: u64 = (0..n as u64).sum();
        assert_eq!(got, want);
    }

    #[test]
    fn zero_chunks_is_noop() {
        let pool = SpinPool::new(4);
        pool.for_each_chunk(0, |_| panic!("must not run"));
    }

    #[test]
    fn single_thread_runs_inline() {
        let pool = SpinPool::new(1);
        let hits: Vec<AtomicU32> = (0..50).map(|_| AtomicU32::new(0)).collect();
        pool.for_each_chunk(hits.len(), |c| {
            hits[c].fetch_add(1, Ordering::Relaxed);
        });
        assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1));
    }

    #[test]
    fn chunk_panic_propagates_and_pool_stays_usable() {
        // A panicking body must (a) NOT hang the barrier (a dead worker would
        // never count its chunk → dispatcher spins forever) and (b) propagate
        // the panic to the dispatcher. Chunk 37 lands on a worker (37 % 4 != 0),
        // exercising the worker-side catch, not just the dispatcher's own.
        let pool = SpinPool::new(4);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pool.for_each_chunk(50, |c| {
                if c == 37 {
                    panic!("boom at chunk {c}");
                }
            });
        }));
        assert!(
            result.is_err(),
            "a panicking chunk body must propagate to the dispatcher"
        );
        // The pool must still work after a panic (not poisoned / not hung).
        let hits: Vec<AtomicU32> = (0..20).map(|_| AtomicU32::new(0)).collect();
        pool.for_each_chunk(hits.len(), |c| {
            hits[c].fetch_add(1, Ordering::Relaxed);
        });
        assert!(
            hits.iter().all(|h| h.load(Ordering::Relaxed) == 1),
            "pool must stay usable after a chunk panic"
        );
    }

    #[test]
    fn concurrent_dispatchers_stay_consistent() {
        // Multiple driver threads dispatching on one shared pool (the
        // `--concurrent N` / multi-threaded-test shape). The dispatch lock
        // serializes them; each dispatch must still complete correctly.
        let pool = SpinPool::new(4);
        std::thread::scope(|s| {
            for _ in 0..3 {
                s.spawn(|| {
                    for round in 1..=50u64 {
                        let acc: Vec<AtomicU64> = (0..20).map(|_| AtomicU64::new(0)).collect();
                        pool.for_each_chunk(20, |c| {
                            acc[c].store(round * (c as u64 + 1), Ordering::Relaxed);
                        });
                        for (c, a) in acc.iter().enumerate() {
                            assert_eq!(a.load(Ordering::Relaxed), round * (c as u64 + 1));
                        }
                    }
                });
            }
        });
    }

    /// Cross-dispatch read-after-write — the real decode pipeline shape
    /// (dispatch A writes a buffer; the *next* dispatch B reads it and writes a
    /// derived buffer). Exercises the visibility the disjoint-write tests don't:
    /// workers running dispatch B must observe ALL of dispatch A's writes (the
    /// `barrier_A.Acquire → epoch_B.Release → worker_B.Acquire` chain). The pool
    /// is oversubscribed (more workers than cores) so the barrier routinely waits
    /// on a descheduled worker. Kept fast (a few hundred rounds) — under EXTREME
    /// oversubscription (2× burners, 4000 rounds) this and the disjoint-write
    /// path stayed correct, so this is a regression guard, not a repro.
    #[test]
    fn stress_cross_dispatch_read_after_write() {
        let cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(8);
        // Oversubscribe the pool itself (more workers than cores) so the barrier
        // routinely waits on a descheduled worker.
        let pool = SpinPool::new((cores + 2).max(4));
        let n = 61usize; // chunks; not a multiple of the thread count
        let mut a = vec![0u64; n];
        let mut b = vec![0u64; n];
        for round in 1..=400u64 {
            // Dispatch A: fill `a` with a round-derived pattern.
            let pa = a.as_mut_ptr() as usize;
            pool.for_each_chunk(n, |c| {
                // SAFETY: chunk c owns element c.
                unsafe { *(pa as *mut u64).add(c) = round.wrapping_mul(c as u64 + 1) | 1 };
            });
            // Dispatch B: read `a`, write `b = f(a)`. If B's workers don't see
            // all of A's writes, `b[c]` is wrong (or derived from a stale 0).
            let pa_r = a.as_ptr() as usize;
            let pb = b.as_mut_ptr() as usize;
            pool.for_each_chunk(n, |c| {
                // SAFETY: read element c (written by A's chunk c), write b[c].
                let av = unsafe { *(pa_r as *const u64).add(c) };
                unsafe { *(pb as *mut u64).add(c) = av.wrapping_mul(31).wrapping_add(7) };
            });
            for c in 0..n {
                let want_a = round.wrapping_mul(c as u64 + 1) | 1;
                assert_eq!(a[c], want_a, "round {round} chunk {c}: A wrong");
                assert_eq!(
                    b[c],
                    want_a.wrapping_mul(31).wrapping_add(7),
                    "round {round} chunk {c}: B read a stale/partial A"
                );
            }
        }
    }

    #[test]
    fn back_to_back_dispatches_reuse_workers() {
        // Exercises the epoch path: many tiny dispatches in a row (the decode
        // loop shape) must each complete fully.
        let pool = SpinPool::new(4);
        for round in 1..=200u64 {
            let acc: Vec<AtomicU64> = (0..16).map(|_| AtomicU64::new(0)).collect();
            pool.for_each_chunk(16, |c| {
                acc[c].store(round * (c as u64 + 1), Ordering::Relaxed);
            });
            for (c, a) in acc.iter().enumerate() {
                assert_eq!(a.load(Ordering::Relaxed), round * (c as u64 + 1));
            }
        }
    }

    // ── SIGSEGV reproduction attempt (three crash reports, all localizing
    // inside `par_chunks_mut`/`run_chunks`, fault addresses matching real
    // model dimensions or `0x1`) ────────────────────────────────────────────
    //
    // Gated on `heavy_tests`: these run a production-scale decode simulation
    // (34 layers × hundreds of tokens, deliberately oversubscribed) and take
    // minutes under coverage instrumentation on a 2-core CI runner — a spin
    // barrier has no spare core to spin on there, so CI pays the worst case
    // of exactly the contention these tests exist to provoke. Run locally via
    // `make larql-compute-test-integration`.
    #[cfg(feature = "heavy_tests")]
    mod stress_decode_shape {
        use super::*;

        const RD_HIDDEN: usize = 2560;
        const RD_INTER: usize = 10240;
        const RD_Q_DIM: usize = 2048;
        const RD_KV_DIM: usize = 1024;
        const RD_ROWS_CHUNK: usize = 32;
        const RD_ELEM_CHUNK: usize = 256;

        fn rd_f32_encode(caller: u64, round: u64, tag: u64, idx: usize) -> f32 {
            // Cheap, collision-resistant-enough encoding of (caller, round, tag,
            // idx) into an f32 so a read of the wrong slot/epoch/caller's data is
            // caught as a mismatch rather than silently looking plausible.
            ((caller.wrapping_mul(998_244_353)
                ^ round.wrapping_mul(1_000_003)
                ^ tag.wrapping_mul(97)
                ^ idx as u64)
                % 1_000_000) as f32
        }
        fn rd_fill_chunk(
            chunk: &mut [f32],
            global_start: usize,
            caller: u64,
            round: u64,
            tag: u64,
        ) {
            for (local_i, v) in chunk.iter_mut().enumerate() {
                // Cheap but non-trivial busy-work per element (~tens of ns) so a
                // chunk's dispatched duration is closer to the real matvec
                // kernel's (many SDOTs per row) than a near-instant synthetic
                // write - in case the bug is timing/duration-dependent (spin ->
                // yield -> park transitions calibrated around real chunk cost).
                let mut acc = 0.0f32;
                for j in 0..64u32 {
                    acc = acc * 1.0000001 + (j as f32).sin();
                }
                *v = rd_f32_encode(caller, round, tag, global_start + local_i) + acc * 0.0;
            }
        }
        fn rd_check(buf: &[f32], caller: u64, round: u64, tag: u64) {
            for (i, &v) in buf.iter().enumerate() {
                let want = rd_f32_encode(caller, round, tag, i);
                assert_eq!(
                    v.to_bits(),
                    want.to_bits(),
                    "caller {caller} round {round} tag {tag} idx {i}: corrupted \
                 (got {v}, want {want}) - buffer len {}",
                    buf.len()
                );
            }
        }

        /// One simulated decode step's worth of dispatches against `par_chunks_mut`
        /// - the actual public entry point production uses, with the REAL
        ///   gemma-3-4b-it row counts (q_dim=2048, kv_dim=1024, hidden=2560,
        ///   intermediate=10240 - all exact multiples of their chunk sizes, same as
        ///   production, so this can't exercise the underflow this file already
        ///   hardened against; it's targeting a different bug). `caller` tags every
        ///   value so concurrent callers (simulating overlapping requests) can tell
        ///   their own data apart from a caller they got mixed up with.
        fn rd_run(caller: u64, rounds: u64, layers: u64) {
            let mut q = vec![0.0f32; RD_Q_DIM];
            let mut k = vec![0.0f32; RD_KV_DIM];
            let mut v = vec![0.0f32; RD_KV_DIM];
            let mut o = vec![0.0f32; RD_HIDDEN];
            let mut gate = vec![0.0f32; RD_INTER];
            let mut up = vec![0.0f32; RD_INTER];
            let mut activated = vec![0.0f32; RD_INTER];
            let mut down = vec![0.0f32; RD_HIDDEN];

            for round in 0..rounds {
                for layer in 0..layers {
                    let tag = round * layers + layer;
                    for (buf, sub) in [
                        (&mut q, 0u64),
                        (&mut k, 1),
                        (&mut v, 2),
                        (&mut o, 3),
                        (&mut gate, 4),
                        (&mut up, 5),
                        (&mut down, 7),
                    ] {
                        let t = tag.wrapping_mul(8) + sub;
                        par_chunks_mut(buf, RD_ROWS_CHUNK, |ci, chunk| {
                            rd_fill_chunk(chunk, ci * RD_ROWS_CHUNK, caller, round, t)
                        });
                        rd_check(buf, caller, round, t);
                    }
                    // Elementwise activation: 256-chunked, INTER-sized - the
                    // production shape at kquant_forward/cached.rs:869.
                    let t = tag.wrapping_mul(8) + 6;
                    par_chunks_mut(&mut activated, RD_ELEM_CHUNK, |ci, chunk| {
                        rd_fill_chunk(chunk, ci * RD_ELEM_CHUNK, caller, round, t)
                    });
                    rd_check(&activated, caller, round, t);
                }
            }
        }

        /// Single sequential caller, at production scale (34 layers x 400
        /// simulated tokens). `--test-threads` contention from other tests
        /// sharing `global()` is part of the reproduction attempt - do not run
        /// this test in isolation only.
        #[test]
        fn stress_realistic_decode_shape_no_corruption() {
            rd_run(0, 40, 34);
        }

        /// Genuinely concurrent callers (unlike `concurrent_dispatchers_stay_
        /// consistent` above, which uses one FIXED dispatch size for every
        /// thread) - each thread runs its own independent `rd_run`-shaped decode
        /// loop against the SAME shared `global()` pool at the SAME time, so
        /// `dispatch_lock` has to actually serialize dispatchers with DIFFERENT
        /// `(total, chunk)` shapes mid-flight, not just identical ones.
        #[test]
        fn stress_concurrent_realistic_decode_shape_no_corruption() {
            std::thread::scope(|s| {
                for caller in 0..6u64 {
                    s.spawn(move || rd_run(caller, 8, 34));
                }
            });
        }

        /// Empty-block straggler pin: alternate a 1-chunk dispatch (most
        /// participants own an empty block, so the barrier passes without
        /// them) with a wide dispatch, back-to-back. Pre-`task_gen` a
        /// straggler from the 1-chunk dispatch could read the re-published
        /// slot, run the wide dispatch's chunks attributed to the old epoch,
        /// then run them AGAIN on re-observing the epoch — over-counting
        /// `completed`, releasing the barrier early, and letting the closure
        /// drop while another worker still executed it (SIGSEGV under
        /// coverage instrumentation, where the window is widest). Exactly-once
        /// per chunk is the observable invariant; the counters are re-created
        /// per dispatch so a late write from a released dispatch is also a
        /// use-after-free the harness can catch.
        #[test]
        fn alternating_block_sizes_run_each_chunk_exactly_once() {
            use std::sync::atomic::AtomicU8;
            const PARTICIPANTS: usize = 4;
            const WIDE_CHUNKS: usize = 7;
            const ITERATIONS: usize = 50_000;
            let pool = SpinPool::new(PARTICIPANTS);
            for iter in 0..ITERATIONS {
                let n = if iter % 2 == 0 { 1 } else { WIDE_CHUNKS };
                let counters: Vec<AtomicU8> = (0..n).map(|_| AtomicU8::new(0)).collect();
                pool.for_each_chunk(n, |c| {
                    counters[c].fetch_add(1, Ordering::Relaxed);
                });
                for (c, counter) in counters.iter().enumerate() {
                    assert_eq!(
                        counter.load(Ordering::Relaxed),
                        1,
                        "iter {iter}: chunk {c} of {n} not run exactly once"
                    );
                }
            }
        }
    }
}
