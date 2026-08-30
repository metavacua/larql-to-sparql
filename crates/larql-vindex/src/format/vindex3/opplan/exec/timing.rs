//! Where a decode token's milliseconds actually go.
//!
//! The projection ledger prices weight traffic; this prices TIME, at the
//! same call sites, so the two describe one execution rather than two
//! stories about it. Together they replace the arithmetic that produced
//! the old "159 ms residue": token wall minus an assumed bandwidth,
//! which is not a measurement of anything and turned out to be wrong by
//! more than a factor of two.
//!
//! **The leaves are disjoint and nothing nests.** A class that wrapped
//! another would double-count, and a sum that double-counts can be made
//! to equal anything. So a timer covers exactly its own arithmetic — the
//! FFN's activation but not the three projections around it, the
//! recurrence but not the five projections beside it — and the
//! reconciliation is:
//!
//! ```text
//! sum(leaf classes) + unattributed = steady token wall
//! ```
//!
//! **`unattributed` is a failing diagnostic, not a bucket.** Above a few
//! percent it means a boundary is missing, and the answer is to find it,
//! not to name it. The moment it becomes somewhere to put the
//! unexplained, this file has recreated the thing it was built to
//! delete.
//!
//! Reconciliation holds for the DECODE path, which runs one position on
//! the caller's thread. The batched driver runs positions in parallel, so
//! its leaves sum across threads and exceed the wall by design; the
//! report says which path it measured.
//!
//! Cost is two `Instant` reads per leaf against roughly 1200 leaves and
//! 480 ms per token — about 60 microseconds, 0.01%. Always on rather than
//! behind a flag that would be off exactly when a number needed
//! explaining.

use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// One kind of work a decode token is made of.
///
/// Deliberately operator-shaped rather than layer-shaped: "layer 7 cost
/// 8 ms" cannot be acted on, and "the recurrence cost 8 ms over 48 calls"
/// can.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum OpClass {
    /// Every dense `y = Wx`, whatever kernel ran it. Timed inside the
    /// executor, at the same site the byte ledger is written, so the two
    /// cannot describe different call sets.
    Projection,
    Embed,
    /// RMS/LayerNorm at every site, including Q/K normalisation.
    Norm,
    Rope,
    /// Softmax over the K/V cache and the weighted sum of V.
    AttentionCore,
    /// The gate's sigmoid and elementwise multiply — NOT its projection.
    OutputGate,
    /// Gated DeltaNet's depthwise causal convolution and the SiLU after
    /// it.
    DeltaConv,
    /// `repeat_interleave` of q/k across value heads.
    DeltaHeadExpand,
    /// beta, softplus, and the decay term.
    DeltaGates,
    /// The delta rule itself — the state update.
    DeltaRecurrence,
    /// Per-head RMS over `Dv`, the norm weight, and the SiLU'd gate.
    DeltaGatedNorm,
    /// GeGLU/SwiGLU/GELU between the FFN's projections.
    FfnActivation,
    /// Residual adds and layer scaling.
    Residual,
    /// Logit multiplier and softcapping over the vocabulary — NOT the
    /// head's projection.
    Logits,
}

impl OpClass {
    /// Every class, so a reader enumerates rather than remembers.
    pub const ALL: [OpClass; 14] = [
        OpClass::Projection,
        OpClass::Embed,
        OpClass::Norm,
        OpClass::Rope,
        OpClass::AttentionCore,
        OpClass::OutputGate,
        OpClass::DeltaConv,
        OpClass::DeltaHeadExpand,
        OpClass::DeltaGates,
        OpClass::DeltaRecurrence,
        OpClass::DeltaGatedNorm,
        OpClass::FfnActivation,
        OpClass::Residual,
        OpClass::Logits,
    ];

    pub fn name(self) -> &'static str {
        match self {
            OpClass::Projection => "Projection",
            OpClass::Embed => "Embed",
            OpClass::Norm => "Norm",
            OpClass::Rope => "RoPE",
            OpClass::AttentionCore => "AttentionCore",
            OpClass::OutputGate => "OutputGate",
            OpClass::DeltaConv => "DeltaConv",
            OpClass::DeltaHeadExpand => "DeltaHeadExpand",
            OpClass::DeltaGates => "DeltaGates",
            OpClass::DeltaRecurrence => "DeltaRecurrence",
            OpClass::DeltaGatedNorm => "DeltaGatedNorm",
            OpClass::FfnActivation => "FfnActivation",
            OpClass::Residual => "Residual",
            OpClass::Logits => "Logits",
        }
    }

    fn index(self) -> usize {
        OpClass::ALL
            .iter()
            .position(|c| *c == self)
            .expect("ALL covers every class")
    }
}

/// One class's tally, read out.
///
/// Calls alongside nanos because they are different problems: 12 ms over
/// 48 calls is an arithmetic cost and 12 ms over 5000 is a dispatch cost,
/// and only one of them is fixed by a faster kernel.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ClassTally {
    pub calls: u64,
    pub nanos: u64,
}

impl ClassTally {
    pub fn nanos_per_call(&self) -> f64 {
        if self.calls == 0 {
            0.0
        } else {
            self.nanos as f64 / self.calls as f64
        }
    }
}

#[derive(Default)]
struct Slot {
    calls: AtomicU64,
    nanos: AtomicU64,
}

/// Every leaf the executor has timed, by class.
pub struct TimingLedger {
    slots: [Slot; 14],
    /// Timers that started while another was already running ON THE SAME
    /// THREAD.
    ///
    /// Counted rather than fatal: a panic here would take down a decode
    /// over an instrumentation mistake. But any overlap at all voids the
    /// reconciliation — the classes would double-count — so the report
    /// must refuse to add up rather than quietly present a total that
    /// is too large.
    nested: AtomicU64,
}

thread_local! {
    /// Whether this thread is already inside a timed leaf.
    static ACTIVE: Cell<bool> = const { Cell::new(false) };
}

impl TimingLedger {
    /// A ledger with nothing in it. `const` so the process ledger is a
    /// static rather than a lazily-initialised one, and so a test can
    /// hold its own instead of racing the shared counters.
    pub(super) const fn new() -> Self {
        #[allow(clippy::declare_interior_mutable_const)]
        const ZERO: Slot = Slot {
            calls: AtomicU64::new(0),
            nanos: AtomicU64::new(0),
        };
        Self {
            slots: [ZERO; 14],
            nested: AtomicU64::new(0),
        }
    }

    pub(super) fn record(&self, class: OpClass, nanos: u64) {
        let slot = &self.slots[class.index()];
        slot.calls.fetch_add(1, Ordering::Relaxed);
        slot.nanos.fetch_add(nanos, Ordering::Relaxed);
    }

    pub fn get(&self, class: OpClass) -> ClassTally {
        let slot = &self.slots[class.index()];
        ClassTally {
            calls: slot.calls.load(Ordering::Relaxed),
            nanos: slot.nanos.load(Ordering::Relaxed),
        }
    }

    pub fn all(&self) -> [(OpClass, ClassTally); 14] {
        OpClass::ALL.map(|c| (c, self.get(c)))
    }

    /// Total timed nanoseconds across every class.
    pub fn total_nanos(&self) -> u64 {
        OpClass::ALL.iter().map(|c| self.get(*c).nanos).sum()
    }

    /// Overlapping timers seen. Non-zero invalidates the reconciliation.
    pub fn nested(&self) -> u64 {
        self.nested.load(Ordering::Relaxed)
    }

    /// Zero everything, so a caller can price ONE step.
    pub fn reset(&self) {
        for slot in &self.slots {
            slot.calls.store(0, Ordering::Relaxed);
            slot.nanos.store(0, Ordering::Relaxed);
        }
        self.nested.store(0, Ordering::Relaxed);
    }
}

/// A running leaf timer. Records on drop.
pub struct Timed {
    class: OpClass,
    started: Instant,
    /// Whether this timer owns the thread's active flag. A nested timer
    /// does not, so it must not clear the flag its parent set.
    outermost: bool,
}

impl Drop for Timed {
    fn drop(&mut self) {
        let nanos = self.started.elapsed().as_nanos() as u64;
        ledger().record(self.class, nanos);
        if self.outermost {
            ACTIVE.with(|a| a.set(false));
        }
    }
}

/// Start timing one leaf. The value must be held for the leaf's extent.
///
/// `let _t = timed(OpClass::Norm);` — binding to `_` instead would drop
/// it immediately and time nothing, which is the one mistake this API
/// makes easy, so bind it to a name.
pub fn timed(class: OpClass) -> Timed {
    let outermost = ACTIVE.with(|a| {
        let was = a.get();
        a.set(true);
        !was
    });
    if !outermost {
        // Counted, never fatal, and identically in every build profile.
        // A `debug_assert` here would make the executor behave one way
        // under test and another in release — the failure mode being
        // guarded against is a silently wrong total, and a panic that
        // only happens in debug does not prevent it.
        ledger().nested.fetch_add(1, Ordering::Relaxed);
    }
    Timed {
        class,
        started: Instant::now(),
        outermost,
    }
}

static LEDGER: TimingLedger = TimingLedger::new();

/// The process's timing ledger.
pub fn ledger() -> &'static TimingLedger {
    &LEDGER
}
