//! DEC-9A — how much external traffic is avoidable by retention alone?
//!
//! Pure. The codec ladder closed on an R4 zero-out: give the lever a perfect
//! score and see what it could possibly buy. This module is the same gate
//! applied to the *other* half of the K3 problem.
//!
//! ## Why retention and not prefetch
//!
//! On the external arm the link is the binding constraint — 25.83 GB/token of
//! routed demand against 3.5 GB/s. **Prefetching cannot reduce bytes there.** A
//! correctly prefetched expert would have been demand-fetched anyway, for the
//! same bytes; a wrongly prefetched one adds bytes and evicts something. Under a
//! bytes-per-token objective the best any prefetcher can do is *tie* no-prefetch,
//! and its real payoff — latency hiding — needs idle link capacity that does not
//! exist at this ratio.
//!
//! **Retention does reduce bytes**, by turning a future fetch into a hit. The
//! optimal retention policy under demand paging with equal-size objects is
//! Belady/MIN, so `MIN` is a true ceiling on what *any* amount of future
//! knowledge — a predictor, a transition graph, a residual lookahead — could
//! ever be worth. If `MIN` barely beats `LRU`, the whole predictive-residency
//! programme has a small ceiling and no cleverness recovers it.
//!
//! ## The decomposition that matters, and the trap it avoids
//!
//! `MIN` versus `LRU` alone conflates two different findings: "LRU is a poor
//! policy here" and "there is exploitable temporal structure". A static resident
//! set makes **no eviction decisions at all**, so it isolates pure popularity:
//!
//! ```text
//!   StaticOracle   best fixed set, chosen with full future knowledge
//!   Belady/MIN     popularity + temporal
//!   MIN - Static   the genuinely TEMPORAL prize — all a predictor can target
//! ```
//!
//! If `MIN ≈ StaticOracle`, every bit of structure is static and the answer is
//! *provision the right set*, not *predict the next one*. That would be the
//! fourth time on this estate that an uninformed policy matched an informed one
//! (`freqmass`, `map-3-4-bridge`, the route-aware hot cache), and it should be
//! read as a regime fact rather than as three unlucky predictors.
//!
//! `Random` eviction is carried as the uninformed control: if it matches `LRU`,
//! recency carries no information either and only the static arm is doing work.
//!
//! ## Compulsory misses are part of the ceiling
//!
//! A slot's first touch must miss at any capacity, so `misses - compulsory` is
//! the only avoidable part. Reporting the compulsory floor keeps a hit-rate
//! improvement from being quoted against a denominator that was never reachable.

use std::collections::{BTreeSet, HashMap};

use serde::Serialize;

use super::rng::SplitMix64;
use super::selection_trace::SelectionTrace;

/// One K3 routed expert, all three branches, MXFP4. The bytes column is a
/// K3-scaled projection whenever the trace is not itself K3 — label it.
pub const K3_ROUTED_EXPERT_BYTES: u64 = 17_547_264;

/// Eviction policies. `StaticOracle` is not an eviction policy at all — it is
/// the provisioning arm, and it is here so the temporal prize can be isolated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum Policy {
    /// Belady/MIN — evict the resident slot whose next use is furthest away.
    /// Optimal for demand paging with equal-size objects, hence a true ceiling.
    Min,
    Lru,
    Lfu,
    /// Best fixed resident set over the whole trace. No evictions, ever.
    StaticOracle,
    /// Uninformed control.
    Random,
}

impl Policy {
    pub fn label(self) -> &'static str {
        match self {
            Self::Min => "Belady/MIN (oracle)",
            Self::Lru => "LRU",
            Self::Lfu => "LFU",
            Self::StaticOracle => "static oracle set",
            Self::Random => "random (control)",
        }
    }

    /// Policies that need future knowledge, and so cannot be deployed.
    pub fn is_oracle(self) -> bool {
        matches!(self, Self::Min | Self::StaticOracle)
    }
}

/// The request sequence, grouped so that simultaneously-needed slots cannot
/// evict one another.
///
/// A group is one `(session, step, stratum)` cell — the `top_k` experts a single
/// layer needs at one token. They are required at the same instant, so a cache
/// must hold all of them at once and none may be chosen as another's victim.
#[derive(Debug, Clone)]
pub struct ReferenceStream {
    flat: Vec<u32>,
    groups: Vec<(usize, usize)>,
    session_of_group: Vec<usize>,
    bank: usize,
    tokens: usize,
    max_group: usize,
    /// Distinct slots needed simultaneously at one token, across all strata.
    peak_token_footprint: usize,
}

impl ReferenceStream {
    pub fn bank(&self) -> usize {
        self.bank
    }
    pub fn tokens(&self) -> usize {
        self.tokens
    }
    pub fn requests(&self) -> usize {
        self.flat.len()
    }
    pub fn max_group(&self) -> usize {
        self.max_group
    }
    pub fn peak_token_footprint(&self) -> usize {
        self.peak_token_footprint
    }
    pub fn distinct_slots(&self) -> usize {
        let mut seen: Vec<u32> = self.flat.clone();
        seen.sort_unstable();
        seen.dedup();
        seen.len()
    }

    /// Build a stream directly from groups, for fixtures that need a specific
    /// temporal shape rather than a captured one.
    #[cfg(test)]
    pub fn from_groups(groups: Vec<Vec<u32>>, session_of_group: Vec<usize>, bank: usize) -> Self {
        let mut flat = Vec::new();
        let mut bounds = Vec::new();
        let mut max_group = 0;
        for g in &groups {
            let start = flat.len();
            flat.extend_from_slice(g);
            bounds.push((start, flat.len()));
            let mut d = g.clone();
            d.sort_unstable();
            d.dedup();
            max_group = max_group.max(d.len());
        }
        let tokens = session_of_group.len().max(1);
        Self {
            flat,
            groups: bounds,
            session_of_group,
            bank,
            tokens,
            max_group,
            peak_token_footprint: max_group,
        }
    }
}

/// Flatten a selection trace into a grouped reference stream.
///
/// A cacheable object is `(stratum, symbol)` — expert 5 of layer 3 is a
/// different physical object from expert 5 of layer 7, and conflating them
/// would invent reuse that no cache could realise.
pub fn stream_from_trace(trace: &SelectionTrace) -> ReferenceStream {
    let alphabet = trace.alphabet();
    let strata = trace.active_strata().to_vec();
    let mut flat = Vec::new();
    let mut groups = Vec::new();
    let mut session_of_group = Vec::new();
    let mut max_group = 0;
    let mut peak_token_footprint = 0;

    for session in 0..trace.sessions() {
        for step in 0..trace.steps() {
            let mut token_slots: Vec<u32> = Vec::new();
            for &stratum in &strata {
                let picks = trace.selections(session, step, stratum);
                if picks.is_empty() {
                    continue;
                }
                let start = flat.len();
                for &sym in picks {
                    let slot = (stratum as u32) * alphabet as u32 + sym;
                    flat.push(slot);
                    token_slots.push(slot);
                }
                let mut group: Vec<u32> = flat[start..].to_vec();
                group.sort_unstable();
                group.dedup();
                max_group = max_group.max(group.len());
                groups.push((start, flat.len()));
                session_of_group.push(session);
            }
            token_slots.sort_unstable();
            token_slots.dedup();
            peak_token_footprint = peak_token_footprint.max(token_slots.len());
        }
    }

    ReferenceStream {
        flat,
        groups,
        session_of_group,
        bank: strata.len() * alphabet,
        tokens: trace.sessions() * trace.steps(),
        max_group,
        peak_token_footprint,
    }
}

/// For each position, the next position referencing the same slot.
fn next_use_table(flat: &[u32]) -> Vec<usize> {
    let mut last: HashMap<u32, usize> = HashMap::new();
    let mut next = vec![usize::MAX; flat.len()];
    for i in (0..flat.len()).rev() {
        if let Some(&j) = last.get(&flat[i]) {
            next[i] = j;
        }
        last.insert(flat[i], i);
    }
    next
}

/// Eviction key: `(primary, secondary, slot)`, always evicting the minimum, so
/// every ordered policy shares one structure and one code path.
struct Ordered {
    set: BTreeSet<(i64, i64, u32)>,
    keys: HashMap<u32, (i64, i64)>,
}

impl Ordered {
    fn new() -> Self {
        Self {
            set: BTreeSet::new(),
            keys: HashMap::new(),
        }
    }
    fn insert(&mut self, slot: u32, key: (i64, i64)) {
        if let Some(old) = self.keys.insert(slot, key) {
            self.set.remove(&(old.0, old.1, slot));
        }
        self.set.insert((key.0, key.1, slot));
    }
    fn remove(&mut self, slot: u32) {
        if let Some(k) = self.keys.remove(&slot) {
            self.set.remove(&(k.0, k.1, slot));
        }
    }
    fn evict_min(&mut self) -> Option<u32> {
        let &(a, b, slot) = self.set.iter().next()?;
        self.set.remove(&(a, b, slot));
        self.keys.remove(&slot);
        Some(slot)
    }
    fn len(&self) -> usize {
        self.set.len()
    }
    fn clear(&mut self) {
        self.set.clear();
        self.keys.clear();
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct SimResult {
    pub policy: Policy,
    pub capacity: usize,
    pub capacity_frac: f64,
    pub requests: u64,
    pub misses: u64,
    /// First touches — unavoidable at any capacity, by any policy.
    pub compulsory: u64,
    pub hit_rate: f64,
    pub misses_per_token: f64,
    pub bytes_per_token: f64,
    /// True when capacity cannot hold one layer's simultaneous demand, so the
    /// result is a thrash floor rather than a policy comparison.
    pub below_group_floor: bool,
    /// True when capacity cannot hold ONE TOKEN's distinct demand across all
    /// strata. Routing is cyclic at token granularity — the same ~240 slots
    /// recur every step — so below this a recency-only policy evicts every slot
    /// exactly before its next use and misses 100% by construction. That is a
    /// known LRU pathology on cyclic references, not a measurement of the
    /// workload, and any MIN-over-LRU gap in such a row is measuring the
    /// pathology. The stricter floor, and the one that governs the comparison.
    pub below_token_floor: bool,
}

/// Simulate one policy at one capacity.
///
/// `warm` keeps the cache across sessions (the serving case); `false` resets it
/// per session (a cold single-stream case). They answer different questions and
/// the pool is a *domain mixture*, which per R10 is anti-conservative for the
/// warm arm — a domain-specific workload would show more cross-session reuse.
pub fn simulate(
    stream: &ReferenceStream,
    policy: Policy,
    capacity: usize,
    warm: bool,
    seed: u64,
    expert_bytes: u64,
) -> SimResult {
    let capacity = capacity.max(1);
    let result = if policy == Policy::StaticOracle {
        static_oracle(stream, capacity, warm)
    } else {
        dynamic(stream, policy, capacity, warm, seed)
    };
    let (misses, compulsory) = result;
    let requests = stream.requests() as u64;
    let tokens = stream.tokens().max(1) as f64;
    SimResult {
        policy,
        capacity,
        capacity_frac: capacity as f64 / stream.bank().max(1) as f64,
        requests,
        misses,
        compulsory,
        hit_rate: if requests == 0 {
            0.0
        } else {
            1.0 - misses as f64 / requests as f64
        },
        misses_per_token: misses as f64 / tokens,
        bytes_per_token: misses as f64 / tokens * expert_bytes as f64,
        below_group_floor: capacity < stream.max_group(),
        below_token_floor: capacity < stream.peak_token_footprint(),
    }
}

/// The provisioning arm: the `capacity` most-referenced slots, resident forever.
///
/// Loading the set is charged as compulsory misses so it is not credited with
/// bytes it never paid — **once** when the cache is warm (a serving process
/// provisions at startup), and **once per session** when cold (a fresh process
/// must re-read it). Charging it once in the cold arm would compare a persistent
/// static set against dynamic caches that get wiped, which is what produced a
/// nonsensical negative temporal prize at high capacity before this was fixed.
fn static_oracle(stream: &ReferenceStream, capacity: usize, warm: bool) -> (u64, u64) {
    let mut freq: HashMap<u32, u64> = HashMap::new();
    for &s in &stream.flat {
        *freq.entry(s).or_default() += 1;
    }
    let mut ranked: Vec<(u32, u64)> = freq.into_iter().collect();
    // Deterministic: frequency descending, then slot id ascending.
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let resident: std::collections::HashSet<u32> =
        ranked.iter().take(capacity).map(|&(s, _)| s).collect();
    let mut ids: Vec<usize> = stream.session_of_group.clone();
    ids.dedup();
    let sessions = if warm { 1 } else { ids.len().max(1) } as u64;
    let load = resident.len() as u64 * sessions;
    let misses = stream.flat.iter().filter(|s| !resident.contains(s)).count() as u64;
    (misses + load, resident.len() as u64)
}

fn dynamic(
    stream: &ReferenceStream,
    policy: Policy,
    capacity: usize,
    warm: bool,
    seed: u64,
) -> (u64, u64) {
    let next_use = if policy == Policy::Min {
        next_use_table(&stream.flat)
    } else {
        Vec::new()
    };
    let mut ordered = Ordered::new();
    let mut rand_items: Vec<u32> = Vec::new();
    let mut rand_pos: HashMap<u32, usize> = HashMap::new();
    let mut rng = SplitMix64(seed);
    let mut freq: HashMap<u32, i64> = HashMap::new();
    let mut ever_seen: std::collections::HashSet<u32> = std::collections::HashSet::new();
    let (mut misses, mut compulsory) = (0u64, 0u64);
    let mut clock: i64 = 0;
    let mut current_session = usize::MAX;

    let is_random = policy == Policy::Random;
    let resident_len = |o: &Ordered, v: &Vec<u32>| if is_random { v.len() } else { o.len() };

    for (g, &(start, end)) in stream.groups.iter().enumerate() {
        if !warm && stream.session_of_group[g] != current_session {
            ordered.clear();
            rand_items.clear();
            rand_pos.clear();
            current_session = stream.session_of_group[g];
        }

        let mut pinned: Vec<u32> = stream.flat[start..end].to_vec();
        pinned.sort_unstable();
        pinned.dedup();

        // Phase 1 — account, before anything is admitted or evicted.
        for &s in &pinned {
            let resident = if is_random {
                rand_pos.contains_key(&s)
            } else {
                ordered.keys.contains_key(&s)
            };
            if !resident {
                misses += 1;
                if ever_seen.insert(s) {
                    compulsory += 1;
                }
            }
        }

        // Phase 2 — pin the group out of the eviction structure so a member
        // cannot be chosen as another member's victim.
        for &s in &pinned {
            if is_random {
                if let Some(i) = rand_pos.remove(&s) {
                    let last = rand_items.pop().expect("non-empty when pos is set");
                    if i < rand_items.len() {
                        rand_items[i] = last;
                        rand_pos.insert(last, i);
                    }
                }
            } else {
                ordered.remove(s);
            }
        }

        // Phase 3 — make room for the whole group, evicting only non-members.
        let room = capacity.saturating_sub(pinned.len());
        while resident_len(&ordered, &rand_items) > room {
            if is_random {
                let i = rng.below(rand_items.len() as u64) as usize;
                let victim = rand_items.swap_remove(i);
                rand_pos.remove(&victim);
                if i < rand_items.len() {
                    let moved = rand_items[i];
                    rand_pos.insert(moved, i);
                }
            } else if ordered.evict_min().is_none() {
                break;
            }
        }

        // Phase 4 — admit the group with refreshed keys.
        for (offset, &s) in pinned.iter().enumerate() {
            clock += 1;
            let f = freq.entry(s).or_insert(0);
            *f += 1;
            if is_random {
                if capacity > rand_items.len() {
                    rand_pos.insert(s, rand_items.len());
                    rand_items.push(s);
                }
                continue;
            }
            let key = match policy {
                Policy::Lru => (clock, 0),
                Policy::Lfu => (*f, clock),
                Policy::Min => {
                    // Position of this slot's occurrence inside the group.
                    let pos = (start..end)
                        .find(|&i| stream.flat[i] == s)
                        .unwrap_or(start + offset);
                    let n = next_use[pos];
                    // Never used again sorts first, so it is evicted first.
                    (
                        if n == usize::MAX {
                            i64::MIN
                        } else {
                            -(n as i64)
                        },
                        0,
                    )
                }
                _ => (clock, 0),
            };
            if capacity > 0 {
                ordered.insert(s, key);
            }
        }
    }
    (misses, compulsory)
}

/// The two gaps that decide the programme.
#[derive(Debug, Clone, Serialize)]
pub struct RetentionGate {
    pub capacity: usize,
    pub capacity_frac: f64,
    /// The LRU-derived columns in this row are a cyclic-thrash artifact.
    pub below_token_floor: bool,
    pub min_misses_per_token: f64,
    pub lru_misses_per_token: f64,
    pub static_misses_per_token: f64,
    /// `(LRU - MIN) / LRU` — how much a better *policy* could buy.
    pub oracle_over_lru: f64,
    /// `(Static - MIN) / Static` — the genuinely TEMPORAL prize, and the only
    /// part any predictor, transition graph or lookahead can target.
    pub temporal_prize: f64,
    /// `(Random - LRU) / Random` — does recency carry information at all?
    pub recency_value: f64,
}

pub(super) fn gain(baseline: f64, improved: f64) -> f64 {
    if baseline <= 0.0 {
        0.0
    } else {
        (baseline - improved) / baseline
    }
}

pub fn gate(results: &[SimResult], capacity: usize) -> Option<RetentionGate> {
    let at = |p: Policy| {
        results
            .iter()
            .find(|r| r.capacity == capacity && r.policy == p)
    };
    let (min, lru, stat) = (
        at(Policy::Min)?,
        at(Policy::Lru)?,
        at(Policy::StaticOracle)?,
    );
    let rnd = at(Policy::Random)?;
    Some(RetentionGate {
        capacity,
        capacity_frac: min.capacity_frac,
        below_token_floor: min.below_token_floor,
        min_misses_per_token: min.misses_per_token,
        lru_misses_per_token: lru.misses_per_token,
        static_misses_per_token: stat.misses_per_token,
        oracle_over_lru: gain(lru.misses_per_token, min.misses_per_token),
        temporal_prize: gain(stat.misses_per_token, min.misses_per_token),
        recency_value: gain(rnd.misses_per_token, lru.misses_per_token),
    })
}

/// Pre-registered bands on the temporal prize, written before the run.
pub const PRIZE_CLOSE_PROGRAMME: f64 = 0.10;
pub const PRIZE_INTERESTING: f64 = 0.33;

pub fn verdict(temporal_prize: f64) -> &'static str {
    if temporal_prize < PRIZE_CLOSE_PROGRAMME {
        "CLOSE — perfect future knowledge barely beats a fixed set; provision, do not predict"
    } else if temporal_prize < PRIZE_INTERESTING {
        "MARGINAL — a real but small temporal prize; only a near-oracle predictor would pay"
    } else {
        "OPEN — future route information is a material resource; target a fraction of MIN"
    }
}
