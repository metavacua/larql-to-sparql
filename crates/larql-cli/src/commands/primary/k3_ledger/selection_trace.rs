//! Generic selection trace — which symbols each position chose, per stratum.
//!
//! "Symbol" is deliberately unnamed. At DEC-8.4 it is an MoE expert id; at
//! DEC-8.1 it is an FFN feature-row id. The frequency-mass estimators in
//! [`super::freqmass`] never learn which, so one instrument grades both the
//! resident-expert bet and the compact-subexpert bet. Nothing here knows about
//! K3, or about any particular model.
//!
//! A **session** is one independent routing context (a prompt and its decode
//! steps). Sessions are the unit the adaptive-cache estimators condition on, so
//! the type keeps them separable rather than pooling at construction.

#[cfg(not(target_arch = "wasm32"))]
use super::super::dec_bench::capture_format::{CapturePool, ROUTING_SENTINEL_EXPERT};

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

/// Marks an unused slot in a fixed-width selection record: a stratum that
/// carried no selection, or a slot freed by zero-weight stripping upstream.
pub const SENTINEL: u32 = u32::MAX;

/// What the symbols in a trace actually are. Carried so an exported
/// distribution says which bank it describes — expert mass and feature mass are
/// not numerically transferable, and a file that does not name its unit invites
/// exactly that substitution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SelectionUnit {
    /// An MoE expert (DEC-8.4).
    Expert,
    /// An FFN feature row inside one expert (DEC-8.1).
    ///
    /// Nothing constructs this yet — only the expert path is wired, so
    /// `from_routing_pool` always labels `Expert`. Kept rather than deleted
    /// because the unit is the thing that stops an expert distribution being
    /// read as a feature one, and a DEC-8.1 export arriving with no way to
    /// name its own bank is exactly the substitution this enum exists to
    /// prevent. Pinned by `every_selection_unit_serialises_under_a_distinct_name`.
    #[allow(dead_code)]
    Feature,
    /// Unlabelled — the estimators do not care, but an export should.
    Symbol,
}

impl SelectionUnit {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Expert => "expert",
            Self::Feature => "feature",
            Self::Symbol => "symbol",
        }
    }
}

/// `[session][step][stratum][width]` selections, sentinel-padded.
#[derive(Debug, Clone)]
pub struct SelectionTrace {
    sessions: usize,
    steps: usize,
    strata: usize,
    width: usize,
    alphabet: usize,
    unit: SelectionUnit,
    ids: Vec<u32>,
    active: Vec<usize>,
}

impl SelectionTrace {
    /// `ids` is the flat `[session][step][stratum][width]` record block.
    ///
    /// `alphabet` is the symbol population — the size of the bank being chosen
    /// from. It is **not** derivable from the trace: a symbol that is never
    /// selected leaves no mark, so inferring it from the maximum observed id
    /// undercounts. Callers that must infer it should say so in their output.
    pub fn new(
        sessions: usize,
        steps: usize,
        strata: usize,
        width: usize,
        alphabet: usize,
        ids: Vec<u32>,
    ) -> Result<Self, String> {
        if width == 0 {
            return Err("selection trace: width 0".into());
        }
        if alphabet == 0 {
            return Err("selection trace: alphabet 0".into());
        }
        let expect = sessions * steps * strata * width;
        if ids.len() != expect {
            return Err(format!(
                "selection trace: {} ids, shape implies {expect}",
                ids.len()
            ));
        }
        if let Some(bad) = ids
            .iter()
            .find(|&&i| i != SENTINEL && i as usize >= alphabet)
        {
            return Err(format!(
                "selection trace: symbol {bad} outside alphabet {alphabet}"
            ));
        }
        let active = (0..strata)
            .filter(|&s| {
                (0..sessions).any(|b| {
                    (0..steps).any(|t| {
                        let base = ((b * steps + t) * strata + s) * width;
                        ids[base..base + width].iter().any(|&i| i != SENTINEL)
                    })
                })
            })
            .collect();
        Ok(Self {
            sessions,
            steps,
            strata,
            width,
            alphabet,
            unit: SelectionUnit::Symbol,
            ids,
            active,
        })
    }

    /// Label what the symbols are. Additive: the estimators ignore it and only
    /// exports read it.
    pub fn with_unit(mut self, unit: SelectionUnit) -> Self {
        self.unit = unit;
        self
    }

    pub fn unit(&self) -> SelectionUnit {
        self.unit
    }

    /// Adapt a DEC residual capture pool captured with `--routing`.
    ///
    /// The alphabet is **inferred** as `max observed id + 1`; an expert never
    /// selected anywhere in the pool is invisible here, so a pool that does not
    /// exercise the whole bank reports a bank smaller than the model's.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_routing_pool(pool: &CapturePool) -> Result<Self, String> {
        let width = pool
            .routing_top_k()
            .ok_or("selection trace: pool has no routing sidecar (capture with --routing)")?;
        let sessions = pool.num_prompts();
        let steps = pool.manifest.steps;
        let strata = pool.manifest.num_layers;

        let mut ids = vec![SENTINEL; sessions * steps * strata * width];
        let mut max_id = 0u32;
        for b in 0..sessions {
            for t in 0..steps {
                for s in 0..strata {
                    let Some(pairs) = pool.routing(b, t, s)? else {
                        continue;
                    };
                    let base = ((b * steps + t) * strata + s) * width;
                    for (slot, &(id, _weight)) in pairs.iter().enumerate() {
                        debug_assert_ne!(id, ROUTING_SENTINEL_EXPERT);
                        ids[base + slot] = id;
                        max_id = max_id.max(id);
                    }
                }
            }
        }
        Ok(
            Self::new(sessions, steps, strata, width, max_id as usize + 1, ids)?
                .with_unit(SelectionUnit::Expert),
        )
    }

    pub fn sessions(&self) -> usize {
        self.sessions
    }

    pub fn steps(&self) -> usize {
        self.steps
    }

    pub fn width(&self) -> usize {
        self.width
    }

    /// Every stratum, including those that carried no selection.
    pub fn strata(&self) -> usize {
        self.strata
    }

    pub fn alphabet(&self) -> usize {
        self.alphabet
    }

    /// Strata that carry at least one selection — MoE layers, in a trace whose
    /// stratum axis is layers.
    pub fn active_strata(&self) -> &[usize] {
        &self.active
    }

    /// Fraction of the bank one position selects. The condition that R2 says
    /// must travel with every union or amortisation figure derived from here.
    pub fn activation_fraction(&self) -> f64 {
        self.width as f64 / self.alphabet as f64
    }

    /// Selections for one cell: the non-sentinel prefix (records are compacted
    /// to the front and sentinel-padded).
    pub fn selections(&self, session: usize, step: usize, stratum: usize) -> &[u32] {
        let base = ((session * self.steps + step) * self.strata + stratum) * self.width;
        let record = &self.ids[base..base + self.width];
        let n = record
            .iter()
            .position(|&i| i == SENTINEL)
            .unwrap_or(self.width);
        &record[..n]
    }

    /// Accumulate selection counts over `steps` into `out` (length `alphabet`).
    pub fn accumulate(
        &self,
        session: usize,
        stratum: usize,
        steps: impl IntoIterator<Item = usize>,
        out: &mut [f64],
    ) {
        for t in steps {
            for &id in self.selections(session, t, stratum) {
                out[id as usize] += 1.0;
            }
        }
    }

    /// Distinct symbols one session touches in its first `steps` steps — the
    /// support (union), as distinct from the frequency mass over that support.
    pub fn support(&self, session: usize, stratum: usize, steps: usize) -> usize {
        let mut seen = vec![false; self.alphabet];
        let mut n = 0;
        for t in 0..steps.min(self.steps) {
            for &id in self.selections(session, t, stratum) {
                if !core::mem::replace(&mut seen[id as usize], true) {
                    n += 1;
                }
            }
        }
        n
    }

    /// Support a memoryless uniform selector would reach — the **null** the
    /// measured curve is read against, never a prediction in its own right.
    /// Measured support at DEC-0 came in 3.4x below this at 16 steps.
    pub fn uniform_support(&self, steps: usize) -> f64 {
        let draws = (steps.min(self.steps) * self.width) as f64;
        let e = self.alphabet as f64;
        e * (1.0 - (1.0 - self.width as f64 / e).powf(draws))
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::dec_bench::capture_format::{CapturePool, RoutingCapture};
    use super::*;

    /// Round-trip a minimal routed pool through the real writer, so the adapter
    /// is pinned against the on-disk format rather than against a hand-built
    /// buffer that could drift from it.
    fn write_pool(dir: &std::path::Path, routing: Vec<Vec<Vec<Vec<(u32, f32)>>>>, top_k: usize) {
        let (prompts, steps, layers, hidden) = (routing.len(), routing[0].len(), 2, 2);
        let plane = vec![vec![vec![0.0f32; hidden]; layers]; steps];
        let per_prompt = vec![plane.clone(); prompts];
        let texts: Vec<String> = (0..prompts).map(|i| format!("p{i}")).collect();
        let rc = RoutingCapture {
            top_k,
            raw: per_prompt.clone(),
            normed: per_prompt.clone(),
            routing,
        };
        CapturePool::write_with_routing(
            dir,
            "test-model",
            hidden,
            layers,
            &texts,
            &per_prompt,
            Some(&rc),
            0,
        )
        .unwrap();
    }

    #[test]
    fn adapts_a_routed_capture_pool_and_infers_the_bank() {
        let dir = tempfile::tempdir().unwrap();
        // 2 prompts x 2 steps x 2 layers, top-2; layer 1 never routes.
        let routing = vec![
            vec![
                vec![vec![(0u32, 0.6), (3, 0.4)], vec![]],
                vec![vec![(1, 0.5), (3, 0.5)], vec![]],
            ],
            vec![
                vec![vec![(2, 0.7), (0, 0.3)], vec![]],
                vec![vec![(0, 0.9), (1, 0.1)], vec![]],
            ],
        ];
        write_pool(dir.path(), routing, 2);

        let pool = CapturePool::open(dir.path()).unwrap();
        let trace = SelectionTrace::from_routing_pool(&pool).unwrap();

        assert_eq!((trace.sessions(), trace.steps(), trace.width()), (2, 2, 2));
        assert_eq!(trace.alphabet(), 4, "inferred as max observed id + 1");
        assert_eq!(trace.active_strata(), &[0], "layer 1 carried no routing");
        assert_eq!(trace.selections(0, 0, 0), &[0, 3]);
        assert_eq!(trace.selections(1, 1, 0), &[0, 1]);
        assert_eq!(trace.selections(0, 0, 1), &[] as &[u32]);
    }

    #[test]
    fn a_bank_the_pool_never_exercises_is_undercounted() {
        let dir = tempfile::tempdir().unwrap();
        // Only expert 0 is ever selected, so the inferred bank is 1 however
        // many experts the model really has. The renderer says so out loud.
        let routing = vec![vec![
            vec![vec![(0u32, 1.0)], vec![]],
            vec![vec![(0, 1.0)], vec![]],
        ]];
        write_pool(dir.path(), routing, 1);

        let pool = CapturePool::open(dir.path()).unwrap();
        let trace = SelectionTrace::from_routing_pool(&pool).unwrap();
        assert_eq!(trace.alphabet(), 1);
        assert_eq!(trace.activation_fraction(), 1.0);
    }

    #[test]
    fn a_pool_without_the_routing_sidecar_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let plane = vec![vec![vec![0.0f32; 2]; 2]; 2];
        CapturePool::write(dir.path(), "m", 2, 2, &["p".to_string()], &[plane], 0).unwrap();

        let pool = CapturePool::open(dir.path()).unwrap();
        let err = SelectionTrace::from_routing_pool(&pool).unwrap_err();
        assert!(err.contains("no routing sidecar"), "{err}");
    }

    /// 2 sessions x 2 steps x 2 strata x width 2; stratum 1 never routes.
    fn fixture() -> SelectionTrace {
        let s = SENTINEL;
        SelectionTrace::new(
            2,
            2,
            2,
            2,
            4,
            vec![
                0, 1, s, s, // b0 t0: st0 -> {0,1}, st1 -> none
                0, 2, s, s, // b0 t1
                3, 1, s, s, // b1 t0
                3, 3, s, s, // b1 t1 (repeat within a record is legal)
            ],
        )
        .unwrap()
    }

    #[test]
    fn active_strata_excludes_the_all_sentinel_stratum() {
        assert_eq!(fixture().active_strata(), &[0]);
    }

    #[test]
    fn selections_take_the_prefix_before_the_sentinel() {
        let t = fixture();
        assert_eq!(t.selections(0, 0, 0), &[0, 1]);
        assert_eq!(t.selections(0, 0, 1), &[] as &[u32]);
    }

    #[test]
    fn accumulate_sums_over_the_requested_steps_only() {
        let t = fixture();
        let mut out = vec![0.0; 4];
        t.accumulate(0, 0, 0..1, &mut out);
        assert_eq!(out, vec![1.0, 1.0, 0.0, 0.0]);
        t.accumulate(0, 0, 1..2, &mut out);
        assert_eq!(out, vec![2.0, 1.0, 1.0, 0.0]);
    }

    #[test]
    fn support_counts_distinct_symbols_not_selections() {
        let t = fixture();
        assert_eq!(t.support(0, 0, 2), 3); // {0,1} u {0,2}
        assert_eq!(t.support(1, 0, 2), 2); // {3,1} u {3,3}
        assert_eq!(t.support(0, 0, 1), 2);
    }

    #[test]
    fn support_saturates_at_the_available_steps() {
        assert_eq!(fixture().support(0, 0, 99), 3);
    }

    #[test]
    fn uniform_support_is_the_memoryless_null() {
        let t = fixture();
        // 4 draws at k/E = 0.5: 4*(1 - 0.5^4) = 3.75, above the measured 3.
        assert!((t.uniform_support(2) - 3.75).abs() < 1e-9);
    }

    #[test]
    fn activation_fraction_carries_the_r2_condition() {
        assert!((fixture().activation_fraction() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn new_rejects_a_shape_mismatch() {
        let err = SelectionTrace::new(1, 1, 1, 2, 4, vec![0]).unwrap_err();
        assert!(err.contains("shape implies"), "{err}");
    }

    #[test]
    fn new_rejects_a_symbol_outside_the_alphabet() {
        let err = SelectionTrace::new(1, 1, 1, 2, 2, vec![0, 7]).unwrap_err();
        assert!(err.contains("outside alphabet"), "{err}");
    }

    #[test]
    fn new_rejects_degenerate_width_and_alphabet() {
        assert!(SelectionTrace::new(1, 1, 1, 0, 4, vec![]).is_err());
        assert!(SelectionTrace::new(1, 1, 1, 1, 0, vec![0]).is_err());
    }

    #[test]
    fn accessors_report_the_constructed_shape() {
        let t = fixture();
        assert_eq!(
            (t.sessions(), t.steps(), t.width(), t.alphabet()),
            (2, 2, 2, 4)
        );
    }

    #[test]
    fn every_selection_unit_serialises_under_a_distinct_name() {
        // The names reach exported files, and the whole point of the enum is
        // that a reader can tell an expert distribution from a feature one.
        // Two units sharing a name would silently permit the substitution.
        let units = [
            SelectionUnit::Expert,
            SelectionUnit::Feature,
            SelectionUnit::Symbol,
        ];
        let mut names: Vec<&str> = units.iter().map(|u| u.as_str()).collect();
        let count = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), count, "unit names must be distinct");
    }
}
