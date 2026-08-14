//! The f32 reference MoE block for **packed** expert stores.
//!
//! [`ExpertWeightFfn`](super::ExpertWeightFfn) resolves one tensor per expert
//! per projection. Gemma 4, GraniteMoE and GPT-OSS instead stack every expert
//! into a single operand per projection, so that backend finds nothing and the
//! scorer used to fall through to the dense FFN and panic on a tensor the
//! architecture never had. This is the sibling for that representation.
//!
//! ## What it deliberately does *not* reuse
//!
//! This is an **oracle**. Its job is to disagree with the production MoE path
//! when the production path is wrong, so it shares no arithmetic with
//! `cpu_moe_forward`: not the dequantisation, not the matmuls, not the
//! accumulation. It re-derives all of them here in plain f32.
//!
//! What it *does* share is addressing — which bytes belong to expert `e`, and
//! (via [`GateUpLayout::row`]) which row of a fused operand is gate and which
//! is up. Those are declarations about the operand's shape, not the operation
//! under test. The failure this boundary prevents is a shared helper carrying
//! an indexing bug into both sides, so oracle and engine agree perfectly with
//! each other and are both wrong — strictly worse than a crash.
//!
//! ## Layout is declared, never inferred
//!
//! A fused `gate_up` row block is read according to the architecture's
//! [`GateUpLayout`]. GraniteMoE is `ContiguousHalves` and GPT-OSS is
//! `Interleaved`, so "packed" alone does not determine it, and reading one as
//! the other silently mixes the branches rather than failing. An architecture
//! that declares a packed format and no layout is refused.

use larql_models::{GateUpBranch, GateUpLayout, ModelArchitecture, ModelWeights};
use ndarray::{s, Array2, Axis};
use std::path::PathBuf;

use super::expert_weight::{gate, router};
use super::{FfnActivations, FfnBackend};

/// Bytes per BF16 element in a packed expert store.
const BF16_BYTES: usize = 2;
/// Gate and up, the two branches sharing a fused operand.
const FUSED_BRANCHES: usize = 2;
/// Names a file for the opt-in layer-0 router capture.
const ROUTE_LOG_ENV: &str = "LARQL_PACKED_ROUTE_LOG";

/// f32 reference MoE over stacked (packed) expert operands.
pub struct PackedExpertWeightFfn<'a> {
    pub weights: &'a ModelWeights,
}

/// One expert's slice of a layer's packed operands, as raw bytes plus the
/// geometry needed to index them. Addressing only — no arithmetic.
struct PackedExpert<'a> {
    gate_up: &'a [u8],
    down: &'a [u8],
    hidden: usize,
    inter: usize,
}

impl<'a> PackedExpertWeightFfn<'a> {
    fn arch(&self) -> &dyn ModelArchitecture {
        &*self.weights.arch
    }

    /// Decode one BF16 element. The upper 16 bits of an f32, zero-extended —
    /// exact, no rounding, so this contributes no error of its own.
    fn bf16(bytes: &[u8], index: usize) -> f32 {
        let lo = index * BF16_BYTES;
        let raw = u16::from_le_bytes([bytes[lo], bytes[lo + 1]]);
        f32::from_bits((raw as u32) << 16)
    }

    /// Router logits for every token: `[tokens, num_experts]`.
    fn router_logits(&self, layer: usize, x: &Array2<f32>) -> Option<Array2<f32>> {
        let arch = self.arch();
        let key = arch.moe_router_key(layer)?;
        let hidden = x.ncols();
        // The router lands in `tensors` as `[experts, hidden]` or in `vectors`
        // as the same matrix flattened, depending on the loader path — the
        // same two places `build_moe_weights` looks. Row-major either way.
        let (w, experts): (&[f32], usize) = match self.weights.tensors.get(&key) {
            Some(t) => (t.as_slice()?, t.nrows()),
            None => {
                let v = self.weights.vectors.get(&key)?;
                if hidden == 0 || v.len() % hidden != 0 {
                    return None;
                }
                (v.as_slice(), v.len() / hidden)
            }
        };
        let mut logits = Array2::<f32>::zeros((x.nrows(), experts));
        for (t, xr) in x.axis_iter(Axis(0)).enumerate() {
            for e in 0..experts {
                let base = e * hidden;
                let mut acc = 0.0f32;
                for j in 0..hidden {
                    acc += xr[j] * w[base + j];
                }
                logits[[t, e]] = acc;
            }
        }
        if let Some(bias) = arch
            .moe_router_bias_key(layer)
            .and_then(|k| self.weights.vectors.get(&k))
        {
            for mut row in logits.axis_iter_mut(Axis(0)) {
                for (e, v) in row.iter_mut().enumerate() {
                    *v += bias[e];
                }
            }
        }
        Some(logits)
    }

    /// Byte slices for expert `e` at `layer`.
    ///
    /// Stride mirrors the writer's own packing: `gate_up` is
    /// `[experts, FUSED_BRANCHES * inter, hidden]` and `down` is
    /// `[experts, hidden, inter]`, both BF16.
    fn expert_bytes(&self, layer: usize, e: usize) -> Option<PackedExpert<'a>> {
        let arch = self.arch();
        let hidden = self.weights.hidden_size;
        let inter = arch.moe_intermediate_size();

        let gu_all = self
            .weights
            .get_packed_bytes(&arch.packed_experts_gate_up_key(layer)?)?;
        let dn_all = self
            .weights
            .get_packed_bytes(&arch.packed_experts_down_key(layer)?)?;

        let gu_stride = FUSED_BRANCHES * inter * hidden * BF16_BYTES;
        let dn_stride = hidden * inter * BF16_BYTES;
        Some(PackedExpert {
            gate_up: gu_all.get(e * gu_stride..(e + 1) * gu_stride)?,
            down: dn_all.get(e * dn_stride..(e + 1) * dn_stride)?,
            hidden,
            inter,
        })
    }

    /// Run one expert over the rows routed to it. `rows` is `[n, hidden]`.
    fn run_expert(
        &self,
        layer: usize,
        e: usize,
        rows: &Array2<f32>,
        layout: GateUpLayout,
    ) -> Option<Array2<f32>> {
        let arch = self.arch();
        let w = self.expert_bytes(layer, e)?;
        let (n, hidden, inter) = (rows.nrows(), w.hidden, w.inter);

        // gate/up: [n, inter] each, read row-by-row through the declared layout.
        let mut gate = Array2::<f32>::zeros((n, inter));
        let mut up = Array2::<f32>::zeros((n, inter));
        for i in 0..inter {
            let g_row = layout.row(GateUpBranch::Gate, i, inter) * hidden;
            let u_row = layout.row(GateUpBranch::Up, i, inter) * hidden;
            for t in 0..n {
                let xr = rows.slice(s![t, ..]);
                let mut g = 0.0f32;
                let mut u = 0.0f32;
                for j in 0..hidden {
                    let xj = xr[j];
                    g += xj * Self::bf16(w.gate_up, g_row + j);
                    u += xj * Self::bf16(w.gate_up, u_row + j);
                }
                gate[[t, i]] = g;
                up[[t, i]] = u;
            }
        }

        let act = gate::apply(&gate, &up, arch.expert_gate_policy(), arch.activation());

        // down: [hidden, inter] → out [n, hidden].
        let mut out = Array2::<f32>::zeros((n, hidden));
        for h in 0..hidden {
            let base = h * inter;
            for t in 0..n {
                let mut acc = 0.0f32;
                for i in 0..inter {
                    acc += act[[t, i]] * Self::bf16(w.down, base + i);
                }
                out[[t, h]] = acc;
            }
        }
        Some(out)
    }

    fn moe_block(&self, layer: usize, x: &Array2<f32>) -> Option<Array2<f32>> {
        // Refuse rather than guess: a packed architecture that declares no
        // layout has not answered a question this backend cannot infer.
        let layout = self.arch().gate_up_layout()?;
        self.moe_block_with_layout(layer, x, layout)
    }

    /// The block with the layout supplied rather than read from the arch.
    ///
    /// Exists so a test can drive the *same bytes* through both layouts and
    /// show the declaration is load-bearing. Production always reaches this
    /// through [`Self::moe_block`], which takes the architecture's answer.
    fn moe_block_with_layout(
        &self,
        layer: usize,
        x: &Array2<f32>,
        layout: GateUpLayout,
    ) -> Option<Array2<f32>> {
        let arch = self.arch();
        let logits = self.router_logits(layer, x)?;
        let top_k = arch.num_experts_per_token();
        let num_experts = arch.num_experts();
        let routing_policy = arch.expert_routing_policy();

        let mut buckets: Vec<(Vec<usize>, Vec<f32>)> = vec![(Vec::new(), Vec::new()); num_experts];
        let mut route_log = RouteLog::open(layer);
        for (token, row) in logits.axis_iter(Axis(0)).enumerate() {
            let routing = router::select(&row.to_vec(), top_k, routing_policy);
            route_log.record(token, &routing);
            for (expert, weight) in routing {
                if let Some(b) = buckets.get_mut(expert) {
                    b.0.push(token);
                    b.1.push(weight);
                }
            }
        }
        route_log.flush(routing_policy);

        let mut out = Array2::<f32>::zeros(x.raw_dim());
        for (e, (tokens, weights)) in buckets.iter().enumerate() {
            if tokens.is_empty() {
                continue;
            }
            let rows = x.select(Axis(0), tokens);
            let expert_out = self.run_expert(layer, e, &rows, layout)?;
            for (i, (&token, &weight)) in tokens.iter().zip(weights.iter()).enumerate() {
                let contribution = &expert_out.slice(s![i, ..]) * weight;
                let mut dst = out.slice_mut(s![token, ..]);
                dst += &contribution;
            }
        }
        Some(out)
    }
}

/// Opt-in capture of the router's decision, written only when
/// `LARQL_PACKED_ROUTE_LOG` names a file. Records the selected expert ids
/// *and* their weights, because the two policies under comparison differ in
/// exactly one of those: softmax and top-k both preserve rank, so the
/// selection should be identical while the weights should not. Capturing only
/// the ids would miss the whole effect; capturing only the score would not
/// show that selection was innocent.
struct RouteLog {
    layer: usize,
    sink: Option<PathBuf>,
    rows: Option<Vec<String>>,
}

impl RouteLog {
    fn open(layer: usize) -> Self {
        // Layer 0 only: the first routed FFN is where a router defect must
        // first appear, and later layers are downstream of it.
        Self::to_sink(layer, std::env::var_os(ROUTE_LOG_ENV).map(PathBuf::from))
    }

    /// The sink as a parameter rather than read from the environment, so a
    /// test can drive the capture without mutating process-global state that
    /// every other test in the binary shares.
    fn to_sink(layer: usize, sink: Option<PathBuf>) -> Self {
        let on = layer == 0 && sink.is_some();
        Self {
            layer,
            sink,
            rows: on.then(Vec::new),
        }
    }

    fn record(&mut self, token: usize, routing: &[(usize, f32)]) {
        if let Some(rows) = self.rows.as_mut() {
            let ids: Vec<String> = routing.iter().map(|(e, _)| e.to_string()).collect();
            let ws: Vec<String> = routing.iter().map(|(_, w)| format!("{w:.9}")).collect();
            rows.push(format!(
                "{{\"token\":{token},\"experts\":[{}],\"weights\":[{}]}}",
                ids.join(","),
                ws.join(",")
            ));
        }
    }

    fn flush(self, policy: larql_models::ExpertRoutingPolicy) {
        let (Some(rows), Some(path)) = (self.rows, self.sink) else {
            return;
        };
        let header = format!("{{\"layer\":{},\"policy\":\"{:?}\"}}", self.layer, policy);
        let body = std::iter::once(header)
            .chain(rows)
            .collect::<Vec<_>>()
            .join("\n");
        let _ = std::fs::write(path, body + "\n");
    }
}

impl FfnBackend for PackedExpertWeightFfn<'_> {
    fn forward(&self, layer: usize, x: &Array2<f32>) -> Array2<f32> {
        self.moe_block(layer, x).unwrap_or_else(|| {
            let arch = self.arch();
            panic!(
                "packed MoE reference could not resolve layer {layer}: \
                 format={:?}, gate_up_layout={:?}, gate_up_key={:?}. A packed \
                 architecture must declare its fused layout — reading the \
                 wrong one mixes gate and up silently rather than failing.",
                arch.expert_format(),
                arch.gate_up_layout(),
                arch.packed_experts_gate_up_key(layer),
            )
        })
    }

    fn forward_observed(&self, layer: usize, x: &Array2<f32>) -> (Array2<f32>, FfnActivations) {
        (self.forward(layer, x), FfnActivations::unobserved())
    }

    fn name(&self) -> &str {
        "packed-expert-weight-f32"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_models::test_fixtures::make_test_gemma4_moe_weights;

    fn input(weights: &larql_models::ModelWeights, rows: usize) -> Array2<f32> {
        Array2::from_shape_fn((rows, weights.hidden_size), |(r, c)| {
            ((r * 31 + c) as f32 * 0.017).sin() * 0.2
        })
    }

    /// The packed backend resolves a stacked BF16 store and produces a finite
    /// block of the right shape — the capability that did not exist before.
    #[test]
    fn packed_store_runs_and_is_finite() {
        let weights = make_test_gemma4_moe_weights();
        assert_eq!(
            weights.arch.expert_format(),
            larql_models::ExpertFormat::PackedBF16
        );
        let x = input(&weights, 3);
        let out = PackedExpertWeightFfn { weights: &weights }
            .moe_block(0, &x)
            .expect("packed layer 0 resolves");
        assert_eq!(out.shape(), x.shape());
        assert!(out.iter().all(|v| v.is_finite()));
    }

    /// Fingerprint every axis of the fused operand.
    ///
    /// A representation test has to contain data that separates every
    /// hypothesis it claims to discriminate. The shared fixture fills
    /// `gate_up` from `i & 0xff`, which is periodic across the row block, so
    /// contiguous and interleaved read *identical numbers* through it — the
    /// first version of the control below passed with `max|delta| == 0`,
    /// proving nothing at all.
    ///
    /// A value that varies only with `(expert, row)` fixes that one case and
    /// still cannot see a transpose or a column-stride error, so the stamp is
    /// a function of all three axes. Accidental symmetry in a fixture is how
    /// a permutation, stride, transpose or interleave mistake stays invisible.
    fn stamp_per_axis_fingerprint(weights: &mut larql_models::ModelWeights) {
        let arch = &*weights.arch;
        let (hidden, inter, experts) = (
            weights.hidden_size,
            arch.moe_intermediate_size(),
            arch.num_experts(),
        );
        let key = arch.packed_experts_gate_up_key(0).expect("packed key");
        let blob = weights
            .raw_bytes
            .get_mut(&key)
            .expect("fixture stores gate_up in raw_bytes");
        let rows = FUSED_BRANCHES * inter;
        for e in 0..experts {
            for r in 0..rows {
                for j in 0..hidden {
                    // Multiplicative, so each axis leaves a signature the
                    // others cannot mask. A purely additive stamp fails here:
                    // summed over `hidden`, every row's dot product comes out
                    // nearly equal and permuting rows moves the result by
                    // 0.03% — under the bar, and the control goes green while
                    // barely discriminating. Row amplitude therefore spans ~3x
                    // (so a row permutation is unmissable) while the column
                    // term keeps a stride or transpose error visible.
                    let row_amp = 0.02 + 0.04 * ((e * 5 + r * 13) % 11) as f32 / 11.0;
                    let col_sig = 0.6 + 0.8 * ((j * 7) % 13) as f32 / 13.0;
                    let v = row_amp * col_sig;
                    let be = ((v.to_bits() >> 16) as u16).to_le_bytes();
                    let idx = ((e * rows + r) * hidden + j) * BF16_BYTES;
                    blob[idx] = be[0];
                    blob[idx + 1] = be[1];
                }
            }
        }
    }

    /// **The layout declaration is load-bearing, not metadata.**
    ///
    /// Same expert bytes, same router, same experts selected — only the rule
    /// for which fused row is gate and which is up differs. On the real model
    /// this separates 0.002% agreement with the HF reference from 89.7%
    /// disagreement, and note what it does *not* do: the wrong reading stays
    /// finite. It does not crash, it computes a different model. That is why
    /// this has to be declared rather than inferred.
    #[test]
    fn reading_the_wrong_fused_layout_changes_the_result() {
        let mut weights = make_test_gemma4_moe_weights();
        stamp_per_axis_fingerprint(&mut weights);
        let ffn = PackedExpertWeightFfn { weights: &weights };
        let x = input(&weights, 3);

        let declared = ffn
            .moe_block_with_layout(0, &x, GateUpLayout::ContiguousHalves)
            .expect("contiguous resolves");
        let wrong = ffn
            .moe_block_with_layout(0, &x, GateUpLayout::Interleaved)
            .expect("interleaved also resolves — it is wrong, not unresolvable");

        assert!(
            wrong.iter().all(|v| v.is_finite()),
            "the wrong layout must stay finite; a crash would make this easy to catch"
        );
        let delta = (&declared - &wrong)
            .iter()
            .fold(0.0f32, |m, v| m.max(v.abs()));
        let scale = declared.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(
            delta > scale * 0.01,
            "layouts must disagree materially: max|delta|={delta}, scale={scale}"
        );
    }

    /// The gate/up indexing rule itself — the one primitive the oracle shares
    /// with any execution path, so it is pinned directly.
    #[test]
    fn gate_up_row_indexing_matches_the_two_reference_splits() {
        // chunk(2, dim=-1): first half gate, second half up.
        let c = GateUpLayout::ContiguousHalves;
        assert_eq!(c.row(GateUpBranch::Gate, 0, 4), 0);
        assert_eq!(c.row(GateUpBranch::Gate, 3, 4), 3);
        assert_eq!(c.row(GateUpBranch::Up, 0, 4), 4);
        assert_eq!(c.row(GateUpBranch::Up, 3, 4), 7);
        // [..., ::2] / [..., 1::2]: even rows gate, odd rows up.
        let i = GateUpLayout::Interleaved;
        assert_eq!(i.row(GateUpBranch::Gate, 0, 4), 0);
        assert_eq!(i.row(GateUpBranch::Gate, 3, 4), 6);
        assert_eq!(i.row(GateUpBranch::Up, 0, 4), 1);
        assert_eq!(i.row(GateUpBranch::Up, 3, 4), 7);
        // Both are permutations of the same row block — no row is lost or
        // visited twice, which is exactly why the wrong one cannot be caught
        // by a shape or bounds check.
        for layout in [c, i] {
            let mut seen: Vec<usize> = (0..4)
                .flat_map(|k| {
                    [
                        layout.row(GateUpBranch::Gate, k, 4),
                        layout.row(GateUpBranch::Up, k, 4),
                    ]
                })
                .collect();
            seen.sort_unstable();
            assert_eq!(seen, (0..8).collect::<Vec<_>>());
        }
    }

    /// The router also arrives as a `[experts, hidden]` tensor rather than a
    /// flattened vector, depending on the loader path. Both readings must give
    /// the same block — a row-major/column-major slip here would reroute every
    /// token while still producing a finite result.
    #[test]
    fn router_resolves_identically_from_tensors_or_vectors() {
        let from_vectors = make_test_gemma4_moe_weights();
        let key = from_vectors.arch.moe_router_key(0).expect("router key");
        let flat = from_vectors
            .vectors
            .get(&key)
            .expect("fixture uses vectors")
            .clone();

        let mut from_tensors = make_test_gemma4_moe_weights();
        let hidden = from_tensors.hidden_size;
        let experts = flat.len() / hidden;
        from_tensors.vectors.remove(&key);
        from_tensors.tensors.insert(
            key,
            Array2::from_shape_vec((experts, hidden), flat)
                .expect("router reshapes to [experts, hidden]")
                .into_shared(),
        );

        let x = input(&from_vectors, 2);
        let a = PackedExpertWeightFfn {
            weights: &from_vectors,
        }
        .moe_block(0, &x)
        .expect("vector router resolves");
        let b = PackedExpertWeightFfn {
            weights: &from_tensors,
        }
        .moe_block(0, &x)
        .expect("tensor router resolves");
        assert_eq!(a, b, "the two router homes must be the same matrix");
    }

    /// The `FfnBackend` surface itself — `forward` is what the scorer calls.
    #[test]
    fn forward_matches_the_block_and_reports_its_name() {
        let weights = make_test_gemma4_moe_weights();
        let ffn = PackedExpertWeightFfn { weights: &weights };
        let x = input(&weights, 2);
        assert_eq!(ffn.forward(0, &x), ffn.moe_block(0, &x).expect("resolves"));
        let (observed, acts) = ffn.forward_observed(0, &x);
        assert_eq!(observed, ffn.forward(0, &x));
        assert!(matches!(acts, FfnActivations::Absent { .. }));
        assert_eq!(ffn.name(), "packed-expert-weight-f32");
    }

    /// The router capture records ids *and* weights, and only for layer 0.
    ///
    /// This is the instrument that showed selection was innocent while the
    /// weights moved, so it is worth pinning that it emits both — an id-only
    /// capture would have missed the entire GraniteMoe effect.
    #[test]
    fn route_log_captures_ids_and_weights_for_layer_zero_only() {
        let dir = std::env::temp_dir().join("larql_route_log_test");
        std::fs::create_dir_all(&dir).expect("tmp dir");
        let path = dir.join("layer0.jsonl");
        let _ = std::fs::remove_file(&path);

        let mut log = RouteLog::to_sink(0, Some(path.clone()));
        log.record(0, &[(3usize, 0.75f32), (1, 0.25)]);
        log.flush(larql_models::ExpertRoutingPolicy::NormalisedOverSelected);

        let body = std::fs::read_to_string(&path).expect("capture written");
        let mut lines = body.lines();
        assert!(lines
            .next()
            .expect("header")
            .contains("NormalisedOverSelected"));
        let row = lines.next().expect("one token row");
        assert!(row.contains("\"experts\":[3,1]"), "{row}");
        assert!(
            row.contains("0.750000000") && row.contains("0.250000000"),
            "{row}"
        );

        // A later layer is downstream of the first router decision, so it is
        // deliberately not captured even with a sink configured.
        let path2 = dir.join("layer3.jsonl");
        let _ = std::fs::remove_file(&path2);
        let mut off = RouteLog::to_sink(3, Some(path2.clone()));
        off.record(0, &[(1usize, 1.0f32)]);
        off.flush(larql_models::ExpertRoutingPolicy::SoftmaxThenSelect);
        assert!(!path2.exists(), "non-zero layers must not write a capture");

        // No sink configured is the default path: nothing is written at all.
        let mut none = RouteLog::to_sink(0, None);
        none.record(0, &[(1usize, 1.0f32)]);
        none.flush(larql_models::ExpertRoutingPolicy::SoftmaxThenSelect);
        let _ = std::fs::remove_file(&path);
    }
}
