## ADDED Requirements: canonical-vindex

### REQ-CANON-001: Covariance estimation

The system SHALL estimate the activation covariance matrix G of shape
[hidden_size, hidden_size] from the token embedding matrix W_E using a
deterministic subsample of at most `max_samples` rows.

G = (1/N) Σ_{v ∈ S} (s · W_E[v])^T (s · W_E[v])

where s = embed_scale and S is a uniformly-strided subsample of the vocabulary.

#### Scenario: identity embeddings produce scaled identity covariance

Given token embeddings equal to the d×d identity matrix and embed_scale=1.0,
when covariance is estimated with max_samples=d,
then G[i,i] = 1/d and G[i,j] = 0 for i≠j.

<!-- test: crates/larql-vindex/src/canonical/covariance.rs::tests::covariance_of_identity_is_identity_scaled -->

#### Scenario: embed_scale squares into covariance

Given any embeddings and embed_scale s,
when covariance is estimated,
then G(s) = s² · G(1).

<!-- test: crates/larql-vindex/src/canonical/covariance.rs::tests::embed_scale_squares_into_covariance -->

#### Scenario: G is symmetric and positive semi-definite

Given any token embeddings,
when covariance is estimated,
then G[i,j] = G[j,i] for all i,j and G[i,j]² ≤ G[i,i]·G[j,j] (Cauchy-Schwarz).

<!-- test: crates/larql-vindex/src/canonical/covariance.rs::tests::covariance_is_symmetric -->
<!-- test: crates/larql-vindex/src/canonical/covariance.rs::tests::covariance_is_positive_semidefinite -->

---

### REQ-CANON-002: Cholesky whitening factor

The system SHALL compute the lower-triangular Cholesky factor L of G such that
L L^T = G + ε·I (with ε = 1e-5 ridge for numerical stability).

The system SHALL provide a packing function that represents L as a flat
Vec<f64> of length d·(d+1)/2 in row-major lower-triangular order:
index(i,j) = i·(i+1)/2 + j for j ≤ i.

The system SHALL provide an unpacking function that recovers the dense L from
the packed form.

#### Scenario: L L^T recovers G up to ridge

Given G, when L = cholesky(G, ridge=1e-5),
then (L L^T)[i,j] ≈ G[i,j] + 1e-5·δ_{ij} within 1e-8 tolerance.

<!-- test: crates/larql-vindex/src/canonical/whitening.rs::tests::cholesky_recovers_g -->

#### Scenario: L is lower-triangular

Given G, when L = cholesky(G, ridge), then L[i,j] = 0 for j > i.

<!-- test: crates/larql-vindex/src/canonical/whitening.rs::tests::l_is_lower_triangular -->

#### Scenario: pack/unpack round-trips

Given a Cholesky factor L, when packed then unpacked,
the result equals the original within f64 precision.

<!-- test: crates/larql-vindex/src/canonical/whitening.rs::tests::unpack_roundtrips_packed -->

#### Scenario: whitened dot product equals Mahalanobis inner product

Given g and h in ℝ^d, let g̃ = L^{-T}g and h̃ = L^{-T}h.
Then g̃·h̃ = g^T G^{-1} h (Mahalanobis inner product).

<!-- test: crates/larql-vindex/src/canonical/whitening.rs::tests::whitening_makes_mahalanobis_a_dot_product -->

---

### REQ-CANON-003: Back-substitution for L^T

The system SHALL provide `back_solve_lt(L, B)` that solves L^T X = B via
back-substitution, where L is lower-triangular. This is the primitive used to
compute whitened vectors g̃ = L^{-T} g.

The system SHALL provide `compute_l_inv_t(L)` that returns the dense d×d
matrix L^{-T} by solving L^T X = I.

#### Scenario: back_solve_lt recovers B

Given lower-triangular L and matrix B,
when Z = back_solve_lt(L, B), then L^T Z = B within 1e-10 tolerance.

<!-- test: crates/larql-compute/src/cpu/ops/linalg.rs::canonical_linalg_tests::back_solve_lt_recovers_rhs -->

#### Scenario: compute_l_inv_t times L^T is identity

Given L, when M = compute_l_inv_t(L), then M @ L^T = I within 1e-10.

<!-- test: crates/larql-compute/src/cpu/ops/linalg.rs::canonical_linalg_tests::compute_l_inv_t_times_l_t_is_identity -->

---

### REQ-CANON-004: On-shell feature filter

The system SHALL compute a boolean on-shell mask for each layer by ranking
features by their c_score (logit of top down-projected token from down_meta)
and marking the top `top_fraction` (default 0.15) as on-shell.

At least one feature SHALL be on-shell even if only one feature exists.
The mask length SHALL equal the number of features in the layer.
An empty feature list SHALL produce an empty mask.

#### Scenario: top 15% of 10 features selects 2

Given 10 features with c_scores 0..9 and top_fraction=0.15 (ceil(1.5)=2),
the on-shell features are those with the two highest c_scores (8 and 9).

<!-- test: crates/larql-vindex/src/canonical/onshell.rs::tests::top_15pct_of_10_is_2 -->

#### Scenario: all-equal c_scores makes all features on-shell

When all c_scores are equal, the threshold equals all scores,
so all features are on-shell.

<!-- test: crates/larql-vindex/src/canonical/onshell.rs::tests::all_same_scores_all_on_shell -->

---

### REQ-CANON-005: Layer regime classifier

The system SHALL classify each layer as Wave, Particle, or Wavelet based on
activation density = (features with c_score > 0.1) / total_features:

- density > 0.5  → Wave
- density < 0.05 → Particle
- otherwise      → Wavelet

An empty feature list SHALL return Wavelet with density 0.0.

#### Scenario: dense layer is Wave

Given 80% of features with c_score > 0.1, regime = Wave.

<!-- test: crates/larql-vindex/src/canonical/regime.rs::tests::dense_layer_is_wave -->

#### Scenario: sparse layer is Particle

Given 2% of features with c_score > 0.1, regime = Particle.

<!-- test: crates/larql-vindex/src/canonical/regime.rs::tests::sparse_layer_is_particle -->

#### Scenario: mid-density is Wavelet

Given 20% of features active, regime = Wavelet.

<!-- test: crates/larql-vindex/src/canonical/regime.rs::tests::mid_density_is_wavelet -->

---

### REQ-CANON-006: Canonical metadata serialization

The system SHALL write `canonical_meta.json` to the vindex directory containing:
- version (u32 = 1)
- model, family, num_layers, hidden_size strings/ints from index.json
- covariance_sample_size and embed_scale
- cholesky_l_packed: Vec<f64> of length hidden_size·(hidden_size+1)/2
- layers: array of {layer, regime, on_shell_count, total_features, mean_density}

The JSON SHALL round-trip losslessly through serde_json.

#### Scenario: regime serializes as snake_case

Wave → "wave", Particle → "particle", Wavelet → "wavelet".

<!-- test: crates/larql-vindex/src/canonical/types.rs::tests::regime_serialises_as_snake_case -->

#### Scenario: CanonicalMeta round-trips

Given a CanonicalMeta value, serializing to JSON and deserializing
reproduces the original value.

<!-- test: crates/larql-vindex/src/canonical/types.rs::tests::canonical_meta_round_trips_through_json -->

---

### REQ-CANON-007: c_score-only down_meta reader

The system SHALL provide `read_cscores_binary(dir)` that reads c_score values
from `down_meta.bin` without constructing token strings (no tokenizer
dependency). It SHALL return Vec<Vec<f32>>, one inner Vec per layer.
Layers with num_features=0 SHALL return an empty inner Vec.

#### Scenario: c_scores match full read

Given a down_meta.bin file, read_cscores_binary returns the same c_score
values as the full read_binary function.

<!-- test: crates/larql-vindex/src/format/down_meta.rs::tests::read_cscores_binary_matches_full_read -->

---

### REQ-CANON-008: CLI command `larql canonicalize`

The system SHALL provide a `larql canonicalize <vindex-path>` subcommand
that loads a vindex directory, runs the canonical pipeline, and writes
`canonical_meta.json`. The command SHALL NOT modify any existing vindex files.

The command SHALL accept:
- `--onshell-fraction <f32>` (default 0.15)
- `--covariance-samples <usize>` (default 4096)

The command SHALL print progress and a summary of on-shell feature counts.

#### Scenario: canonical_meta.json is written

Given a valid vindex directory, after `larql canonicalize`,
canonical_meta.json exists and is valid JSON matching the schema.

<!-- test: integration: cargo run -- canonicalize <vindex> then check file exists -->

#### Scenario: existing files are untouched

After `larql canonicalize`, all pre-existing vindex files have the same
content as before.

<!-- test: integration: checksum vindex files before/after -->
