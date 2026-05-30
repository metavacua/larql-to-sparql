# `larql convert quantize` — CLI surface spec

**Status:** FP4 + Q4K shipped (exp 26). Future formats extensible
through the same grammar.
**Scope:** CLI shape for converting a loaded vindex into a quantised
variant. Each format is a sibling subcommand under `quantize`, with
its own flag surface. FP4 and Q4K are wired today; future formats
land as additional subcommands without changing the grammar.
**Format-specific references:**
- FP4: [`fp4-format-spec.md`](fp4-format-spec.md) (byte layout),
  [`fp4-precision-policy.md`](fp4-precision-policy.md) (A/B/C
  policies + compliance gate).
- Q4K: GGML "Q4_K_M" mix (Q4_K gate/up + Q6_K down), Ollama-
  compatible. Library entry: `larql_vindex::quant::vindex_to_q4k`
  on top of `format::weights::write_model_weights_q4k_with_opts`.

---

## 0. The umbrella

`larql convert quantize <format>` is the family entry point:

```
larql convert quantize fp4   [fp4 flags]         ← wired today
larql convert quantize q4k   [q4k flags]         ← wired today
larql convert quantize fp6   [fp6 flags]         ← future
larql convert quantize ...   [format-specific]
```

Format-specific flag sets stay isolated (FP4's `--policy` /
`--compliance-floor` / `--threshold` don't clutter Q4K's
invocation), but users have one mental model: "quantise a vindex."

**Adding a new format is three edits:**

1. One `QuantizeCommand::FooBar { ... }` variant in `convert_cmd.rs`.
2. One `run_quantize_foobar` fn delegating to the format's library
   entry.
3. One library fn `larql_vindex::quant::vindex_to_foobar(src, dst, config)`
   mirroring the shape of `vindex_to_fp4`.

No other CLI or library code touches. Other formats' flag surfaces
are unaffected. This is the structural payoff of the nested-
subcommand grammar: the CLI grows linearly, not combinatorially.

## 1. Why a spec before code

The example binary (`crates/larql-vindex/examples/fp4_convert.rs`)
already did the work. Promoting it to `larql convert quantize fp4`
was mostly mechanical, but a few things needed pinning before we
wrote the clap subcommand so the output is stable across format
revisions:

- **Flag surface** — which knobs are user-facing, which are internal,
  which get deprecated later.
- **Self-policing gate** — what happens when a projection fails the
  compliance floor, how it's reported, whether the run is allowed to
  continue or is treated as an error.
- **Output directory layout** — what files land, what gets hard-linked
  from the source, what's optional.
- **Failure modes** — what a non-success run looks like (what's
  written, what's emitted to stderr, what the exit code is).
- **Diagnostics** — where the dispatch trace / describe helpers
  integrate so a user can tell at a glance whether the output will
  actually be FP4 end-to-end.

Pinning these now means the first real `larql convert` run that ships
to someone outside the repo produces output whose schema is stable.

## 2. FP4 invocation

```
larql convert quantize fp4 \
    --input  SRC                               # existing vindex directory
    --output DST                               # new vindex directory
    [--policy option-a | option-b | option-c]  # default: option-b
    [--compliance-floor FRAC]                  # default: 0.99
    [--threshold RATIO]                        # default: 16.0 (format-derived)
    [--force]                                  # overwrite DST if present
    [--strict]                                 # fail on any compliance-floor miss
    [--no-sidecar]                             # skip fp4_compliance.json emission
    [--quiet]                                  # suppress backend-describe output
```

**Defaults are the "just works for the common case" path.** Running
`larql convert quantize fp4 --input X --output Y` produces an
Option B vindex (source-dtype gate + FP4 up + FP8 down), with the Q1
compliance scan written to `DST/fp4_compliance.json` and the one-line
backend summary printed on stdout. The defaults match the policy
spec's recommended Option B, so users who just want "the default FP4
vindex" don't need any flags.

**`--threshold` help text must explain the default, not leave it as a
number.** The 16.0 default is the format-derived E4M3-vs-E2M1 exponent
budget (see `FP4_FORMAT_SPEC.md` §5.1 and the DeepSeek reference).
Users who raise it are being more permissive about FP4 block
compliance; users who lower it are being stricter. Example help
text: `--threshold RATIO    max/min sub-block scale ratio for the
FP4 compliance gate (default: 16.0, the E4M3/E2M1 exponent budget;
lower = stricter, higher = more permissive)`.

## 3. FP4 behavior sketch

```
> larql convert quantize fp4 --input output/gemma3-4b-f16.vindex --output output/gemma3-4b-fp4.vindex

== quantize fp4 ==
  in     : output/gemma3-4b-f16.vindex
  out    : output/gemma3-4b-fp4.vindex
  model  : google/gemma-3-4b-it
  policy : option-b (gate=source, up=FP4, down=FP8)
  floor  : 99.0% compliance at R<16.0

→ scanning reference vindex …
    gate  : 99.91%   → keep as f32 (gate stays at source dtype; FP4 gate blocked on FP4-aware KNN path)
    up    : 99.93%   → FP4         (meets floor)
    down  : 99.65%   → FP8         (policy: down is always FP8 under option-b; compliance floor N/A for FP8)

→ writing output …
    gate_vectors.bin         (hard-link, 3.32 GB)
    up_features_fp4.bin      (new,  0.44 GB)
    down_features_fp8.bin    (new,  0.85 GB)
    fp4_compliance.json      (new)
    index.json               (new, fp4 manifest attached)
    [auxiliary files hard-linked: attn_weights.bin, down_meta.bin, embeddings.bin, …]

── summary ──
  FFN storage : 9.96 GB → 4.60 GB  (2.17× compression)
  Walk backend: FP4 sparse (gate=f32, up=fp4, down=fp8), gate KNN (F32 mmap)
  Wall time   : 12.3s

  → load output with LARQL_VINDEX_DESCRIBE=1 to verify the backend at runtime.
```

Compliance failures (projection targeted for FP4 falls below floor):

```
    down  : 98.42%   → FP8 (policy: down is always FP8 under option-b; floor N/A for FP8)
    up    : 97.80%   ⚠ DOWNGRADE: FP4 floor (99.0%) missed → writing as FP8 (fallback_precision from manifest)

⚠ compliance floor missed on 1 projection; see fp4_compliance.json for details.
(Use --strict to treat this as a fatal error.)
```

The compliance floor is a **precision-FP4 gate**, not a per-projection
gate. It only applies where the policy says "write this projection
as FP4"; projections targeted for FP8 or F16 skip the check entirely
(FP8 doesn't use the max/min-sub-block-scale distributional
assumption, and F16 is bit-identical to source). That's why the down
line above reads "floor N/A for FP8" — it's not a bug in the log
output, it's the honest description of what the floor measures.

Under `--strict`, the same scenario exits non-zero after writing the
compliance sidecar. Under default, the converter downgrades the
affected projection to the fallback precision from the manifest's
`compliance_gate` and continues.

## 4. Q4K invocation + behavior

```
larql convert quantize q4k \
    --input  SRC                  # existing vindex with full f32/f16 weights
    --output DST                  # new vindex directory
    [--down-q4k]                  # FFN down at Q4_K instead of Q6_K (Q4_K_M default keeps it at Q6_K)
    [--force]                     # overwrite DST if present
    [--quiet]                     # suppress backend-describe output
```

**The default produces an Ollama-compatible Q4_K_M mix:** attention
Q/K/O at Q4_K, attention V at Q6_K, FFN gate/up at Q4_K, FFN down at
Q6_K. `--down-q4k` switches FFN down to Q4_K uniformly — saves ~30 MB
per layer on a 31B model (~1.8 GB total) at modest precision cost
that the empirical scatter-sum averages across the intermediate
dimension (validated by `walk_correctness`, which auto-relaxes its
prob-delta gate from 0.02 to 0.035 when Q4_K down is detected).

**Precondition:** the source vindex must have full model weights
(`extract_level: inference` or `all`). The Q4K writer reads every
attention and FFN tensor from the source and rewrites them as
quantised blocks; a browse-only vindex (no `attn_weights.bin` /
`up_weights.bin` / `down_weights.bin`) is rejected with a clear
error pointing at `--level inference`. Quantised sources (`quant !=
none`) are also rejected — re-quantising an already-quantised vindex
is a no-op or worse.

```
> larql convert quantize q4k --input output/gemma3-4b-f16.vindex --output output/gemma3-4b-q4k.vindex

== quantize q4k ==
  in       : output/gemma3-4b-f16.vindex
  out      : output/gemma3-4b-q4k.vindex
  down_q4k : false (Q6_K down (Q4_K_M mix))

── summary ──
  FFN storage : 6.64 GB → 4.94 GB  (1.35× compression)
  Linked aux  : 6 files (4.63 GB)
  Wall time   : 13.5s
  Walk backend: Q4K interleaved, gate KNN (F32 mmap)

→ output/gemma3-4b-q4k.vindex
```

Q4K's compression ratio is more modest than FP4's because (a) the
4-bit nibble is paired with a richer per-block scale + min layout
(GGML Q4_K is 144 B per 256-element super-block vs FP4's 137 B), and
(b) the V-projection and FFN down stay at Q6_K by default. The
tradeoff is precision: Q4K is the same format llama.cpp / Ollama
ship with and is validated against the Gemma walk-correctness gate;
FP4 is an experimental spatially-sparser layout with its own
compliance regime.

### Output layout (Q4K)

```
DST/
├── index.json                        # quant=q4k, has_model_weights=true
│
│  # ── Hard-linked from SRC (zero-copy, no rewrite) ──
├── gate_vectors.bin                  # gate matrix (KNN still wants the dense float view)
├── embeddings.bin
├── down_meta.bin
├── feature_labels.json
├── tokenizer.json
├── README.md                         # if SRC carried one
│
│  # ── Written by this run ──
├── attn_weights_q4k.bin              # Q/K/O at Q4_K, V at Q6_K
├── attn_weights_q4k_manifest.json
├── interleaved_q4k.bin               # gate + up at Q4_K, down at Q6_K (or Q4_K with --down-q4k)
├── interleaved_q4k_manifest.json
├── lm_head_q4.bin                    # output projection at Q4_K
├── norms.bin                         # layer + final norms (always f32)
└── weight_manifest.json
```

The float weight files (`attn_weights.bin`, `up_weights.bin`,
`down_weights.bin`, `interleaved.bin`, `lm_head.bin`) from the
source are **not** hard-linked — the Q4K weight files replace them.
Hard-linking the floats too would inflate the output by 6+ GB on a
4B model with no consumer for those bytes.

### Atomic write

Like FP4, the writer stages into `DST.tmp/` and renames on success.
Partial output never carries a valid `index.json`, so a crashed run
is unambiguously distinguishable from a complete one.

## 5. Exit codes

| Code | Meaning                                                            |
| ---- | ------------------------------------------------------------------ |
| 0    | Output produced; all policy-specified projections written.         |
| 1    | Input vindex invalid, missing files, or unsupported geometry.      |
| 2    | Compliance floor missed on ≥ 1 projection AND `--strict` was set.  |
| 3    | I/O error writing output.                                          |
| 4    | Output exists and `--force` not provided.                          |

Non-success codes always leave `DST` either absent (on early failure)
or with a partial output clearly tagged by the absence of
`index.json` (written atomically at the end of the run).

## 6. Self-policing gate integration (FP4 only)

The Q1 scanner (`crates/larql-vindex/examples/fp4_q1_scan.rs`)
currently lives as an example. For `larql convert quantize fp4` it
is promoted to `larql_vindex::quant::scan` — a library entry the
convert subcommand calls directly, producing an in-memory
`ComplianceReport` that the converter consults before deciding the
per-projection precision.

Scanner-as-library invariants:
- No filesystem I/O inside the scanner itself (reads come from the
  `VectorIndex` accessors, which already mmap the data).
- Pure function: `scan(index, threshold) -> ComplianceReport`.
- Report is the same JSON shape the example emits, minus any CLI-only
  framing.

This makes the Q1 scanner usable anywhere — the convert subcommand
today, future `larql verify --fp4` tomorrow, regression tests next
week. One implementation, multiple consumers.

## 7. FP4 output layout

```
DST/
├── index.json                  # updated: fp4 manifest attached, checksums refreshed
├── fp4_compliance.json         # per-projection scan + action taken
│
│  # ── Hard-linked from SRC (zero-copy, no rewrite) ──
├── attn_weights.bin            # attention
├── down_meta.bin               # per-feature output token metadata
├── embeddings.bin              # embed
├── feature_labels.json         # labels
├── gate_vectors.bin            # gate kept at source dtype (policy default)
├── norms.bin                   # layer norms
├── tokenizer.json
├── weight_manifest.json
│
│  # ── Written by this run ──
├── up_features_fp4.bin         # FP4 E2M1, 256-elem blocks
└── down_features_fp8.bin       # FP8 E4M3, 256-elem blocks
```

Files are listed in the same order the converter's summary prints
them, so the stdout output can be diffed against `ls DST/` to
confirm the write.

### Hard-link fallback

On filesystems that don't support hard links (cross-filesystem, some
network mounts), the converter falls back to file copy and emits a
one-line notice. The output is functionally identical; size on disk
doubles for the hard-linked portion. Should be rare in practice.

## 8. Diagnostics that ship with the subcommand

Three observability hooks, all default-on:

1. **Backend summary line** (already implemented via
   `VectorIndex::describe_ffn_backend()`). Printed on stdout after
   the write. Suppressed with `--quiet`.
2. **Compliance sidecar path** echoed in the summary. Makes it
   obvious where to look when investigating a compliance miss.
3. **One-liner suggesting `LARQL_VINDEX_DESCRIBE=1`** for users who
   want to double-check the backend at runtime (not just at convert
   time).

This is deliberately conservative — we're not emitting verbose trace
by default. Users running into trouble enable `LARQL_WALK_TRACE=1` at
runtime. The convert subcommand itself should be quiet by default
and only noisy on anomalies.

## 9. Testing surface

The existing tests mostly transfer:

| Existing test                                                | Covers |
| ------------------------------------------------------------ | ------ |
| `tests/test_fp4_synthetic` (7 tests)                         | Per-feature round-trip through a loaded FP4 vindex — the kind `larql convert` produces. |
| `tests/test_fp4_storage` (4 tests, real fixture)             | End-to-end against `gemma3-4b-fp4.vindex`. Switching to `larql convert`-produced output changes nothing. |
| `format::fp4_storage::tests` (7 tests)                       | File-level writer/reader. The converter uses these via `write_fp4_projection` / `write_fp8_projection`. |
| `index::fp4_storage::tests` (13 tests)                       | Per-projection storage — same abstraction. |
| `walk_ffn::routing_tests` (3 tests)                          | Predicate ladder, including the Q2-regression guard. |

New tests the CLI subcommand needs:

1. **Smoke:** invoke the CLI with a small synthetic input vindex,
   assert stdout contains the expected summary lines and that DST
   has the expected filenames.
2. **Exit codes:** invoke with `--force` absent when DST exists →
   exit 4. Invoke with `--strict` and a synthetic input rigged to
   miss compliance → exit 2.
3. **Self-policing:** invoke with a synthetic input that has a
   projection below the floor (inject a pathological block) →
   verify the output manifest records the downgrade and the stored
   file is the fallback precision.
4. **Round-trip parity:** convert synthetic SRC → DST, load DST,
   compare row reads to SRC f32 data within the expected FP4 bound.

Four tests, ~200 LOC total, all using the tempdir pattern already
established in `tests/test_fp4_synthetic.rs`.

## 10. What this does NOT do (v1)

- **Safetensors-direct FP4 extract.** Two-step (`extract` then
  `quantize fp4`) remains the workflow. The reason is decoupling:
  the FP4 writer should never need to know about extract-time
  concerns (HuggingFace format quirks, model-specific weight
  reorganisation, tied-embedding detection, PLE handling for
  Gemma 4 E2B). The vindex is the stable intermediate — if FP4
  conversion is a function of a vindex, it composes cleanly with
  whatever extract path produced that vindex, now and in the future.
  Merging the two into a single "safetensors-to-FP4" entry point
  would duplicate extract logic and couple the FP4 writer to
  loader-specific surprises.
- **Mixed-precision override per-layer.** `--layers 0..12 down=fp4,
  13.. down=fp8` style is deferred. Data doesn't yet say it buys
  anything; revisit after cross-model Q2.
- **In-place conversion.** No `--in-place` flag. The existing vindex
  stays untouched; the FP4 copy is separate. Reversibility matters.
- **GGUF / MLX interop.** Out of scope; this operates on LARQL
  vindexes only.

## 11. Shipping checklist

- [x] Promote `fp4_q1_scan` from example to library
      (`larql_vindex::quant::scan`). Preserve the example binary as a
      thin wrapper so existing scripts keep working.
- [x] Promote `fp4_convert` logic to a library fn
      (`larql_vindex::quant::vindex_to_fp4`). Example binary becomes
      a thin wrapper.
- [x] Add `ConvertCommand::Quantize(QuantizeCommand)` + `Fp4` and
      `Q4k` variants in
      `crates/larql-cli/src/commands/extraction/convert_cmd.rs` with
      the flag surfaces above.
- [x] Wire `run_quantize_fp4` and `run_quantize_q4k` to the library
      fns.
- [x] Add the 4 CLI-level tests listed in §9 (FP4) plus 4 lifecycle
      tests for Q4K (preconditions + force/no-force + already-q4k).
- [ ] Update `docs/cli.md` and `docs/specs/vindex-format-spec.md`
      §12.1 with the new subcommands and example invocations.
- [x] Smoke: run on `gemma3-4b-f16.vindex` for both FP4 and Q4K,
      verify the converted vindex loads and decodes ("Paris is the
      capital of" → " France …").

Deferred until shipping:

- [ ] Integrate a progress callback (currently `vindex_to_q4k` /
      `vindex_to_fp4` use silent callbacks; the CLI should print
      per-stage timing without needing `eprintln!` spam). Reuse the
      existing `larql_vindex::IndexLoadCallbacks`-style trait shape.

## 12. v1 decisions closed + open items

### Closed by this spec

1. **Subcommand name: `quantize fp4`** (nested under `convert
   quantize`). Replaces the earlier draft's `vindex-to-fp4` flat
   subcommand. The nested shape extends to other formats without
   the CLI growing a new top-level entry per format. Matches the
   existing
   `gguf-to-vindex` / `safetensors-to-vindex` pattern. Keep.

2. **Atomic conversion: write to `DST.tmp/`, fsync, rename to `DST/`
   on success.** Moved from "open / defer" to v1 baseline. Rationale:
   partial output that *looks* complete (some files written,
   `index.json` absent or stale) is a foot-gun for users scripting
   against this tool. Atomic-rename is the right pattern for any
   tool that produces a directory of related files, and the cost is
   trivial (~20 LOC). On filesystems where `rename` would cross a
   mount boundary (rare), the converter falls back to in-place write
   with a warning.

3. **Compliance sidecar: always-on by default, `--no-sidecar`
   opt-out.** Sidecar is ~1 KB and removes the foot-gun of "why did
   my FP4 vindex get reshaped?" Silence is a CI-only concern.

### Still open

1. **Should the default policy be settable globally?** e.g. via
   `~/.larql/config.toml` or `LARQL_FP4_POLICY=option-a`. Not obvious
   Option A will ever be the common default (Q2 ablation confirms B
   as default); defer until a concrete use case emerges.

2. **Should the Q1 scan output the full JSON sidecar even when the
   scan is run standalone (not through convert)?** The example
   binary already does this. Library version should expose both a
   `ComplianceReport` struct (for programmatic use) and a `to_json`
   helper (for CLI write). Non-blocking.
