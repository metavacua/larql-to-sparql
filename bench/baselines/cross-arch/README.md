# bench/baselines/cross-arch

Per-architecture decode baselines for `make bench-cross-arch --compare`.

## What's here

One JSON file per architecture in the bench matrix, named `<vindex-shorthand>.json`. Files mirror the schema from `larql bench --output json`. Created by `make bench-cross-arch ARGS=--save-baseline`.

## When to update

After **any kernel/dispatch change** that's been A/B'd on Gemma 3 4B and is being promoted to default (per [ADR-017](../../../crates/larql-compute/docs/adr/017-shader-retention-model-agnosticity.md)). The 2026-05-09 QKV defuse was an example — promoted on Gemma 3 4B alone, then validated on Gemma 4 26B A4B post-hoc; baselines should be re-saved after such validations land.

Don't update on routine code changes that don't touch dispatch or shaders.

## Workflow

**1. Save baselines on a cool machine.** Thermal artifacts can fake 1.5-3× regressions ([feedback_thermal_perf_artifacts.md](../../../../README.md) memory entry). Run when load avg is low:

```bash
# Confirm thermals
uptime          # load 1m < ~5
ollama ps       # no models loaded; if so, ollama stop them

# Save baselines (~3-5 min total)
make bench-cross-arch ARGS=--save-baseline
```

**2. Check for regressions before landing a kernel change:**

```bash
make bench-cross-arch ARGS=--compare
```

Exit code 1 if any arch drops below `tok_per_s × (1 - LARQL_TOK_PER_S_THRESHOLD)` (default 5%).

**3. If every arch regresses simultaneously**: suspect thermal first. The script's failure-tip explicitly flags this. Re-bench on a cooled machine before bisecting.

## Thresholds

- Default: `LARQL_TOK_PER_S_THRESHOLD=0.05` (5%).
- Tighten to `0.02` (2%) for kernel work where small wins are meaningful.
- Loosen to `0.10` (10%) only if you're explicitly accepting some perf regression for correctness/feature work.

## Architecture matrix

| shorthand | family | hidden | layers | notes |
|---|---|---|---|---|
| `gemma3-4b-q4k-v2` | gemma3 | 2560 | 34 | canonical Gemma 3 4B Q4_K |
| `gemma4-31b-q4k` | gemma4 | 5376 | 60 | Gemma 4 31B dense |
| `llama2-7b-q4k` | llama | 4096 | 32 | Llama 2 7B |
| `mistral-7b-v0.1-q4k` | mistral | 4096 | 32 | Mistral 7B v0.1 base |
| `gemma4-26b-a4b` | gemma4 | 5376 | 60 | Gemma 4 26B A4B (MoE; large file, optional) |

Missing vindexes are skipped gracefully — the script reports "skipped: vindex not found" and continues.

## Adding a new architecture

When a new model family lands in `larql-models/architectures/`:
1. Add a row to `ARCH_MATRIX` (or `MOE_MATRIX`) in `scripts/bench-cross-arch.sh`.
2. Run `make bench-cross-arch ARGS=--save-baseline` on a cool machine.
3. Commit the new `<shorthand>.json` baseline to this directory.
4. Update `crates/larql-compute/docs/architecture-shader-map.md` with the architecture's row in the bench table.
