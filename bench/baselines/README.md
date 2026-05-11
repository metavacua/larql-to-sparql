# bench/baselines

Committed baseline JSON files for the GT10 CI regression gate (ADR-0012 §Layer 3).

## File naming

`grid-{model-shorthand}.json` — one file per model, committed after a known-good run.

Example: `grid-gemma3-4b-q4k.json`

## Update workflow

When you intentionally improve performance (e.g. after a wire format change or kernel speedup),
update the baseline by re-running the bench and committing the new JSON:

```bash
LARQL_BENCH_VINDEX=output/gemma3-4b-q4k.vindex \
LARQL_BENCH_FFN_URL=http://localhost:8080 \
./scripts/bench-grid-regress.sh gemma3-4b-q4k
# First run with no baseline → saves current as baseline automatically.
# Subsequent runs → compare and fail if regression detected.
```

Or to force-update an existing baseline:

```bash
larql bench output/gemma3-4b-q4k.vindex \
    --ffn http://localhost:8080 \
    --wire f32,f16 \
    --tokens 30 \
    --output json \
    --output-file bench/baselines/grid-gemma3-4b-q4k.json
git add bench/baselines/grid-gemma3-4b-q4k.json
git commit -m "bench: update gemma3-4b-q4k baseline after f16 wire improvement"
```

## CI integration

Add to your CI pipeline:

```yaml
- name: Grid regression gate
  env:
    LARQL_BENCH_VINDEX: output/gemma3-4b-q4k.vindex
    LARQL_BENCH_FFN_URL: http://shard-server:8080
  run: make bench-grid MODEL=gemma3-4b-q4k
```

## Baseline JSON schema (ADR-0012)

```json
{
  "timestamp": "1234567890",
  "model": "output/gemma3-4b-q4k.vindex",
  "prompt": "The capital of France is",
  "tokens": 30,
  "wire": "f32,f16",
  "concurrent": 1,
  "results": [
    {
      "backend": "remote-ffn-stream (http://...)",
      "prefill_ms": 12.3,
      "ms_per_tok": { "mean": 52.1, "p50": 51.8, "p99": 67.2 },
      "tok_per_s": 19.2,
      "wire_bytes_per_tok": 300000,
      "n_steps": 30,
      "note": ""
    }
  ]
}
```
