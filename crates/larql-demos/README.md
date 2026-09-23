# larql-demos

Runnable demonstrations of larql's shipped capabilities — one home for
"how do I use this", filed under the crate whose capability each one
shows.

Each folder has its own README listing every demo, what it shows, and
what it needs to run — including measured runtimes, so a nine-minute demo
is not mistaken for a hung one.

| folder | | demos | weight-free |
|---|---|---:|---:|
| [`boundary/`](examples/boundary/README.md) | Boundary codec | 2 | 2 |
| [`compute/`](examples/compute/README.md) | Compute kernels and solvers | 3 | 3 |
| [`core/`](examples/core/README.md) | Knowledge-graph core | 5 | 5 |
| [`inference/`](examples/inference/README.md) | Inference engine | 14 | 6 |
| [`kv/`](examples/kv/README.md) | KV engines | 1 | 1 |
| [`lql/`](examples/lql/README.md) | LQL query layer | 6 | 3 |
| [`models/`](examples/models/README.md) | Architecture detection | 3 | 3 |
| [`server/`](examples/server/README.md) | Serving surface | 4 | 3 |
| [`vindex/`](examples/vindex/README.md) | Vindex format and store | 5 | 5 |
| | | **43** | **31** |

```sh
cargo run -p larql-demos --example chat_demo
```

Folders are not auto-discovered by cargo, so every demo is declared as an
explicit `[[example]]` in `Cargo.toml` with its path. Adding a demo means
adding four lines there — deliberately, so the inventory stays visible.

The weight-free demos run in CI on every platform. The rest compile in CI
but need a real vindex to execute; they take `--vindex PATH` and fail by
name when it is missing, rather than surfacing a bare `NotFound`.

## What is deliberately not here

**Benchmarks, diagnostics and parity harnesses** stay in their own
crate's `examples/` — `bench_*`, `debug_*`, `profile_*`, `*_parity`,
`compare_*`, `membw_probe` and friends. They exercise the engine rather
than showing how to use it, so they belong next to the code they measure,
where a change and its benchmark move together.

**Research probes** live in `chris-experiments/larql_probes`, pinned to
the larql revision that produced their recorded verdict. A probe answers
a question once; a demo is documentation that has to keep working. The
two have opposite maintenance contracts, which is why they no longer
share a directory.

`apollo_rd_backend` was briefly filed here and has moved to the probes:
it is named after a backend but is really the compute half of a
chris-experiments script, and is useless without it.

The dividing question when adding something here: *would a new user run
this to understand larql?* If it only makes sense while chasing a
specific result, it is a probe, not a demo.
