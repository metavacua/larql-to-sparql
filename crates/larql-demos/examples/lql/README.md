# LQL query layer demos

Run any of these from a larql checkout:

```sh
cargo run -p larql-demos --example compact_demo
```

| demo | what it shows | run |
|---|---|---|
| `compact_demo` | Storage-tier walkthrough for the LSM-style storage engine. | weight-free · 0.2s |
| `compile_demo` | End-to-end COMPILE. | needs a vindex · 88s |
| `lql_demo` | Parse, session, execute, error handling. | weight-free · 0.3s |
| `parser_demo` | Every statement type in spec v0.4, with its AST. | weight-free · 0.3s |
| `refine_demo` | End-to-end INSERT + COMPILE — Rust port of `experiments/14_vindex_compilation`. | needs a vindex · **541s** |
| `trace_demo` | Residual-stream decomposition. | needs a vindex · 124s |

3 of these need no model weights and run in well under a second, so they are the quickest way to see the surface working.

## `refine_demo` is slow on purpose

It runs a real INSERT + COMPILE against Gemma 3 4B and takes about **nine
minutes**. That is the demo working, not hanging — it holds ~86% of one
core throughout. Budget accordingly before assuming it has stalled.

---

Demos that need a vindex take `--vindex PATH`; point them at any model you have under `output/`. They fail by name if the path is missing rather than surfacing a bare `NotFound`.
