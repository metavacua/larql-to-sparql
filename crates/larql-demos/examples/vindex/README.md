# Vindex format and store demos

Run any of these from a larql checkout:

```sh
cargo run -p larql-demos --example demo_features
```

| demo | what it shows | run |
|---|---|---|
| `demo_features` | Showcase of the complete larql-vindex API. | weight-free · 0.2s |
| `demo_memit_solve` | `memit_solve` + `MemitStore` — the COMPACT MAJOR pipeline in miniature. | weight-free · 0.1s |
| `mmap_demo` | Vindex mmap memory behaviour and model-scaling projections. | weight-free · 0.7s |
| `q4k_demo` | Streaming Q4_K extract. | weight-free · 0.3s |
| `walker_demo` | All three build-time graph extractors against a tiny mock model. | weight-free · 0.3s |

None of these need model weights — they run in well under a second and are the quickest way to see the surface working.

---

Demos that need a vindex take `--vindex PATH`; point them at any model you have under `output/`. They fail by name if the path is missing rather than surfacing a bare `NotFound`.
