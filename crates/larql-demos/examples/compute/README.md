# Compute kernels and solvers demos

Run any of these from a larql checkout:

```sh
cargo run -p larql-demos --example demo_architecture
```

| demo | what it shows | run |
|---|---|---|
| `demo_architecture` | Guided tour of larql-compute's major design decisions. | weight-free · 0.5s |
| `demo_basic` | Auto-detect the backend and run basic operations. | weight-free · 0.2s |
| `demo_ridge_solve` | `ridge_decomposition_solve` — the closed-form ridge solve underlying MEMIT-style weight edits. | weight-free · 0.3s |

None of these need model weights — they run in well under a second and are the quickest way to see the surface working.

---

Demos that need a vindex take `--vindex PATH`; point them at any model you have under `output/`. They fail by name if the path is missing rather than surfacing a bare `NotFound`.
