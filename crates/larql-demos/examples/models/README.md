# Architecture detection demos

Run any of these from a larql checkout:

```sh
cargo run -p larql-demos --example architecture_demo
```

| demo | what it shows | run |
|---|---|---|
| `architecture_demo` | Detection and configuration for all 12 supported architectures. | weight-free · 0.2s |
| `demo_loading` | Loading from a directory or a GGUF file. | weight-free · 0.1s |
| `demo_tensor_keys` | Tensor-key patterns compared across architectures. | weight-free · 0.2s |

None of these need model weights — they run in well under a second and are the quickest way to see the surface working.

---

Demos that need a vindex take `--vindex PATH`; point them at any model you have under `output/`. They fail by name if the path is missing rather than surfacing a bare `NotFound`.
