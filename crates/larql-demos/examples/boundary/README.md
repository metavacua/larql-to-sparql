# Boundary codec demos

Run any of these from a larql checkout:

```sh
cargo run -p larql-demos --example encode_decode
```

| demo | what it shows | run |
|---|---|---|
| `encode_decode` | Encode and decode a synthetic residual with both `bf16` and `int8_clip3sigma`. | weight-free · 0.1s |
| `gate_decision` | Gate decisions for four boundary types, matching the Exp 43 continuation tests. | weight-free · 0.1s |

None of these need model weights — they run in well under a second and are the quickest way to see the surface working.

---

Demos that need a vindex take `--vindex PATH`; point them at any model you have under `output/`. They fail by name if the path is missing rather than surfacing a bare `NotFound`.
