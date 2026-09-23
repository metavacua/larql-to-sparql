# KV engines demos

Run any of these from a larql checkout:

```sh
cargo run -p larql-demos --example engine_ladder
```

| demo | what it shows | run |
|---|---|---|
| `engine_ladder` | Every shipped engine end to end on synthetic weights. | weight-free · 0.2s |

---

Demos that need a vindex take `--vindex PATH`; point them at any model you have under `output/`. They fail by name if the path is missing rather than surfacing a bare `NotFound`.
