# Knowledge-graph core demos

Run any of these from a larql checkout:

```sh
cargo run -p larql-demos --example algorithm_demo
```

| demo | what it shows | run |
|---|---|---|
| `algorithm_demo` | Shortest path, merge, subgraph, connected components. | weight-free · 0.2s |
| `edge_demo` | Edge construction, metadata, compact serialization. | weight-free · 0.1s |
| `filter_demo` | Filter a graph — select edges by confidence, layer, relation. | weight-free · 0.2s |
| `graph_demo` | Build, query, traverse and serialize a knowledge graph. | weight-free · 0.2s |
| `serialization_demo` | JSON vs MessagePack, packed binary, CSV, format detection, bytes API. | weight-free · 0.2s |

None of these need model weights — they run in well under a second and are the quickest way to see the surface working.

---

Demos that need a vindex take `--vindex PATH`; point them at any model you have under `output/`. They fail by name if the path is missing rather than surfacing a bare `NotFound`.
