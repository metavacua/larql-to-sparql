# Serving surface demos

Run any of these from a larql checkout:

```sh
cargo run -p larql-demos --example embed_demo
```

| demo | what it shows | run |
|---|---|---|
| `embed_demo` | What the embed endpoints return, with synthetic data. | weight-free · 0.2s |
| `openai_demo` | Boots an in-process server and exercises `/v1/models`, `/v1/embeddings`, `/v1/completions`, `/v1/chat/completions`. | takes a vindex path |
| `vindex3_serve_demo` | Serves a VINDEX3 container over `/v1/models` + `/v1/completions` (buffered + SSE) through the real router — self-encoded miniature, or pass a container path. | weight-free · <1s |
| `server_demo` | Builds a synthetic vindex and shows what the server would return. | weight-free · 0.2s |
| `shard_query_demo` | Exp 53 `ShardService`, end to end. | weight-free · 0.3s |

3 of these need no model weights and run in well under a second, so they are the quickest way to see the surface working.

---

Demos that need a vindex take `--vindex PATH`; point them at any model you have under `output/`. They fail by name if the path is missing rather than surfacing a bare `NotFound`.
