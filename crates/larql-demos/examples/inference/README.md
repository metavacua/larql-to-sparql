# Inference engine demos

Run any of these from a larql checkout:

```sh
cargo run -p larql-demos --example attention_demo
```

| demo | what it shows | run |
|---|---|---|
| `attention_demo` | The fused online-softmax attention kernel in action. | weight-free · 0.2s |
| `ave_demo` | Arithmetic Virtual Expert against a real Q4_K vindex. Writes `bench/aim-validation/`. | needs a vindex · 51s |
| `backend_demo` | Auto-calibrated hybrid CPU/Metal dispatch. | weight-free · 0.4s |
| `chat_demo` | Multi-turn conversation through `ChatSession`. | needs a vindex · 16s |
| `clustering_demo` | Clustering and relation discovery. | weight-free · 0.2s |
| `detok_demo` | Preserve word spacing across streamed tokens. | weight-free · 0.2s |
| `eos_demo` | The EOS detector halting generation correctly. | `--vindex PATH` · 10s |
| `experts_demo` | WASM expert registry — structured op+args calls across all experts. | needs the wasm build, below |
| `ffn_cache_demo` | FFN L1 cache behaviour, hit/miss stats, patch safety. | `--model ID --vindex PATH` |
| `inference_demo` | Forward pass from safetensors weights. | needs weights · 6s |
| `mech_interp_demo` | Capture, lens, neighbours, ablate, steer, patch. | weight-free · 0.1s |
| `pair_matching_demo` | Pair-based relation matching. | weight-free · 0.2s |
| `sampling_demo` | Greedy vs temperature vs top-p on one prompt. | `--vindex PATH` · 9s |
| `streaming_demo` | Print each token as the model emits it. | needs a vindex · 5s |

6 of these need no model weights and run in well under a second, so they are the quickest way to see the surface working.

## `experts_demo` needs a build first

It loads WASM experts from `crates/larql-experts/target/wasm32-wasip1/release`,
which is not produced by a normal workspace build:

```sh
cd crates/larql-experts && cargo build --target wasm32-wasip1 --release
```

Without it the demo exits naming the directory it looked in.

---

Demos that need a vindex take `--vindex PATH`; point them at any model you have under `output/`. They fail by name if the path is missing rather than surfacing a bare `NotFound`.
