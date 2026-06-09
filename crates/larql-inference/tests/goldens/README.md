# DSv4-Flash HF parity reference dumps

`dsv4_flash_hf_reference.json` (gitignored until generated) is the
HuggingFace `transformers` reference consumed by
`tests/test_dsv4_hf_parity.rs` (`dsv4-hf-parity`, tasks #63/#69).

Generate it out-of-band on a host with the HF weights + transformers:

```sh
python scripts/dsv4_hf_reference.py \
    --model deepseek-ai/DeepSeek-V4-Flash \
    --prompt "The capital of France is" \
    --top-k 10 \
    --out crates/larql-inference/tests/goldens/dsv4_flash_hf_reference.json
```

Format:

```json
{ "model": "...", "prompt": "...", "dtype": "bfloat16",
  "token_ids": [464, 3139, ...],
  "top_k": [ {"token_id": 6342, "logit": 21.3}, ... ] }
```

The Rust test runs the DSv4 GGUF forward on `token_ids` and asserts the
greedy next-token matches the reference top-1 (+ top-K overlap + a loose
top-1 logit tolerance). Absent the dump, the test skips.
