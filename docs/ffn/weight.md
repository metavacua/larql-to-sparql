# WeightFfn — Dense Architecture-Correct FFN

**File:** `crates/larql-inference/src/ffn/weight.rs`
**Status:** Production
**Speed:** 6ms/layer (baseline)
**Accuracy:** 100% (ground truth)

## Description

Dense FFN that follows the model architecture exactly. Reads the `ModelArchitecture` trait to
determine FFN type (gated/standard), activation function (SiLU/GELU), and bias handling.
Supports all model families: Gemma, Llama, Mistral, Qwen, DeepSeek, etc.

## Computation

For gated models (Gemma, Llama):
```
gate = x @ W_gate.T
up   = x @ W_up.T
activation = SiLU(gate) * up
output = activation @ W_down.T
```

For non-gated models:
```
projected = activation(x @ W_up.T + bias)
output = projected @ W_down.T + bias
```

## Usage

```rust
use larql_inference::{WeightFfn, predict_with_ffn};

let ffn = WeightFfn { weights };
let result = predict_with_ffn(weights, tokenizer, &token_ids, 5, &ffn);
```

## When to use

- Default inference path
- Ground truth for comparing other backends
- Any time exact model reproduction is required
