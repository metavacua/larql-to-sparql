# SparseFfn — Top-K Sparse FFN

**File:** `crates/larql-inference/src/ffn/sparse.rs`
**Status:** Production
**Speed:** 8-22ms/layer (always slower than dense)
**Accuracy:** 100% at K=8092+ for Gemma-3-4b

## Description

Computes the full gate matmul to find which features activate, then only computes up/down
projections for the top-K features. Falls back to dense BLAS when K >= 80% of features.

## Why it's slower than dense

The gate matmul alone costs 1.9ms (31% of FFN). SparseFfn still does this full scan, then
adds gather + sparse computation overhead. The sparse up/down savings don't offset the gate
cost plus the overhead.

## Usage

```rust
use larql_inference::{SparseFfn, predict_with_ffn};

let ffn = SparseFfn { weights, top_k: 8092 };
let result = predict_with_ffn(weights, tokenizer, &token_ids, 5, &ffn);
```

## Benchmarks

| K | FFN time | vs Dense | Match rate |
|---|----------|----------|------------|
| 64 | 8.0ms | 0.75x | 10% |
| 512 | 9.4ms | 0.64x | 20% |
| 4096 | 21.9ms | 0.28x | 70% |
| 8092 | 6.0ms | 1.01x (fallback) | 100% |
