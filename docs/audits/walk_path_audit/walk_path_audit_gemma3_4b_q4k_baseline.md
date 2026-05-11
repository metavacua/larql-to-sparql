# walk_path_audit

**Model:** `google/gemma-3-4b-it`  
**Vindex:** `/Users/christopherhay/chris-source/larql/output/gemma3-4b-q4k-v2.vindex`  
**Prompts:** 3

**Metrics.** Assertion: `min cos`, `max rel L2 = L2 / ‖primary‖` — both magnitude-invariant. Diagnostic: `max abs L2`, `max|Δ|` — vary with residual magnitude, included for triage of outlier observations (e.g. residual-norm spikes at specific (layer, token) pairs).

## Summary

| path | bound | min cos (assert) | max rel L2 (assert) | top-1 ok | Paris ΔP | max abs L2 (diag) | worst rel-L2 layer | worst rel-L2 prompt | verdict |
|---|---|---|---|---|---|---|---|---|---|
| `sparse` | quantized (cos≥0.99000, rel_L2≤5e-1) | 0.996306 | 9.562e-2 | ✓ | 4.171e-3 | 1.562e2 | 14 | paris | **PASS** |
| `interleaved_q4k` | quantized (cos≥0.99000, rel_L2≤5e-1) | 0.992737 | 1.205e-1 | ✓ | 2.576e-2 | 2.305e2 | 10 | code | **PASS** |

## `sparse`

**Mask:** fp4=false q4=false interleaved=false full_mmap=false q4k=false down_features=false  
**Sparse K:** MAX  
**Bound (quantized):** cos ≥ 0.99000, rel_L2 ≤ 5e-1  
**Assertion aggregate:** min cos = 0.996306, max rel_L2 = 9.562e-2 (layer 14, prompt paris, pos 1)  
**Diagnostic aggregate:** max abs_L2 = 1.562e2 (layer 11, prompt code, pos 1), max|Δ| = 2.291e1, n_obs = 1326  
**Dispatch counts:** `sparse:gemv_full_k`=102  

### Per-prompt

| prompt | walk top-1 | dense top-1 | match | walk P | dense P | ΔP |
|---|---|---|---|---|---|---|
| `apollo` | ` Neil` | ` Neil` | ✓ | 0.999705 | 0.999688 | 1.788e-5 |
| `code` | `
` | `
` | ✓ | 0.998335 | 0.998293 | 4.143e-5 |
| `paris` | ` Paris` | ` Paris` | ✓ | 0.802410 | 0.806580 | 4.171e-3 |

### Per-layer

| layer | dispatch | min cos (assert) | max rel L2 (assert) | rel L2 worst (prompt/pos) | max abs L2 (diag) | max\|Δ\| (diag) | abs L2 worst (prompt/pos) | n |
|---|---|---|---|---|---|---|---|---|
| 0 | `sparse:gemv_full_k` | 0.998125 | 6.161e-2 | paris/2 | 2.055e0 | 2.318e-1 | apollo/17 | 39 |
| 1 | `sparse:gemv_full_k` | 0.996887 | 7.923e-2 | apollo/24 | 7.787e-1 | 1.854e-1 | code/3 | 39 |
| 2 | `sparse:gemv_full_k` | 0.998347 | 5.836e-2 | apollo/5 | 9.176e-1 | 2.878e-1 | code/3 | 39 |
| 3 | `sparse:gemv_full_k` | 0.998545 | 5.528e-2 | apollo/4 | 1.290e0 | 1.938e-1 | code/1 | 39 |
| 4 | `sparse:gemv_full_k` | 0.997445 | 7.144e-2 | paris/3 | 7.284e-1 | 8.630e-2 | apollo/4 | 39 |
| 5 | `sparse:gemv_full_k` | 0.998721 | 5.054e-2 | apollo/4 | 5.749e-1 | 7.858e-2 | code/2 | 39 |
| 6 | `sparse:gemv_full_k` | 0.998194 | 6.020e-2 | apollo/20 | 7.745e-1 | 7.582e-2 | paris/2 | 39 |
| 7 | `sparse:gemv_full_k` | 0.997430 | 7.201e-2 | apollo/10 | 7.752e-1 | 1.103e-1 | code/2 | 39 |
| 8 | `sparse:gemv_full_k` | 0.996894 | 7.903e-2 | apollo/15 | 1.858e0 | 1.452e-1 | paris/1 | 39 |
| 9 | `sparse:gemv_full_k` | 0.996699 | 8.123e-2 | apollo/21 | 4.605e0 | 5.926e-1 | paris/1 | 39 |
| 10 | `sparse:gemv_full_k` | 0.996862 | 7.933e-2 | apollo/21 | 3.196e1 | 2.776e0 | code/1 | 39 |
| 11 | `sparse:gemv_full_k` | 0.996992 | 7.753e-2 | apollo/15 | 1.562e2 | 2.291e1 | code/1 | 39 |
| 12 | `sparse:gemv_full_k` | 0.997630 | 6.890e-2 | apollo/21 | 1.133e0 | 1.414e-1 | paris/1 | 39 |
| 13 | `sparse:gemv_full_k` | 0.997662 | 6.845e-2 | apollo/15 | 1.311e0 | 4.340e-1 | code/1 | 39 |
| 14 | `sparse:gemv_full_k` | 0.997381 | 9.562e-2 | paris/1 | 2.627e0 | 1.374e0 | paris/1 | 39 |
| 15 | `sparse:gemv_full_k` | 0.998138 | 6.107e-2 | code/5 | 1.629e0 | 4.894e-1 | paris/1 | 39 |
| 16 | `sparse:gemv_full_k` | 0.997572 | 7.403e-2 | code/1 | 6.925e0 | 1.920e0 | code/1 | 39 |
| 17 | `sparse:gemv_full_k` | 0.998379 | 6.375e-2 | code/1 | 3.630e0 | 3.843e-1 | code/1 | 39 |
| 18 | `sparse:gemv_full_k` | 0.998466 | 5.645e-2 | code/5 | 1.556e0 | 3.273e-1 | code/1 | 39 |
| 19 | `sparse:gemv_full_k` | 0.998544 | 5.394e-2 | code/5 | 1.488e0 | 5.347e-1 | code/1 | 39 |
| 20 | `sparse:gemv_full_k` | 0.998362 | 5.733e-2 | paris/0 | 1.635e0 | 1.709e-1 | paris/1 | 39 |
| 21 | `sparse:gemv_full_k` | 0.998207 | 5.986e-2 | paris/0 | 1.554e0 | 4.122e-1 | paris/1 | 39 |
| 22 | `sparse:gemv_full_k` | 0.997382 | 7.230e-2 | paris/0 | 1.836e0 | 2.566e-1 | code/1 | 39 |
| 23 | `sparse:gemv_full_k` | 0.998058 | 6.232e-2 | code/6 | 1.409e0 | 4.069e-1 | code/1 | 39 |
| 24 | `sparse:gemv_full_k` | 0.997901 | 6.507e-2 | apollo/14 | 6.699e-1 | 1.496e-1 | paris/1 | 39 |
| 25 | `sparse:gemv_full_k` | 0.997730 | 6.744e-2 | code/6 | 1.121e0 | 9.484e-2 | code/1 | 39 |
| 26 | `sparse:gemv_full_k` | 0.996705 | 8.120e-2 | apollo/14 | 7.822e-1 | 6.112e-2 | paris/1 | 39 |
| 27 | `sparse:gemv_full_k` | 0.997405 | 7.771e-2 | apollo/4 | 1.064e0 | 1.909e-1 | paris/1 | 39 |
| 28 | `sparse:gemv_full_k` | 0.997313 | 7.340e-2 | code/4 | 7.852e-1 | 2.053e-1 | paris/1 | 39 |
| 29 | `sparse:gemv_full_k` | 0.998239 | 5.932e-2 | apollo/25 | 3.940e-1 | 4.788e-2 | paris/1 | 39 |
| 30 | `sparse:gemv_full_k` | 0.996306 | 8.600e-2 | paris/0 | 1.024e0 | 2.279e-1 | code/1 | 39 |
| 31 | `sparse:gemv_full_k` | 0.998610 | 6.197e-2 | code/1 | 1.477e0 | 2.120e-1 | code/1 | 39 |
| 32 | `sparse:gemv_full_k` | 0.998157 | 6.148e-2 | code/2 | 9.521e-1 | 2.022e-1 | apollo/18 | 39 |
| 33 | `sparse:gemv_full_k` | 0.998693 | 5.125e-2 | apollo/18 | 2.876e0 | 2.781e-1 | apollo/18 | 39 |

## `interleaved_q4k`

**Mask:** fp4=true q4=true interleaved=true full_mmap=true q4k=false down_features=false  
**Sparse K:** —  
**Bound (quantized):** cos ≥ 0.99000, rel_L2 ≤ 5e-1  
**Assertion aggregate:** min cos = 0.992737, max rel_L2 = 1.205e-1 (layer 10, prompt code, pos 1)  
**Diagnostic aggregate:** max abs_L2 = 2.305e2 (layer 11, prompt code, pos 1), max|Δ| = 1.842e1, n_obs = 1326  
**Dispatch counts:** `interleaved_q4k:dequant`=102  

### Per-prompt

| prompt | walk top-1 | dense top-1 | match | walk P | dense P | ΔP |
|---|---|---|---|---|---|---|
| `apollo` | ` Neil` | ` Neil` | ✓ | 0.999575 | 0.999688 | 1.126e-4 |
| `code` | `
` | `
` | ✓ | 0.998205 | 0.998293 | 8.863e-5 |
| `paris` | ` Paris` | ` Paris` | ✓ | 0.832341 | 0.806580 | 2.576e-2 |

### Per-layer

| layer | dispatch | min cos (assert) | max rel L2 (assert) | rel L2 worst (prompt/pos) | max abs L2 (diag) | max\|Δ\| (diag) | abs L2 worst (prompt/pos) | n |
|---|---|---|---|---|---|---|---|---|
| 0 | `interleaved_q4k:dequant` | 0.996821 | 8.383e-2 | paris/2 | 2.445e0 | 3.301e-1 | code/2 | 39 |
| 1 | `interleaved_q4k:dequant` | 0.995108 | 1.018e-1 | apollo/24 | 1.046e0 | 1.471e-1 | code/3 | 39 |
| 2 | `interleaved_q4k:dequant` | 0.997131 | 7.597e-2 | apollo/5 | 1.336e0 | 4.093e-1 | code/3 | 39 |
| 3 | `interleaved_q4k:dequant` | 0.996318 | 8.748e-2 | apollo/5 | 1.732e0 | 2.932e-1 | code/3 | 39 |
| 4 | `interleaved_q4k:dequant` | 0.995099 | 9.892e-2 | paris/3 | 1.074e0 | 1.010e-1 | apollo/4 | 39 |
| 5 | `interleaved_q4k:dequant` | 0.997021 | 8.736e-2 | apollo/15 | 8.401e-1 | 9.131e-2 | code/2 | 39 |
| 6 | `interleaved_q4k:dequant` | 0.996035 | 8.958e-2 | paris/3 | 1.294e0 | 9.903e-2 | paris/2 | 39 |
| 7 | `interleaved_q4k:dequant` | 0.994850 | 1.024e-1 | apollo/15 | 1.161e0 | 2.550e-1 | code/2 | 39 |
| 8 | `interleaved_q4k:dequant` | 0.993650 | 1.136e-1 | apollo/15 | 1.940e0 | 1.363e-1 | paris/1 | 39 |
| 9 | `interleaved_q4k:dequant` | 0.994395 | 1.066e-1 | apollo/21 | 4.917e0 | 5.676e-1 | paris/1 | 39 |
| 10 | `interleaved_q4k:dequant` | 0.992737 | 1.205e-1 | code/1 | 5.471e1 | 4.525e0 | code/1 | 39 |
| 11 | `interleaved_q4k:dequant` | 0.994069 | 1.087e-1 | apollo/15 | 2.305e2 | 1.842e1 | code/1 | 39 |
| 12 | `interleaved_q4k:dequant` | 0.995630 | 9.357e-2 | apollo/15 | 1.515e0 | 1.353e-1 | code/1 | 39 |
| 13 | `interleaved_q4k:dequant` | 0.995506 | 9.490e-2 | apollo/15 | 1.714e0 | 5.591e-1 | code/1 | 39 |
| 14 | `interleaved_q4k:dequant` | 0.995504 | 1.103e-1 | code/1 | 3.016e0 | 1.375e0 | code/1 | 39 |
| 15 | `interleaved_q4k:dequant` | 0.996433 | 8.480e-2 | code/1 | 2.322e0 | 3.794e-1 | code/1 | 39 |
| 16 | `interleaved_q4k:dequant` | 0.995368 | 1.014e-1 | code/1 | 9.484e0 | 2.410e0 | code/1 | 39 |
| 17 | `interleaved_q4k:dequant` | 0.997188 | 7.864e-2 | code/1 | 4.477e0 | 7.362e-1 | code/1 | 39 |
| 18 | `interleaved_q4k:dequant` | 0.997350 | 7.274e-2 | code/5 | 2.480e0 | 3.205e-1 | code/1 | 39 |
| 19 | `interleaved_q4k:dequant` | 0.997191 | 7.513e-2 | code/5 | 1.851e0 | 5.497e-1 | paris/1 | 39 |
| 20 | `interleaved_q4k:dequant` | 0.997369 | 7.433e-2 | paris/0 | 2.314e0 | 2.410e-1 | code/1 | 39 |
| 21 | `interleaved_q4k:dequant` | 0.996933 | 7.857e-2 | paris/0 | 2.129e0 | 3.264e-1 | code/1 | 39 |
| 22 | `interleaved_q4k:dequant` | 0.996112 | 8.836e-2 | paris/0 | 2.374e0 | 3.421e-1 | code/1 | 39 |
| 23 | `interleaved_q4k:dequant` | 0.996525 | 8.364e-2 | code/4 | 2.347e0 | 6.035e-1 | code/1 | 39 |
| 24 | `interleaved_q4k:dequant` | 0.995331 | 9.703e-2 | apollo/14 | 8.891e-1 | 1.794e-1 | paris/1 | 39 |
| 25 | `interleaved_q4k:dequant` | 0.995180 | 9.814e-2 | apollo/4 | 1.321e0 | 1.003e-1 | code/1 | 39 |
| 26 | `interleaved_q4k:dequant` | 0.993065 | 1.176e-1 | apollo/14 | 9.884e-1 | 1.012e-1 | paris/1 | 39 |
| 27 | `interleaved_q4k:dequant` | 0.994323 | 1.076e-1 | code/4 | 1.499e0 | 1.931e-1 | code/1 | 39 |
| 28 | `interleaved_q4k:dequant` | 0.993989 | 1.095e-1 | code/4 | 1.146e0 | 2.188e-1 | code/1 | 39 |
| 29 | `interleaved_q4k:dequant` | 0.996498 | 8.381e-2 | apollo/9 | 5.736e-1 | 5.159e-2 | code/1 | 39 |
| 30 | `interleaved_q4k:dequant` | 0.994191 | 1.093e-1 | paris/0 | 1.299e0 | 3.259e-1 | code/1 | 39 |
| 31 | `interleaved_q4k:dequant` | 0.997006 | 7.756e-2 | code/2 | 1.677e0 | 1.368e-1 | apollo/13 | 39 |
| 32 | `interleaved_q4k:dequant` | 0.996598 | 8.304e-2 | code/2 | 1.313e0 | 3.345e-1 | apollo/18 | 39 |
| 33 | `interleaved_q4k:dequant` | 0.996819 | 7.993e-2 | code/2 | 4.009e0 | 4.479e-1 | apollo/18 | 39 |

