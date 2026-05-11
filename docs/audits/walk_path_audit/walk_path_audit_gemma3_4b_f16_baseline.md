# walk_path_audit

**Model:** `google/gemma-3-4b-it`  
**Vindex:** `/Users/christopherhay/chris-source/larql/output/gemma3-4b-f16.vindex`  
**Prompts:** 3

**Metrics.** Assertion: `min cos`, `max rel L2 = L2 / ‖primary‖` — both magnitude-invariant. Diagnostic: `max abs L2`, `max|Δ|` — vary with residual magnitude, included for triage of outlier observations (e.g. residual-norm spikes at specific (layer, token) pairs).

## Summary

| path | bound | min cos (assert) | max rel L2 (assert) | top-1 ok | Paris ΔP | max abs L2 (diag) | worst rel-L2 layer | worst rel-L2 prompt | verdict |
|---|---|---|---|---|---|---|---|---|---|
| `sparse` | exact (cos≥0.99999, rel_L2≤1e-2) | 0.999997 | 1.881e-3 | ✓ | 1.229e-4 | 2.317e0 | 32 | paris | **PASS** |
| `full_mmap` | exact (cos≥0.99999, rel_L2≤1e-2) | 0.999997 | 1.881e-3 | ✓ | 1.229e-4 | 2.317e0 | 32 | paris | **PASS** |
| `exact` | exact (cos≥0.99999, rel_L2≤1e-2) | 0.999996 | 1.881e-3 | ✓ | 1.425e-4 | 2.213e0 | 32 | paris | **PASS** |

## `sparse`

**Mask:** fp4=false q4=false interleaved=false full_mmap=false q4k=false down_features=false  
**Sparse K:** MAX  
**Bound (exact):** cos ≥ 0.99999, rel_L2 ≤ 1e-2  
**Assertion aggregate:** min cos = 0.999997, max rel_L2 = 1.881e-3 (layer 32, prompt paris, pos 0)  
**Diagnostic aggregate:** max abs_L2 = 2.317e0 (layer 11, prompt code, pos 1), max|Δ| = 5.480e-1, n_obs = 1326  
**Dispatch counts:** `sparse:gemv_full_k`=102  

### Per-prompt

| prompt | walk top-1 | dense top-1 | match | walk P | dense P | ΔP |
|---|---|---|---|---|---|---|
| `apollo` | ` Neil` | ` Neil` | ✓ | 0.999687 | 0.999688 | 7.749e-7 |
| `code` | `
` | `
` | ✓ | 0.998288 | 0.998293 | 4.947e-6 |
| `paris` | ` Paris` | ` Paris` | ✓ | 0.806458 | 0.806580 | 1.229e-4 |

### Per-layer

| layer | dispatch | min cos (assert) | max rel L2 (assert) | rel L2 worst (prompt/pos) | max abs L2 (diag) | max\|Δ\| (diag) | abs L2 worst (prompt/pos) | n |
|---|---|---|---|---|---|---|---|---|
| 0 | `sparse:gemv_full_k` | 0.999998 | 1.881e-3 | paris/0 | 6.464e-2 | 1.418e-2 | paris/0 | 39 |
| 1 | `sparse:gemv_full_k` | 0.999997 | 1.071e-3 | paris/0 | 9.465e-3 | 2.219e-3 | apollo/5 | 39 |
| 2 | `sparse:gemv_full_k` | 0.999999 | 4.821e-4 | apollo/21 | 1.181e-2 | 1.839e-3 | code/1 | 39 |
| 3 | `sparse:gemv_full_k` | 0.999999 | 4.993e-4 | paris/0 | 1.289e-2 | 3.440e-3 | code/1 | 39 |
| 4 | `sparse:gemv_full_k` | 0.999998 | 1.677e-3 | paris/0 | 1.074e-2 | 2.827e-3 | paris/1 | 39 |
| 5 | `sparse:gemv_full_k` | 0.999998 | 1.877e-3 | paris/0 | 9.049e-3 | 1.968e-3 | paris/2 | 39 |
| 6 | `sparse:gemv_full_k` | 0.999999 | 6.507e-4 | paris/0 | 5.092e-3 | 1.108e-3 | paris/2 | 39 |
| 7 | `sparse:gemv_full_k` | 0.999999 | 6.504e-4 | paris/0 | 9.787e-3 | 3.249e-3 | paris/1 | 39 |
| 8 | `sparse:gemv_full_k` | 0.999998 | 1.367e-3 | code/1 | 5.796e-2 | 9.023e-3 | paris/1 | 39 |
| 9 | `sparse:gemv_full_k` | 0.999998 | 9.395e-4 | code/1 | 9.678e-2 | 2.339e-2 | paris/1 | 39 |
| 10 | `sparse:gemv_full_k` | 0.999998 | 8.775e-4 | paris/0 | 1.973e-1 | 5.815e-2 | paris/1 | 39 |
| 11 | `sparse:gemv_full_k` | 0.999997 | 1.742e-3 | paris/0 | 2.317e0 | 5.480e-1 | code/1 | 39 |
| 12 | `sparse:gemv_full_k` | 0.999999 | 7.411e-4 | paris/0 | 1.111e-2 | 2.605e-3 | paris/1 | 39 |
| 13 | `sparse:gemv_full_k` | 0.999998 | 1.099e-3 | paris/0 | 1.207e-2 | 2.915e-3 | code/1 | 39 |
| 14 | `sparse:gemv_full_k` | 0.999998 | 1.777e-3 | paris/1 | 4.882e-2 | 8.987e-3 | paris/1 | 39 |
| 15 | `sparse:gemv_full_k` | 0.999997 | 1.384e-3 | paris/0 | 2.652e-2 | 6.041e-3 | paris/1 | 39 |
| 16 | `sparse:gemv_full_k` | 0.999998 | 6.268e-4 | code/1 | 5.960e-2 | 1.278e-2 | paris/1 | 39 |
| 17 | `sparse:gemv_full_k` | 0.999998 | 7.348e-4 | paris/0 | 3.121e-2 | 7.481e-3 | paris/1 | 39 |
| 18 | `sparse:gemv_full_k` | 0.999999 | 4.111e-4 | apollo/15 | 1.532e-2 | 2.185e-3 | paris/1 | 39 |
| 19 | `sparse:gemv_full_k` | 0.999998 | 4.818e-4 | code/4 | 1.241e-2 | 2.339e-3 | paris/1 | 39 |
| 20 | `sparse:gemv_full_k` | 0.999999 | 8.811e-4 | code/1 | 3.623e-2 | 9.584e-3 | paris/1 | 39 |
| 21 | `sparse:gemv_full_k` | 0.999997 | 5.445e-4 | paris/1 | 2.024e-2 | 3.606e-3 | paris/1 | 39 |
| 22 | `sparse:gemv_full_k` | 0.999998 | 7.465e-4 | paris/1 | 2.963e-2 | 8.500e-3 | paris/1 | 39 |
| 23 | `sparse:gemv_full_k` | 0.999998 | 4.629e-4 | code/1 | 1.947e-2 | 3.304e-3 | paris/1 | 39 |
| 24 | `sparse:gemv_full_k` | 0.999998 | 6.635e-4 | apollo/14 | 7.960e-3 | 9.918e-4 | paris/1 | 39 |
| 25 | `sparse:gemv_full_k` | 0.999998 | 9.148e-4 | paris/1 | 2.617e-2 | 7.834e-3 | paris/1 | 39 |
| 26 | `sparse:gemv_full_k` | 0.999999 | 1.212e-3 | code/1 | 2.552e-2 | 6.826e-3 | paris/1 | 39 |
| 27 | `sparse:gemv_full_k` | 0.999999 | 3.566e-4 | paris/1 | 1.149e-2 | 3.290e-3 | paris/1 | 39 |
| 28 | `sparse:gemv_full_k` | 0.999998 | 7.206e-4 | paris/1 | 1.206e-2 | 3.516e-3 | paris/1 | 39 |
| 29 | `sparse:gemv_full_k` | 0.999998 | 3.680e-4 | paris/1 | 3.432e-3 | 6.620e-4 | paris/1 | 39 |
| 30 | `sparse:gemv_full_k` | 0.999999 | 8.747e-4 | code/1 | 2.296e-2 | 5.402e-3 | paris/1 | 39 |
| 31 | `sparse:gemv_full_k` | 0.999998 | 7.797e-4 | paris/0 | 1.002e-2 | 3.058e-3 | apollo/13 | 39 |
| 32 | `sparse:gemv_full_k` | 0.999998 | 1.881e-3 | paris/0 | 1.711e-2 | 3.771e-3 | apollo/18 | 39 |
| 33 | `sparse:gemv_full_k` | 0.999999 | 6.178e-4 | paris/0 | 2.646e-2 | 4.828e-3 | apollo/18 | 39 |

## `full_mmap`

**Mask:** fp4=true q4=true interleaved=true full_mmap=false q4k=false down_features=false  
**Sparse K:** —  
**Bound (exact):** cos ≥ 0.99999, rel_L2 ≤ 1e-2  
**Assertion aggregate:** min cos = 0.999997, max rel_L2 = 1.881e-3 (layer 32, prompt paris, pos 0)  
**Diagnostic aggregate:** max abs_L2 = 2.317e0 (layer 11, prompt code, pos 1), max|Δ| = 5.480e-1, n_obs = 1326  
**Dispatch counts:** `full_mmap`=102  

### Per-prompt

| prompt | walk top-1 | dense top-1 | match | walk P | dense P | ΔP |
|---|---|---|---|---|---|---|
| `apollo` | ` Neil` | ` Neil` | ✓ | 0.999687 | 0.999688 | 7.749e-7 |
| `code` | `
` | `
` | ✓ | 0.998288 | 0.998293 | 4.947e-6 |
| `paris` | ` Paris` | ` Paris` | ✓ | 0.806458 | 0.806580 | 1.229e-4 |

### Per-layer

| layer | dispatch | min cos (assert) | max rel L2 (assert) | rel L2 worst (prompt/pos) | max abs L2 (diag) | max\|Δ\| (diag) | abs L2 worst (prompt/pos) | n |
|---|---|---|---|---|---|---|---|---|
| 0 | `full_mmap` | 0.999998 | 1.881e-3 | paris/0 | 6.464e-2 | 1.418e-2 | paris/0 | 39 |
| 1 | `full_mmap` | 0.999997 | 1.071e-3 | paris/0 | 9.465e-3 | 2.219e-3 | apollo/5 | 39 |
| 2 | `full_mmap` | 0.999999 | 4.821e-4 | apollo/21 | 1.181e-2 | 1.839e-3 | code/1 | 39 |
| 3 | `full_mmap` | 0.999999 | 4.993e-4 | paris/0 | 1.289e-2 | 3.440e-3 | code/1 | 39 |
| 4 | `full_mmap` | 0.999998 | 1.677e-3 | paris/0 | 1.074e-2 | 2.827e-3 | paris/1 | 39 |
| 5 | `full_mmap` | 0.999998 | 1.877e-3 | paris/0 | 9.049e-3 | 1.968e-3 | paris/2 | 39 |
| 6 | `full_mmap` | 0.999999 | 6.507e-4 | paris/0 | 5.092e-3 | 1.108e-3 | paris/2 | 39 |
| 7 | `full_mmap` | 0.999999 | 6.504e-4 | paris/0 | 9.787e-3 | 3.249e-3 | paris/1 | 39 |
| 8 | `full_mmap` | 0.999998 | 1.367e-3 | code/1 | 5.796e-2 | 9.023e-3 | paris/1 | 39 |
| 9 | `full_mmap` | 0.999998 | 9.395e-4 | code/1 | 9.678e-2 | 2.339e-2 | paris/1 | 39 |
| 10 | `full_mmap` | 0.999998 | 8.775e-4 | paris/0 | 1.973e-1 | 5.815e-2 | paris/1 | 39 |
| 11 | `full_mmap` | 0.999997 | 1.742e-3 | paris/0 | 2.317e0 | 5.480e-1 | code/1 | 39 |
| 12 | `full_mmap` | 0.999999 | 7.411e-4 | paris/0 | 1.111e-2 | 2.605e-3 | paris/1 | 39 |
| 13 | `full_mmap` | 0.999998 | 1.099e-3 | paris/0 | 1.207e-2 | 2.915e-3 | code/1 | 39 |
| 14 | `full_mmap` | 0.999998 | 1.777e-3 | paris/1 | 4.882e-2 | 8.987e-3 | paris/1 | 39 |
| 15 | `full_mmap` | 0.999997 | 1.384e-3 | paris/0 | 2.652e-2 | 6.041e-3 | paris/1 | 39 |
| 16 | `full_mmap` | 0.999998 | 6.268e-4 | code/1 | 5.960e-2 | 1.278e-2 | paris/1 | 39 |
| 17 | `full_mmap` | 0.999998 | 7.348e-4 | paris/0 | 3.121e-2 | 7.481e-3 | paris/1 | 39 |
| 18 | `full_mmap` | 0.999999 | 4.111e-4 | apollo/15 | 1.532e-2 | 2.185e-3 | paris/1 | 39 |
| 19 | `full_mmap` | 0.999998 | 4.818e-4 | code/4 | 1.241e-2 | 2.339e-3 | paris/1 | 39 |
| 20 | `full_mmap` | 0.999999 | 8.811e-4 | code/1 | 3.623e-2 | 9.584e-3 | paris/1 | 39 |
| 21 | `full_mmap` | 0.999997 | 5.445e-4 | paris/1 | 2.024e-2 | 3.606e-3 | paris/1 | 39 |
| 22 | `full_mmap` | 0.999998 | 7.465e-4 | paris/1 | 2.963e-2 | 8.500e-3 | paris/1 | 39 |
| 23 | `full_mmap` | 0.999998 | 4.629e-4 | code/1 | 1.947e-2 | 3.304e-3 | paris/1 | 39 |
| 24 | `full_mmap` | 0.999998 | 6.635e-4 | apollo/14 | 7.960e-3 | 9.918e-4 | paris/1 | 39 |
| 25 | `full_mmap` | 0.999998 | 9.148e-4 | paris/1 | 2.617e-2 | 7.834e-3 | paris/1 | 39 |
| 26 | `full_mmap` | 0.999999 | 1.212e-3 | code/1 | 2.552e-2 | 6.826e-3 | paris/1 | 39 |
| 27 | `full_mmap` | 0.999999 | 3.566e-4 | paris/1 | 1.149e-2 | 3.290e-3 | paris/1 | 39 |
| 28 | `full_mmap` | 0.999998 | 7.206e-4 | paris/1 | 1.206e-2 | 3.516e-3 | paris/1 | 39 |
| 29 | `full_mmap` | 0.999998 | 3.680e-4 | paris/1 | 3.432e-3 | 6.620e-4 | paris/1 | 39 |
| 30 | `full_mmap` | 0.999999 | 8.747e-4 | code/1 | 2.296e-2 | 5.402e-3 | paris/1 | 39 |
| 31 | `full_mmap` | 0.999998 | 7.797e-4 | paris/0 | 1.002e-2 | 3.058e-3 | apollo/13 | 39 |
| 32 | `full_mmap` | 0.999998 | 1.881e-3 | paris/0 | 1.711e-2 | 3.771e-3 | apollo/18 | 39 |
| 33 | `full_mmap` | 0.999999 | 6.178e-4 | paris/0 | 2.646e-2 | 4.828e-3 | apollo/18 | 39 |

## `exact`

**Mask:** fp4=true q4=true interleaved=true full_mmap=true q4k=true down_features=false  
**Sparse K:** —  
**Bound (exact):** cos ≥ 0.99999, rel_L2 ≤ 1e-2  
**Assertion aggregate:** min cos = 0.999996, max rel_L2 = 1.881e-3 (layer 32, prompt paris, pos 0)  
**Diagnostic aggregate:** max abs_L2 = 2.213e0 (layer 11, prompt code, pos 1), max|Δ| = 5.469e-1, n_obs = 1326  
**Dispatch counts:** `exact`=102  

### Per-prompt

| prompt | walk top-1 | dense top-1 | match | walk P | dense P | ΔP |
|---|---|---|---|---|---|---|
| `apollo` | ` Neil` | ` Neil` | ✓ | 0.999687 | 0.999688 | 6.557e-7 |
| `code` | `
` | `
` | ✓ | 0.998288 | 0.998293 | 5.662e-6 |
| `paris` | ` Paris` | ` Paris` | ✓ | 0.806438 | 0.806580 | 1.425e-4 |

### Per-layer

| layer | dispatch | min cos (assert) | max rel L2 (assert) | rel L2 worst (prompt/pos) | max abs L2 (diag) | max\|Δ\| (diag) | abs L2 worst (prompt/pos) | n |
|---|---|---|---|---|---|---|---|---|
| 0 | `exact` | 0.999999 | 1.881e-3 | paris/0 | 6.464e-2 | 1.415e-2 | paris/0 | 39 |
| 1 | `exact` | 0.999997 | 1.071e-3 | paris/0 | 9.435e-3 | 2.226e-3 | apollo/5 | 39 |
| 2 | `exact` | 0.999998 | 4.764e-4 | apollo/21 | 1.172e-2 | 1.822e-3 | code/1 | 39 |
| 3 | `exact` | 0.999999 | 4.987e-4 | paris/0 | 1.280e-2 | 3.412e-3 | code/1 | 39 |
| 4 | `exact` | 0.999998 | 1.676e-3 | paris/0 | 1.073e-2 | 2.833e-3 | paris/1 | 39 |
| 5 | `exact` | 0.999998 | 1.877e-3 | paris/0 | 8.982e-3 | 1.961e-3 | paris/2 | 39 |
| 6 | `exact` | 0.999997 | 6.508e-4 | paris/0 | 4.935e-3 | 1.110e-3 | paris/1 | 39 |
| 7 | `exact` | 0.999998 | 6.504e-4 | paris/0 | 9.764e-3 | 3.241e-3 | paris/1 | 39 |
| 8 | `exact` | 0.999999 | 1.366e-3 | code/1 | 5.796e-2 | 9.003e-3 | paris/1 | 39 |
| 9 | `exact` | 0.999998 | 9.392e-4 | code/1 | 9.675e-2 | 2.345e-2 | paris/1 | 39 |
| 10 | `exact` | 0.999997 | 8.768e-4 | paris/0 | 1.832e-1 | 5.749e-2 | paris/1 | 39 |
| 11 | `exact` | 0.999998 | 1.741e-3 | paris/0 | 2.213e0 | 5.469e-1 | code/1 | 39 |
| 12 | `exact` | 0.999998 | 7.399e-4 | paris/0 | 1.096e-2 | 2.581e-3 | paris/1 | 39 |
| 13 | `exact` | 0.999998 | 1.098e-3 | paris/0 | 1.194e-2 | 2.933e-3 | code/1 | 39 |
| 14 | `exact` | 0.999997 | 1.773e-3 | paris/1 | 4.869e-2 | 8.955e-3 | paris/1 | 39 |
| 15 | `exact` | 0.999998 | 1.383e-3 | paris/0 | 2.626e-2 | 5.947e-3 | paris/1 | 39 |
| 16 | `exact` | 0.999998 | 4.993e-4 | paris/1 | 4.779e-2 | 1.310e-2 | paris/1 | 39 |
| 17 | `exact` | 0.999998 | 7.352e-4 | paris/0 | 3.091e-2 | 7.601e-3 | paris/1 | 39 |
| 18 | `exact` | 0.999998 | 4.051e-4 | apollo/15 | 1.500e-2 | 2.151e-3 | paris/1 | 39 |
| 19 | `exact` | 0.999999 | 4.712e-4 | code/4 | 1.238e-2 | 2.346e-3 | paris/1 | 39 |
| 20 | `exact` | 0.999998 | 8.791e-4 | code/1 | 3.611e-2 | 9.591e-3 | paris/1 | 39 |
| 21 | `exact` | 0.999996 | 5.420e-4 | paris/1 | 2.015e-2 | 3.624e-3 | paris/1 | 39 |
| 22 | `exact` | 0.999998 | 7.402e-4 | paris/1 | 2.938e-2 | 8.521e-3 | paris/1 | 39 |
| 23 | `exact` | 0.999999 | 4.590e-4 | code/1 | 1.937e-2 | 3.263e-3 | paris/1 | 39 |
| 24 | `exact` | 0.999998 | 6.522e-4 | apollo/14 | 7.908e-3 | 9.792e-4 | paris/1 | 39 |
| 25 | `exact` | 0.999998 | 9.127e-4 | paris/1 | 2.611e-2 | 7.819e-3 | paris/1 | 39 |
| 26 | `exact` | 0.999998 | 1.211e-3 | code/1 | 2.550e-2 | 6.807e-3 | paris/1 | 39 |
| 27 | `exact` | 0.999998 | 3.532e-4 | paris/1 | 1.138e-2 | 3.313e-3 | paris/1 | 39 |
| 28 | `exact` | 0.999999 | 7.169e-4 | paris/1 | 1.200e-2 | 3.494e-3 | paris/1 | 39 |
| 29 | `exact` | 0.999998 | 3.623e-4 | paris/1 | 3.379e-3 | 6.689e-4 | paris/1 | 39 |
| 30 | `exact` | 0.999998 | 8.717e-4 | code/1 | 2.289e-2 | 5.394e-3 | paris/1 | 39 |
| 31 | `exact` | 0.999998 | 7.792e-4 | paris/0 | 9.782e-3 | 3.040e-3 | apollo/13 | 39 |
| 32 | `exact` | 0.999998 | 1.881e-3 | paris/0 | 1.703e-2 | 3.788e-3 | apollo/18 | 39 |
| 33 | `exact` | 0.999999 | 6.176e-4 | paris/0 | 2.572e-2 | 4.705e-3 | apollo/18 | 39 |


