# Quantization Formats

`larql-models` handles data format encoding/decoding. Compute-path quantized operations (GPU matvec, shader dispatch) are in `larql-compute`.

## f16 / bf16 (half.rs)

IEEE 754 half-precision and Google Brain bfloat16.

### f16 (binary16)

```
Sign: 1 bit | Exponent: 5 bits | Mantissa: 10 bits
Range: ±65504, precision: ~3.3 decimal digits
```

Conversion: bit manipulation (no hardware f16 required).

### bf16 (bfloat16)

```
Sign: 1 bit | Exponent: 8 bits | Mantissa: 7 bits
Range: same as f32, precision: ~2.4 decimal digits
```

Decode: shift left 16 bits → reinterpret as f32. Fast but less precise than f16.

### API

```rust
use larql_models::quant::half;

let f16_bytes = half::encode_f16(&[1.0, -2.0, 3.14]);  // Vec<u8>, 2 bytes each
let f32_vals = half::decode_f16(&f16_bytes);             // Vec<f32>
let f32_vals = half::decode_bf16(&bf16_bytes);           // Vec<f32>
```

## GGML Block Quantization (ggml.rs)

GGML uses block quantization: groups of 32 elements share a scale factor, reducing storage while preserving relative magnitudes within each block.

### Q4_0

```
Block size: 32 elements
Storage: 2 bytes (f16 scale) + 16 bytes (32 × 4-bit values) = 18 bytes
Bits per weight: 4.5
```

Encoding: scale = max(abs(block)) / 7, quantize each value to 4-bit signed int.
Decoding: value = scale × (nibble - 8).

### Q4_1

```
Block size: 32 elements
Storage: 2 bytes (f16 scale) + 2 bytes (f16 min) + 16 bytes (4-bit values) = 20 bytes
Bits per weight: 5.0
```

Like Q4_0 but with a per-block minimum offset. Decoding: value = scale × nibble + min.

### Q5_0

```
Block size: 32 elements
Storage: 2 bytes (f16 scale) + 4 bytes (32 high bits) + 16 bytes (4-bit low values) = 22 bytes
Bits per weight: 5.5
```

5-bit quantization: 4 low bits packed as Q4_0, plus one extra bit per element stored in a separate bitfield.

### Q5_1

```
Block size: 32 elements
Storage: 2 bytes (f16 scale) + 2 bytes (f16 min) + 4 bytes (high bits) + 16 bytes (low values) = 24 bytes
Bits per weight: 6.0
```

Q5_0 with per-block minimum offset.

### Q8_0

```
Block size: 32 elements
Storage: 2 bytes (f16 scale) + 32 bytes (32 × int8 values) = 34 bytes
Bits per weight: 8.5
```

Encoding: scale = max(abs(block)) / 127, quantize to int8.
Decoding: value = scale × int8_value.

Higher quality than Q4 but 2x larger. Used for intermediate quantization in compute paths.

### API

```rust
use larql_models::quant::ggml;

// Quantize
let q4_bytes = ggml::quantize_q4_0(&f32_data);      // f32 → packed Q4_0
let q8_bytes = ggml::quantize_q8_0(&f32_data);      // f32 → packed Q8_0

// Dequantize (any supported type). Returns ModelError::Parse if `bytes`
// is shorter than the declared element count implies, or if num_elements
// is not a multiple of the block size (32 for Q4/Q5/Q8, 256 for Q4_K/Q6_K).
let f32_data = ggml::dequantize(&bytes, ggml::TYPE_Q4_0, num_elements)?;
let f32_data = ggml::dequantize_q4_0(&bytes, num_elements)?;  // type-specific

// Format info
let size = ggml::tensor_data_size(ggml::TYPE_Q4_0, 1024);  // bytes for 1024 elements
let name = ggml::type_name(ggml::TYPE_Q8_0);                // "Q8_0"
```

### Type Constants

| Constant | Value | Name |
|----------|-------|------|
| `TYPE_F32` | 0 | F32 |
| `TYPE_F16` | 1 | F16 |
| `TYPE_Q4_0` | 2 | Q4_0 |
| `TYPE_Q4_1` | 3 | Q4_1 |
| `TYPE_Q8_0` | 6 | Q8_0 |
| `TYPE_Q5_0` | 8 | Q5_0 |
| `TYPE_Q5_1` | 9 | Q5_1 |
| `TYPE_Q4_K` | 12 | Q4_K |
| `TYPE_Q6_K` | 14 | Q6_K |
| `TYPE_BF16` | 30 | BF16 |

## MXFP4 (mxfp4.rs)

Microscaling FP4 format used by GPT-OSS / OpenAI models for packed MoE expert weights.

### Format

```
Block size: 32 elements
Scale: 1 byte e8m0 (8-bit exponent, no mantissa — pure power of 2)
Values: 32 × 4-bit floats (16 bytes), packed 2 per byte

Total: 1 + 16 = 17 bytes per 32 elements = 4.25 bits per weight
```

### e8m0 Scale

The scale is a pure power of 2 encoded as an 8-bit unsigned exponent:

```
scale = 2^(exponent - 127)

Examples:
  exponent=0   → 2^(-127) ≈ 5.88e-39 (denorm)
  exponent=126 → 2^(-1) = 0.5
  exponent=127 → 2^0 = 1.0
  exponent=128 → 2^1 = 2.0
  exponent=255 → 2^128 ≈ 3.4e38
```

### FP4 Values

Each 4-bit value encodes a small float: 1 sign bit + 2 exponent bits + 1 mantissa bit.

| Nibble | Value | Nibble | Value |
|--------|-------|--------|-------|
| 0000 | 0.0 | 1000 | -0.0 |
| 0001 | 0.5 | 1001 | -0.5 |
| 0010 | 1.0 | 1010 | -1.0 |
| 0011 | 1.5 | 1011 | -1.5 |
| 0100 | 2.0 | 1100 | -2.0 |
| 0101 | 3.0 | 1101 | -3.0 |
| 0110 | 4.0 | 1110 | -4.0 |
| 0111 | 6.0 | 1111 | -6.0 |

Final value = e8m0_scale × fp4_value.

### API

```rust
use larql_models::quant::mxfp4;

let scale = mxfp4::e8m0_to_f32(128);  // 2.0

// Dequantize a single expert's projection:
//   blocks = out_features × groups × 16 bytes (each byte = 2 × 4-bit values)
//   scales = out_features × groups bytes (one e8m0 per group of 32 elements)
let f32_row = mxfp4::dequantize_expert(&blocks, &scales, out_features, groups)?;

// Dequantize all experts from packed [num_experts, out_features, groups, 16] tensors:
let experts: Vec<Vec<f32>> =
    mxfp4::dequantize_all_experts(&blocks, &scales, num_experts, out_features, groups)?;
```

Both functions return `ModelError::Parse` if `blocks` or `scales` is too short
for the declared shape — truncated inputs surface as clean errors rather than
panicking on a slice OOB.

## Size Comparison

For a 10240×2560 FFN weight matrix (26.2M elements):

| Format | Size | Ratio vs f32 |
|--------|------|-------------|
| f32 | 105 MB | 1.0x |
| f16 | 52.4 MB | 0.50x |
| Q8_0 | 27.9 MB | 0.27x |
| Q5_1 | 19.7 MB | 0.19x |
| Q5_0 | 18.0 MB | 0.17x |
| Q4_1 | 16.4 MB | 0.16x |
| Q4_0 | 14.7 MB | 0.14x |
| MXFP4 | 13.9 MB | 0.13x |
