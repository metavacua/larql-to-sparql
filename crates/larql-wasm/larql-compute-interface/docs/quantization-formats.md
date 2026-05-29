# Quantization Formats — larql-compute

## Summary

| Format | Size per value | Block size | Use case | Origin |
|--------|---------------|------------|----------|--------|
| Q4_0 | 0.5625 B | 32 values = 18 bytes | FFN gate/up/down | GGUF standard |
| Q4_K | 0.578 B | 256 values = 148 bytes | Attention Q/K/O | GGUF standard (our layout) |
| Q4_KF | 0.625 B | 256 values = 160 bytes | Fast decode (experimental) | LARQL original |
| Q6_K | 0.820 B | 256 values = 210 bytes | Attention V, FFN down | GGUF standard |
| Q8_0 | 1.125 B | 32 values = 36 bytes | Higher precision fallback | GGUF standard |

## Q4_0 (Production FFN)

```
Block: 32 values → 18 bytes
  [0..1]   f16 scale (2 bytes)
  [2..17]  16 packed nibble bytes (32 × 4-bit values)

Dequant: val = scale × (nibble - 8)
GPU kernel: q4_matvec_v4 (integer inner loop, 57 GB/s)
```

## Q4_K (Production Attention — our 148-byte layout)

```
Superblock: 256 values → 148 bytes
  [0..1]    f16 d (delta scale)
  [2..3]    f16 dmin (minimum scale)
  [4..15]   12 bytes: 8 × 6-bit sub-block scales (packed)
  [16..19]  4 bytes: 8 × 4-bit sub-block mins (packed)
  [20..147] 128 bytes: 256 × 4-bit values (packed nibbles)

Dequant: val = d × scale_j × nibble - dmin × min_j
  where j = sub-block index (0..7), each sub-block = 32 values

GPU kernel: q4k_qkv_proj (fused QKV, sub-block lanes)
CPU reference: cpu/ops/q4k_matvec.rs
```

**Difference from GGUF**: Our layout separates scales (12 bytes) and mins (4 bytes).
GGUF packs both into 12 bytes using 6-bit precision for both.
See Q4_K GGUF below.

## Q4_K GGUF (144-byte, Ollama-compatible)

```
Superblock: 256 values → 144 bytes
  [0..1]    half d
  [2..3]    half dmin
  [4..15]   12 bytes: scales AND mins packed together
            bytes 0-3: lower 6 bits = scales[0..3], upper 2 bits = mins[0..3] low
            bytes 4-7: lower 6 bits = scales[4..7], upper 2 bits = mins[4..7] low
            bytes 8-11: upper 4 bits of all 8 mins packed
  [16..143] 128 bytes nibbles

GPU kernel: q4kf_qkv_proj (llama.cpp-style, register-based input)
Quantizer: quantize_q4_k_gguf()
```

## Q4_KF (Pre-baked scales — experimental)

```
Superblock: 256 values → 160 bytes
  [0..15]   8 × half pre-computed d*scale_j
  [16..31]  8 × half pre-computed dmin*min_j
  [32..159] 128 bytes nibbles

Eliminates header decode + scale unpack from inference hot loop.
Measured: no speed improvement vs Q4_K (scale decode is <10% of ALU).
Converter: q4k_to_q4kf()
```

## Q6_K (Higher precision)

```
Superblock: 256 values → 210 bytes
  [0..127]   128 bytes: lower 4 bits (packed nibbles)
  [128..191] 64 bytes: upper 2 bits (4 per byte)
  [192..207] 16 × int8 scales (one per 16-value sub-block)
  [208..209] f16 super-block scale

Dequant: val = d × scale_j × ((lo4 | (hi2 << 4)) - 32)
GPU kernel: q6k_matvec
CPU reference: cpu/ops/q6k_matvec.rs
```

## Q8_0 (Intermediate precision)

```
Block: 32 values → separate storage
  Values: int8 array (1 byte per value)
  Scales: f32 array (1 per 32-element block)

Dequant: val = scale × int8_value
GPU kernel: q8_matvec, q8_qkv_proj
```

## Quantization Strategy (matching Ollama Q4_K_M)

| Component | Ollama | LARQL | Format |
|-----------|--------|-------|--------|
| Attention Q/K/O | Q4_K | Q4_K | q4k_qkv_proj |
| Attention V | Q6_K | Q6_K | q6k_matvec |
| FFN gate/up | Q4_K | Q4_0 | q4_matvec_v4 |
| FFN down | Q6_K | Q4_0 | q4_f32_matvec |
| Norms | f32 | f32 | rms_norm |
| Embeddings | Q6_K | Q6_K | — |

## Quantize Functions (cpu/ops/q4_common.rs)

| Function | Input | Output | Notes |
|----------|-------|--------|-------|
| `quantize_q4_0(data)` | `&[f32]` | `Vec<u8>` | 18 bytes per 32 values |
| `quantize_q4_k(data)` | `&[f32]` | `Vec<u8>` | 148 bytes per 256 values |
| `quantize_q4_k_gguf(data)` | `&[f32]` | `Vec<u8>` | 144 bytes, GGUF-compatible |
| `quantize_q4_kf(data)` | `&[f32]` | `Vec<u8>` | 160 bytes, pre-baked scales |
| `quantize_q6_k(data)` | `&[f32]` | `Vec<u8>` | 210 bytes per 256 values |
| `quantize_to_q8(x)` | `&[f32]` | `(Vec<i8>, Vec<f32>)` | Values + per-block scales |
| `q4k_to_q4kf(data, rows, hidden)` | Q4_K bytes | `Vec<u8>` | Format conversion |
| `f16_to_f32(bits)` | `u16` | `f32` | Shared helper |
