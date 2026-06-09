## ADDED Requirements

### Requirement: f16-to-f32 subnormal decoding MUST be bit-exact vs the powi reference

`larql_compute::cpu::ops::q4_common::f16_to_f32` SHALL decode every f16
input bit pattern to the SAME f32 bits as the `mant / 1024 * 2^(-14)`
powi-based reference, for all 65,536 possible inputs (excluding NaN
payload differences, which are unobservable in Q4_K decode). This
applies to normal, subnormal, signed-zero, and infinity inputs alike.

The previous subnormal formula `(127u32 - 14 - lz) << 23` was off by
one in the biased exponent, producing values exactly 2× the correct
magnitude. The corrected formula is `(112u32 - lz) << 23`, equivalent
to `127 + (-14 - 1 - lz)` (biased exponent for `mant * 2^(-14-shifts)`
where `shifts` is the number of left-shifts needed to normalise the
mantissa).

#### Scenario: All 65,536 f16 inputs decode bit-exact vs the powi reference

- **GIVEN** the powi-based reference implementation
  `f16_to_f32_powi_reference` (mirroring llama.cpp's mathematical
  definition of f16 → f32)
- **WHEN** `f16_to_f32(bits)` is called for every `bits` in
  `0..=u16::MAX`
- **THEN** the f32 bit-representation SHALL match `f16_to_f32_powi_reference(bits)`
  for every non-NaN input; for NaN inputs the result SHALL be a
  representable f32 NaN (payload preservation is allowed but not
  required)
<!-- test: larql_compute::cpu::ops::q4_common::tests::f16_to_f32_bit_exact_for_all_inputs -->

### Requirement: K-quant matvec output MUST match canonical-dequant within Q8_K activation noise on real on-disk bytes

The Q4_K × Q8_K and Q6_K × Q8_K matvec kernels SHALL produce output
within ≤ 1.5 % relative error of the canonical-dequant + f32 dot
product on real vindex bytes, including when the f16 super-block scale
`d` falls in subnormal range. The previous f16 subnormal decode bug
caused Q6_K matvec output to be 2× the canonical-dequant reference on
any tensor whose `d` was small enough to be subnormal (e.g., Gemma 3
4B V-projection and FFN_DOWN at small weight magnitudes).

#### Scenario: Q6_K matvec on real Gemma 3 4B V bytes matches canonical dequant

- **GIVEN** a Q4_K vindex for Gemma 3 4B with the V projection stored
  as Q6_K
- **WHEN** `q6k_q8k_matvec_into(out, q8k_x, v_bytes, kv_dim, hidden)`
  is called with a Q8_K-quantised activation `q8k_x`
- **THEN** the output SHALL match `dequantize_q6_k(v_bytes) @ x`
  row-wise within ≤ 1.5 % relative error (Q8_K activation noise
  envelope)
<!-- test: unbacked -->
