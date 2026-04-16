//! Q4 matvec v2: optimised for throughput.
//!
//! Changes from v1:
//! 1. Remove threadgroup shared memory (Q8 input fits in L1 cache at 2560B)
//! 2. Process 4 rows per thread (coalesced access across simdgroup)
//! 3. Unroll inner loop fully
//! 4. Use float accumulation throughout (avoid int→float at block boundary)
//!
//! Target: 0.57ms → <0.2ms on 14.7MB matrix.

pub const SHADER: &str = r#"
// Q4 matvec v2: 4 rows per thread, no threadgroup memory, fully unrolled.
// Grid: N/4 threads. Each thread computes 4 output scores.
// Adjacent threads process adjacent groups of 4 rows = coalesced reads.

kernel void q4_matvec_v2(
    device const uchar* Q4    [[buffer(0)]],
    device const float* x_f32 [[buffer(1)]],   // f32 input (not Q8)
    device float*       out   [[buffer(2)]],
    constant uint&      N     [[buffer(3)]],   // num rows (must be multiple of 4)
    constant uint&      K     [[buffer(4)]],   // hidden dim
    uint tid [[thread_position_in_grid]])
{
    uint row_base = tid * 4;
    if (row_base >= N) return;

    uint blocks = K / 32;
    uint bytes_per_row = blocks * 18;

    device const uchar* r0 = Q4 + (row_base + 0) * bytes_per_row;
    device const uchar* r1 = Q4 + (row_base + 1) * bytes_per_row;
    device const uchar* r2 = Q4 + (row_base + 2) * bytes_per_row;
    device const uchar* r3 = Q4 + (row_base + 3) * bytes_per_row;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    for (uint b = 0; b < blocks; b++) {
        // Decode scales for 4 rows
        float s0 = decode_f16_metal(ushort(r0[b*18]) | (ushort(r0[b*18+1]) << 8));
        float s1 = decode_f16_metal(ushort(r1[b*18]) | (ushort(r1[b*18+1]) << 8));
        float s2 = decode_f16_metal(ushort(r2[b*18]) | (ushort(r2[b*18+1]) << 8));
        float s3 = decode_f16_metal(ushort(r3[b*18]) | (ushort(r3[b*18+1]) << 8));

        device const uchar* q0 = r0 + b * 18 + 2;
        device const uchar* q1 = r1 + b * 18 + 2;
        device const uchar* q2 = r2 + b * 18 + 2;
        device const uchar* q3 = r3 + b * 18 + 2;

        // x values for this block
        device const float* xb = x_f32 + b * 32;

        // Process 16 bytes (32 values) per row
        float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;

        for (uint j = 0; j < 16; j++) {
            float x_lo = xb[j * 2];
            float x_hi = xb[j * 2 + 1];

            uchar byte0 = q0[j];
            sum0 += (float(int(byte0 & 0x0F) - 8)) * x_lo + (float(int(byte0 >> 4) - 8)) * x_hi;

            uchar byte1 = q1[j];
            sum1 += (float(int(byte1 & 0x0F) - 8)) * x_lo + (float(int(byte1 >> 4) - 8)) * x_hi;

            uchar byte2 = q2[j];
            sum2 += (float(int(byte2 & 0x0F) - 8)) * x_lo + (float(int(byte2 >> 4) - 8)) * x_hi;

            uchar byte3 = q3[j];
            sum3 += (float(int(byte3 & 0x0F) - 8)) * x_lo + (float(int(byte3 >> 4) - 8)) * x_hi;
        }

        acc0 += sum0 * s0;
        acc1 += sum1 * s1;
        acc2 += sum2 * s2;
        acc3 += sum3 * s3;
    }

    if (row_base + 0 < N) out[row_base + 0] = acc0;
    if (row_base + 1 < N) out[row_base + 1] = acc1;
    if (row_base + 2 < N) out[row_base + 2] = acc2;
    if (row_base + 3 < N) out[row_base + 3] = acc3;
}
"#;
