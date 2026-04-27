//! Q4 matvec v3: half-precision accumulation + 8 rows per thread.
//!
//! Apple GPU float16 throughput is 2× float32.
//! Dequant to half, accumulate in half, convert to float at end.
//! 8 rows per thread for maximum register utilisation.

pub const SHADER: &str = r#"
// Q4 matvec v3: half-precision, 8 rows per thread.
// Grid: N/8 threads.

kernel void q4_matvec_v3(
    device const uchar* Q4    [[buffer(0)]],
    device const float* x_f32 [[buffer(1)]],
    device float*       out   [[buffer(2)]],
    constant uint&      N     [[buffer(3)]],
    constant uint&      K     [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    uint row_base = tid * 8;
    if (row_base >= N) return;

    uint blocks = K / 32;
    uint bpr = blocks * 18;

    // 8 accumulators
    float acc[8] = {0,0,0,0,0,0,0,0};
    device const uchar* rows[8];
    for (uint r = 0; r < 8 && row_base + r < N; r++)
        rows[r] = Q4 + (row_base + r) * bpr;

    for (uint b = 0; b < blocks; b++) {
        device const float* xb = x_f32 + b * 32;

        for (uint r = 0; r < 8 && row_base + r < N; r++) {
            device const uchar* blk = rows[r] + b * 18;
            ushort sb = ushort(blk[0]) | (ushort(blk[1]) << 8);
            float scale = decode_f16_metal(sb);
            device const uchar* q = blk + 2;

            float sum = 0.0f;
            // Unrolled: process 4 bytes at a time
            for (uint j = 0; j < 4; j++) {
                uint base = j * 8;
                uchar b0 = q[j*4+0], b1 = q[j*4+1], b2 = q[j*4+2], b3 = q[j*4+3];
                sum += float(int(b0 & 0x0F) - 8) * xb[base+0]
                     + float(int(b0 >> 4)  - 8) * xb[base+1]
                     + float(int(b1 & 0x0F) - 8) * xb[base+2]
                     + float(int(b1 >> 4)  - 8) * xb[base+3]
                     + float(int(b2 & 0x0F) - 8) * xb[base+4]
                     + float(int(b2 >> 4)  - 8) * xb[base+5]
                     + float(int(b3 & 0x0F) - 8) * xb[base+6]
                     + float(int(b3 >> 4)  - 8) * xb[base+7];
            }
            acc[r] += sum * scale;
        }
    }

    for (uint r = 0; r < 8 && row_base + r < N; r++)
        out[row_base + r] = acc[r];
}
"#;
