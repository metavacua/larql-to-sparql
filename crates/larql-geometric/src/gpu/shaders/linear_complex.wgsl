// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Ian Douglas Lawrence Norman McLean
//
// Complex linear attention: two-pass computation
// Pass 1: KtV[i,j] = sum_t { cos(K[t,i]) * V[t,j],  -sin(K[t,i]) * V[t,j] }
// Pass 2: out[pos,dim] = (sum_d { cos(Q[pos,d])*KtV_re[d,dim] - sin(Q[pos,d])*KtV_im[d,dim] }) / seq_len

struct Params {
    seq_len: u32,
    head_dim: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> Q: array<f32>;
@group(0) @binding(2) var<storage, read> K: array<f32>;
@group(0) @binding(3) var<storage, read> V: array<f32>;
@group(0) @binding(4) var<storage, read_write> KtV_re: array<f32>;
@group(0) @binding(5) var<storage, read_write> KtV_im: array<f32>;
@group(0) @binding(6) var<storage, read_write> out_buf: array<f32>;

@compute @workgroup_size(8, 8)
fn pass1_ktv(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let j = gid.y;
    if i >= p.head_dim || j >= p.head_dim { return; }
    var re = 0.0f;
    var im = 0.0f;
    for (var t = 0u; t < p.seq_len; t++) {
        let k_val = K[t * p.head_dim + i];
        let v_val = V[t * p.head_dim + j];
        re += cos(k_val) * v_val;
        im -= sin(k_val) * v_val;
    }
    KtV_re[i * p.head_dim + j] = re;
    KtV_im[i * p.head_dim + j] = im;
}

@compute @workgroup_size(64)
fn pass2_out(@builtin(global_invocation_id) gid: vec3<u32>) {
    let flat = gid.x;
    let pos = flat / p.head_dim;
    let dim = flat % p.head_dim;
    if pos >= p.seq_len { return; }
    var s = 0.0f;
    for (var d = 0u; d < p.head_dim; d++) {
        let q_val = Q[pos * p.head_dim + d];
        s += cos(q_val) * KtV_re[d * p.head_dim + dim]
           - sin(q_val) * KtV_im[d * p.head_dim + dim];
    }
    out_buf[pos * p.head_dim + dim] = s / f32(p.seq_len);
}
