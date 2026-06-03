// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Ian Douglas Lawrence Norman McLean
import init, { GeometricAttn } from './larql_geometric.js';

let wasmReady = false;
async function ensureInit() {
    if (!wasmReady) { await init(); wasmReady = true; }
}

function log(msg) {
    const el = document.getElementById('log');
    el.textContent += msg + '\n';
    el.scrollTop = el.scrollHeight;
}

window.runBench = async function() {
    await ensureInit();
    const backend = document.getElementById('backend').value;
    const n = parseInt(document.getElementById('seqlen').value, 10);
    const d = parseInt(document.getElementById('headdim').value, 10);

    const rand = () => (Math.random() - 0.5) * 2;
    const q = Float32Array.from({length: n*d}, rand);
    const k = Float32Array.from({length: n*d}, rand);
    const v = Float32Array.from({length: n*d}, rand);

    const attn = new GeometricAttn(backend, n, d);
    attn.run(q, k, v); // warm-up

    const REPS = 10;
    const t0 = performance.now();
    let out;
    for (let i = 0; i < REPS; i++) out = attn.run(q, k, v);
    const avg = (performance.now() - t0) / REPS;
    log(`[${backend}] n=${n} d=${d}: ${avg.toFixed(2)}ms/call`);
    drawHeatmap(out, n, d);
};

function drawHeatmap(data, n, d) {
    const canvas = document.getElementById('heatmap');
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    const rows = Math.min(n, 32), cols = Math.min(d, 64);
    const cw = Math.floor(W / cols), ch = Math.floor(H / rows);
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < rows * cols; i++) {
        const v = data[i] ?? 0;
        if (v < lo) lo = v;
        if (v > hi) hi = v;
    }
    const range = Math.max(hi - lo, 1e-8);
    ctx.clearRect(0, 0, W, H);
    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            const t = ((data[r * d + c] ?? 0) - lo) / range;
            const red = Math.round(t * 220);
            const blue = Math.round((1 - t) * 220);
            ctx.fillStyle = `rgb(${red},40,${blue})`;
            ctx.fillRect(c * cw, r * ch, cw - 1, ch - 1);
        }
    }
}
