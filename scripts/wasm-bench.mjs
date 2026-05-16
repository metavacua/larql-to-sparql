#!/usr/bin/env node
/**
 * WASM parallel-vs-serial benchmark runner.
 *
 * Downloads pre-built WASM artifacts from the `build` CI job and measures
 * throughput for pagerank and BFS on graphs of increasing size.
 *
 * Artifacts layout expected (set by CI download-artifact steps):
 *   ./wasm-artifacts/serial/   ← larql-wasm-serial pkg dir
 *   ./wasm-artifacts/parallel/ ← larql-wasm-parallel pkg dir
 *
 * Outputs: bench-results.json + Markdown table to stdout.
 */

import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import path from "node:path";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, "..");

// ── Locate artifact directories ───────────────────────────────────────────────

const ARTIFACT_BASE = path.join(repoRoot, "wasm-artifacts");
const SERIAL_PKG = path.join(ARTIFACT_BASE, "serial");
const PARALLEL_PKG = path.join(ARTIFACT_BASE, "parallel");

function checkArtifacts() {
  const missing = [];
  if (!existsSync(SERIAL_PKG)) missing.push(SERIAL_PKG);
  if (!existsSync(PARALLEL_PKG)) missing.push(PARALLEL_PKG);
  if (missing.length) {
    console.error("Missing artifact directories:");
    missing.forEach((p) => console.error("  " + p));
    console.error(
      "\nRun the CI `build` jobs first, or download artifacts to wasm-artifacts/{serial,parallel}/"
    );
    process.exit(1);
  }
}

// ── Benchmark configuration ───────────────────────────────────────────────────

const CONFIGS = [
  { label: "small", nEdges: 200, rounds: 20 },
  { label: "medium", nEdges: 1000, rounds: 10 },
  { label: "large", nEdges: 5000, rounds: 5 },
];

const THREAD_COUNT = 4;

// ── Helpers ───────────────────────────────────────────────────────────────────

function median(arr) {
  const s = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

function fmtMs(ms) {
  return ms.toFixed(1) + " ms";
}

function speedup(serial, parallel) {
  if (parallel <= 0) return "N/A";
  return (serial / parallel).toFixed(2) + "×";
}

// ── Main ──────────────────────────────────────────────────────────────────────

async function main() {
  checkArtifacts();

  // Dynamic imports resolve at runtime from the downloaded pkg directories.
  const require = createRequire(import.meta.url);

  console.log("Loading serial WASM module...");
  const serialPkgJs = path.join(SERIAL_PKG, "larql_wasm.js");
  const serialWasm = await import(serialPkgJs);
  // wasm-pack web target: call default export to init.
  await serialWasm.default(
    readFileSync(path.join(SERIAL_PKG, "larql_wasm_bg.wasm"))
  );

  console.log("Loading parallel WASM module...");
  const parallelPkgJs = path.join(PARALLEL_PKG, "larql_wasm.js");
  const parallelWasm = await import(parallelPkgJs);
  await parallelWasm.default(
    readFileSync(path.join(PARALLEL_PKG, "larql_wasm_bg.wasm"))
  );

  console.log(`Initialising parallel thread pool (${THREAD_COUNT} threads)...`);
  await parallelWasm.initThreadPool(THREAD_COUNT);

  // ── Run benchmarks ────────────────────────────────────────────────────────

  const results = {};

  for (const { label, nEdges, rounds } of CONFIGS) {
    console.log(`\nBenchmarking config: ${label} (${nEdges} edges, ${rounds} rounds each)`);

    // PageRank
    const prSerial = serialWasm.benchmark_pagerank_serial(nEdges, rounds);
    const prParallel = parallelWasm.benchmark_pagerank_parallel(nEdges, rounds);

    // BFS
    const bfsSerial = serialWasm.benchmark_bfs_serial(nEdges, rounds);
    const bfsParallel = parallelWasm.benchmark_bfs_parallel(nEdges, rounds);

    results[label] = {
      nEdges,
      rounds,
      pagerank: { serial_ms: prSerial, parallel_ms: prParallel },
      bfs: { serial_ms: bfsSerial, parallel_ms: bfsParallel },
    };
  }

  // ── Print Markdown table ──────────────────────────────────────────────────

  console.log("\n## WASM Serial vs Parallel Benchmark Results\n");
  console.log(
    "| Config | Op | Serial | Parallel | Speedup |"
  );
  console.log(
    "|--------|----|--------|----------|---------|"
  );

  for (const [label, r] of Object.entries(results)) {
    const pr = r.pagerank;
    const bfs = r.bfs;
    console.log(
      `| ${label} (${r.nEdges} edges) | pagerank | ${fmtMs(pr.serial_ms)} | ${fmtMs(pr.parallel_ms)} | ${speedup(pr.serial_ms, pr.parallel_ms)} |`
    );
    console.log(
      `| ${label} (${r.nEdges} edges) | bfs      | ${fmtMs(bfs.serial_ms)} | ${fmtMs(bfs.parallel_ms)} | ${speedup(bfs.serial_ms, bfs.parallel_ms)} |`
    );
  }

  // ── Write JSON output ─────────────────────────────────────────────────────

  const output = {
    timestamp: new Date().toISOString(),
    thread_count: THREAD_COUNT,
    results,
  };

  writeFileSync("bench-results.json", JSON.stringify(output, null, 2));
  console.log("\nWrote bench-results.json");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
