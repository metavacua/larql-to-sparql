#!/usr/bin/env python3
"""Long-context bench for the qwen35 KV-cache compression arc (Phase 3).

Compares f16 vs RotorQuant Iso3 device-resident KV cache across
varying prompt sizes. Records:
  - tok/s for the 32-token decode that follows prefill
  - VRAM peak during generation (sampled via nvidia-smi)
  - prompt_tokens / completion_tokens from the OpenAI response usage

Assumes a server is already running on http://localhost:8181 with the
desired KV-format env var set. Run twice (once per format) and diff
the outputs:

    # Terminal 1 — f16 baseline
    LARQL_QWEN35_GPU=1 ./target/release/larql-server …

    # Terminal 2 — bench
    python scripts/bench-long-context.py f16 > bench-f16.csv

    # Restart server with iso3
    LARQL_QWEN35_GPU=1 LARQL_QWEN35_KV_FORMAT=iso3 \\
        LARQL_QWEN35_KV_MAX_SEQ=8192 \\
        ./target/release/larql-server …

    python scripts/bench-long-context.py iso3 > bench-iso3.csv

The chat completion API tokenises the prompt server-side; the script
sends increasingly-long lorem-ipsum chunks and records the `usage`
counts the server returns. Char-to-token ratio depends on tokenizer
(Qwen ≈ 3-4 chars/token), so prompt size targets are approximate.

Each row of output CSV:
  mode,target_prompt_tokens,actual_prompt_tokens,decode_tokens,prefill_s,decode_s,tok_per_s,vram_pre_mib,vram_peak_mib
"""

import json
import os
import subprocess
import sys
import threading
import time
import urllib.request

PORT = int(os.environ.get("LARQL_BENCH_PORT", "8181"))
URL = f"http://localhost:{PORT}/v1/chat/completions"
MODEL = os.environ.get("LARQL_BENCH_MODEL", "Qwen3.6-35B-A3B")
DECODE_TOKENS = int(os.environ.get("LARQL_BENCH_DECODE", "32"))
HTTP_TIMEOUT = float(os.environ.get("LARQL_BENCH_HTTP_TIMEOUT", "600"))

# Target prompt token counts. Qwen tokeniser averages ~3.5 chars/tok
# on lorem-style English, so we send char counts close to 3.5×.
# Override with `LARQL_BENCH_TARGETS=32000,64000,128000`.
_DEFAULT_TARGETS = "128,2048,4096"
PROMPT_TARGETS = [
    int(x) for x in os.environ.get("LARQL_BENCH_TARGETS", _DEFAULT_TARGETS).split(",") if x
]

LOREM = (
    "The quick brown fox jumps over the lazy dog. "
    "In a hole in the ground there lived a hobbit. "
    "Far over the misty mountains cold to dungeons deep and caverns old. "
    "Two roads diverged in a yellow wood, and sorry I could not travel both. "
    "It was the best of times, it was the worst of times. "
)


def vram_used_mib():
    """Read GPU 0 memory.used in MiB from nvidia-smi."""
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
    )
    return int(out.decode().splitlines()[0].strip())


class VramSampler(threading.Thread):
    """Background sampler that records peak VRAM during a request."""
    def __init__(self, interval=0.1):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak = 0
        self.running = True

    def run(self):
        while self.running:
            try:
                v = vram_used_mib()
                if v > self.peak:
                    self.peak = v
            except Exception:
                pass
            time.sleep(self.interval)

    def stop(self):
        self.running = False
        self.join(timeout=1.0)
        return self.peak


def make_prompt_chars(target_chars):
    base = LOREM
    out = ""
    while len(out) < target_chars:
        out += base
    return out[:target_chars]


def call(prompt_chars, decode_tokens=DECODE_TOKENS, timeout=HTTP_TIMEOUT):
    """POST a chat completion. Return (data, elapsed_s)."""
    body = json.dumps(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt_chars}],
            "max_tokens": decode_tokens,
            "temperature": 0.0,
        }
    ).encode()
    req = urllib.request.Request(
        URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    t1 = time.perf_counter()
    return json.loads(raw), t1 - t0


def bench_one(mode, target_tok):
    # Estimate char count from target tokens (over-approximate; we
    # measure actual_prompt_tokens from the response).
    chars = int(target_tok * 4.5)
    prompt = make_prompt_chars(chars)

    pre_vram = vram_used_mib()
    sampler = VramSampler()
    sampler.start()
    try:
        data, elapsed = call(prompt)
    finally:
        peak_vram = sampler.stop()
    usage = data.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    # Crude split: assume prefill dominates the wall time at long
    # context; tok/s is decode-only via completion_tokens / time
    # spent after prefill. We don't have a clean prefill/decode split
    # from the API, so emit the simpler decode_per_token_avg:
    decode_per_tok = elapsed / max(completion_tokens, 1)
    decode_tok_per_s = completion_tokens / max(elapsed, 1e-9)
    return {
        "mode": mode,
        "target_prompt_tokens": target_tok,
        "actual_prompt_tokens": prompt_tokens,
        "decode_tokens": completion_tokens,
        "wall_s": elapsed,
        "decode_per_tok_avg_s": decode_per_tok,
        "tok_per_s_overall": decode_tok_per_s,
        "vram_pre_mib": pre_vram,
        "vram_peak_mib": peak_vram,
        "content": (data.get("choices", [{}])[0].get("message", {}).get("content", ""))[:60],
    }


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "unknown"
    # Warmup so we don't measure lazy weight load.
    print("[warmup]", file=sys.stderr)
    bench_one(mode, 32)

    cols = [
        "mode",
        "target_prompt_tokens",
        "actual_prompt_tokens",
        "decode_tokens",
        "wall_s",
        "decode_per_tok_avg_s",
        "tok_per_s_overall",
        "vram_pre_mib",
        "vram_peak_mib",
        "content",
    ]
    print(",".join(cols))
    for tgt in PROMPT_TARGETS:
        try:
            row = bench_one(mode, tgt)
        except Exception as e:
            print(f"# target_tok={tgt} FAILED: {e}", file=sys.stderr)
            continue
        print(",".join(str(row[c]).replace(",", " ") for c in cols))
        sys.stdout.flush()


if __name__ == "__main__":
    main()
