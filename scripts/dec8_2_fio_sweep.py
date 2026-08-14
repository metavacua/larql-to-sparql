#!/usr/bin/env python3
"""DEC-8.2 — row-granular random read on an external NVMe, driven through fio.

Row-granular expert fetch turns one large sequential read per expert into
thousands of small scattered ones. DEC-8.0 said the resulting REQUEST RATE
(~7.7M IOPS aggregate for the interleaved layout) already exceeds what the link
could do at 4 KiB even if it were purely bandwidth-limited. This measures the
actual ceiling.

WHY FIO AND NOT A PYTHON HARNESS. The first attempt at this used Python threads
issuing pread(). It reported 875k IOPS at 1.0 us p50 latency and 32 GB/s on
5.85 MB reads -- physically impossible for NVMe, and QD32 came out SLOWER than
QD1. Two instrument failures: the just-written file was resident in the page
cache (F_NOCACHE on the read fd does not evict pages already cached by the
write), and CPython's GIL caps a thread pool at roughly one million syscalls per
second regardless of the device. An instrument that tops out below the kill bar
cannot measure the kill bar. Recorded because "the tool was the bottleneck" is
the failure mode this ladder's rules exist to catch (R0: name the condition).

Both failures are fixed here: fio forks real processes (psync engine -- macOS
has no libaio/io_uring), and the test file is 200 GB against 137 GB of RAM, so
the page cache cannot hold it even if --direct=1 were ignored.

LINK CONVENTION (R0). The pre-registration says "USB4". The measured link is
Thunderbolt with PCIe TUNNELLING: the drive enumerates as native NVMe over PCIe
x4 at 16 GT/s through a 40 Gb/s enclosure, not as USB mass-storage. Native NVMe
queues make this the FAVOURABLE case for small random reads. A result here is an
upper bound on what a USB4/UASP enclosure would do and must not be quoted as one.
"""

import argparse
import json
import subprocess
from pathlib import Path

# ── Geometry from DEC-8.0's measurement of the real checkpoint ─────────────
PACKED_ROW = 1792                 # w1/w2 packed row per feature
SCALE_ROW = 112                   # paired MXFP4 scale row per feature
FEATURE_ROW = PACKED_ROW + SCALE_ROW      # 1904, arm B's single extent
EXPERT_PAYLOAD = 5_849_088        # one branch of one expert
MOE_LAYERS, TOP_K, FEATURES = 92, 16, 3072
FEATURE_FRAC, MATRICES, LOCALITY = 0.064, 2, 1.5


def required():
    per_tok = FEATURE_FRAC * FEATURES * MATRICES * TOP_K * MOE_LAYERS / LOCALITY
    return {"features_per_token": per_tok,
            "arm_B_iops_at_20": per_tok * 20,
            "arm_A_iops_at_20": per_tok * 20 * 2}


MAX_JOBS_PER_FIO = 16   # macOS kern.sysv.shmmax caps a single fio's job table


def fio(path, name, runtime, jobs, bs=None, bssplit=None, extra=None):
    """Total concurrency `jobs`, split across parallel fio instances.

    macOS caps the SysV shm segment fio allocates for its job table, so a single
    invocation dies with "failed to setup shm segment" past ~16 jobs. Raising
    kern.sysv.shmmax needs root; running several fio processes side by side does
    not, and the device sees the same number of outstanding requests either way.
    Rates are summed; latency is reported as the worst instance's percentile,
    which is the honest reading when several instances share one device.
    """
    if jobs <= MAX_JOBS_PER_FIO:
        return _fio_one(path, name, runtime, jobs, bs, bssplit, extra)
    n_inst = (jobs + MAX_JOBS_PER_FIO - 1) // MAX_JOBS_PER_FIO
    per = jobs // n_inst
    procs = [_fio_spawn(path, f"{name}_{i}", runtime, per, bs, bssplit, extra)
             for i in range(n_inst)]
    parts = [_fio_collect(p, per, bs) for p in procs]
    return {"iops": sum(p["iops"] for p in parts),
            "gb_s": sum(p["gb_s"] for p in parts),
            "lat_us_p50": max(p["lat_us_p50"] for p in parts),
            "lat_us_p99": max(p["lat_us_p99"] for p in parts),
            "jobs": per * n_inst, "instances": n_inst}


def _fio_cmd(path, name, runtime, jobs, bs, bssplit, extra):
    cmd = ["fio", f"--name={name}", f"--filename={path}", "--rw=randread",
           "--direct=1", "--ioengine=psync", "--thread", f"--numjobs={jobs}",
           f"--runtime={runtime}", "--time_based", "--norandommap",
           "--randrepeat=0", "--group_reporting", "--output-format=json"]
    cmd.append(f"--bs={bs}" if bs else f"--bssplit={bssplit}")
    return cmd + (extra or [])


def _fio_spawn(path, name, runtime, jobs, bs, bssplit, extra):
    return subprocess.Popen(_fio_cmd(path, name, runtime, jobs, bs, bssplit, extra),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True)


def _fio_collect(proc, jobs, bs):
    so, se = proc.communicate()
    if proc.returncode != 0 or not so.strip().startswith("{"):
        raise RuntimeError(f"fio failed (jobs={jobs}, bs={bs}): {(se or so)[:300]}")
    r = json.loads(so)["jobs"][0]["read"]
    return {"iops": r["iops"], "gb_s": r["bw_bytes"] / 1e9,
            "lat_us_p50": r["clat_ns"]["percentile"].get("50.000000", 0) / 1e3,
            "lat_us_p99": r["clat_ns"]["percentile"].get("99.000000", 0) / 1e3}


def _fio_one(path, name, runtime, jobs, bs=None, bssplit=None, extra=None):
    p = _fio_spawn(path, name, runtime, jobs, bs, bssplit, extra)
    r = _fio_collect(p, jobs, bs)
    r["jobs"] = jobs
    r["instances"] = 1
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--path", default="/Volumes/model-drive/dec8_2.bin")
    ap.add_argument("--runtime", type=int, default=10)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    req = required()

    print("=" * 78)
    print("DEC-8.2 — row-granular random read (fio, psync, direct=1, 200 GB file)")
    print("=" * 78)
    print("LINK: Thunderbolt PCIe tunnelling / native NVMe — NOT USB4/UASP.")
    print(f"needed at 20 tok/s, single drive: arm B "
          f"{req['arm_B_iops_at_20']/1e6:.2f}M IOPS, arm A "
          f"{req['arm_A_iops_at_20']/1e6:.2f}M IOPS")
    print()

    res = {}

    print(f"--- concurrency sweep, arm B (bs={FEATURE_ROW}) ---")
    print(f"{'jobs':>5} {'IOPS':>12} {'GB/s':>8} {'p50 us':>9} {'p99 us':>9}")
    res["B_concurrency"] = []
    # 16 is an INSTRUMENT ceiling, not a device one: macOS ships
    # kern.sysv.shmall=1024 pages = 4 MB of SysV shm system-wide, and fio sizes
    # its job table from that. Raising it needs root. Little's Law below checks
    # whether we are latency-bound (measuring the link) or tool-bound.
    for j in (1, 2, 4, 8, 16):
        r = fio(args.path, "armB", args.runtime, j, bs=FEATURE_ROW)
        res["B_concurrency"].append(r)
        print(f"{j:>5} {r['iops']:>12,.0f} {r['gb_s']:>8.2f} "
              f"{r['lat_us_p50']:>9.1f} {r['lat_us_p99']:>9.1f}")
    print()

    best_j = max(res["B_concurrency"], key=lambda r: r["iops"])["jobs"]
    print(f"--- block-size sweep at numjobs={best_j} ---")
    print(f"{'bs':>10} {'IOPS':>12} {'GB/s':>8} {'p50 us':>9}")
    res["blocksize"] = []
    for bs in (512, 1024, FEATURE_ROW, 4096, 16384, 65536, 1 << 20):
        r = fio(args.path, "bssweep", args.runtime, best_j, bs=bs)
        r["bs"] = bs
        res["blocksize"].append(r)
        print(f"{bs:>10,} {r['iops']:>12,.0f} {r['gb_s']:>8.2f} "
              f"{r['lat_us_p50']:>9.1f}")
    print()

    print(f"--- arm A (two extents/feature: {PACKED_ROW}+{SCALE_ROW}) "
          f"vs arm C (whole expert) ---")
    res["A"] = fio(args.path, "armA", args.runtime, best_j,
                   bssplit=f"{PACKED_ROW}/50:{SCALE_ROW}/50")
    res["C"] = fio(args.path, "armC", args.runtime, min(best_j, 8),
                   bs=EXPERT_PAYLOAD)
    print(f"  A  {res['A']['iops']:>12,.0f} IOPS  {res['A']['gb_s']:>6.2f} GB/s "
          f"-> {res['A']['iops']/2:,.0f} features/s")
    print(f"  C  {res['C']['iops']:>12,.0f} IOPS  {res['C']['gb_s']:>6.2f} GB/s "
          f"(sequential-equivalent baseline)")
    print()

    best_b = max(res["B_concurrency"], key=lambda r: r["iops"])
    feats_b = best_b["iops"]
    feats_a = res["A"]["iops"] / 2
    print("--- verdict ---")
    print(f"  arm B peak   {best_b['iops']/1e6:.3f}M IOPS @ {best_b['jobs']} jobs, "
          f"{best_b['gb_s']:.2f} GB/s")
    print(f"    vs {req['arm_B_iops_at_20']/1e6:.2f}M needed -> "
          f"{req['arm_B_iops_at_20']/best_b['iops']:.1f}x SHORT")
    print(f"  arm A peak   {res['A']['iops']/1e6:.3f}M IOPS -> "
          f"{req['arm_A_iops_at_20']/res['A']['iops']:.1f}x SHORT")
    print(f"  relayout gain (B features/s over A): {feats_b/feats_a:.2f}x")
    print(f"  tok/s supported by arm B: "
          f"{feats_b/req['features_per_token']:.2f}")
    print(f"  arm C sequential: {res['C']['gb_s']:.2f} GB/s")
    print()

    # Little's Law: concurrency = throughput x latency. If the implied
    # concurrency matches the jobs actually issued, we are latency-bound on the
    # link and the number is the link's, not the harness's.
    implied = best_b["iops"] * best_b["lat_us_p50"] / 1e6
    print("--- is this the link or the instrument? (Little's Law) ---")
    print(f"  {best_b['iops']:,.0f} IOPS x {best_b['lat_us_p50']:.1f} us "
          f"= {implied:.1f} concurrent I/Os, against {best_b['jobs']} issued "
          f"-> {'LATENCY-BOUND on the link' if implied > 0.8*best_b['jobs'] else 'harness-bound, distrust'}")
    need_conc = req["arm_B_iops_at_20"] * best_b["lat_us_p50"] / 1e6
    print(f"  reaching {req['arm_B_iops_at_20']/1e6:.2f}M IOPS at this latency "
          f"would need ~{need_conc:,.0f} concurrent I/Os in flight.")
    print("=" * 78)

    out = {"link_convention": "Thunderbolt PCIe tunnelling (native NVMe), "
                              "NOT USB4/UASP — favourable case, upper bound",
           "drive": "Samsung SSD 990 PRO 2TB, Ugreen 40Gb/s enclosure",
           "method": "fio 3.42, psync, direct=1, 200 GB file vs 137 GB RAM",
           "required": req, "results": res,
           "tok_s_supported_by_B": feats_b / req["features_per_token"],
           "shortfall_B_x": req["arm_B_iops_at_20"] / best_b["iops"]}
    p = args.out or Path(__file__).parent / ".k3_budget_cache" / "dec8_2_fio.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
