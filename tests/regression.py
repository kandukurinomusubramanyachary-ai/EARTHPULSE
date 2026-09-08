#!/usr/bin/env python3
"""
EarthPulse regression suite.

Runs the full HTTP pipeline for the known-good India case, the previously
crashing Nepal case, and a set of edge cases that must degrade gracefully
rather than kill the server.

Usage:  python3 tests/regression.py [base_url]
"""
from __future__ import annotations

import json
import subprocess
import sys
import time

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"

CASES = [
    # (label, query, expectation)
    ("TEST A  India Terai (must keep working)",
     "Check flood/water expansion changes in Terai, Khaniyadhana Tahsil, Shivpuri, "
     "Madhya Pradesh, India between 2021 and 2025 using Sentinel-2 imagery.",
     "result"),
    ("TEST B  Nepal (was crashing)",
     "Check flood/water expansion changes in Nepal between 2021 and 2025 using Sentinel-2 imagery.",
     "result"),
    ("EDGE    Kathmandu, Nepal",
     "Check flood/water expansion changes in Kathmandu, Nepal between 2021 and 2025.",
     "any"),
    ("EDGE    Terai, Nepal",
     "Check flood/water expansion changes in Terai, Nepal between 2021 and 2025.",
     "any"),
    ("EDGE    Bangladesh",
     "Check flood/water expansion changes in Bangladesh between 2021 and 2025.",
     "any"),
    ("EDGE    no imagery (Antarctica, short window)",
     "Show water expansion in Antarctica between 2021 and 2022.",
     "any"),
    ("EDGE    invalid location",
     "Show water expansion in Zzzqqxx Nowhereland between 2021 and 2025.",
     "graceful_error"),
    ("EDGE    no location given",
     "Show me water expansion",
     "graceful_error"),
]


def server_pid() -> str | None:
    r = subprocess.run(["pgrep", "-f", "app.main:app"], capture_output=True, text=True)
    pids = [p for p in r.stdout.split() if p.strip()]
    return pids[0] if pids else None


def rss_mb(pid: str) -> float:
    try:
        for line in open(f"/proc/{pid}/status"):
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1024
    except Exception:
        pass
    return -1.0


def run(query: str, grid: int = 480, timeout: int = 400) -> dict:
    t0 = time.time()
    p = subprocess.run(
        ["curl", "-sS", "-N", "-X", "POST", f"{BASE}/api/analyze",
         "-H", "Content-Type: application/json",
         "-d", json.dumps({"query": query, "grid": grid})],
        capture_output=True, text=True, timeout=timeout,
    )
    acc: dict = {}
    err = None
    frames = 0
    for line in p.stdout.splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        frames += 1
        acc.update(d)
        if d.get("status") == "error":
            err = d
    acc["_elapsed"] = time.time() - t0
    acc["_frames"] = frames
    acc["_error"] = err
    acc["_truncated"] = "transfer closed" in (p.stderr or "")
    return acc


def main() -> int:
    pid = server_pid()
    if not pid:
        print("SERVER NOT RUNNING")
        return 2
    print(f"server pid {pid} | baseline RSS {rss_mb(pid):.0f} MB")
    print("=" * 92)

    failures = 0
    for label, query, expect in CASES:
        if not server_pid():
            print(f"{label:44s} >>> SERVER DEAD BEFORE THIS TEST <<<")
            failures += 1
            break
        r = run(query)
        alive = server_pid() is not None
        rss = rss_mb(pid) if alive else -1

        if not alive or r["_truncated"]:
            status, detail = "CRASH", "server died / stream truncated"
            failures += 1
        elif r["_error"]:
            msg = str(r["_error"].get("message", ""))[:58]
            if expect == "result":
                status, detail = "FAIL", f"expected result, got error: {msg}"
                failures += 1
            else:
                status, detail = "OK(err)", f"graceful: {msg}"
        elif r.get("narrative"):
            s = r.get("stats", {})
            detail = (f"{s.get('changed_area_ha', 0):.1f} ha | "
                      f"{s.get('cluster_count', 0)} clusters | "
                      f"conf {r.get('confidence', 0):.2f} | "
                      f"{s.get('epoch_count', 0)} epochs")
            if expect == "graceful_error":
                status = "OK(res)"
            else:
                status = "PASS"
        else:
            status, detail = "FAIL", "no narrative and no error frame"
            failures += 1

        print(f"{label:44s} {status:8s} {r['_elapsed']:5.1f}s RSS {rss:4.0f}MB  {detail}")

    print("=" * 92)
    alive = server_pid() is not None
    print(f"server {'ALIVE' if alive else 'DEAD'} | final RSS {rss_mb(pid):.0f} MB | failures: {failures}")
    return 0 if failures == 0 and alive else 1


if __name__ == "__main__":
    sys.exit(main())
