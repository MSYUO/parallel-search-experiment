"""
calibrate.py — Find suitable DIFFICULTY for the experiment.
Target: N=1, k=10 finishes in 30s~120s.
Usage: python calibrate.py
"""

import hashlib
import time
import os
import sys

K_TARGET    = 10
WORKER_ID   = 1
TRIAL       = 1
DIFFICULTIES = [4, 5, 6]

TARGET_LOW  = 30    # seconds
TARGET_HIGH = 120   # seconds


def run_once(difficulty: int) -> tuple[float, int]:
    """Return (elapsed_seconds, total_hashes)."""
    seed   = f"worker_{WORKER_ID}_{TRIAL}"
    prefix = "0" * difficulty

    found  = 0
    nonce  = 0
    start  = time.perf_counter()

    while found < K_TARGET:
        digest = hashlib.sha256(f"{seed}:{nonce}".encode()).hexdigest()
        if digest.startswith(prefix):
            found += 1
        nonce += 1

    elapsed = time.perf_counter() - start
    return elapsed, nonce


def fmt(n: int) -> str:
    return f"{n:,}"


def main():
    print("=" * 62)
    print(f"  Calibration: DIFFICULTY scan (k={K_TARGET}, N=1, trial=1)")
    print(f"  Target range: {TARGET_LOW}s ~ {TARGET_HIGH}s")
    print("=" * 62)

    results = []   # list of (diff, elapsed, hps, tag)

    for diff in DIFFICULTIES:
        print(f"\nDIFFICULTY={diff}  running...", end="", flush=True)

        elapsed, total_hashes = run_once(diff)
        hps = int(total_hashes / elapsed)

        if elapsed < TARGET_LOW:
            tag = "too fast"
        elif elapsed <= TARGET_HIGH:
            tag = "GOOD <<<"
        else:
            tag = "too slow"

        results.append((diff, elapsed, hps, tag))
        print(f"\rDIFFICULTY={diff} | k={K_TARGET} 찾는데 {elapsed:,.1f}초"
              f" | 초당 해시: {fmt(hps)}"
              f" | [{tag}]")

        # stop early if way too slow (> 5 min)
        if elapsed > 300:
            print(f"  (DIFFICULTY={diff+1}+ would take ~{elapsed*16:.0f}s -- skipping)")
            break

    # pick best: GOOD first, else closest to target midpoint
    target_mid = (TARGET_LOW + TARGET_HIGH) / 2
    good = [r for r in results if r[3] == "GOOD <<<"]
    if good:
        recommendation = good[0][0]
    else:
        recommendation = min(results, key=lambda r: abs(r[1] - target_mid))[0]

    rec = next(r for r in results if r[0] == recommendation)

    print()
    print("=" * 62)
    print(f"  Recommended DIFFICULTY = {recommendation}  ({rec[1]:.1f}s per trial)")
    if rec[3] != "GOOD <<<":
        if rec[1] < TARGET_LOW:
            print(f"  Note: slightly fast ({rec[1]:.1f}s < {TARGET_LOW}s target).")
            print(f"        Try increasing k or use DIFFICULTY={recommendation+1}.")
        else:
            print(f"  Note: slightly slow ({rec[1]:.1f}s > {TARGET_HIGH}s target).")
            print(f"        Acceptable for experiment (N>=2 will be faster).")
    print(f"  Run experiment with:")
    print(f"    bash run_experiment.sh --difficulty {recommendation} --k {K_TARGET}")
    print("=" * 62)


if __name__ == "__main__":
    main()
