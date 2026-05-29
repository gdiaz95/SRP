#!/usr/bin/env python3
"""table1_accuracy.py

Reproduce Table 1: synthetic maze test accuracy (%) with and without TC,
for 2_green through 8_green at 140 iterations (homeless-Sok model).

Usage:
    CUDA_VISIBLE_DEVICES=0 python table1_accuracy.py
"""

import os
import re
import subprocess
import sys
import time

DATA_TYPES = ["2_green", "3_green", "4_green", "5_green", "6_green", "7_green", "8_green"]


def run_test(data_type, use_tc, iterations=140):
    """Run test_model.py and return accuracy (%).
    With TC:    termination_condition=True,  iterations 20-140, returns -1 key.
    Without TC: termination_condition=False, iterations fixed at 140, returns 140 key.
    """
    env = os.environ.copy()

    cmd = [
        sys.executable, "test_model.py",
        f"problem.data_type={data_type}",
        f"termination_condition={'true' if use_tc else 'false'}",
        f"problem.model.test_iterations.low={20 if use_tc else iterations}",
        f"problem.model.test_iterations.high={iterations}",
    ]

    t0 = time.time()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        env=env,
        bufsize=1,
    )

    accuracy = None
    for raw in proc.stdout:
        line = raw.rstrip()
        if 'Testing accuracy' in line:
            if use_tc:
                m = re.search(r'-1:\s*([0-9.]+)', line)
            else:
                m = re.search(rf'{iterations}:\s*([0-9.]+)', line)
            if m:
                accuracy = float(m.group(1))

    proc.wait()
    elapsed = time.time() - t0
    label = "TC" if use_tc else f"no TC ({iterations} iters)"
    print(f"    {data_type:8s}  {label:20s}  →  {accuracy:.2f}%  ({elapsed:.1f}s)")
    return accuracy


def main():
    print("=" * 65)
    print("  Table 1: Synthetic maze accuracy — with vs without TC")
    print("  Model: homeless-Sok  |  TC iterations: 20-140  |  No-TC cap: 59 (val-selected)  |  threshold: 0.65")
    print("=" * 65)

    results_tc    = {}
    results_no_tc = {}

    # NO_TC_ITERS = 59: peak no-TC accuracy on 5-terminal validation set
    # observed in the training log for the selected model (homeless-Sok epoch 16).
    # This is the validation-selected iteration cap for the no-TC baseline.
    NO_TC_ITERS = 59

    for data_type in DATA_TYPES:
        print(f"\n  [{data_type}]")
        results_tc[data_type]    = run_test(data_type, use_tc=True,  iterations=140)
        results_no_tc[data_type] = run_test(data_type, use_tc=False, iterations=NO_TC_ITERS)

    # ── Print table ───────────────────────────────────────────────────────────
    splits = [dt.split('_')[0] for dt in DATA_TYPES]

    print(f"\n{'=' * 65}")
    print("  Table 1 Results")
    print(f"{'=' * 65}")
    print(f"  {'':15s}" + "".join(f"{s:>8s}" for s in splits))
    print(f"  {'TC module':15s}" + "".join(
        f"{results_tc.get(dt, float('nan')):8.2f}" for dt in DATA_TYPES))
    print(f"  {'No TC module':15s}" + "".join(
        f"{results_no_tc.get(dt, float('nan')):8.2f}" for dt in DATA_TYPES))
    print(f"{'=' * 65}\n")


if __name__ == "__main__":
    main()
