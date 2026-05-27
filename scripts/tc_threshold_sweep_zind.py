#!/usr/bin/env python3
"""tc_threshold_sweep_zind.py

Sweep TC path-intensity threshold on ZInD floor plan datasets:
  5-terminal → validation set (threshold selection)
  6-terminal, 7-terminal → test sets (clean evaluation)

Usage (from project root):
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/tc_threshold_sweep_zind.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepthinking.utils.testing import apply_mask, check_termination_multiple
from deepthinking.models.dt_net_2d_parallel import DTNet, BasicBlock

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_PATH = "outputs/training_default/training-unlost-Tamar/model_best.pth"

DATASETS = {
    "5-term (val)":  "data/zind/big_floorplans/5_green/maze_data_test_21",
    "6-term (test)": "data/zind/big_floorplans/6_green/maze_data_test_21",
    "7-term (test)": "data/zind/big_floorplans/7_green/maze_data_test_21",
}

THRESHOLDS = [round(v, 2) for v in np.arange(0.30, 0.85, 0.05)]

TC_FIRST_BATCH = 10   # floor plans: uniform batch size of 10
TC_BATCH       = 10
MAX_ROUNDS     = 13   # 14 rounds × 10 iters = 140 total, matching test_zind_config high:140

# ── Model ─────────────────────────────────────────────────────────────────────

def load_model(ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    net_state = state.get("net", state)
    clean = {k.replace("module.", ""): v for k, v in net_state.items()}
    proj_key = next((k for k in clean if "projection" in k and "weight" in k), None)
    width = clean[proj_key].shape[0] if proj_key else 400
    net = DTNet(BasicBlock, [2], width=width, in_channels=3, recall=True)
    net.load_state_dict(clean, strict=True)
    return net.to(device).eval()


def tc_correct_multi(net, inp_np, sol_np, thresholds, device):
    inp   = torch.from_numpy(inp_np.astype(np.float32)).unsqueeze(0).to(device)
    n_gt  = float(torch.from_numpy(sol_np.astype(np.float32)).sum())

    results  = {thr: None for thr in thresholds}
    interims = None
    first    = True

    for rnd in range(MAX_ROUNDS + 1):
        n_iters = TC_FIRST_BATCH if first else TC_BATCH
        all_out, interims = net(inp, n_iters, interims)
        first = False
        single_image = apply_mask(inp, all_out)

        for thr in thresholds:
            if results[thr] is not None:
                continue
            try:
                single_target, finish, _ = check_termination_multiple(single_image, thr)
            except Exception:
                results[thr] = False
                continue
            if finish:
                n_pred = single_target.squeeze().sum().item()
                results[thr] = n_pred <= (n_gt + 8)
            elif rnd == MAX_ROUNDS:
                results[thr] = False

        if all(v is not None for v in results.values()):
            break

    return {thr: (v if v is not None else False) for thr, v in results.items()}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading model: {MODEL_PATH}")
    net = load_model(MODEL_PATH, device)

    datasets = {}
    for name, path in DATASETS.items():
        inp = np.load(Path(path) / "inputs.npy")
        sol = np.load(Path(path) / "solutions.npy")
        datasets[name] = (inp, sol)
        print(f"  {name}: {len(inp)} samples")

    print()

    for ds_name, (inputs, solutions) in datasets.items():
        N = len(inputs)
        print(f"\n── {ds_name} ({N} samples) ──")
        t0 = time.time()

        correct = {thr: 0 for thr in THRESHOLDS}

        for i in range(N):
            res = tc_correct_multi(net, inputs[i], solutions[i], THRESHOLDS, device)
            for thr, ok in res.items():
                if ok:
                    correct[thr] += 1
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{N}  [{time.time()-t0:.0f}s]", flush=True)

        print(f"  Done [{time.time()-t0:.0f}s]")
        print(f"  {'Threshold':>10}  {'Accuracy':>10}")
        print(f"  {'-'*23}")
        for thr in THRESHOLDS:
            acc = 100.0 * correct[thr] / N
            print(f"  {thr:>10.2f}  {acc:>9.1f}%")


if __name__ == "__main__":
    main()
