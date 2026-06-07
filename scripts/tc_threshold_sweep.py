#!/usr/bin/env python3
"""tc_threshold_sweep.py

Sweep TC path-intensity threshold on both synthetic and ZInD datasets.
Saves results to results/tc_threshold_sweep/{synthetic,zind}.npy —
each a dict {threshold: {ds_name: accuracy_%}}.

Usage (from project root):
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/tc_threshold_sweep.py
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

THRESHOLDS = [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80]

CONFIGS = {
    "synthetic": {
        "model":    "outputs/training_default/training-homeless-Sok/model_best.pth",
        "datasets": {
            "5-term (val)":  "data/5_green/maze_data_test_21",
            "6-term (test)": "data/6_green/maze_data_test_21",
            "7-term (test)": "data/7_green/maze_data_test_21",
        },
        "tc_first": 20,
        "tc_batch": 10,
        "max_rounds": 20,
    },
    "zind": {
        "model":    "outputs/training_default/training-unlost-Tamar/model_best.pth",
        "datasets": {
            "5-term (val)":  "data/zind/big_floorplans/5_green/maze_data_test_21",
            "6-term (test)": "data/zind/big_floorplans/6_green/maze_data_test_21",
            "7-term (test)": "data/zind/big_floorplans/7_green/maze_data_test_21",
        },
        "tc_first": 10,
        "tc_batch": 10,
        "max_rounds": 13,
    },
}

OUT_DIR = Path("results/tc_threshold_sweep")

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


def tc_correct_multi(net, inp_np, sol_np, thresholds, tc_first, tc_batch, max_rounds, device):
    inp      = torch.from_numpy(inp_np.astype(np.float32)).unsqueeze(0).to(device)
    n_gt     = float(torch.from_numpy(sol_np.astype(np.float32)).sum())
    results  = {thr: None for thr in thresholds}
    interims = None
    first    = True

    for rnd in range(max_rounds + 1):
        n_iters = tc_first if first else tc_batch
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
            elif rnd == max_rounds:
                results[thr] = False

        if all(v is not None for v in results.values()):
            break

    return {thr: (v if v is not None else False) for thr, v in results.items()}


# ── Main ─────────────────────────────────────────────────────────────────────

def run_config(name, cfg, device):
    print(f"\n{'='*50}")
    print(f"  {name.upper()}")
    print(f"{'='*50}")
    net = load_model(cfg["model"], device)
    print(f"  Model: {cfg['model']}")

    results = {}   # {ds_name: {thr: accuracy_%}}

    for ds_name, ds_path in cfg["datasets"].items():
        inputs    = np.load(Path(ds_path) / "inputs.npy")
        solutions = np.load(Path(ds_path) / "solutions.npy")
        N         = len(inputs)
        print(f"\n── {ds_name} ({N} samples) ──")
        t0 = time.time()

        correct = {thr: 0 for thr in THRESHOLDS}
        for i in range(N):
            res = tc_correct_multi(
                net, inputs[i], solutions[i], THRESHOLDS,
                cfg["tc_first"], cfg["tc_batch"], cfg["max_rounds"], device
            )
            for thr, ok in res.items():
                if ok:
                    correct[thr] += 1
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{N}  [{time.time()-t0:.0f}s]", flush=True)

        print(f"  Done [{time.time()-t0:.0f}s]")
        print(f"  {'Threshold':>10}  {'Accuracy':>10}")
        print(f"  {'-'*23}")
        results[ds_name] = {}
        for thr in THRESHOLDS:
            acc = 100.0 * correct[thr] / N
            results[ds_name][thr] = round(acc, 1)
            print(f"  {thr:>10.2f}  {acc:>9.1f}%")

    out_path = OUT_DIR / f"{name}.npy"
    np.save(out_path, results)
    print(f"\n  Saved → {out_path}")
    return results


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, cfg in CONFIGS.items():
        run_config(name, cfg, device)


if __name__ == "__main__":
    main()
