#!/usr/bin/env python3
"""figure8_test_iterations.py

Figure 8: accuracy vs. iteration count and per-sample runtime for synthetic mazes,
using MazeNet with the Termination Condition (TC).

For each terminal count in [2..8]:

  Accuracy pass (acc_eval, only plotted for 5/6/7):
    Runs TC with max_iters=20 (200 individual iterations tracked).
    Saves: results/figure8_test_iterations/TC_accuracy_{n}G_rerun.npy

  Runtime pass (time_eval):
    Runs TC with max_iters=4 (40 total iterations = 4 rounds of 10).
    Saves: results/figure8_test_iterations/TC_{n}_G_40_iters_rerun.npy

After testing, generates two plots:
  Plot 1 — Accuracy vs. iterations (5G, 6G, 7G), x=0..140, y=0..100%
  Plot 2 — Log runtime vs. terminals (2..8): MazeNet + approx methods + Dijkstra

Approximation timings (Wavefront MST, Mehlhorn, Kou) are computed once and
cached in results/figure8_test_iterations/approx_timings/.
Dijkstra timings are loaded from results/durations_Dijkstras/ (too expensive to rerun).

Model:  outputs/training_default/training-homeless-Sok/model_best.pth
        (dt_net_2d_parallel, width=128, max_iters=30)

TC:     tc_threshold=0.65, tc_first_batch=10  (this experiment)

Usage (run from project root):
    CUDA_VISIBLE_DEVICES=2 python scripts/figure8_test_iterations.py
"""

import os
import sys
import time
from types import SimpleNamespace

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from networkx.algorithms.approximation.steinertree import steiner_tree
from torch.utils import data
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import deepthinking as dt
from deepthinking.utils.testing import apply_mask, check_termination_multiple

# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_PATH      = os.path.join("outputs", "training_default",
                                "training-homeless-Sok", "model_best.pth")
DATA_ROOT       = "data"
RESULT_DIR      = os.path.join("results", "figure8_test_iterations")
APPROX_CACHE    = os.path.join(RESULT_DIR, "approx_timings")
DIJ_DUR_DIR     = os.path.join("results", "durations_Dijkstras")

TERMINALS       = list(range(2, 9))    # run MazeNet TC for all 2–8
TERMINALS_ACC   = [5, 6, 7]           # shown in accuracy plot
TC_THRESHOLD    = 0.65
TC_FIRST_BATCH  = 10                   # first-batch size for this experiment
TEST_SIZE       = 21
N_APPROX_SAMPLE = 1000                 # mazes sampled per terminal for approx timing

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(APPROX_CACHE, exist_ok=True)


# ── Model and data helpers ────────────────────────────────────────────────────

def load_model(device):
    model_args = SimpleNamespace(
        model="dt_net_2d_parallel",
        model_path=MODEL_PATH,
        width=128,
        max_iters=30,
    )
    net, _, _ = dt.utils.load_model_from_checkpoint(
        "mazes", model_args, device, testing=True
    )
    return net


def get_testloader(n_terminals):
    from easy_to_hard_data import MazeDataset
    data_dir = os.path.join(DATA_ROOT, f"{n_terminals}_green")
    testset  = MazeDataset(data_dir, train=False, size=TEST_SIZE, download=False)
    return data.DataLoader(testset, batch_size=1, shuffle=False, drop_last=False)


# ── TC test loop ──────────────────────────────────────────────────────────────

def run_tc_test(net, testloader, device, acc_eval=False, time_eval=False):
    """Run TC-gated forward passes and return (accuracy_array, durations).

    acc_eval=True  → max_iters=20, tracks 200 iteration positions.
    time_eval=True → max_iters=4,  per-sample solve time (40 iter cap).
    """
    max_iters = 20 if acc_eval else (4 if time_eval else 14)

    net.eval()
    total                  = 0
    corrects_per_iteration = torch.zeros(max_iters * 10)
    durations              = []

    with torch.no_grad():
        for inputs, targets in tqdm(testloader, leave=False):
            inputs, targets = inputs.to(device), targets.to(device)

            corrects  = torch.zeros(targets.size(0), max_iters * 10)
            interims  = None
            first     = True

            for i in range(max_iters + 1):
                start_time       = time.time()
                iters_this_batch = TC_FIRST_BATCH if first else 10
                all_outputs, interims = net(inputs, iters_this_batch, interims)
                first = False

                single_image                = apply_mask(inputs, all_outputs)
                single_target, finishidx, _ = check_termination_multiple(
                    single_image, TC_THRESHOLD
                )

                if i == max_iters and not finishidx:
                    durations.append(time.time() - start_time)
                    break

                if finishidx:
                    durations.append(time.time() - start_time)
                    num_ones_predicted = single_target.squeeze().sum(dim=[0, 1]).to(device)
                    num_ones           = targets.sum(dim=[1, 2]).to(device)
                    correct            = int(num_ones_predicted <= (num_ones + 8))
                    corrects[:, i:max_iters * 10] = correct
                    break

            for j in range(max_iters * 10):
                corrects_per_iteration[j] += corrects[:, j].sum().item()
            total += targets.size(0)

    accuracy = 100.0 * corrects_per_iteration / total
    return np.array(accuracy), durations


# ── Graph utilities for approximation timing ──────────────────────────────────

def image_into_graph(pixels):
    pixels = pixels.transpose(1, 2, 0)
    n_x, n_y = pixels.shape[0] - 2, pixels.shape[1] - 2
    WHITE, GREEN = [1, 1, 1], [0, 1, 0]
    G, green_cells = nx.Graph(), []
    for i in range(3, n_x, 4):
        for j in range(3, n_y, 4):
            cell = pixels[i, j]
            is_white = (cell == WHITE).all()
            is_green = (cell == GREEN).all()
            if not (is_white or is_green):
                continue
            node_id = (j // 4, i // 4)
            G.add_node(node_id)
            def _add(nb):
                if nb not in G: G.add_node(nb)
            if i > 3 and (pixels[i-2, j] == WHITE).all():
                nb = (node_id[0], node_id[1]-1); _add(nb); G.add_edge(node_id, nb)
            if j < n_y-2 and (pixels[i, j+2] == WHITE).all():
                nb = (node_id[0]+1, node_id[1]); _add(nb); G.add_edge(node_id, nb)
            if i < n_x-2 and (pixels[i+2, j] == WHITE).all():
                nb = (node_id[0], node_id[1]+1); _add(nb); G.add_edge(node_id, nb)
            if j > 3 and (pixels[i, j-2] == WHITE).all():
                nb = (node_id[0]-1, node_id[1]); _add(nb); G.add_edge(node_id, nb)
            if is_green:
                green_cells.append(node_id)
    return G, green_cells


def _time_wavefront(G, gc):
    t0 = time.time()
    complete = nx.Graph()
    for i in range(len(gc)):
        for j in range(i+1, len(gc)):
            l, path = nx.single_source_dijkstra(G, gc[i], gc[j])
            complete.add_edge(gc[i], gc[j], weight=l, path=path)
    mst = nx.minimum_spanning_tree(complete, weight='weight', algorithm='prim')
    unique = set()
    for _, _, d in mst.edges(data=True): unique.update(d['path'])
    return time.time() - t0


def _time_steiner(G, gc, method):
    t0 = time.time()
    steiner_tree(G, gc, method=method)
    return time.time() - t0


def compute_approx_timings():
    """Compute or load cached mean per-maze timing for Wavefront, Mehlhorn, Kou (2–8 G)."""
    methods = ["wavefront", "mehlhorn", "kou"]
    means   = {m: [] for m in methods}

    for n in TERMINALS:
        cache = {m: os.path.join(APPROX_CACHE, f"{m}_{n}G.npy") for m in methods}
        missing = [m for m in methods if not os.path.exists(cache[m])]

        if missing:
            print(f"  [{n}G] computing approx timings for: {missing} ...")
            inputs = np.load(os.path.join(DATA_ROOT, f"{n}_green",
                                          "maze_data_test_21", "inputs.npy"))
            idx = np.random.default_rng(42).choice(
                len(inputs), size=min(N_APPROX_SAMPLE, len(inputs)), replace=False
            )
            wave_t, meh_t, kou_t = [], [], []
            for i in tqdm(idx, desc=f"  {n}G", leave=False):
                G, gc = image_into_graph(inputs[i])
                if not nx.is_connected(G):
                    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
                try:
                    if "wavefront" in missing: wave_t.append(_time_wavefront(G, gc))
                    if "mehlhorn"  in missing: meh_t.append(_time_steiner(G, gc, "mehlhorn"))
                    if "kou"       in missing: kou_t.append(_time_steiner(G, gc, "kou"))
                except Exception:
                    continue
            for m, arr in zip(methods, [wave_t, meh_t, kou_t]):
                if m in missing and arr:
                    np.save(cache[m], np.array(arr))

        for m in methods:
            means[m].append(float(np.mean(np.load(cache[m])))
                            if os.path.exists(cache[m]) else float("nan"))

    return means   # each value is a list of 7 means (one per terminal 2–8)


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_accuracy():
    """Plot 1: cumulative accuracy vs. iterations for 5G, 6G, 7G."""
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {5: "steelblue", 6: "darkorange", 7: "forestgreen"}

    for n in TERMINALS_ACC:
        path = os.path.join(RESULT_DIR, f"TC_accuracy_{n}G_rerun.npy")
        if not os.path.exists(path):
            print(f"  [WARNING] Missing {path} — skipping {n}G in accuracy plot.")
            continue
        acc  = np.load(path).astype(float)
        vals = np.concatenate([[0.0], acc[:14]])   # prepend 0 at x=0
        x    = np.arange(0, 141, 10)               # 0, 10, ..., 140
        ax.plot(x, vals, marker="o", markersize=4,
                color=colors[n], label=f"{n} terminals")

    ax.set_xlim(0, 140)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Iterations", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("MazeNet (TC): Accuracy vs. Iterations — Synthetic Mazes", fontsize=12)
    ax.set_xticks(range(0, 141, 20))
    ax.set_yticks(range(0, 101, 10))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    out = os.path.join(RESULT_DIR, "figure8_accuracy.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


def plot_time(approx_means):
    """Plot 2: log runtime vs. terminals (2–8) for MazeNet, approx methods, Dijkstra."""
    x = np.array(TERMINALS)

    # MazeNet — freshly computed rerun durations
    net_means = []
    for n in TERMINALS:
        p = os.path.join(RESULT_DIR, f"TC_{n}_G_40_iters_rerun.npy")
        net_means.append(float(np.mean(np.load(p))) if os.path.exists(p) else float("nan"))

    # Dijkstra — load pre-computed (too expensive to rerun)
    dij_means = []
    for n in TERMINALS:
        p = os.path.join(DIJ_DUR_DIR, f"{n}_G_21.npy")
        dij_means.append(float(np.mean(np.load(p))) if os.path.exists(p) else float("nan"))

    fig, ax = plt.subplots(figsize=(7, 5))
    kw = dict(linestyle="-", marker="v", markersize=6)

    ax.semilogy(x, net_means,                 label="MazeNet",             color="blue",    **kw)
    ax.semilogy(x, approx_means["wavefront"], label="Wavefront MST",       color="green",   **kw)
    ax.semilogy(x, approx_means["mehlhorn"],  label="Mehlhorn",            color="magenta", **kw)
    ax.semilogy(x, approx_means["kou"],       label="Kou",                 color="orange",  **kw)
    ax.semilogy(x, dij_means,                 label="Dijkstra exhaustive", color="red",     **kw)

    ax.set_xlim(1.5, 8.5)
    ax.set_ylim(1e-4, 1e2)
    ax.set_xticks(TERMINALS)
    ax.set_xlabel("Number of Terminals", fontsize=12)
    ax.set_ylabel("Time per maze (s, log scale)", fontsize=12)
    ax.set_title("Runtime vs. Number of Terminals — Synthetic Mazes", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)

    out = os.path.join(RESULT_DIR, "figure8_time.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print("=" * 72)
    print("  Figure 8: test_iterations — synthetic mazes, terminals 2–8")
    print("=" * 72)

    net = load_model(device)

    for n in TERMINALS:
        print(f"\n  [{n}_green]")
        testloader = get_testloader(n)

        # ── Accuracy pass (run for all, plot only 5/6/7) ──────────────────
        print(f"    Accuracy pass (max_iters=20 → 200 iterations)...")
        t0 = time.time()
        acc_array, _ = run_tc_test(net, testloader, device, acc_eval=True)
        print(f"    Done in {time.time()-t0:.1f}s  |  final: {acc_array[-1]:.2f}%")
        out_acc = os.path.join(RESULT_DIR, f"TC_accuracy_{n}G_rerun.npy")
        np.save(out_acc, acc_array)
        print(f"    Saved → {out_acc}")

        # ── Runtime pass ──────────────────────────────────────────────────
        print(f"    Runtime pass (max_iters=4 → 40 iterations)...")
        t0 = time.time()
        _, durations = run_tc_test(net, testloader, device, time_eval=True)
        dur_arr = np.array(durations)
        print(f"    Done in {time.time()-t0:.1f}s  |  mean: {dur_arr.mean()*1000:.2f}ms")
        out_dur = os.path.join(RESULT_DIR, f"TC_{n}_G_40_iters_rerun.npy")
        np.save(out_dur, dur_arr)
        print(f"    Saved → {out_dur}")

    print(f"\n{'=' * 72}")
    print(f"  All MazeNet results saved to {RESULT_DIR}/")
    print(f"{'=' * 72}")

    # ── Approximation timings (compute once, cache) ───────────────────────────
    print("  Computing / loading approximation timings...")
    approx_means = compute_approx_timings()

    # ── Generate plots ────────────────────────────────────────────────────────
    print("\n  Generating plots...")
    plot_accuracy()
    plot_time(approx_means)

    print(f"\n{'=' * 72}")
    print(f"  Plots saved to {RESULT_DIR}/")
    print(f"{'=' * 72}\n")


if __name__ == "__main__":
    main()
