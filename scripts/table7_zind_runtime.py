#!/usr/bin/env python3
"""table7_zind_runtime.py

Table 7: runtime comparison on ZInD (big floorplan) test sets for 5, 6, 7 terminals.

Methods:
  MazeNet   — rerun TC with time_eval (max_iters=4, 40-iter cap)
  Wavefront — computed and cached (1 000-sample subsample)
  Mehlhorn  — computed and cached
  Kou       — computed and cached
  Dijkstra  — loaded from results/ratios_calculation_rooms/times.npy (too expensive)

Model:  outputs/training_default/training-unmeet-Kearstin/model_best.pth
        (dt_net_2d_parallel, width=128, max_iters=30)
TC:     tc_threshold=0.50, tc_first_batch=10  (ZInD defaults)

Results saved to results/table7_zind_runtime/

Usage (run from project root):
    CUDA_VISIBLE_DEVICES=2 python scripts/table7_zind_runtime.py
"""

import os
import sys
import time
from types import SimpleNamespace

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
                                "training-unmeet-Kearstin", "model_best.pth")
DATA_ROOT       = os.path.join("data", "zind", "big_floorplans")
RESULT_DIR      = os.path.join("results", "table7_zind_runtime")
APPROX_CACHE    = os.path.join(RESULT_DIR, "approx_timings")
DIJKSTRA_TIMES  = os.path.join("results", "ratios_calculation_rooms", "times.npy")
DIJKSTRA_ARRAYS = os.path.join("results", "ratios_calculation_rooms", "all_array_except_NET_{n}.npy")

TERMINALS       = [5, 6, 7]
TC_THRESHOLD    = 0.50    # ZInD threshold
TC_FIRST_BATCH  = 10      # ZInD first-batch size
TEST_SIZE       = 21
N_APPROX_SAMPLE = 1000    # mazes sampled per terminal for approx timing

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


# ── TC runtime test ───────────────────────────────────────────────────────────

def run_tc_runtime(net, testloader, device):
    """TC with time_eval: max_iters=4 → 40-iteration cap. Returns per-sample durations."""
    max_iters = 4

    net.eval()
    durations = []

    with torch.no_grad():
        for inputs, targets in tqdm(testloader, leave=False):
            inputs, targets = inputs.to(device), targets.to(device)

            interims = None
            first    = True

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
                    break

    return np.array(durations)


# ── Graph utilities for approximation timing ──────────────────────────────────

def image_into_graph(pixels):
    pixels = pixels.transpose(1, 2, 0)
    n_x, n_y = pixels.shape[0] - 2, pixels.shape[1] - 2
    WHITE, GREEN = [1, 1, 1], [0, 1, 0]
    G, green_cells = nx.Graph(), []
    for i in range(3, n_x, 4):
        for j in range(3, n_y, 4):
            cell     = pixels[i, j]
            is_white = (cell == WHITE).all()
            is_green = (cell == GREEN).all()
            if not (is_white or is_green):
                continue
            node_id = (j // 4, i // 4)
            G.add_node(node_id)
            def _add(nb):
                if nb not in G: G.add_node(nb)
            if i > 3     and (pixels[i-2, j] == WHITE).all():
                nb = (node_id[0], node_id[1]-1); _add(nb); G.add_edge(node_id, nb)
            if j < n_y-2 and (pixels[i, j+2] == WHITE).all():
                nb = (node_id[0]+1, node_id[1]); _add(nb); G.add_edge(node_id, nb)
            if i < n_x-2 and (pixels[i+2, j] == WHITE).all():
                nb = (node_id[0], node_id[1]+1); _add(nb); G.add_edge(node_id, nb)
            if j > 3     and (pixels[i, j-2] == WHITE).all():
                nb = (node_id[0]-1, node_id[1]); _add(nb); G.add_edge(node_id, nb)
            if is_green:
                green_cells.append(node_id)
    return G, green_cells


def _time_wavefront(G, gc):
    t0 = time.time()
    complete = nx.Graph()
    for i in range(len(gc)):
        for j in range(i + 1, len(gc)):
            length, path = nx.single_source_dijkstra(G, gc[i], gc[j])
            complete.add_edge(gc[i], gc[j], weight=length, path=path)
    mst = nx.minimum_spanning_tree(complete, weight='weight', algorithm='prim')
    unique = set()
    for _, _, d in mst.edges(data=True): unique.update(d['path'])
    return time.time() - t0


def _time_steiner(G, gc, method):
    t0 = time.time()
    steiner_tree(G, gc, method=method)
    return time.time() - t0


def compute_approx_timings():
    """Compute or load cached mean per-maze timing (s) for approx methods on ZInD."""
    methods = ["wavefront", "mehlhorn", "kou"]
    means   = {m: {} for m in methods}

    for n in TERMINALS:
        cache   = {m: os.path.join(APPROX_CACHE, f"{m}_{n}G.npy") for m in methods}
        missing = [m for m in methods if not os.path.exists(cache[m])]

        if missing:
            print(f"  [{n}G] computing approx timings for: {missing} ...")
            inputs = np.load(os.path.join(DATA_ROOT, f"{n}_green",
                                          "maze_data_test_21", "inputs.npy"))
            idx = np.random.default_rng(42).choice(
                len(inputs), size=min(N_APPROX_SAMPLE, len(inputs)), replace=False
            )
            wave_t, meh_t, kou_t = [], [], []
            for i in tqdm(idx, desc=f"  {n}G approx", leave=False):
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
            if os.path.exists(cache[m]):
                arr = np.load(cache[m])
                means[m][n] = (float(np.mean(arr)), float(np.std(arr)))
            else:
                means[m][n] = float("nan")

    return means


# ── Table printer ─────────────────────────────────────────────────────────────

def print_table(title, rows, terminals):
    """rows: list of (label, {n: mean_s | (mean_s, std_s)}) pairs."""
    w = 20
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")
    print(f"  {'Method':22s}" + "".join(f"{f'{n} terminals':>{w}s}" for n in terminals))
    print(f"  {'-'*22}" + "-" * (w * len(terminals)))
    for label, values in rows:
        row = f"  {label:22s}"
        for n in terminals:
            v = values.get(n, float("nan"))
            if isinstance(v, tuple):
                mean_ms, std_ms = v[0] * 1000, v[1] * 1000
                row += f"{f'{mean_ms:.2f}±{std_ms:.2f}':>{w}s}"
            elif not np.isnan(v):
                row += f"{v*1000:>{w}.2f}"
            else:
                row += f"{'N/A':>{w}s}"
        print(row)
    print(f"  {'(values in ms)':22s}")
    print(f"{'=' * 80}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print("=" * 72)
    print("  Table 7: ZInD runtime — MazeNet vs. approximation methods (5/6/7G)")
    print("=" * 72)

    # ── MazeNet TC runtime ────────────────────────────────────────────────────
    net = load_model(device)
    net_dur = {}

    for n in TERMINALS:
        print(f"\n  [{n}G] MazeNet TC runtime (max_iters=4 → 40 iterations)...")
        t0 = time.time()
        testloader = get_testloader(n)
        dur_arr    = run_tc_runtime(net, testloader, device)
        elapsed    = time.time() - t0
        net_dur[n] = dur_arr
        print(f"    Done in {elapsed:.1f}s  |  n={len(dur_arr)}  "
              f"mean={dur_arr.mean()*1000:.2f}ms  std={dur_arr.std()*1000:.2f}ms")
        out = os.path.join(RESULT_DIR, f"TC_{n}_G_40_iters_rerun.npy")
        np.save(out, dur_arr)
        print(f"    Saved → {out}")

    # ── Approximation timings ─────────────────────────────────────────────────
    print(f"\n  Computing / loading approximation timings (n_sample={N_APPROX_SAMPLE})...")
    approx = compute_approx_timings()

    # ── Load Dijkstra times (mean from times.npy, std from per-sample arrays) ──
    dij_dur = {}
    for n in TERMINALS:
        arr_path = DIJKSTRA_ARRAYS.format(n=n)
        if os.path.exists(arr_path):
            arr = np.load(arr_path)          # shape (N, 4, 2): col0=exhaustive
            times_s = arr[:, 0, 1]
            dij_dur[n] = (float(np.mean(times_s)), float(np.std(times_s)))
        elif os.path.exists(DIJKSTRA_TIMES):
            dij_times = np.load(DIJKSTRA_TIMES, allow_pickle=True).item()
            if n in dij_times and "exhaustive" in dij_times[n]:
                dij_dur[n] = dij_times[n]["exhaustive"]  # scalar fallback
    if dij_dur:
        print(f"\n  MT-Dijkstra times loaded (mean±std from per-sample arrays)")
    else:
        print(f"\n  [WARNING] MT-Dijkstra timing files not found — row will be empty.")

    # ── Build rerun table ─────────────────────────────────────────────────────
    rows = [
        ("MazeNet (TC)",        {n: (float(np.mean(net_dur[n])), float(np.std(net_dur[n]))) for n in TERMINALS}),
        ("Wavefront MST",       approx["wavefront"]),
        ("Mehlhorn",            approx["mehlhorn"]),
        ("Kou",                 approx["kou"]),
        ("MT-Dijkstra",         dij_dur),
    ]
    print_table("Table 7 — ZInD Runtime", rows, TERMINALS)

    print(f"  All results saved to {RESULT_DIR}/\n")


if __name__ == "__main__":
    main()
