#!/usr/bin/env python3
"""table2_approx_accuracies.py

Recompute Table 2 graph-approximation accuracies (Wavefront MST, Mehlhorn, Kou)
on the synthetic maze test sets (2_green through 7_green).

For each terminal count:
  - Loads inputs.npy / solutions.npy from data/{n}_green/maze_data_test_21/
  - Parses the maze image into a NetworkX graph
  - Runs Wavefront MST, Mehlhorn, and Kou on each maze
  - Computes accuracy = fraction of mazes where approx length <= optimal length

Results are saved to results/approximation_accuracies/ and a Table 2 summary
is printed at the end.

Usage (run from project root):
    python scripts/table2_approx_accuracies.py
"""

import itertools
import os
import time

import networkx as nx
import numpy as np
from networkx.algorithms.approximation.steinertree import steiner_tree
from tqdm import tqdm

DATA_ROOT  = "data"
RESULT_DIR = os.path.join("results", "approximation_accuracies")
DATA_TYPES = ["2_green", "3_green", "4_green", "5_green", "6_green", "7_green"]

os.makedirs(RESULT_DIR, exist_ok=True)


# ── Graph utilities (ported from Lydia/get_time_ratios_accuracies.ipynb) ──────

def image_into_graph(pixels):
    """Convert a (3, H, W) float image to a NetworkX graph + list of green nodes."""
    pixels = pixels.transpose(1, 2, 0)
    n_x = pixels.shape[0] - 2
    n_y = pixels.shape[1] - 2

    WHITE = [1, 1, 1]
    GREEN = [0, 1, 0]
    G = nx.Graph()
    green_cells = []

    for i in range(3, n_x, 4):
        for j in range(3, n_y, 4):
            cell = pixels[i, j]
            is_white = (cell == WHITE).all()
            is_green = (cell == GREEN).all()
            if not (is_white or is_green):
                continue

            node_id = (j // 4, i // 4)
            G.add_node(node_id, is_green=is_green)

            def _add(nb):
                if nb not in G:
                    G.add_node(nb, is_green=False)

            if i > 3 and (pixels[i - 2, j] == WHITE).all():
                nb = (node_id[0], node_id[1] - 1); _add(nb); G.add_edge(node_id, nb)
            if j < n_y - 2 and (pixels[i, j + 2] == WHITE).all():
                nb = (node_id[0] + 1, node_id[1]); _add(nb); G.add_edge(node_id, nb)
            if i < n_x - 2 and (pixels[i + 2, j] == WHITE).all():
                nb = (node_id[0], node_id[1] + 1); _add(nb); G.add_edge(node_id, nb)
            if j > 3 and (pixels[i, j - 2] == WHITE).all():
                nb = (node_id[0] - 1, node_id[1]); _add(nb); G.add_edge(node_id, nb)

            if is_green:
                green_cells.append(node_id)

    return G, green_cells


def create_best_path_graph(pixels):
    """Convert a (H, W) solution mask to a NetworkX graph (white pixels = path)."""
    n_x = pixels.shape[0] - 2
    n_y = pixels.shape[1] - 2
    WHITE = 1
    G = nx.Graph()

    for i in range(3, n_x, 4):
        for j in range(3, n_y, 4):
            if not (pixels[i, j] == WHITE):
                continue
            node_id = (j // 4, i // 4)
            G.add_node(node_id)

            def _add(nb):
                if nb not in G:
                    G.add_node(nb)

            if i > 3 and pixels[i - 2, j] == WHITE:
                nb = (node_id[0], node_id[1] - 1); _add(nb); G.add_edge(node_id, nb)
            if j < n_y - 2 and pixels[i, j + 2] == WHITE:
                nb = (node_id[0] + 1, node_id[1]); _add(nb); G.add_edge(node_id, nb)
            if i < n_x - 2 and pixels[i + 2, j] == WHITE:
                nb = (node_id[0], node_id[1] + 1); _add(nb); G.add_edge(node_id, nb)
            if j > 3 and pixels[i, j - 2] == WHITE:
                nb = (node_id[0] - 1, node_id[1]); _add(nb); G.add_edge(node_id, nb)

    return G


def get_steiner_length(G, green_cells, method):
    """Return edge count of the Steiner tree approximation."""
    T = steiner_tree(G, green_cells, method=method)
    return T.number_of_edges()


def get_wavefront_length(G, green_cells):
    """Return unique-node count minus 1 for the Wavefront MST solution."""
    complete = nx.Graph()
    for i in range(len(green_cells)):
        for j in range(i + 1, len(green_cells)):
            src, tgt = green_cells[i], green_cells[j]
            length, path = nx.single_source_dijkstra(G, src, tgt)
            complete.add_edge(src, tgt, weight=length, path=path)

    mst = nx.minimum_spanning_tree(complete, weight='weight', algorithm='prim')

    unique = set()
    for _, _, data in mst.edges(data=True):
        unique.update(data['path'])
    return len(unique) - 1


# ── Accuracy helper ────────────────────────────────────────────────────────────

def compute_accuracy(optimal_lengths, approx_lengths):
    """Fraction of samples where approx_length <= optimal_length."""
    opt = np.array(optimal_lengths, dtype=float)
    app = np.array(approx_lengths, dtype=float)
    return float(np.mean(app <= opt)) * 100.0


def compute_ratio_on_mistakes(optimal_lengths, approx_lengths):
    """Mean(approx/opt) conditioned on mistake cases (approx > opt).
    Returns 1.0 if there are no mistakes."""
    opt = np.array(optimal_lengths, dtype=float)
    app = np.array(approx_lengths, dtype=float)
    mask = app > opt
    if not mask.any():
        return 1.0
    return float(np.mean(app[mask] / opt[mask]))


# ── Per-data-type runner ───────────────────────────────────────────────────────

def run_data_type(data_type):
    data_dir = os.path.join(DATA_ROOT, data_type, "maze_data_test_21")
    inputs    = np.load(os.path.join(data_dir, "inputs.npy"))
    solutions = np.load(os.path.join(data_dir, "solutions.npy"))
    n = inputs.shape[0]

    opt_lengths  = []
    wave_lengths = []
    kou_lengths  = []
    meh_lengths  = []

    for i in tqdm(range(n), desc=data_type, leave=False):
        image = inputs[i]
        sol   = solutions[i]

        G, green_cells = image_into_graph(image)
        if not nx.is_connected(G):
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()

        target_graph = create_best_path_graph(sol)
        opt_len = target_graph.number_of_edges()

        try:
            wave_len = get_wavefront_length(G, green_cells)
            kou_len  = get_steiner_length(G, green_cells, 'kou')
            meh_len  = get_steiner_length(G, green_cells, 'mehlhorn')
        except Exception:
            # skip broken samples
            continue

        opt_lengths.append(opt_len)
        wave_lengths.append(wave_len)
        kou_lengths.append(kou_len)
        meh_lengths.append(meh_len)

    acc_wave = compute_accuracy(opt_lengths, wave_lengths)
    acc_kou  = compute_accuracy(opt_lengths, kou_lengths)
    acc_meh  = compute_accuracy(opt_lengths, meh_lengths)

    ratio_wave = compute_ratio_on_mistakes(opt_lengths, wave_lengths)
    ratio_kou  = compute_ratio_on_mistakes(opt_lengths, kou_lengths)
    ratio_meh  = compute_ratio_on_mistakes(opt_lengths, meh_lengths)

    return acc_wave, acc_kou, acc_meh, ratio_wave, ratio_kou, ratio_meh


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("  Table 2: Synthetic maze accuracy — graph approximation algorithms")
    print("=" * 72)

    results = {}

    for data_type in DATA_TYPES:
        print(f"\n  [{data_type}]")
        t0 = time.time()
        acc_wave, acc_kou, acc_meh, ratio_wave, ratio_kou, ratio_meh = run_data_type(data_type)
        elapsed = time.time() - t0
        results[data_type] = (acc_wave, acc_kou, acc_meh, ratio_wave, ratio_kou, ratio_meh)
        print(f"    Wavefront MST : {acc_wave:.2f}%  ratio|mistakes={ratio_wave:.4f}")
        print(f"    Kou           : {acc_kou:.2f}%  ratio|mistakes={ratio_kou:.4f}")
        print(f"    Mehlhorn      : {acc_meh:.2f}%  ratio|mistakes={ratio_meh:.4f}")
        print(f"    ({elapsed:.1f}s)")

    # ── Save results ──────────────────────────────────────────────────────────
    wave_arr = np.array([results[dt][0] for dt in DATA_TYPES])
    kou_arr  = np.array([results[dt][1] for dt in DATA_TYPES])
    meh_arr  = np.array([results[dt][2] for dt in DATA_TYPES])

    wave_ratio_arr = np.array([results[dt][3] for dt in DATA_TYPES])
    kou_ratio_arr  = np.array([results[dt][4] for dt in DATA_TYPES])
    meh_ratio_arr  = np.array([results[dt][5] for dt in DATA_TYPES])

    np.save(os.path.join(RESULT_DIR, "wave_accuracies_rerun.npy"),  wave_arr / 100.0)
    np.save(os.path.join(RESULT_DIR, "kou_accuracies_rerun.npy"),   kou_arr)
    np.save(os.path.join(RESULT_DIR, "meh_accuracies_rerun.npy"),   meh_arr)
    np.save(os.path.join(RESULT_DIR, "wave_ratios_rerun.npy"),      wave_ratio_arr)
    np.save(os.path.join(RESULT_DIR, "kou_ratios_rerun.npy"),       kou_ratio_arr)
    np.save(os.path.join(RESULT_DIR, "meh_ratios_rerun.npy"),       meh_ratio_arr)
    print(f"\n  Saved to {RESULT_DIR}/")

    # ── Print Table 2 ─────────────────────────────────────────────────────────
    splits = [d.split("_")[0] for d in DATA_TYPES]

    print(f"\n{'=' * 72}")
    print("  Table 2: Accuracy (%)")
    print(f"{'=' * 72}")
    print(f"  {'':15s}" + "".join(f"{s:>8s}" for s in splits))
    print(f"  {'MazeNet (TC)':15s}" + "".join(f"{'100.00':>8s}" for _ in DATA_TYPES))
    print(f"  {'Wavefront MST':15s}" + "".join(f"{results[dt][0]:8.2f}" for dt in DATA_TYPES))
    print(f"  {'Mehlhorn':15s}" + "".join(f"{results[dt][2]:8.2f}" for dt in DATA_TYPES))
    print(f"  {'Kou':15s}" + "".join(f"{results[dt][1]:8.2f}" for dt in DATA_TYPES))
    print(f"{'=' * 72}")

    # ── Print Table 3 ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("  Table 3: Norm. edge count conditioned on mistakes")
    print(f"{'=' * 72}")
    print(f"  {'':15s}" + "".join(f"{s:>8s}" for s in splits))
    print(f"  {'MazeNet (TC)':15s}" + "".join(f"{'1.00':>8s}" for _ in DATA_TYPES))
    print(f"  {'Wavefront MST':15s}" + "".join(f"{results[dt][3]:8.2f}" for dt in DATA_TYPES))
    print(f"  {'Mehlhorn':15s}" + "".join(f"{results[dt][5]:8.2f}" for dt in DATA_TYPES))
    print(f"  {'Kou':15s}" + "".join(f"{results[dt][4]:8.2f}" for dt in DATA_TYPES))
    print(f"{'=' * 72}\n")


if __name__ == "__main__":
    main()
