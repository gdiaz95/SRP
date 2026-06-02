#!/usr/bin/env python3
"""compute_maze_difficulty.py

Compute per-maze difficulty statistics for the synthetic maze test sets
(2–8 terminals) and report mean ± std per terminal count.

Statistics per maze:
  1. graph_size      — number of passable nodes
  2. path_length     — number of edges in the optimal solution
  3. path_density    — path_length / graph_size
  4. mean_term_dist  — mean pairwise shortest-path distance between terminals
  5. junctions       — number of degree-≥3 nodes in the maze graph

Results saved to results/maze_difficulty/difficulty_stats.npy

Usage (from project root):
    uv run python scripts/compute_maze_difficulty.py
"""

import os
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
from tqdm import tqdm

DATA_ROOT      = "data"
ZIND_DATA_PATH = "data/zind/big_floorplans/unordered/created/all_created_final.npy"
RESULT_DIR     = Path("results/maze_difficulty")
DATA_TYPES     = ["2_green", "3_green", "4_green", "5_green",
                  "6_green", "7_green", "8_green"]

RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ── Graph parsing (same as table2 / table4) ────────────────────────────────────

def image_into_graph(pixels):
    """(3, H, W) float → NetworkX graph + list of terminal (green) nodes."""
    pixels = pixels.transpose(1, 2, 0)
    n_x, n_y = pixels.shape[0] - 2, pixels.shape[1] - 2
    WHITE, GREEN = [1, 1, 1], [0, 1, 0]
    G, terminals = nx.Graph(), []
    for i in range(3, n_x, 4):
        for j in range(3, n_y, 4):
            cell = pixels[i, j]
            is_white = (cell == WHITE).all()
            is_green = (cell == GREEN).all()
            if not (is_white or is_green):
                continue
            nid = (j // 4, i // 4)
            G.add_node(nid)
            def _add(nb):
                if nb not in G: G.add_node(nb)
            if i > 3     and (pixels[i-2, j] == WHITE).all():
                nb = (nid[0], nid[1]-1); _add(nb); G.add_edge(nid, nb)
            if j < n_y-2 and (pixels[i, j+2] == WHITE).all():
                nb = (nid[0]+1, nid[1]); _add(nb); G.add_edge(nid, nb)
            if i < n_x-2 and (pixels[i+2, j] == WHITE).all():
                nb = (nid[0], nid[1]+1); _add(nb); G.add_edge(nid, nb)
            if j > 3     and (pixels[i, j-2] == WHITE).all():
                nb = (nid[0]-1, nid[1]); _add(nb); G.add_edge(nid, nb)
            if is_green:
                terminals.append(nid)
    return G, terminals


def solution_edge_count(sol_2d):
    """Count edges in the solution graph (white pixel connectivity)."""
    H, W = sol_2d.shape[0] - 2, sol_2d.shape[1] - 2
    count = 0
    for i in range(3, H, 4):
        for j in range(3, W, 4):
            if sol_2d[i, j] < 0.5:
                continue
            if i > 3     and sol_2d[i-2, j] >= 0.5: count += 1
            if j < W - 2 and sol_2d[i, j+2] >= 0.5: count += 1
    return count


def mean_terminal_distance(G, terminals):
    """Mean pairwise shortest-path distance between terminals."""
    if len(terminals) < 2:
        return 0.0
    dists = []
    for a in range(len(terminals)):
        for b in range(a + 1, len(terminals)):
            try:
                d = nx.shortest_path_length(G, terminals[a], terminals[b])
                dists.append(d)
            except nx.NetworkXNoPath:
                pass
    return float(np.mean(dists)) if dists else 0.0


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    all_stats = {}   # {n_terminals: {stat_name: [values]}}

    for dt in DATA_TYPES:
        n = int(dt.split("_")[0])
        data_dir = Path(DATA_ROOT) / dt / "maze_data_test_21"
        inputs    = np.load(data_dir / "inputs.npy")
        solutions = np.load(data_dir / "solutions.npy")
        N = len(inputs)

        stats = {k: [] for k in [
            "graph_size", "path_length", "path_density",
            "mean_term_dist", "junctions"
        ]}

        t0 = time.time()
        for i in tqdm(range(N), desc=dt, leave=False):
            G, terminals = image_into_graph(inputs[i])
            if not nx.is_connected(G):
                G = G.subgraph(max(nx.connected_components(G), key=len)).copy()

            graph_size = G.number_of_nodes()
            path_len   = solution_edge_count(solutions[i])
            junctions  = sum(1 for n in G.nodes if G.degree(n) >= 3)
            mean_dist  = mean_terminal_distance(G, terminals)

            stats["graph_size"].append(graph_size)
            stats["path_length"].append(path_len)
            stats["path_density"].append(path_len / graph_size if graph_size > 0 else 0)
            stats["mean_term_dist"].append(mean_dist)
            stats["junctions"].append(junctions)

        elapsed = time.time() - t0
        all_stats[n] = {k: np.array(v) for k, v in stats.items()}
        np.save(RESULT_DIR / f"stats_{n}G.npy", all_stats[n])
        print(f"  {dt} ({N} mazes, {elapsed:.1f}s)")

    # ── Print summary table ────────────────────────────────────────────────────
    terminals_list = [int(dt.split("_")[0]) for dt in DATA_TYPES]
    stat_labels = [
        ("graph_size",     "Graph size (nodes)"),
        ("path_length",    "Path length (edges)"),
        ("path_density",   "Path density"),
        ("mean_term_dist", "Mean terminal dist"),
        ("junctions",      "Junctions (deg≥3)"),
    ]

    print(f"\n{'='*90}")
    print(f"  Maze difficulty statistics (mean ± std)")
    print(f"{'='*90}")
    header = f"  {'Statistic':25s}" + "".join(f"{n:>12d}" for n in terminals_list)
    print(header)
    print(f"  {'-'*25}" + "-"*12*len(terminals_list))

    for key, label in stat_labels:
        row = f"  {label:25s}"
        for n in terminals_list:
            arr = all_stats[n][key]
            row += f"  {arr.mean():5.1f}±{arr.std():4.1f}"
        print(row)

    print(f"{'='*90}")

    # ── ZInD floor plan instances (path planning evaluation set) ─────────────
    print("\nProcessing ZInD floor plan instances (all_created_final.npy)…")
    zind_data = np.load(ZIND_DATA_PATH, allow_pickle=True)
    zind_stats = {}

    for n in [5, 6, 7]:
        rows = [zind_data[i] for i in range(len(zind_data)) if zind_data[i, 0] == n]
        stats = {k: [] for k in ["path_length", "path_density",
                                  "mean_term_dist", "junctions"]}
        for row in tqdm(rows, desc=f"ZInD {n}-term", leave=False):
            inp, sol = row[1], row[2]
            G, terminals = image_into_graph(inp)
            if not nx.is_connected(G):
                G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
            graph_size = G.number_of_nodes()
            path_len   = solution_edge_count(sol)
            junctions  = sum(1 for nd in G.nodes if G.degree(nd) >= 3)
            mean_dist  = mean_terminal_distance(G, terminals)
            stats["path_length"].append(path_len)
            stats["path_density"].append(path_len / graph_size if graph_size > 0 else 0)
            stats["mean_term_dist"].append(mean_dist)
            stats["junctions"].append(junctions)
            stats.setdefault("graph_size", []).append(graph_size)
        zind_stats[n] = {k: np.array(v) for k, v in stats.items()}
        print(f"  ZInD {n}-term: {len(rows)} instances")

    # ── Print ZInD table ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  ZInD floor plan difficulty (mean ± std) — path planning set")
    print(f"{'='*60}")
    print(f"  {'Statistic':25s}" + "".join(f"{n:>12d}-term" for n in [5, 6, 7]))
    for key, label in stat_labels:
        if key == "graph_size": continue
        row = f"  {label:25s}"
        for n in [5, 6, 7]:
            arr = zind_stats[n][key]
            row += f"  {arr.mean():6.1f}±{arr.std():4.1f}"
        print(row)
    print(f"{'='*60}")

    # Save combined
    np.save(RESULT_DIR / "all_difficulty_stats.npy", all_stats)
    np.save(RESULT_DIR / "zind_difficulty_stats.npy", zind_stats)
    print(f"\nSaved to {RESULT_DIR}/")


if __name__ == "__main__":
    main()
