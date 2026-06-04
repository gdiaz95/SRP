#!/usr/bin/env python3
"""table5_path_lengths.py

Recompute Tables 5–8 for ZInD floorplan path planning experiments.

Table 5: path lengths (m) for A*, ARA*, RRT* across 5 masking filters.
Table 6: % path length change vs Original, conditioned on deviation cases.
Table 9: % SRP runtime change vs Original, per filter/planner.
Table 8: full SRP execution time (s) = filter generation + path planning.

Data:
  ZInD big floorplans — data/zind/big_floorplans/unordered/created/all_created_final.npy
  90 selected configurations: 68 (5-terminal) + 17 (6-terminal) + 5 (7-terminal)

Table 8/9 MazeNet filter times: loaded from results/table7_zind_runtime/TC_{n}_G_40_iters_rerun.npy

Pixel-to-metre conversion: 1/240 m/pixel ≈ 0.4167 cm/pixel.

Raw pixel lengths saved to results/path_planning/table5_lengths_rerun.npz

Usage (run from project root):
    python scripts/table5_path_lengths.py
"""

import math
import os
import random
import sys
import time
from heapq import heappush, heappop

import cv2
import networkx as nx
import numpy as np
from networkx.algorithms.approximation.steinertree import steiner_tree
from tqdm import tqdm

# ── Configuration ──────────────────────────────────────────────────────────────

DATA_PATH  = os.path.join("data", "zind", "big_floorplans", "unordered",
                           "created", "all_created_final.npy")
RESULT_DIR = os.path.join("results", "path_planning")
TERMINALS  = [5, 6, 7]

PIXEL_TO_M = 1.0 / 240.0   # 0.4167 cm/px (≈ 15m wide at 3600 px)

os.makedirs(RESULT_DIR, exist_ok=True)


# ── Graph utilities (ported from Lydia/path_planning_correct.py) ───────────────

def image_into_graph(pixels):
    pixels = pixels.transpose(1, 2, 0)
    n_x, n_y = pixels.shape[0] - 2, pixels.shape[1] - 2
    WHITE, GREEN = [1, 1, 1], [0, 1, 0]
    G, green_cells = nx.Graph(), []
    for i in range(3, n_x, 4):
        for j in range(3, n_y, 4):
            is_white = (pixels[i, j] == WHITE).all()
            is_green = (pixels[i, j] == GREEN).all()
            if not (is_white or is_green):
                continue
            node_id = (j // 4, i // 4)
            G.add_node(node_id, is_green=is_green, is_red=False)
            def _add(nb):
                if nb not in G:
                    G.add_node(nb, is_green=False)
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


def create_best_path_graph(pixels):
    n_x, n_y = pixels.shape[0] - 2, pixels.shape[1] - 2
    WHITE = 1
    G = nx.Graph()
    for i in range(3, n_x, 4):
        for j in range(3, n_y, 4):
            if not (pixels[i, j] == WHITE).all():
                continue
            node_id = (j // 4, i // 4)
            G.add_node(node_id, is_green=False, is_red=False)
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
    return G


def get_red_cell(input_graph, target_graph, green_cells):
    for node in input_graph.nodes():
        if node in green_cells:
            target_graph.nodes[node]['is_green'] = True
            if target_graph.degree[node] == 1:
                red_cell = node
                target_graph.nodes[node]['is_green'] = False
                input_graph.nodes[node]['is_green'] = False
                target_graph.nodes[node]['is_red'] = True
                input_graph.nodes[node]['is_red'] = True
                green_cells.remove(node)
                return red_cell, green_cells
    return None, green_cells


def sort_indices(input_graph, target_graph, green_cells, red_cell):
    return sorted(
        range(len(green_cells)),
        key=lambda i: nx.shortest_path_length(target_graph, source=red_cell, target=green_cells[i])
    )


def graph_into_image(G, green_cells, image_size=(50, 90), red_cell=None):
    WHITE = np.array([255, 255, 255], dtype=np.uint8)
    GREEN = np.array([0, 255, 0], dtype=np.uint8)
    RED   = np.array([255, 0, 0], dtype=np.uint8)
    height, width = image_size
    image = np.zeros((height, width, 3), dtype=np.uint8)
    for node1, node2 in G.edges:
        i1, j1 = 4*node1[1]+2, 4*node1[0]+2
        i2, j2 = 4*node2[1]+2, 4*node2[0]+2
        npts = max(abs(i2-i1), abs(j2-j1)) + 1
        for i, j in zip(np.linspace(i1, i2, npts, dtype=int),
                        np.linspace(j1, j2, npts, dtype=int)):
            for di, dj in [(0,0),(0,1),(1,0),(1,1)]:
                ii, jj = i+di, j+dj
                if 0 <= ii < height and 0 <= jj < width:
                    image[ii, jj] = WHITE
    for node in G.nodes:
        i, j = 4*node[1]+2, 4*node[0]+2
        color = GREEN if node in green_cells else (RED if red_cell and node == red_cell else WHITE)
        for di, dj in [(0,0),(0,1),(1,0),(1,1)]:
            ii, jj = i+di, j+dj
            if 0 <= ii < height and 0 <= jj < width:
                image[ii, jj] = color
    return image


def remove_black_squares(image):
    result = image.copy()
    H, W, _ = image.shape
    black_mask = np.all(image == [0, 0, 0], axis=2)
    allowed = np.array([[255, 255, 255], [0, 255, 0], [255, 0, 0]])
    def border_ok(sl):
        return all(any(np.all(px == a) for a in allowed) for px in sl)
    for i in range(1, H-2):
        for j in range(1, W-2):
            if black_mask[i,j] and black_mask[i,j+1] and black_mask[i+1,j] and black_mask[i+1,j+1]:
                if (border_ok(image[i-1, j-1:j+3]) and border_ok(image[i+2, j-1:j+3]) and
                        border_ok(image[i:i+2, j-1]) and border_ok(image[i:i+2, j+2])):
                    result[i:i+2, j:j+2] = [255, 255, 255]
    return result


def bestgraph_to_image_widen(G, image_mask, green_cells, red_cell=None,
                              image_size=(50, 90), width_line=8):
    WHITE = np.array([255, 255, 255], dtype=np.uint8)
    GREEN = np.array([0, 255, 0], dtype=np.uint8)
    RED   = np.array([255, 0, 0], dtype=np.uint8)
    height, width = image_size
    image = np.zeros((height, width, 3), dtype=np.uint8)
    w1, w2 = int(-(width_line/2)+1), int((width_line/2)+1)
    def n2p(node): return 4*node[0]+2, 4*node[1]+2
    for s, e in G.edges:
        x0, y0 = n2p(s); x1, y1 = n2p(e)
        npts = max(abs(x1-x0), abs(y1-y0)) + 1
        for x, y in zip(np.linspace(x0, x1, npts, dtype=int),
                        np.linspace(y0, y1, npts, dtype=int)):
            for dx in range(2):
                for dy in range(w1, w2):
                    xi, yi = x+dx, y+dy
                    if 0 <= xi < width and 0 <= yi < height:
                        image[yi, xi] = WHITE
    for node in G.nodes:
        x, y = n2p(node)
        for dx in range(w1, w2):
            for dy in range(w1, w2):
                xi, yi = x+dx, y+dy
                if 0 <= xi < width and 0 <= yi < height:
                    image[yi, xi] = WHITE
    for node in (list(green_cells) + ([red_cell] if red_cell else [])):
        x, y = n2p(node)
        color = GREEN if node in green_cells else RED
        for dx in range(2):
            for dy in range(2):
                xi, yi = x+dx, y+dy
                if 0 <= xi < width and 0 <= yi < height:
                    image[yi, xi] = color
    image_mask = remove_black_squares(image_mask)
    image[np.all(image_mask == [0,0,0], axis=-1)] = [0,0,0]
    return image


def graph_sol_into_image_widen(G, image_mask, green_cells,
                                image_size=(50, 90), width_line=8):
    WHITE = np.array([255, 255, 255], dtype=np.uint8)
    GREEN = np.array([0, 255, 0], dtype=np.uint8)
    height, width = image_size
    image = np.zeros((height, width, 3), dtype=np.uint8)
    w1, w2 = int(-(width_line/2)+1), int((width_line/2)+1)
    def n2p(node): return 4*node[0]+2, 4*node[1]+2
    for s, e in G.edges:
        x0, y0 = n2p(s); x1, y1 = n2p(e)
        npts = max(abs(x1-x0), abs(y1-y0)) + 1
        for x, y in zip(np.linspace(x0, x1, npts, dtype=int),
                        np.linspace(y0, y1, npts, dtype=int)):
            for dx in range(2):
                for dy in range(w1, w2):
                    xi, yi = x+dx, y+dy
                    if 0 <= xi < width and 0 <= yi < height:
                        image[yi, xi] = WHITE
    for node in G.nodes:
        x, y = n2p(node)
        for dx in range(w1, w2):
            for dy in range(w1, w2):
                xi, yi = x+dx, y+dy
                if 0 <= xi < width and 0 <= yi < height:
                    image[yi, xi] = WHITE
    for node in G.nodes:
        if node in green_cells or G.nodes[node].get('is_green', False):
            x, y = n2p(node)
            for dx in range(2):
                for dy in range(2):
                    xi, yi = x+dx, y+dy
                    if 0 <= xi < width and 0 <= yi < height:
                        image[yi, xi] = GREEN
    image_mask = remove_black_squares(image_mask)
    image[np.all(image_mask == [0,0,0], axis=-1)] = [0,0,0]
    return image


def get_steiner_tree_approx(G, green_cells, method):
    while not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    T = steiner_tree(G, green_cells, method=method)
    return T


def get_wavefront(G, green_cells):
    complete = nx.Graph()
    for i in range(len(green_cells)):
        for j in range(i+1, len(green_cells)):
            length, path = nx.single_source_dijkstra(G, green_cells[i], green_cells[j])
            complete.add_edge(green_cells[i], green_cells[j], weight=length, path=path)
    mst = nx.minimum_spanning_tree(complete, weight='weight', algorithm='prim')
    target = nx.Graph()
    for u, v, data in mst.edges(data=True):
        path = data['path']
        for k in range(len(path)-1):
            if not target.has_edge(path[k], path[k+1]):
                target.add_edge(path[k], path[k+1], **G.get_edge_data(path[k], path[k+1], default={}))
    return target


def get_terminals_in_resized(image_resized, sorted_indices):
    hsv = cv2.cvtColor(image_resized, cv2.COLOR_RGB2HSV)
    mask_green = cv2.inRange(hsv,
                             np.array([50, 100, 100], dtype=np.uint8),
                             np.array([70, 255, 255], dtype=np.uint8))
    mask_white = cv2.inRange(image_resized,
                             np.array([200, 200, 200], dtype=np.uint8),
                             np.array([255, 255, 255], dtype=np.uint8))
    mask_red   = cv2.inRange(hsv,
                             np.array([0, 200, 200], dtype=np.uint8),
                             np.array([10, 255, 255], dtype=np.uint8))
    mask_free  = cv2.bitwise_or(cv2.bitwise_or(mask_white, mask_green), mask_red)
    _, _, _, centroids_g = cv2.connectedComponentsWithStats(mask_green)
    green_terminals = [(int(cx), int(cy)) for cx, cy in centroids_g[1:]]
    green_terminals = [green_terminals[i] for i in sorted_indices]
    _, _, _, centroids_r = cv2.connectedComponentsWithStats(mask_red)
    red_terminals = [(int(cx), int(cy)) for cx, cy in centroids_r[1:]]
    return green_terminals, red_terminals, mask_free


# ── Path planners (ported from Lydia/path_planning_correct.py) ────────────────

def astar_path_anytime(start, goal, free_mask, epsilon):
    start, goal = tuple(start), tuple(goal)
    def h(p, q): return abs(q[0]-p[0]) + abs(q[1]-p[1])
    visited, dist, parent = set(), {start: 0}, {start: None}
    pq = [(epsilon * h(start, goal), start)]
    moves = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
    height, width = free_mask.shape
    free = free_mask.astype(bool)
    while pq:
        _, cur = heappop(pq)
        if cur in visited: continue
        visited.add(cur)
        if cur == goal:
            path = []
            while cur is not None: path.append(cur); cur = parent[cur]
            return path[::-1]
        x, y = cur
        for dx, dy in moves:
            nx_, ny = x+dx, y+dy
            if 0 <= nx_ < width and 0 <= ny < height and free[ny, nx_]:
                nb = (nx_, ny)
                cost = dist[cur] + (math.hypot(dx, dy) if dx*dy != 0 else 1.0)
                if cost < dist.get(nb, float('inf')):
                    dist[nb] = cost; parent[nb] = cur
                    heappush(pq, (cost + epsilon*h(nb, goal), nb))
    return None


def A_star_anytime(green_terminals, red_terminals, free_mask, epsilon):
    start = red_terminals[0]; path = [start]; current = start
    for goal in green_terminals:
        if goal == current: continue
        seg = astar_path_anytime(current, goal, free_mask, epsilon)
        if seg is None: return None
        path.extend(seg[1:]); current = goal
    return path


def is_collision_free(p1, p2, free_mask):
    x1, y1 = p1; x2, y2 = p2
    dx, dy = abs(x2-x1), abs(y2-y1)
    x, y = x1, y1; n = 1+dx+dy
    xi = 1 if x2 > x1 else -1; yi = 1 if y2 > y1 else -1
    err = dx-dy; dx *= 2; dy *= 2
    for _ in range(n):
        if free_mask[y, x] == 0: return False
        if err > 0: x += xi; err -= dy
        else:       y += yi; err += dx
    return True


def rrt_star_plan(start, goal, free_mask,
                   max_iter=50000, step=50, goal_bias=0.9, neighbor_radius=75):
    height, width = free_mask.shape
    Node = lambda pt, parent=None, cost=0: {'pt': pt, 'parent': parent, 'cost': cost}
    tree = [Node(start)]
    for _ in range(max_iter):
        q_rand = goal if random.random() < goal_bias else None
        while q_rand is None:
            xr, yr = random.randrange(width), random.randrange(height)
            if free_mask[yr, xr]: q_rand = (xr, yr)
        q_near = min(tree, key=lambda nd: (nd['pt'][0]-q_rand[0])**2 + (nd['pt'][1]-q_rand[1])**2)
        xn, yn = q_near['pt']
        theta = math.atan2(q_rand[1]-yn, q_rand[0]-xn)
        newx = max(0, min(int(xn + step*math.cos(theta)), width-1))
        newy = max(0, min(int(yn + step*math.sin(theta)), height-1))
        if not free_mask[newy, newx]: continue
        q_new_pt = (newx, newy)
        if not is_collision_free(q_near['pt'], q_new_pt, free_mask): continue
        neighbors = [nd for nd in tree
                     if math.hypot(nd['pt'][0]-newx, nd['pt'][1]-newy) <= neighbor_radius]
        q_min = q_near; cost_min = q_near['cost'] + math.hypot(newx-xn, newy-yn)
        for nd in neighbors:
            if is_collision_free(nd['pt'], q_new_pt, free_mask):
                c = nd['cost'] + math.hypot(newx-nd['pt'][0], newy-nd['pt'][1])
                if c < cost_min: q_min = nd; cost_min = c
        new_node = Node(q_new_pt, parent=q_min, cost=cost_min)
        tree.append(new_node)
        for nd in neighbors:
            if nd is not q_min and is_collision_free(q_new_pt, nd['pt'], free_mask):
                nc = new_node['cost'] + math.hypot(nd['pt'][0]-newx, nd['pt'][1]-newy)
                if nc < nd['cost']: nd['parent'] = new_node; nd['cost'] = nc
        if math.hypot(newx-goal[0], newy-goal[1]) <= step:
            if is_collision_free(q_new_pt, goal, free_mask):
                gn = Node(goal, parent=new_node,
                          cost=new_node['cost']+math.hypot(newx-goal[0], newy-goal[1]))
                tree.append(gn)
                path = []
                nd = gn
                while nd: path.append(nd['pt']); nd = nd['parent']
                return path[::-1]
    return None


def RRT_star(green_terminals, red_terminals, free_mask):
    start = red_terminals[0]; path = [start]; current = start
    for goal in green_terminals:
        if goal == current: continue
        seg = rrt_star_plan(current, goal, free_mask)
        if seg is None: return None
        if seg[-1] != goal: seg.append(goal)
        path.extend(seg[1:]); current = goal
    return path


def get_distance(path):
    if path is None or len(path) < 2: return 0
    return sum(math.hypot(path[i+1][0]-path[i][0], path[i+1][1]-path[i][1])
               for i in range(len(path)-1))


# ── Per-floor-plan runner ─────────────────────────────────────────────────────

def process_floor_plan(img, sol, floor, idx):
    """
    Returns (lengths, ftimes, ptimes).
    lengths: dict (planner, mask) → pixel path length
    ftimes:  dict mask → filter generation time (s)  [NET=0.0, filled from TC cache in main]
    ptimes:  dict (planner, mask) → path planning time (s)
    planners: astar, ara_star, rrt_star
    masks:    NET, Kou, Meh, Wavefront, Original
    """
    graph, green_cells = image_into_graph(img)
    target_graph_NET   = create_best_path_graph(sol)

    while not nx.is_connected(graph):
        graph = graph.subgraph(max(nx.connected_components(graph), key=len)).copy()

    t0 = time.time()
    target_graph_kou = get_steiner_tree_approx(graph, list(green_cells), 'kou')
    ftime_kou = time.time() - t0

    t0 = time.time()
    target_graph_meh = get_steiner_tree_approx(graph, list(green_cells), 'mehlhorn')
    ftime_meh = time.time() - t0

    t0 = time.time()
    target_graph_wavefront = get_wavefront(graph, list(green_cells))
    ftime_wave = time.time() - t0

    ftimes = {"NET": 0.0, "Kou": ftime_kou, "Meh": ftime_meh,
              "Wavefront": ftime_wave, "Original": 0.0}

    red_cell, green_cells = get_red_cell(graph, target_graph_NET, green_cells)
    sorted_idx_NET  = sort_indices(graph, target_graph_NET,       green_cells, red_cell)
    sorted_idx_kou  = sort_indices(graph, target_graph_kou,       green_cells, red_cell)
    sorted_idx_meh  = sort_indices(graph, target_graph_meh,       green_cells, red_cell)
    sorted_idx_wave = sort_indices(graph, target_graph_wavefront, green_cells, red_cell)

    base_image = graph_into_image(graph, green_cells, image_size=(50, 90), red_cell=red_cell)
    base_image = remove_black_squares(base_image)

    sol_NET  = bestgraph_to_image_widen(target_graph_NET,       base_image, green_cells,
                                         red_cell=red_cell, image_size=(50, 90), width_line=8)
    sol_Kou  = graph_sol_into_image_widen(target_graph_kou,       base_image, green_cells,
                                           image_size=(50, 90), width_line=8)
    sol_Meh  = graph_sol_into_image_widen(target_graph_meh,       base_image, green_cells,
                                           image_size=(50, 90), width_line=8)
    sol_Wave = graph_sol_into_image_widen(target_graph_wavefront, base_image, green_cells,
                                           image_size=(50, 90), width_line=8)

    W, H = 3600, 2000
    orig_r = cv2.resize(base_image, (W, H), interpolation=cv2.INTER_NEAREST)
    net_r  = cv2.resize(sol_NET,    (W, H), interpolation=cv2.INTER_NEAREST)
    kou_r  = cv2.resize(sol_Kou,    (W, H), interpolation=cv2.INTER_NEAREST)
    meh_r  = cv2.resize(sol_Meh,    (W, H), interpolation=cv2.INTER_NEAREST)
    wave_r = cv2.resize(sol_Wave,   (W, H), interpolation=cv2.INTER_NEAREST)

    gt_net,  rt_net,  mask_net  = get_terminals_in_resized(net_r,  sorted_idx_NET)
    gt_kou,  _,       mask_kou  = get_terminals_in_resized(kou_r,  sorted_idx_kou)
    gt_meh,  _,       mask_meh  = get_terminals_in_resized(meh_r,  sorted_idx_meh)
    gt_wave, _,       mask_wave = get_terminals_in_resized(wave_r, sorted_idx_wave)
    _,       _,       mask_orig = get_terminals_in_resized(orig_r, sorted_idx_NET)

    def run_astar(gt, rt, mask, eps):
        return get_distance(A_star_anytime(gt, rt, mask, eps))

    def run_rrt_star(gt, rt, mask):
        return get_distance(RRT_star(gt, rt, mask))

    lengths = {}
    ptimes  = {}
    for mask_name, gt, rt, mask in [
        ("NET",       gt_net, rt_net, mask_net),
        ("Kou",       gt_net, rt_net, mask_kou),
        ("Meh",       gt_net, rt_net, mask_meh),
        ("Wavefront", gt_net, rt_net, mask_wave),
        ("Original",  gt_net, rt_net, mask_orig),
    ]:
        t0 = time.time()
        lengths[("astar",    mask_name)] = run_astar(gt, rt, mask, eps=1.0)
        ptimes[("astar",    mask_name)] = time.time() - t0

        t0 = time.time()
        lengths[("ara_star", mask_name)] = run_astar(gt, rt, mask, eps=2.5)
        ptimes[("ara_star", mask_name)] = time.time() - t0

        t0 = time.time()
        lengths[("rrt_star", mask_name)] = run_rrt_star(gt, rt, mask)
        ptimes[("rrt_star", mask_name)] = time.time() - t0

    return lengths, ftimes, ptimes


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("  Table 5: ZInD floorplan path lengths (m) — A*, ARA*, RRT*")
    print("=" * 72)

    if not os.path.exists(DATA_PATH):
        print(f"\n  [ERROR] Data not found: {DATA_PATH}")
        sys.exit(1)

    all_data = np.load(DATA_PATH, allow_pickle=True)

    planners  = ["astar", "ara_star", "rrt_star"]
    masks     = ["NET", "Kou", "Meh", "Wavefront", "Original"]
    px_lengths = {(p, m): [] for p in planners for m in masks}
    all_fp_data = []   # (n, lengths, ftimes, ptimes) per floor plan

    for n in TERMINALS:
        imgs = all_data[all_data[:, 0] == n, 1]
        sols = all_data[all_data[:, 0] == n, 2]
        print(f"\n  [{n} terminals — {len(imgs)} floor plans]")
        t0 = time.time()

        for idx in tqdm(range(len(imgs)), desc=f"  {n}G", leave=False):
            try:
                lengths, ftimes, ptimes = process_floor_plan(imgs[idx], sols[idx], n, idx)
                all_fp_data.append((n, lengths, ftimes, ptimes))
                for key, val in lengths.items():
                    if val > 0:
                        px_lengths[key].append(val)
            except Exception as e:
                print(f"    [skip {n}_{idx}: {e}]")

        elapsed = time.time() - t0
        print(f"    done in {elapsed:.1f}s  |  collected "
              f"{len(px_lengths[('astar','NET')])} A*/NET samples so far")

    # ── Save raw pixel lengths ─────────────────────────────────────────────────
    save_path = os.path.join(RESULT_DIR, "table5_lengths_rerun.npz")
    np.savez(save_path, **{f"{p}__{m}": np.array(v)
                            for (p, m), v in px_lengths.items()})
    print(f"\n  Raw pixel lengths saved → {save_path}")

    # ── Convert to metres and print table ─────────────────────────────────────
    def mean_m(p, m):
        arr = np.array(px_lengths[(p, m)])
        return float(np.mean(arr)) * PIXEL_TO_M if len(arr) > 0 else float("nan")

    col_labels = ["MazeNet", "Kou", "Mehlhorn", "Wavefront MST", "Original"]
    mask_keys  = ["NET",     "Kou", "Meh",      "Wavefront",     "Original"]
    planner_labels = [("A*", "astar"), ("ARA*", "ara_star"), ("RRT*", "rrt_star")]

    col_w = 14

    def print_table(title, rows):
        print(f"\n{'=' * 72}")
        print(f"  {title}")
        print(f"{'=' * 72}")
        print(f"  {'':8s}" + "".join(f"{c:>{col_w}s}" for c in col_labels))
        print(f"  {'-'*8}" + "-" * (col_w * len(col_labels)))
        for label, vals in rows:
            row = f"  {label:8s}"
            for v in vals:
                row += f"{v:>{col_w}.2f}" if not (isinstance(v, float) and np.isnan(v)) else f"{'N/A':>{col_w}s}"
            print(row)
        print(f"{'=' * 72}")

    new_rows = [(lbl, [mean_m(pk, mk) for mk in mask_keys])
                for lbl, pk in planner_labels]

    print_table("Table 5 — path lengths (m)", new_rows)

    # ── Table 6: % path length change vs Original (conditioned on deviation) ──────

    masks_cond  = ["NET", "Kou", "Meh", "Wavefront"]
    cond_labels = ["MazeNet", "Kou", "Mehlhorn", "Wavefront MST"]
    cond_pct    = {(p, m): [] for p in planners for m in masks_cond}
    TOL         = 0.5   # pixel tolerance for "deviation"

    for _, lengths_fp, _, _ in all_fp_data:
        for p in planners:
            len_orig = lengths_fp.get((p, "Original"), 0)
            if len_orig <= 0:
                continue
            for m in masks_cond:
                len_m = lengths_fp.get((p, m), 0)
                if len_m > 0 and abs(len_m - len_orig) > TOL:
                    cond_pct[(p, m)].append((len_m - len_orig) / len_orig * 100)

    planner_labels6 = [("A*", "astar"), ("ARA*", "ara_star"), ("RRT*", "rrt_star")]

    w6 = 14

    def print_t6(title, rows6):
        print(f"\n{'=' * 72}")
        print(f"  {title}")
        print(f"{'=' * 72}")
        print(f"  {'':8s}" + "".join(f"{c:>{w6}s}" for c in cond_labels))
        print(f"  {'-'*8}" + "-" * (w6 * len(cond_labels)))
        for lbl, vals in rows6:
            row = f"  {lbl:8s}"
            for v in vals:
                row += f"{v:>{w6}.2f}"
            print(row)
        print(f"{'=' * 72}")

    def mean_cond(pk, mk):
        v = cond_pct[(pk, mk)]
        return float(np.mean(v)) if v else 0.0

    new_rows6 = [(lbl, [mean_cond(pk, mk) for mk in masks_cond]) for lbl, pk in planner_labels6]

    print_t6("Table 6 — % path length change vs Original (conditioned)", new_rows6)

    save_t6 = os.path.join(RESULT_DIR, "table6_conditioned_pct_rerun.npz")
    np.savez(save_t6, **{f"{p}__{m}": np.array(cond_pct[(p, m)])
                         for p in planners for m in masks_cond})
    print(f"\n  Table 6 conditioned samples saved → {save_t6}")

    n_dev_str = "  n_deviated"
    for mk in masks_cond:
        n_dev = sum(len(cond_pct[(pk, mk)]) for _, pk in planner_labels6)
        n_dev_str += f"     n={n_dev:>3d}   "
    print(n_dev_str)

    # ── Table 8: full SRP execution time (s) ─────────────────────────────────────
    # (Table 7 is computed separately in scripts/table7_zind_runtime.py)
    #
    # Filter gen times use fixed per-terminal-count means (same approach as NET):
    #   NET      → mean TC time from table7 rerun cache
    #   Kou/Meh/Wave → mean Steiner time from table7 approx_timings cache
    # This avoids per-floor-plan outliers skewing Table 9 percentages.

    TC_DIR = os.path.join("results", "table7_zind_runtime")
    APPROX_DIR = os.path.join(TC_DIR, "approx_timings")

    tc_s   = {}
    kou_s  = {}
    meh_s  = {}
    wave_s = {}
    BREAKDOWN_DIR = os.path.join("results", "runtime_breakdown")
    for n in TERMINALS:
        tc_path = os.path.join(TC_DIR, f"TC_{n}_G_40_iters_rerun.npy")
        tc_s[n] = float(np.mean(np.load(tc_path))) if os.path.exists(tc_path) else 0.0

        # Add mask gen + HD upscale + decode + transfer steps missing from Table 7
        bd_path = os.path.join(BREAKDOWN_DIR, f"breakdown_{n}G.npy")
        if os.path.exists(bd_path):
            bd = np.load(bd_path, allow_pickle=True).item()
            extra_s = (np.mean(bd["mask_gen"]) + np.mean(bd["hd_upscale"])
                       + np.mean(bd["decode"]) + np.mean(bd["transfer_to_cpu"]))
            tc_s[n] += extra_s

        kou_path  = os.path.join(APPROX_DIR, f"kou_{n}G.npy")
        meh_path  = os.path.join(APPROX_DIR, f"mehlhorn_{n}G.npy")
        wave_path = os.path.join(APPROX_DIR, f"wavefront_{n}G.npy")
        kou_s[n]  = float(np.mean(np.load(kou_path)))  if os.path.exists(kou_path)  else 0.0
        meh_s[n]  = float(np.mean(np.load(meh_path)))  if os.path.exists(meh_path)  else 0.0
        wave_s[n] = float(np.mean(np.load(wave_path))) if os.path.exists(wave_path) else 0.0

    print("\n  Fixed mean filter gen times loaded (s) — MazeNet includes full pipeline:")
    for n in TERMINALS:
        print(f"    {n}G: NET={tc_s[n]*1000:.1f}ms  Kou={kou_s[n]*1000:.1f}ms  "
              f"Meh={meh_s[n]*1000:.1f}ms  Wave={wave_s[n]*1000:.1f}ms")

    srp_times = {(p, m): [] for p in planners for m in masks}
    for (n, lengths_fp, ftimes, ptimes) in all_fp_data:
        ft_fixed = {
            "NET":      tc_s[n],
            "Kou":      kou_s[n],
            "Meh":      meh_s[n],
            "Wavefront": wave_s[n],
            "Original": 0.0,
        }
        for p in planners:
            for m in masks:
                srp_times[(p, m)].append(ft_fixed[m] + ptimes.get((p, m), 0.0))

    w8 = 14
    col_labels8 = ["MazeNet", "Kou", "Mehlhorn", "Wavefront MST", "Original"]
    mask_keys8  = ["NET",     "Kou", "Meh",      "Wavefront",     "Original"]

    def print_t8(title, rows8):
        print(f"\n{'=' * 72}")
        print(f"  {title}")
        print(f"{'=' * 72}")
        print(f"  {'':8s}" + "".join(f"{c:>{w8}s}" for c in col_labels8))
        print(f"  {'-'*8}" + "-" * (w8 * len(col_labels8)))
        for lbl, vals in rows8:
            row = f"  {lbl:8s}"
            for v in vals:
                row += f"{v:>{w8}.2f}"
            print(row)
        print(f"  {'(values in s)':8s}")
        print(f"{'=' * 72}")

    new_rows8 = [(lbl, [float(np.mean(srp_times[(pk, mk)])) for mk in mask_keys8])
                 for lbl, pk in planner_labels6]

    print_t8("Table 8 — full SRP time (s)", new_rows8)

    save_t8 = os.path.join(RESULT_DIR, "table8_srp_times_rerun.npz")
    np.savez(save_t8, **{f"{p}__{m}": np.array(srp_times[(p, m)])
                         for p in planners for m in masks})
    print(f"  Table 8 SRP times saved → {save_t8}")

    # ── Table 9: % runtime change vs Original ─────────────────────────────────────

    masks_cond9  = ["NET", "Kou", "Meh", "Wavefront"]
    col_labels9  = ["MazeNet", "Kou", "Mehlhorn", "Wavefront MST"]

    w9 = 14

    def print_t9(title, rows9):
        print(f"\n{'=' * 72}")
        print(f"  {title}")
        print(f"{'=' * 72}")
        print(f"  {'':8s}" + "".join(f"{c:>{w9}s}" for c in col_labels9))
        print(f"  {'-'*8}" + "-" * (w9 * len(col_labels9)))
        for lbl, vals in rows9:
            row = f"  {lbl:8s}"
            for v in vals:
                row += f"{v:>{w9}.2f}"
            print(row)
        print(f"  {'(values in %)':8s}")
        print(f"{'=' * 72}")

    def pct_vs_orig(pk, mk):
        mean_m = float(np.mean(srp_times[(pk, mk)]))
        mean_o = float(np.mean(srp_times[(pk, "Original")]))
        return (mean_m - mean_o) / mean_o * 100

    new_rows9 = [(lbl, [pct_vs_orig(pk, mk) for mk in masks_cond9])
                 for lbl, pk in planner_labels6]

    print_t9("Table 9 — % runtime change vs Original", new_rows9)

    save_t9 = os.path.join(RESULT_DIR, "table9_pct_srp_rerun.npz")
    pct_arr = {f"{pk}__{mk}": np.array([pct_vs_orig(pk, mk)])
               for lbl, pk in planner_labels6 for mk in masks_cond9}
    np.savez(save_t9, **pct_arr)
    print(f"  Table 9 % runtime saved → {save_t9}")

    print(f"  All results saved to {RESULT_DIR}/\n")


if __name__ == "__main__":
    main()
