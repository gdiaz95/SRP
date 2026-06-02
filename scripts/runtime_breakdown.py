#!/usr/bin/env python3
"""runtime_breakdown.py

Separate runtime accounting for each stage of the SRP pipeline on ZInD
floor plan test sets (5, 6, 7 terminals):

  1. CPU → GPU transfer
  2. MazeNet GPU inference          (CUDA events)
  3. apply_mask                     (GPU)
  4. TC verification                (CPU graph traversal)
  5. GPU → CPU transfer
  6. Output decode                  (binary mask → grid graph)
  7. Mask generation                (graph → 50×90 RGB constraint image)
  8. HD upscale + terminal extract  (cv2.resize to 3600×2000 + mask extract)
  9. A* planning on HD image        (3600×2000 free mask)
 10. ARA* planning on HD image
 11. RRT* planning on HD image

Planners in steps 9–11 use the identical functions as tables5-6-8-9_path_lengths.py
so timings are directly comparable to Tables 6/8/9 in the paper.

Usage (from project root):
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/runtime_breakdown.py
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/runtime_breakdown.py --no-planning
"""

import argparse
import heapq
import math
import os
import random
import sys
import time

import cv2
import networkx as nx
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deepthinking.utils.testing import apply_mask, check_termination_multiple
from deepthinking.models.dt_net_2d_parallel import DTNet, BasicBlock

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_PATH     = "outputs/training_default/training-unlost-Tamar/model_best.pth"
# Same dataset as Tables 5-9 (linear-topology floor plans, 90 instances total)
DATA_PATH      = os.path.join("data", "zind", "big_floorplans",
                               "unordered", "created", "all_created_final.npy")
RESULT_DIR     = "results/runtime_breakdown"
TERMINALS      = [5, 6, 7]

TC_THRESHOLD   = 0.50
TC_FIRST_BATCH = 10
TC_BATCH       = 10
MAX_ROUNDS     = 13          # 14 × 10 = 140 iters, matches test_zind_config

COMP_W, COMP_H = 3600, 2000  # HD planning resolution (same as tables5-6-8-9)
WIDTH_LINE     = 8           # mask widening (same as tables5-6-8-9)

os.makedirs(RESULT_DIR, exist_ok=True)


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


# ── Image ↔ graph (same as table4/table2) ────────────────────────────────────

def image_into_graph(pixels):
    """Convert (3, H, W) float image to NetworkX graph + green terminal list."""
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
                green_cells.append(nid)
    return G, green_cells


def create_best_path_graph_from_binary(pred_bin_2d):
    """Convert (H, W) binary float array → NetworkX graph with grid-coord nodes.
    Equivalent to create_best_path_graph from table4_zind_accuracies.py."""
    n_x = pred_bin_2d.shape[0] - 2
    n_y = pred_bin_2d.shape[1] - 2
    G = nx.Graph()
    for i in range(3, n_x, 4):
        for j in range(3, n_y, 4):
            if pred_bin_2d[i, j] < 0.5:
                continue
            nid = (j // 4, i // 4)
            G.add_node(nid)
            def _add(nb):
                if nb not in G: G.add_node(nb)
            if i > 3     and pred_bin_2d[i-2, j] >= 0.5:
                nb = (nid[0], nid[1]-1); _add(nb); G.add_edge(nid, nb)
            if j < n_y-2 and pred_bin_2d[i, j+2] >= 0.5:
                nb = (nid[0]+1, nid[1]); _add(nb); G.add_edge(nid, nb)
            if i < n_x-2 and pred_bin_2d[i+2, j] >= 0.5:
                nb = (nid[0], nid[1]+1); _add(nb); G.add_edge(nid, nb)
            if j > 3     and pred_bin_2d[i, j-2] >= 0.5:
                nb = (nid[0]-1, nid[1]); _add(nb); G.add_edge(nid, nb)
    return G


# ── Graph rendering (ported from tables5-6-8-9_path_lengths.py) ───────────────

def remove_black_squares(image):
    result = image.copy()
    H, W, _ = image.shape
    black_mask = np.all(image == [0, 0, 0], axis=2)
    allowed = np.array([[255, 255, 255], [0, 255, 0], [255, 0, 0]])
    def border_ok(sl):
        return all(any(np.all(px == a) for a in allowed) for px in sl)
    for i in range(1, H-2):
        for j in range(1, W-2):
            if (black_mask[i, j] and black_mask[i, j+1]
                    and black_mask[i+1, j] and black_mask[i+1, j+1]):
                if (border_ok(image[i-1, j-1:j+3]) and
                        border_ok(image[i+2, j-1:j+3]) and
                        border_ok(image[i:i+2, j-1]) and
                        border_ok(image[i:i+2, j+2])):
                    result[i:i+2, j:j+2] = [255, 255, 255]
    return result


def bestgraph_to_image_widen(G, image_mask, green_cells, red_cell=None,
                               image_size=(50, 90), width_line=WIDTH_LINE):
    """Render MazeNet solution graph to (H, W, 3) uint8 constraint image.
    Identical to bestgraph_to_image_widen in tables5-6-8-9_path_lengths.py."""
    WHITE = np.array([255, 255, 255], dtype=np.uint8)
    GREEN = np.array([0, 255, 0],   dtype=np.uint8)
    RED   = np.array([255, 0, 0],   dtype=np.uint8)
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
    # apply obstacle mask (same as paper script)
    image[np.all(image_mask == [0, 0, 0], axis=-1)] = [0, 0, 0]
    return image


# ── HD mask extraction (ported from tables5-6-8-9_path_lengths.py) ────────────

def get_terminals_in_resized(image_resized, sorted_indices):
    """Extract terminal positions and free-space mask from HD (3600×2000) image."""
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
    green_terminals = [green_terminals[i] for i in sorted_indices
                       if i < len(green_terminals)]
    _, _, _, centroids_r = cv2.connectedComponentsWithStats(mask_red)
    red_terminals = [(int(cx), int(cy)) for cx, cy in centroids_r[1:]]
    return green_terminals, red_terminals, mask_free


# ── HD planners (ported from tables5-6-8-9_path_lengths.py) ──────────────────

def astar_path_anytime(start, goal, free_mask, epsilon):
    start, goal = tuple(start), tuple(goal)
    def h(p, q): return abs(q[0]-p[0]) + abs(q[1]-p[1])
    visited, dist, parent = set(), {start: 0}, {start: None}
    pq = [(epsilon * h(start, goal), start)]
    moves = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
    height, width = free_mask.shape
    free = free_mask.astype(bool)
    while pq:
        _, cur = heapq.heappop(pq)
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
                    heapq.heappush(pq, (cost + epsilon*h(nb, goal), nb))
    return None


def A_star_anytime(green_terminals, red_terminals, free_mask, epsilon):
    if not red_terminals: return None
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
                return True   # found path — return quickly (we only need timing)
    return False


def RRT_star(green_terminals, red_terminals, free_mask):
    if not red_terminals: return None
    start = red_terminals[0]; current = start
    for goal in green_terminals:
        if goal == current: continue
        rrt_star_plan(current, goal, free_mask)
        current = goal
    return True


# ── GPU-safe timer ────────────────────────────────────────────────────────────

class CUDATimer:
    def __init__(self, device):
        self.use_cuda = (device.type == 'cuda') if hasattr(device, 'type') \
                        else ('cuda' in str(device))
        if self.use_cuda:
            self.start_ev = torch.cuda.Event(enable_timing=True)
            self.end_ev   = torch.cuda.Event(enable_timing=True)

    def start(self):
        if self.use_cuda: self.start_ev.record()
        else:             self._t = time.perf_counter()

    def stop(self):
        if self.use_cuda:
            self.end_ev.record()
            torch.cuda.synchronize()
            return self.start_ev.elapsed_time(self.end_ev) / 1000.0  # → seconds
        else:
            return time.perf_counter() - self._t


# ── Per-sample breakdown ──────────────────────────────────────────────────────

def time_sample(net, inp_cpu, device, run_planning=True):
    """Time each SRP pipeline component for one ZInD floor plan sample."""
    tmr = CUDATimer(device)
    t = {k: 0.0 for k in [
        'transfer_to_gpu', 'inference', 'apply_mask',
        'tc_check', 'transfer_to_cpu',
        'decode', 'mask_gen', 'hd_upscale',
        'astar', 'arastar', 'rrtstar',
    ]}

    # ── 1. CPU → GPU ──────────────────────────────────────────────────────────
    tmr.start()
    inp = torch.from_numpy(inp_cpu.astype(np.float32)).unsqueeze(0).to(device)
    t['transfer_to_gpu'] = tmr.stop()

    # ── 2/3/4. Inference + apply_mask + TC (batched loop) ────────────────────
    interims = None; first = True; final_target = None; all_out = None
    for rnd in range(MAX_ROUNDS + 1):
        n_iters = TC_FIRST_BATCH if first else TC_BATCH

        tmr.start()
        all_out, interims = net(inp, n_iters, interims)
        t['inference'] += tmr.stop()
        first = False

        tmr.start()
        single_image = apply_mask(inp, all_out)
        t['apply_mask'] += tmr.stop()

        tc0 = time.perf_counter()
        try:
            single_target, finish, _ = check_termination_multiple(single_image, TC_THRESHOLD)
        except Exception:
            finish = False; single_target = None
        t['tc_check'] += time.perf_counter() - tc0

        if finish or rnd == MAX_ROUNDS:
            final_target = single_target
            break

    # ── 5. GPU → CPU ─────────────────────────────────────────────────────────
    tmr.start()
    if final_target is not None:
        pred_cpu = final_target.cpu().squeeze().numpy()
    else:
        pred_cpu = all_out.cpu().squeeze().numpy()
    t['transfer_to_cpu'] = tmr.stop()

    # ── 6. Decode: binary mask → grid graph ──────────────────────────────────
    t0 = time.perf_counter()
    if pred_cpu.ndim == 3:
        pred_bin = (np.argmax(pred_cpu, axis=0) >= 0.5).astype(np.float32)
    else:
        pred_bin = (pred_cpu > 0.5).astype(np.float32)
    pred_graph = create_best_path_graph_from_binary(pred_bin)
    t['decode'] = time.perf_counter() - t0

    # ── 7. Mask generation (grid graph → 50×90 RGB constraint image) ─────────
    t0 = time.perf_counter()
    _, green_cells = image_into_graph(inp_cpu)
    green_cells = list(green_cells)
    # determine start terminal (red): pick a degree-1 node of pred_graph in terminals
    red_cell = None
    for node in green_cells:
        if node in pred_graph and pred_graph.degree(node) == 1:
            red_cell = node; break
    if red_cell is None and green_cells:
        red_cell = green_cells[0]
    green_goals = [c for c in green_cells if c != red_cell]
    # base image from input (float [0,1] → uint8)
    base_image = (inp_cpu.transpose(1, 2, 0) * 255).astype(np.uint8)
    base_image = remove_black_squares(base_image)
    # render MazeNet constraint image
    constraint_50x90 = bestgraph_to_image_widen(
        pred_graph, base_image, green_goals,
        red_cell=red_cell, image_size=(50, 90), width_line=WIDTH_LINE,
    )
    t['mask_gen'] = time.perf_counter() - t0

    # ── 8. HD upscale + terminal/mask extraction ──────────────────────────────
    t0 = time.perf_counter()
    constraint_hd = cv2.resize(constraint_50x90, (COMP_W, COMP_H),
                                interpolation=cv2.INTER_NEAREST)
    # Sort green terminals by path distance from red_cell along pred_graph,
    # matching tables5-6-8-9_path_lengths.py sort_indices() exactly.
    if red_cell is not None and red_cell in pred_graph and len(green_goals) > 0:
        def _path_len(node):
            try:
                return nx.shortest_path_length(pred_graph, source=red_cell, target=node)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                return float('inf')
        sorted_idx = sorted(range(len(green_goals)), key=lambda i: _path_len(green_goals[i]))
    else:
        sorted_idx = list(range(len(green_goals)))
    green_hd, red_hd, mask_hd = get_terminals_in_resized(constraint_hd, sorted_idx)
    t['hd_upscale'] = time.perf_counter() - t0

    # ── 9/10/11. Downstream planning on 3600×2000 free mask ──────────────────
    if run_planning and red_hd and green_hd:
        t0 = time.perf_counter()
        A_star_anytime(green_hd, red_hd, mask_hd, epsilon=1.0)
        t['astar'] = time.perf_counter() - t0

        t0 = time.perf_counter()
        A_star_anytime(green_hd, red_hd, mask_hd, epsilon=2.5)
        t['arastar'] = time.perf_counter() - t0

        t0 = time.perf_counter()
        RRT_star(green_hd, red_hd, mask_hd)
        t['rrtstar'] = time.perf_counter() - t0

    return t


# ── Reporting ─────────────────────────────────────────────────────────────────

COMPONENTS = [
    ('transfer_to_gpu', 'CPU → GPU transfer'),
    ('inference',       'MazeNet inference (GPU)'),
    ('apply_mask',      'apply_mask (GPU)'),
    ('tc_check',        'TC verification (CPU)'),
    ('transfer_to_cpu', 'GPU → CPU transfer'),
    ('decode',          'Output decode (CPU)'),
    ('mask_gen',        'Mask generation (CPU)'),
    ('hd_upscale',      'HD upscale + extract (CPU)'),
    ('astar',           'A* on HD 3600×2000 (CPU)'),
    ('arastar',         'ARA* on HD 3600×2000 (CPU)'),
    ('rrtstar',         'RRT* on HD 3600×2000 (CPU)'),
]

MAZENET_KEYS = [
    'transfer_to_gpu', 'inference', 'apply_mask', 'tc_check',
    'transfer_to_cpu', 'decode', 'mask_gen', 'hd_upscale',
]


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-planning", action="store_true",
                        help="Skip downstream planning")
    parser.add_argument("--n-warmup", type=int, default=5,
                        help="Warmup samples before timing (default 5)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading model: {MODEL_PATH}")
    net = load_model(MODEL_PATH, device)

    # Load the same 90-instance dataset used for Tables 5-9
    all_data = np.load(DATA_PATH, allow_pickle=True)

    all_results = {}

    for n in TERMINALS:
        rows   = [all_data[i] for i in range(len(all_data)) if all_data[i, 0] == n]
        inputs = np.stack([r[1] for r in rows])   # (N, 3, 50, 90) float64
        N = len(inputs)
        print(f"\n── {n}-terminal  ({N} samples) ──")

        component_times = {k: [] for k, _ in COMPONENTS}

        print(f"  Warming up ({args.n_warmup} samples)...", flush=True)
        for i in range(min(args.n_warmup, N)):
            time_sample(net, inputs[i], device,
                        run_planning=not args.no_planning)

        print(f"  Timing {N} samples...", flush=True)
        t_wall = time.time()
        for i in range(N):
            t = time_sample(net, inputs[i], device,
                            run_planning=not args.no_planning)
            for k, _ in COMPONENTS:
                component_times[k].append(t[k])
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{N}  [{time.time()-t_wall:.0f}s]", flush=True)

        all_results[n] = {k: np.array(v) for k, v in component_times.items()}
        np.save(os.path.join(RESULT_DIR, f"breakdown_{n}G.npy"),
                {k: v for k, v in all_results[n].items()})
        print(f"  Saved → {RESULT_DIR}/breakdown_{n}G.npy")

    # ── Print table ───────────────────────────────────────────────────────────
    W = 100
    print(f"\n{'='*W}")
    print(f"  Runtime breakdown (mean ± std, ms)"
          f"{'':30s}5-term       6-term       7-term")
    print(f"{'='*W}")
    for key, label in COMPONENTS:
        if args.no_planning and key in ('astar', 'arastar', 'rrtstar'):
            continue
        row = f"  {label:45s}"
        for n in TERMINALS:
            arr = all_results[n][key] * 1000  # → ms
            if arr.mean() < 0.001:
                row += f"  {'<0.001':>12s}"
            else:
                row += f"  {arr.mean():6.2f}±{arr.std():5.2f}"
        print(row)
    print(f"{'='*W}")

    print(f"\n  {'MazeNet pipeline (excl. planning)':45s}", end="")
    for n in TERMINALS:
        r = all_results[n]
        total = sum(r[k].mean() for k in MAZENET_KEYS) * 1000
        print(f"  {total:>12.2f}ms", end="")
    print()


if __name__ == "__main__":
    main()
