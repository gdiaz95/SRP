#!/usr/bin/env python3
"""visualize_trajectories.py

Visualizes Full A* vs SRP-masked trajectories on the same room-like floor plan
image used in the paper experiments 

Pipeline (matching the notebook):
  1. Load all_created_final.npy (same 90 instances as Tables 5-6-8-9).
  2. Remove 2×2 black artifacts  →  remove_black_squares()
  3. Crop 2-px border, resize to 3600×2000, run contour-erosion to get a
     clean architectural floor plan  →  get_original_complete()  (cv2-based).
  4. Plan the wavefront-MST paths on G_full (unmasked) and G_srp (masked).
  5. Project graph-node paths onto the 3600×2000 canvas and overlay with colour.
  6. Save per-instance side-by-side PNGs and an overview grid.

Run from project root:
    uv run python scripts/visualize_trajectories.py              # 10 instances
    uv run python scripts/visualize_trajectories.py --all        # all 90
    uv run python scripts/visualize_trajectories.py --idx 0 5 42
"""

import argparse
import csv
import math
import shutil
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import pybullet as p
import pybullet_data

# ── Constants ──────────────────────────────────────────────────────────────────

DATA_PATH = Path("data/zind/big_floorplans/unordered/created/all_created_final.npy")
OUT_DIR   = Path("results/path_planning/visualizations")

IMG_H, IMG_W   = 50, 90
GRID_W, GRID_H = 22, 12

# Complete-image canvas (matches notebook)
CROP      = 2                     # pixels cropped from each edge
COMP_W    = 3600                  # output width  (pixels)
COMP_H    = 2000                  # output height (pixels)
CROP_W    = IMG_W - 2 * CROP      # 86
CROP_H    = IMG_H - 2 * CROP      # 46
INNER_GAP = 65                    # erosion kernel size for wall thickening

COL_FULL = '#2563EB'              # blue  — Full A*
COL_SRP  = '#EA580C'              # orange-red — SRP masked
LW_PATH  = 4.0
LW_GRID  = 2.0

SCALE_X  = COMP_W / CROP_W       # ≈ 41.86 px per original px (horizontal)
SCALE_Y  = COMP_H / CROP_H       # ≈ 43.48 px per original px (vertical)

# Physical scale: PIXEL_TO_M = 1/240 at 3600-px render width
# → original 90-px image uses scale 40/240 = 1/6 m/px
# → one graph hop = 4 original pixels = 4/6 ≈ 0.667 m
SCALE = 1.0 / 6.0                 # m/px in the original 90-px image

# PyBullet robot parameters
# ROBOT_R must be < corridor half-gap (≈ 0.083 m) to avoid false-positive collisions.
# Corridor = 2 px = 0.333 m → wall box right-edge gap to node centre = 0.083 m.
ROBOT_R               = 0.07     # m  (TurtleBot-class disc, fits in 2-px corridors)
WALL_H                = 0.60     # m
PENETRATION_THRESHOLD = -0.005   # m  (contacts shallower than this are ignored)


# ── Graph-to-image rendering ───────────────────────────────────────────────────

def render_graph_to_image(G, terminals):
    """Render graph G as (IMG_H, IMG_W, 3) uint8 image using the same
    node/corridor pixel layout as the ZInD encoding.
    Terminals are green; other nodes and corridors are white; rest is black.
    """
    img = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
    term_set = set(map(tuple, terminals))

    for node in G.nodes():
        x, y = node
        r, c = 4 * y + 2, 4 * x + 2
        img[r:r+2, c:c+2] = [0, 255, 0] if node in term_set else [255, 255, 255]

    for n1, n2 in G.edges():
        (x1, y1), (x2, y2) = n1, n2
        if x1 > x2 or (x1 == x2 and y1 > y2):
            (x1, y1), (x2, y2) = (x2, y2), (x1, y1)
        r, c = 4 * y1 + 2, 4 * x1 + 2
        if x2 == x1 + 1:
            img[r:r+2, c+2:c+4] = [255, 255, 255]
        elif y2 == y1 + 1:
            img[r+2:r+4, c:c+2] = [255, 255, 255]
    return img


# ── Image cleaning (exact port from notebook) ─────────────────────────────────

def remove_black_squares(image):
    """Replace isolated 2×2 black corner artifacts with white (from notebook)."""
    result = image.copy()
    H, W, _ = image.shape
    black = np.all(image == [0, 0, 0], axis=2)
    allowed = np.array([[255, 255, 255], [0, 255, 0], [255, 0, 0]])

    def _ok(sl):
        flat = sl.reshape(-1, 3)
        return all(np.any(np.all(px == allowed, axis=1)) for px in flat)

    for i in range(1, H - 2):
        for j in range(1, W - 2):
            if black[i, j] and black[i, j+1] and black[i+1, j] and black[i+1, j+1]:
                if (_ok(image[i-1, j-1:j+3]) and _ok(image[i+2, j-1:j+3]) and
                        _ok(image[i:i+2, j-1:j]) and _ok(image[i:i+2, j+2:j+3])):
                    result[i:i+2, j:j+2] = [255, 255, 255]
    return result


# ── Room-like floor plan rendering (matches notebook's get_original_complete) ──

def _build_complete(img_hwc):
    """(H,W,3) uint8 → (COMP_H,COMP_W,3) uint8 room-like image.

    Replicates notebook get_original_complete():
      remove_black_squares → crop 2-px border → resize 3600×2000 →
      detect black-wall contours, erode by INNER_GAP/2 px each side,
      draw eroded fills as black on white → architectural wall look.

    Crucially, we then RESTORE all originally-walkable pixels to white so that
    over-erosion never makes navigable corridors appear as walls (fixing the
    path-crosses-wall visual artefact seen in dense floor plans like inst 004).
    """
    img = remove_black_squares(img_hwc)
    img_crop = img[CROP:-CROP, CROP:-CROP]
    img_big = cv2.resize(img_crop, (COMP_W, COMP_H), interpolation=cv2.INTER_NEAREST)

    # Remember which pixels were walkable in the original encoding
    walkable_mask = np.any(img_big >= 128, axis=2)   # True = floor/corridor

    gray = cv2.cvtColor(img_big, cv2.COLOR_RGB2GRAY)
    _, binary_inv = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary_inv, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    output = np.ones((COMP_H, COMP_W, 3), dtype=np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (INNER_GAP, INNER_GAP))
    for cnt in contours:
        mask_c = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask_c, [cnt], -1, 255, thickness=-1)
        eroded = cv2.erode(mask_c, kernel, iterations=1)
        inner, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        if inner:
            cv2.drawContours(output, inner, -1, (0, 0, 0), thickness=cv2.FILLED)

    # Guarantee: every pixel that was walkable in the original stays white.
    # This prevents over-erosion from blacking-out narrow corridors.
    output[walkable_mask] = [255, 255, 255]

    cv2.rectangle(output, (0, 0), (COMP_W - 1, COMP_H - 1), (0, 0, 0), thickness=5)
    return output


def get_original_complete(inp):
    """(3,50,90) float32 → (COMP_H,COMP_W,3) room-like image (full floor plan)."""
    img = (inp.transpose(1, 2, 0).clip(0, 1) * 255).astype(np.uint8)
    return _build_complete(img)


def get_srp_complete(inp, G_srp, terminals):
    """Full floor plan with non-SRP areas darkened to show the reduced search space.

    Uses the full-resolution floor plan as background (avoiding the all-black
    artefact that occurs when eroding a very sparse G_srp image), then darkens
    every pixel that falls outside the G_srp graph to make the restricted
    search space visually apparent.
    """
    complete = get_original_complete(inp)   # full floor plan (COMP_H, COMP_W, 3)

    # Build SRP walkable mask at canvas resolution
    srp_img  = render_graph_to_image(G_srp, terminals)       # (50, 90, 3)
    srp_crop = srp_img[CROP:-CROP, CROP:-CROP]               # (46, 86, 3)
    srp_big  = cv2.resize(srp_crop, (COMP_W, COMP_H),
                          interpolation=cv2.INTER_NEAREST)    # (COMP_H, COMP_W, 3)
    srp_walkable = np.any(srp_big >= 128, axis=2)            # True = in G_srp

    # Darken non-SRP areas (walls stay black; non-SRP floors go dark gray)
    overlay = complete.astype(np.float32)
    overlay[~srp_walkable] *= 0.20
    overlay = overlay.clip(0, 255).astype(np.uint8)

    cv2.rectangle(overlay, (0, 0), (COMP_W - 1, COMP_H - 1), (0, 0, 0), thickness=5)
    return overlay


# ── Coordinate projection ──────────────────────────────────────────────────────

def node_to_comp(x, y):
    """Graph node (x,y) → (plot_x, plot_y) in the 3600×2000 canvas.

    Original (90-wide) px centre: col = 4x+2.5, row = 4y+2.5
    After 2-px crop:              col = 4x+0.5, row = 4y+0.5
    After resize ×SCALE:          plot_x = (4x+0.5)*SCALE_X,
                                  plot_y = (4y+0.5)*SCALE_Y
    """
    return (4 * x + 0.5) * SCALE_X, (4 * y + 0.5) * SCALE_Y


# ── Graph decoding ─────────────────────────────────────────────────────────────

def decode_graphs(inp, sol):
    ch0, ch1 = inp[0], inp[1]
    G_full, terminals = nx.Graph(), []

    for y in range(GRID_H):
        for x in range(GRID_W):
            r, c = 4 * y + 2, 4 * x + 2
            if ch1[r, c] > 0.5:
                is_term = ch0[r, c] < 0.5
                G_full.add_node((x, y), is_terminal=is_term)
                if is_term:
                    terminals.append((x, y))

    for y in range(GRID_H):
        for x in range(GRID_W):
            if (x, y) not in G_full:
                continue
            r, c = 4 * y + 2, 4 * x + 2
            if x + 1 < GRID_W and (x+1, y) in G_full and ch1[r, c+2] > 0.5:
                G_full.add_edge((x, y), (x+1, y))
            if y + 1 < GRID_H and (x, y+1) in G_full and ch1[r+2, c] > 0.5:
                G_full.add_edge((x, y), (x, y+1))

    G_srp = nx.Graph()
    for node, data in G_full.nodes(data=True):
        x, y = node
        r, c = 4 * y + 2, 4 * x + 2
        if sol[r, c] > 0.5 or data["is_terminal"]:
            G_srp.add_node(node, **data)
    for u, v in G_full.edges():
        if u not in G_srp or v not in G_srp:
            continue
        (x1, y1), (x2, y2) = u, v
        r, c = 4 * y1 + 2, 4 * x1 + 2
        cr, cc = (r, c + 2) if x2 == x1 + 1 else (r + 2, c)
        if sol[cr, cc] > 0.5:
            G_srp.add_edge(u, v)

    return G_full, G_srp, terminals


# ── PyBullet helpers ──────────────────────────────────────────────────────────

def node_to_world(x, y):
    """Graph node (x,y) → PyBullet world (wx, wy) in metres."""
    col = 4 * x + 2.5
    row = 4 * y + 2.5
    return col * SCALE, (IMG_H - 1 - row) * SCALE


def corridor_to_world(n1, n2):
    """Midpoint world coord of the 2-px corridor between adjacent nodes n1, n2."""
    (x1, y1), (x2, y2) = n1, n2
    c1, r1 = 4 * x1 + 2.5, 4 * y1 + 2.5
    c2, r2 = 4 * x2 + 2.5, 4 * y2 + 2.5
    return ((c1 + c2) / 2) * SCALE, (IMG_H - 1 - (r1 + r2) / 2) * SCALE


def build_wall_mesh(ch1, client):
    """Build the floor-plan walls as a single triangle-mesh rigid body.

    Row-by-row run-length encoding: each horizontal run of wall pixels
    becomes one AABB box, all merged into one mesh collision shape.
    """
    def _box_mesh(cx, cy, cz, hx, hy, hz):
        v = [
            [cx-hx, cy-hy, cz-hz], [cx+hx, cy-hy, cz-hz],
            [cx+hx, cy+hy, cz-hz], [cx-hx, cy+hy, cz-hz],
            [cx-hx, cy-hy, cz+hz], [cx+hx, cy-hy, cz+hz],
            [cx+hx, cy+hy, cz+hz], [cx-hx, cy+hy, cz+hz],
        ]
        t = [0,1,2, 0,2,3, 4,6,5, 4,7,6,
             0,5,1, 0,4,5, 2,6,3, 3,6,7,
             0,3,4, 3,7,4, 1,5,2, 2,5,6]
        return v, t

    all_verts, all_tris, offset = [], [], 0
    for r in range(IMG_H):
        c = 0
        while c < IMG_W:
            if ch1[r, c] < 0.5:
                c0 = c
                while c < IMG_W and ch1[r, c] < 0.5:
                    c += 1
                run_w = (c - c0) * SCALE
                cx_   = (c0 + (c - c0) / 2) * SCALE
                cy_   = (IMG_H - 1 - r) * SCALE
                vs, ts = _box_mesh(cx_, cy_, WALL_H / 2,
                                   run_w / 2, SCALE / 2, WALL_H / 2)
                all_verts.extend(vs)
                all_tris.extend([idx + offset for idx in ts])
                offset += 8
            else:
                c += 1

    cid  = p.createCollisionShape(p.GEOM_MESH,
                                  vertices=all_verts, indices=all_tris,
                                  physicsClientId=client)
    return p.createMultiBody(0, cid, -1, [0, 0, 0], physicsClientId=client)


def _has_wall_collision(robot, wall_body, client):
    contacts = p.getContactPoints(robot, wall_body, physicsClientId=client)
    return any(c[8] < PENETRATION_THRESHOLD for c in contacts)


def navigate_mission(robot, wall_body, terminals, ordered_edges, sp_paths, client):
    """Navigate each MST segment independently. Returns (success, distance_m).

    Each pairwise shortest-path segment is checked independently so DFS
    backtracking gaps never teleport the robot through walls.
    """
    total_m = 0.0
    for i, j in ordered_edges:
        key  = (min(i, j), max(i, j))
        path = sp_paths.get(key) or list(reversed(sp_paths.get((j, i), [])))
        if i > j:
            path = list(reversed(path))

        for k, node in enumerate(path):
            wx, wy = node_to_world(node[0], node[1])
            check_pts = [(wx, wy)]
            if k > 0:
                cwx, cwy = corridor_to_world(path[k - 1], node)
                check_pts = [(cwx, cwy), (wx, wy)]

            for px, py in check_pts:
                p.resetBasePositionAndOrientation(
                    robot, [px, py, ROBOT_R], [0, 0, 0, 1],
                    physicsClientId=client)
                p.performCollisionDetection(physicsClientId=client)
                if _has_wall_collision(robot, wall_body, client):
                    return False, total_m

            if k > 0:
                wx0, wy0 = node_to_world(path[k-1][0], path[k-1][1])
                total_m += math.hypot(wx - wx0, wy - wy0)

    return True, total_m


# ── Path planning (wavefront MST) ─────────────────────────────────────────────

def plan_mission(G, terminals):
    """Returns (ordered_edges, sp_paths, elapsed_ms, ok)."""
    n = len(terminals)
    metric, sp_paths = nx.Graph(), {}
    t0 = time.perf_counter()
    for i in range(n):
        for j in range(i + 1, n):
            try:
                length, path = nx.single_source_dijkstra(
                    G, terminals[i], terminals[j])
                metric.add_edge(i, j, weight=length)
                sp_paths[(i, j)] = path
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                return None, None, 0.0, False
    mst = nx.minimum_spanning_tree(metric, weight="weight")
    ordered = list(nx.dfs_edges(mst, source=0))
    elapsed_ms = (time.perf_counter() - t0) * 1e3
    return ordered, sp_paths, elapsed_ms, True


# ── Path drawing ───────────────────────────────────────────────────────────────

def draw_mission_paths(ax, terminals, ordered_edges, sp_paths,
                       color, linestyle, lw):
    """Draw each MST segment independently to avoid DFS-backtracking jumps.

    Each pairwise path is retrieved from sp_paths and converted to canvas
    coordinates.  Consecutive graph nodes are adjacent in the grid, so the
    straight line between their canvas projections passes through the
    corridor pixels — no explicit corridor midpoints are needed.
    """
    for i, j in ordered_edges:
        key  = (min(i, j), max(i, j))
        path = sp_paths.get(key) or list(reversed(sp_paths.get((j, i), [])))
        if len(path) < 2:
            continue
        xs = [node_to_comp(n[0], n[1])[0] for n in path]
        ys = [node_to_comp(n[0], n[1])[1] for n in path]
        ax.plot(xs, ys, linestyle=linestyle, color=color, linewidth=lw,
                alpha=0.90, zorder=3,
                solid_capstyle='round', solid_joinstyle='round')


def draw_terminals(ax, terminals, marker_size=200):
    for k, t in enumerate(terminals):
        px, py = node_to_comp(t[0], t[1])
        if k == 0:
            ax.scatter(px, py, s=marker_size * 1.5, marker='*',
                       c='yellow', edgecolors='black', linewidths=1.0,
                       zorder=6)
        else:
            ax.scatter(px, py, s=marker_size, marker='o',
                       c='lime', edgecolors='black', linewidths=0.8,
                       zorder=5)


# ── Per-instance figure ───────────────────────────────────────────────────────

def visualize_instance(idx, inp, sol, n_t, client=None, robot=None):
    """Returns a stats dict on success, None on failure."""
    G_full, G_srp, terminals = decode_graphs(inp, sol)
    if len(terminals) < 2:
        return None

    full_ord, full_sp, full_ms, full_ok = plan_mission(G_full, terminals)
    srp_ord,  srp_sp,  srp_ms,  srp_ok  = plan_mission(G_srp,  terminals)
    if not full_ok or not srp_ok:
        return None

    n_full  = G_full.number_of_nodes()
    n_srp   = G_srp.number_of_nodes()
    red_pct = 100.0 * (1.0 - n_srp / max(n_full, 1))
    speedup = full_ms / max(srp_ms, 1e-9)

    # ── PyBullet simulation ───────────────────────────────────────────────────
    full_nav_ok = srp_nav_ok = None
    if client is not None and robot is not None:
        wall_body   = build_wall_mesh(inp[1], client)
        full_nav_ok, _ = navigate_mission(
            robot, wall_body, terminals, full_ord, full_sp, client)
        srp_nav_ok,  _ = navigate_mission(
            robot, wall_body, terminals, srp_ord,  srp_sp,  client)
        p.removeBody(wall_body, physicsClientId=client)

    def _nav_tag(ok):
        if ok is None:
            return ""
        return "    ✓ Nav OK" if ok else "    ✗ Nav fail"

    # ── backgrounds: full (left) and SRP-masked (right) ──────────────────────
    complete_full = get_original_complete(inp)
    complete_srp  = get_srp_complete(inp, G_srp, terminals)

    # ── figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(18, 5.4), facecolor='#1a1a1a')
    fig.suptitle(
        f"Instance {idx}  |  {n_t} terminals  |  "
        f"Speedup {speedup:.1f}×  |  Search-space reduction {red_pct:.0f}%",
        fontsize=11, color='white', y=1.01)

    panel_data = [
        (complete_full, full_ord, full_sp, COL_FULL, '-',
         f"Full A*  ({n_full} nodes)",
         f"Plan: {full_ms:.1f} ms{_nav_tag(full_nav_ok)}"),
        (complete_srp,  srp_ord,  srp_sp,  COL_SRP,  '--',
         f"SRP Masked  ({n_srp} nodes,  −{red_pct:.0f}% search space)",
         f"Plan: {srp_ms:.1f} ms{_nav_tag(srp_nav_ok)}"),
    ]

    for ax, (bg, ord_, sp_, col, ls, title, stats_str) in zip(axes, panel_data):
        ax.imshow(bg, interpolation='bilinear', aspect='equal')
        ax.set_title(title, fontsize=10, fontweight='bold', color=col, pad=4)
        ax.axis('off')
        draw_mission_paths(ax, terminals, ord_, sp_, col, ls, LW_PATH)
        draw_terminals(ax, terminals)
        ax.text(0.01, 0.02, stats_str,
                transform=ax.transAxes, fontsize=9,
                color='white', va='bottom', ha='left',
                bbox=dict(boxstyle='round,pad=0.3', fc='black', alpha=0.6))

    handles = [
        mpatches.Patch(color=COL_FULL, label='Full A* path'),
        mpatches.Patch(color=COL_SRP,  label='SRP path'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='yellow',
                   markersize=10, label='Start terminal'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lime',
                   markersize=7,  label='Goal terminal'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=4,
               framealpha=0.3, fontsize=9, labelcolor='white',
               bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout(pad=0.5)
    out = OUT_DIR / f"instance_{idx:03d}_n{n_t}.png"
    plt.savefig(out, dpi=100, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)

    return dict(
        idx=idx,        n_terms=n_t,
        n_full=n_full,  n_srp=n_srp,
        reduction_pct=red_pct,
        full_plan_ms=full_ms,  srp_plan_ms=srp_ms,
        speedup=speedup,
        full_nav_ok=full_nav_ok,  srp_nav_ok=srp_nav_ok,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def print_summary(records):
    if not records:
        print("No records.")
        return

    def a(key):
        return np.array([r[key] for r in records], dtype=float)

    n         = len(records)
    red       = a("reduction_pct")
    full_ms   = a("full_plan_ms")
    srp_ms    = a("srp_plan_ms")
    speedup   = a("speedup")
    n_terms_a = a("n_terms").astype(int)
    has_nav   = any(r.get("full_nav_ok") is not None for r in records)
    if has_nav:
        full_nav = np.array([r["full_nav_ok"] for r in records], dtype=bool)
        srp_nav  = np.array([r["srp_nav_ok"]  for r in records], dtype=bool)

    sep = "=" * 62
    print(f"\n{sep}")
    print("  ZInD Robot Simulation — Runtime & Feasibility")
    print(f"  ({n} instances, same data as Tables 5-6-8-9)")
    print(sep)
    print(f"  Search-space reduction  : {red.mean():.1f} ± {red.std():.1f} %")
    print()
    print(f"  Full A* planning time   : {full_ms.mean():.2f} ± {full_ms.std():.2f} ms")
    print(f"  SRP   planning time     : {srp_ms.mean():.2f} ± {srp_ms.std():.2f} ms")
    print(f"  Planning speedup (SRP)  : {speedup.mean():.2f}× ± {speedup.std():.2f}")
    if has_nav:
        print()
        print(f"  Full A* nav success     : {full_nav.sum()} / {n}  ({100*full_nav.mean():.1f} %)")
        print(f"  SRP   nav success       : {srp_nav.sum()} / {n}  ({100*srp_nav.mean():.1f} %)")
    print()
    hdr = f"  {'N_terms':>7}  {'Count':>5}  {'Reduction%':>11}  {'Speedup×':>9}"
    hdr += f"  {'NavOK%':>7}" if has_nav else ""
    print(hdr)
    print("  " + "-"*7 + "  " + "-"*5 + "  " + "-"*11 + "  " + "-"*9
          + ("  " + "-"*7 if has_nav else ""))
    for nt in sorted(set(n_terms_a.tolist())):
        mask = n_terms_a == nt
        row  = (f"  {nt:>7}  {mask.sum():>5}  "
                f"{red[mask].mean():>10.1f}%  "
                f"{speedup[mask].mean():>9.2f}")
        if has_nav:
            row += f"  {100*srp_nav[mask].mean():>6.1f}%"
        print(row)
    print(sep)


def save_csv(records, path):
    if not records:
        return
    fields = list(records[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(records)
    print(f"  CSV saved → {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', action='store_true',
                        help='Visualize all 90 instances')
    parser.add_argument('--n',   type=int, default=10,
                        help='Number of instances (default 10)')
    parser.add_argument('--idx', type=int, nargs='+',
                        help='Specific indices')
    args = parser.parse_args()

    # ── clean output directory ────────────────────────────────────────────────
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)

    print(f"Loading {DATA_PATH} …")
    raw       = np.load(DATA_PATH, allow_pickle=True)
    inputs    = np.stack(raw[:, 1].tolist()).astype(np.float32)
    solutions = np.stack(raw[:, 2].tolist()).astype(np.float32)
    n_terms   = raw[:, 0].astype(int)
    N         = len(inputs)
    counts    = {int(nt): int((n_terms == nt).sum()) for nt in np.unique(n_terms)}
    print(f"  {N} instances: {counts}\n")

    if args.idx:
        indices = sorted(set(args.idx))
    elif args.all:
        indices = list(range(N))
    else:
        indices = []
        for nt in sorted(set(n_terms.tolist())):
            cands = [i for i in range(N) if n_terms[i] == nt]
            indices += cands[:max(1, args.n // len(set(n_terms.tolist())))]
        indices = sorted(indices[:args.n])

    # ── PyBullet setup (one client reused across all instances) ──────────────
    print("Initialising PyBullet (DIRECT / headless) …")
    pb_client = p.connect(p.DIRECT)
    p.setGravity(0, 0, 0, physicsClientId=pb_client)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=pb_client)
    p.loadURDF("plane.urdf", physicsClientId=pb_client)
    robot_shape = p.createCollisionShape(
        p.GEOM_SPHERE, radius=ROBOT_R, physicsClientId=pb_client)
    pb_robot = p.createMultiBody(
        1.0, robot_shape, -1, [0.1, 0.1, ROBOT_R], physicsClientId=pb_client)

    records = []
    ok = 0
    for idx in indices:
        stats = visualize_instance(idx, inputs[idx], solutions[idx],
                                   int(n_terms[idx]),
                                   client=pb_client, robot=pb_robot)
        if stats is not None:
            records.append(stats)
            ok += 1
            nf = "✓" if stats["full_nav_ok"] else "✗"
            ns = "✓" if stats["srp_nav_ok"]  else "✗"
            print(f"  [{ok:3d}] inst_{idx:03d}_n{n_terms[idx]}  "
                  f"full={stats['full_plan_ms']:.1f}ms nav={nf}  "
                  f"srp={stats['srp_plan_ms']:.1f}ms nav={ns}  "
                  f"speedup={stats['speedup']:.1f}×  reduction={stats['reduction_pct']:.0f}%")
        else:
            print(f"  [---] skipped {idx}")

    p.disconnect(pb_client)
    print(f"\n  Saved {ok}/{len(indices)} figures → {OUT_DIR}/")

    if records:
        print_summary(records)
        save_csv(records, OUT_DIR / "simulation_stats.csv")


if __name__ == "__main__":
    main()
