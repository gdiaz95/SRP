#!/usr/bin/env python3
"""gif_tc_failure_8term.py

For 8-terminal failure cases, save a three-panel PNG:
  Panel 1 — soft intensity at iter 10  (not yet converged)
  Panel 2 — soft intensity at iter 59  (no TC, failure case)
  Panel 3 — TC output (binary, white paths)

Labels at the top of each panel. No other text.

Usage (from project root):
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/gif_tc_failure_8term.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepthinking.models.dt_net_2d_parallel import DTNet, BasicBlock
from deepthinking.utils.testing import apply_mask, check_termination_multiple

# ── Config ─────────────────────────────────────────────────────────────────────

MODEL_PATH    = "outputs/training_default/training-homeless-Sok/model_best.pth"
DATA_DIR      = "data/8_green/maze_data_test_21"
OUT_DIR       = Path("images/tc_failure_8term")
ITER_EARLY    = 10
ITER_LATE     = 59
TC_THRESHOLD  = 0.65
TC_FIRST      = 20
TC_BATCH      = 10
TC_MAX_ROUNDS = 5
N_IMAGES      = 2    # save only the 2nd and 3rd TC failures (skip first)
SCALE         = 10    # 48 → 480 px
GAP           = 20    # white space between panels
PADDING       = 16    # white border left/right/bottom
LABEL_H       = 52
FONT_PATH     = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_SIZE     = 28


# ── Model ──────────────────────────────────────────────────────────────────────

def load_model(path, device):
    state = torch.load(path, map_location=device, weights_only=False)
    net_state = state.get("net", state)
    clean = {k.replace("module.", ""): v for k, v in net_state.items()}
    proj_key = next((k for k in clean if "projection" in k and "weight" in k), None)
    width = clean[proj_key].shape[0] if proj_key else 128
    net = DTNet(BasicBlock, [2], width=width, in_channels=3, recall=True)
    net.load_state_dict(clean, strict=True)
    return net.to(device).eval()


def run_iter_then_tc(net, inp_np, n_pre_iters, device):
    """
    Run n_pre_iters iterations, capture intensity, then continue with TC
    from that same hidden state.
    Returns (intensity (H,W) float32, tc_binary (H,W) uint8).
    """
    x = torch.from_numpy(inp_np.astype(np.float32)).unsqueeze(0).to(device)
    passable = (inp_np[0] > 0.5) | (inp_np[1] > 0.5)

    with torch.no_grad():
        # ── phase 1: run to n_pre_iters ──────────────────────────────────────
        all_out, interims = net(x, iters_to_do=n_pre_iters)
        logits    = all_out[0, -1]
        soft      = torch.softmax(logits, dim=0)[1].cpu().numpy().clip(0, 1)
        intensity = (soft * passable).astype(np.float32)

        # ── phase 2: continue with TC from the same hidden state ─────────────
        final = None
        for rnd in range(TC_MAX_ROUNDS + 1):
            n = TC_BATCH
            all_out, interims = net(x, n, interims)
            sm = apply_mask(x, all_out)
            try:
                target, finish, _ = check_termination_multiple(sm, TC_THRESHOLD)
            except Exception:
                finish = False; target = None
            if finish or rnd == TC_MAX_ROUNDS:
                final = target; break

    if final is not None:
        tc_bin = (final.cpu().squeeze().numpy() > 0.5).astype(np.uint8)
    else:
        tc_bin = all_out[0, -1].argmax(dim=0).cpu().numpy().astype(np.uint8)

    return intensity, tc_bin


def notc_binary_at(net, inp_np, n_iter, device):
    """Binary argmax at exactly n_iter iters (for failure detection)."""
    x = torch.from_numpy(inp_np.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        all_out, _ = net(x, iters_to_do=n_iter)
    return all_out[0, -1].argmax(dim=0).cpu().numpy().astype(np.uint8)


# ── Rendering ──────────────────────────────────────────────────────────────────

def render_intensity(inp_np, intensity):
    """
    (3,H,W) float input + (H,W) float intensity → (H,W,3) uint8.
    Walls = black. Free space = intensity as white brightness.
    Terminals = green on top.
    """
    H, W = inp_np.shape[1], inp_np.shape[2]
    out = np.zeros((H, W, 3), dtype=np.uint8)
    passable = (inp_np[0] > 0.5) | (inp_np[1] > 0.5)
    # free space floor = 70 (always visible grey), path peak = 255 (white)
    brightness = (intensity.clip(0, 1) * 185 + 70).astype(np.uint8)
    out[..., 0][passable] = brightness[passable]
    out[..., 1][passable] = brightness[passable]
    out[..., 2][passable] = brightness[passable]
    # terminals in green
    terminals = (inp_np[1] > 0.5) & (inp_np[0] < 0.5)
    out[terminals] = [0, 220, 80]
    return out


def render_binary(inp_np, pred):
    """Binary prediction: predicted path = white, terminals = green, walls = black."""
    H, W = inp_np.shape[1], inp_np.shape[2]
    out = np.zeros((H, W, 3), dtype=np.uint8)
    passable = (inp_np[0] > 0.5) | (inp_np[1] > 0.5)
    out[passable] = [45, 45, 45]
    out[pred > 0] = [255, 255, 255]
    terminals = (inp_np[1] > 0.5) & (inp_np[0] < 0.5)
    out[terminals] = [0, 220, 80]
    return out


def upscale(arr, scale):
    return arr.repeat(scale, axis=0).repeat(scale, axis=1)


def add_top_label(img_array, text, text_color=(0, 0, 0), bg=(255, 255, 255)):
    """Add a bold black text label banner on white background at the TOP."""
    img = Image.fromarray(img_array)
    w, h = img.size
    new = Image.new("RGB", (w, h + LABEL_H), color=bg)
    new.paste(img, (0, LABEL_H))
    draw = ImageDraw.Draw(new)
    try:
        font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text(((w - tw) // 2, (LABEL_H - th) // 2), text, fill=text_color, font=font)
    return np.array(new)


def make_figure(inp_np, sol_np, intensity_59, pred_tc, out_path):
    """Three panels: Correct Solution | Before TC | After TC."""
    gt_binary = (sol_np > 0.5).astype(np.uint8)
    p1 = add_top_label(upscale(render_binary(inp_np, gt_binary),       SCALE), "Correct Solution")
    p2 = add_top_label(upscale(render_intensity(inp_np, intensity_59), SCALE), "Before TC")
    p3 = add_top_label(upscale(render_binary(inp_np, pred_tc),         SCALE), "After TC")

    H = max(p1.shape[0], p2.shape[0], p3.shape[0])
    W = p1.shape[1] + GAP + p2.shape[1] + GAP + p3.shape[1]
    canvas = np.full((H + PADDING, W + 2*PADDING, 3), 255, dtype=np.uint8)
    x = PADDING
    for panel in [p1, p2, p3]:
        canvas[:panel.shape[0], x:x + panel.shape[1]] = panel
        x += panel.shape[1] + GAP
    Image.fromarray(canvas).save(out_path)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    inputs    = np.load(Path(DATA_DIR) / "inputs.npy")
    solutions = np.load(Path(DATA_DIR) / "solutions.npy")
    print(f"Loaded {len(inputs)} samples  |  device: {device}")

    net = load_model(MODEL_PATH, device)

    found = 0   # count TC failures; skip first, save next N_IMAGES
    saved = 0
    for i in range(len(inputs)):
        if saved >= N_IMAGES:
            break
        inp = inputs[i]
        sol = solutions[i]
        n_gt = int((sol > 0.5).sum())

        intensity_59, tc_59 = run_iter_then_tc(net, inp, ITER_LATE, device)
        if tc_59.sum() == n_gt:
            continue   # TC correct — skip

        found += 1
        if found == 1:
            continue   # skip the first failure case

        tag = f"failure_{saved+1:02d}_s{i:04d}"
        make_figure(inp, sol, intensity_59, tc_59, OUT_DIR / f"{tag}_fig59.png")
        print(f"  saved {tag}  (TC@59 px={tc_59.sum()}, GT={n_gt})")
        saved += 1

    print(f"\nDone — {saved} images saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
