#!/usr/bin/env python3
"""train_unet_baseline.py

Train a simple U-Net segmentation baseline on the same synthetic maze data
as homeless-Sok (mixed_2_3_4, 500k mazes, 2-4 terminals) and evaluate on
the same test splits (2–8 terminals) using the same accuracy metric as Table 1.

Usage (from project root):
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/train_unet_baseline.py

Saves checkpoint to outputs/unet_baseline/model_best.pth
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

# ── Config ─────────────────────────────────────────────────────────────────────

TRAIN_DATA_PATH = "/mnt/drive2/gabriel_data/Lydia/train/mixed_2_3_4/maze_data_train_21"
TEST_DATA_ROOT   = "data"
VAL_DATA_TYPE    = "5_green"          # validation set (same as homeless-Sok)
TEST_DATA_TYPES  = ["2_green", "3_green", "4_green",
                    "5_green", "6_green", "7_green", "8_green"]

OUT_DIR    = Path("outputs/unet_baseline")
EPOCHS     = 20
BATCH_SIZE = 25
LR         = 1e-3
VAL_SPLIT  = 0.2
SEED       = 42

# accuracy threshold: same as Table 1 (correct if n_pred <= n_gt + 8)
ACC_MARGIN = 8

# ── Dataset ────────────────────────────────────────────────────────────────────

class MazeNpy(Dataset):
    def __init__(self, inp_path, sol_path):
        self.inputs  = torch.from_numpy(np.load(inp_path).astype(np.float32))
        sols         = np.load(sol_path).astype(np.float32)
        self.targets = torch.from_numpy(sols).unsqueeze(1)   # (N,1,H,W)

    def __len__(self):  return len(self.inputs)
    def __getitem__(self, i): return self.inputs[i], self.targets[i]


# ── U-Net ──────────────────────────────────────────────────────────────────────

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_ch=3, base=32):
        super().__init__()
        # Encoder
        self.enc1 = DoubleConv(in_ch, base)       # 48×48
        self.enc2 = DoubleConv(base,   base*2)    # 24×24
        self.enc3 = DoubleConv(base*2, base*4)    # 12×12
        self.bot  = DoubleConv(base*4, base*8)    # 6×6
        self.pool = nn.MaxPool2d(2)
        # Decoder
        self.up3  = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.dec3 = DoubleConv(base*8, base*4)
        self.up2  = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.dec2 = DoubleConv(base*4, base*2)
        self.up1  = nn.ConvTranspose2d(base*2, base,   2, stride=2)
        self.dec1 = DoubleConv(base*2, base)
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b  = self.bot(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b),  e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)   # logits (N,1,H,W)


# ── Loss (BCE + Dice) ──────────────────────────────────────────────────────────

def dice_loss(pred, target, eps=1e-6):
    pred   = torch.sigmoid(pred)
    inter  = (pred * target).sum(dim=(2, 3))
    union  = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return 1 - (2 * inter + eps) / (union + eps)


def loss_fn(logits, target):
    bce  = F.binary_cross_entropy_with_logits(logits, target)
    dice = dice_loss(logits, target).mean()
    return bce + dice


# ── Accuracy (same metric as Table 1) ─────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for inp, tgt in loader:
        inp, tgt = inp.to(device), tgt.to(device)
        logits = model(inp)
        pred   = (torch.sigmoid(logits) > 0.5).float()
        n_pred = pred.sum(dim=(1, 2, 3))
        n_gt   = tgt.sum(dim=(1, 2, 3))
        correct += (n_pred <= n_gt + ACC_MARGIN).sum().item()
        total   += inp.size(0)
    return 100.0 * correct / total


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    print(f"Loading training data from {TRAIN_DATA_PATH}…")
    full_ds = MazeNpy(
        os.path.join(TRAIN_DATA_PATH, "inputs.npy"),
        os.path.join(TRAIN_DATA_PATH, "solutions.npy"),
    )
    n_val   = int(VAL_SPLIT * len(full_ds))
    n_train = len(full_ds) - n_val
    gen     = torch.Generator().manual_seed(SEED)
    train_ds, _ = random_split(full_ds, [n_train, n_val], generator=gen)

    # Use 5-terminal test set as validation (same as homeless-Sok)
    val_ds = MazeNpy(
        os.path.join(TEST_DATA_ROOT, VAL_DATA_TYPE, "maze_data_test_21", "inputs.npy"),
        os.path.join(TEST_DATA_ROOT, VAL_DATA_TYPE, "maze_data_test_21", "solutions.npy"),
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=100, shuffle=False,
                              num_workers=2, pin_memory=True)

    print(f"Train: {len(train_ds):,}  |  Val (5-term): {len(val_ds):,}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = UNet(in_ch=3, base=32).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"U-Net parameters: {n_params/1e6:.3f}M")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    best_val = 0.0
    log_path = OUT_DIR / "train.log"

    with open(log_path, "w") as logf:
        logf.write(f"U-Net baseline  |  params={n_params/1e6:.3f}M  |  device={device}\n\n")

    for epoch in range(EPOCHS):
        model.train()
        t0   = time.time()
        loss_sum = n_batches = 0

        for inp, tgt in train_loader:
            inp, tgt = inp.to(device), tgt.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(inp), tgt)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            n_batches += 1

        scheduler.step()
        train_loss = loss_sum / n_batches
        val_acc    = evaluate(model, val_loader, device)
        elapsed    = time.time() - t0

        msg = (f"Epoch {epoch:2d}  loss={train_loss:.4f}  "
               f"val_acc(5-term)={val_acc:.2f}%  [{elapsed:.0f}s]")
        print(msg, flush=True)
        with open(log_path, "a") as logf:
            logf.write(msg + "\n")

        if val_acc > best_val:
            best_val = val_acc
            torch.save({"net": model.state_dict(), "epoch": epoch,
                        "val_acc": val_acc}, OUT_DIR / "model_best.pth")
            print(f"  → saved best  ({best_val:.2f}%)")

    # ── Final evaluation on all test splits ───────────────────────────────────
    print(f"\nLoading best model (val_acc={best_val:.2f}%)…")
    ckpt = torch.load(OUT_DIR / "model_best.pth", map_location=device)
    model.load_state_dict(ckpt["net"])

    print("\n" + "="*60)
    print("  U-Net baseline — Table 1 comparison")
    print("="*60)
    results = {}
    for dt in TEST_DATA_TYPES:
        ds = MazeNpy(
            os.path.join(TEST_DATA_ROOT, dt, "maze_data_test_21", "inputs.npy"),
            os.path.join(TEST_DATA_ROOT, dt, "maze_data_test_21", "solutions.npy"),
        )
        loader = DataLoader(ds, batch_size=100, shuffle=False, num_workers=2)
        acc = evaluate(model, loader, device)
        n   = dt.split("_")[0]
        results[n] = acc
        print(f"  {dt:12s}: {acc:.2f}%")

    splits = [dt.split("_")[0] for dt in TEST_DATA_TYPES]
    print("\n" + "="*60)
    print(f"  {'':15s}" + "".join(f"{s:>8s}" for s in splits))
    print(f"  {'TC module':15s}" + "".join(f"{'100.00':>8s}" for _ in splits))
    print(f"  {'No TC (59 itr)':15s}" +
          "99.40  96.13  89.06  80.40  68.30  55.00  59.00".replace("  ", "      "))
    print(f"  {'U-Net':15s}" + "".join(f"{results[s]:8.2f}" for s in splits))
    print("="*60)

    # save results
    np.save(OUT_DIR / "test_accuracies.npy", results)
    print(f"\nSaved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
