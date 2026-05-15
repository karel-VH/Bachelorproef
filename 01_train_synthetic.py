import csv
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from common import (
    DistanceCNN,
    normalise_iq, dist_to_target, target_to_dist,
    DIST_MIN, DIST_MAX,
    N_USED_CHANNELS, CHANNELS_TO_SKIP,
    FREQUENCY_START, FREQUENCY_STEP, SPEED_OF_LIGHT,
)
from synthetic import generate_synthetic_iq

ROOT    = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)


class SyntheticCSDataset(Dataset):
    def __init__(self, n_samples, seed=0):
        self.n_samples = n_samples
        self._seed = seed

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        rng = np.random.default_rng(self._seed * 1_000_003 + idx)
        d = float(rng.uniform(DIST_MIN, DIST_MAX))
        iq = generate_synthetic_iq(d, rng)
        x = normalise_iq(iq)
        y = np.array([dist_to_target(d)], dtype=np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)


def main():
    N_TRAIN      = 120_000
    N_VAL        =  12_000
    BATCH_SIZE   =     256
    EPOCHS       =      60
    LR           =    1e-3
    WEIGHT_DECAY =    1e-5
    SEED         =      42

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device = {device}")
    print(f"[info] training distance range: [{DIST_MIN}, {DIST_MAX}] m")
    print(f"[info] N_TRAIN={N_TRAIN}  N_VAL={N_VAL}  BATCH={BATCH_SIZE}  EPOCHS={EPOCHS}")

    train_ds = SyntheticCSDataset(N_TRAIN, seed=SEED)
    val_ds   = SyntheticCSDataset(N_VAL,   seed=SEED + 9999)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    model     = DistanceCNN().to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=EPOCHS)
    criterion = nn.HuberLoss(delta=0.05)

    print(f"[info] CNN parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_log = []
    best_val = math.inf

    for epoch in range(1, EPOCHS + 1):
        model.train()
        sum_loss, n_seen = 0.0, 0
        t0 = time.time()
        for x, y in train_dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimiser.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()
            sum_loss += loss.item() * x.size(0)
            n_seen   += x.size(0)
        train_loss = sum_loss / max(n_seen, 1)

        model.eval()
        sum_loss, n_seen = 0.0, 0
        all_pred_d, all_true_d = [], []
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                sum_loss += criterion(pred, y).item() * x.size(0)
                n_seen   += x.size(0)
                all_pred_d.append(target_to_dist(pred.cpu().numpy()).ravel())
                all_true_d.append(target_to_dist(y.cpu().numpy()).ravel())
        val_loss = sum_loss / max(n_seen, 1)
        all_pred_d = np.concatenate(all_pred_d)
        all_true_d = np.concatenate(all_true_d)
        val_mae   = float(np.mean(  np.abs(all_pred_d - all_true_d)))
        val_medae = float(np.median(np.abs(all_pred_d - all_true_d)))
        val_bias  = float(np.mean(       all_pred_d - all_true_d))

        scheduler.step()
        dt = time.time() - t0
        print(f"[epoch {epoch:3d}/{EPOCHS}] train={train_loss:.4f}  val={val_loss:.4f}  "
              f"MAE={val_mae:5.2f} m  MedAE={val_medae:4.2f} m  "
              f"bias={val_bias:+5.2f} m  ({dt:5.1f}s)")
        train_log.append(dict(
            epoch=epoch, train_loss=train_loss, val_loss=val_loss,
            val_mae_m=val_mae, val_medae_m=val_medae, val_bias_m=val_bias,
            lr=optimiser.param_groups[0]["lr"],
        ))

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": {
                    "dist_min":           DIST_MIN,
                    "dist_max":           DIST_MAX,
                    "n_used_channels":    N_USED_CHANNELS,
                    "channels_to_skip":   list(CHANNELS_TO_SKIP),
                    "frequency_start_hz": FREQUENCY_START,
                    "frequency_step_hz":  FREQUENCY_STEP,
                    "speed_of_light":     SPEED_OF_LIGHT,
                    "in_channels":        2,
                    "epoch":              epoch,
                    "val_loss":           val_loss,
                    "val_mae_m":          val_mae,
                    "val_medae_m":        val_medae,
                    "val_bias_m":         val_bias,
                },
            }, OUT_DIR / "cnn_distance_model.pt")

        _save_log_and_curves(train_log, OUT_DIR, best_val)

    _save_log_and_curves(train_log, OUT_DIR, best_val)
    print(f"[done] best model -> {OUT_DIR / 'cnn_distance_model.pt'}  (val Huber = {best_val:.4f})")


def _save_log_and_curves(train_log, out_dir, best_val):
    if not train_log:
        return

    with open(out_dir / "training_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(train_log[0].keys()))
        w.writeheader()
        w.writerows(train_log)

    epochs   = [r["epoch"]       for r in train_log]
    tr_loss  = [r["train_loss"]  for r in train_log]
    va_loss  = [r["val_loss"]    for r in train_log]
    va_mae   = [r["val_mae_m"]   for r in train_log]
    va_medae = [r["val_medae_m"] for r in train_log]
    va_bias  = [r["val_bias_m"]  for r in train_log]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, tr_loss, label="train")
    axes[0].plot(epochs, va_loss, label="val")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("Huber loss")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, va_mae,   label="val MAE")
    axes[1].plot(epochs, va_medae, label="val MedAE")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("error (m)")
    axes[1].set_title("Validation error"); axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].plot(epochs, va_bias, color="tab:red", label="val bias")
    axes[2].axhline(0, color="k", lw=0.6)
    axes[2].set_xlabel("epoch"); axes[2].set_ylabel("bias (m)")
    axes[2].set_title("Validation bias (predicted - true)")
    axes[2].legend(); axes[2].grid(alpha=0.3)

    fig.suptitle(f"Training -- distance range [{DIST_MIN}, {DIST_MAX}] m   "
                 f"(epoch {train_log[-1]['epoch']}, best validation Huber = {best_val:.4f})")
    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()