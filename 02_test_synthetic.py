import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from common import (
    DistanceCNN,
    normalise_iq, filter_outlier_tones, target_to_dist,
    mc_dropout_predict,
    DIST_MIN, DIST_MAX, N_USED_CHANNELS,
)
from synthetic import generate_synthetic_iq

ROOT       = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "outputs" / "cnn_distance_model.pt"
OUT_DIR    = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)


def main():
    if not MODEL_PATH.exists():
        raise SystemExit(f"Model not found: {MODEL_PATH}\nRun 01_train_synthetic.py first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device = {device}")

    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    cfg  = ckpt.get("config", {})
    print(f"[info] loaded checkpoint epoch {cfg.get('epoch')}, "
          f"validation MAE = {cfg.get('val_mae_m', float('nan')):.2f} m")

    model = DistanceCNN().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    N_TEST = 4_000
    SEED   = 20260427
    N_MC   = 15
    BATCH  = 512
    rng    = np.random.default_rng(SEED)

    true_d = rng.uniform(DIST_MIN, DIST_MAX, size=N_TEST).astype(np.float64)
    X = np.empty((N_TEST, 2, N_USED_CHANNELS), dtype=np.float32)
    for i, d in enumerate(true_d):
        iq = generate_synthetic_iq(d, rng)
        iq = filter_outlier_tones(iq)
        X[i] = normalise_iq(iq)

    pred_d = np.empty(N_TEST, dtype=np.float64)
    for s in range(0, N_TEST, BATCH):
        xb = torch.from_numpy(X[s:s + BATCH]).to(device)
        t  = mc_dropout_predict(model, xb, n_passes=N_MC)
        pred_d[s:s + BATCH] = target_to_dist(t)

    err     = pred_d - true_d
    abs_err = np.abs(err)
    mae   = float(np.mean(abs_err))
    medae = float(np.median(abs_err))
    p90   = float(np.percentile(abs_err, 90))

    print(f"[result] N={N_TEST}  range=[{DIST_MIN}, {DIST_MAX}] m  "
          f"bias={err.mean():+.2f} m  MAE={mae:.2f} m  "
          f"MedAE={medae:.2f} m  P90={p90:.2f} m")

    csv_path = OUT_DIR / "test_synthetic.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "true_distance_m", "predicted_distance_m",
                    "abs_error_m", "rel_error_pct"])
        for i in range(N_TEST):
            rel_pct = abs_err[i] / max(true_d[i], 1e-6) * 100.0
            w.writerow([i, f"{true_d[i]:.4f}", f"{pred_d[i]:.4f}",
                        f"{abs_err[i]:.4f}", f"{rel_pct:.3f}"])
    print(f"[info] CSV  -> {csv_path}")

    centres = list(range(1, int(round(DIST_MAX)) + 1))
    data_per_bin = []
    for c in centres:
        mask = (true_d >= c - 0.5) & (true_d < c + 0.5)
        data_per_bin.append(pred_d[mask] if mask.any() else np.array([np.nan]))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.boxplot(data_per_bin, positions=centres, widths=0.6,
               showfliers=True,
               flierprops=dict(marker="o", markersize=3, alpha=0.4))
    ax.plot([0, DIST_MAX + 1], [0, DIST_MAX + 1], "k--", lw=1.2, label="ideaal (y = x)")
    ax.set_xticks(centres)
    ax.set_xticklabels([str(c) for c in centres])
    ax.set_xlim(0, DIST_MAX + 1)
    ax.set_ylim(max(0, DIST_MIN - 1), DIST_MAX + 5)
    ax.set_xlabel("werkelijke afstand bin-centrum (m)")
    ax.set_ylabel("voorspelde afstand (m)")
    ax.set_title(f"CNN-evaluatie op synthetische testset (N = {N_TEST}, MC-dropout passes = {N_MC})\n"
                 f"bias = {err.mean():+.2f} m, MAE = {mae:.2f} m, MedAE = {medae:.2f} m")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()

    out_png = OUT_DIR / "test_synthetic.png"
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"[info] PNG  -> {out_png}")
    print("[done]")


if __name__ == "__main__":
    main()