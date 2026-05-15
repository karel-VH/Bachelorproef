import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


SPEED_OF_LIGHT   = 299_792_458.0
FREQUENCY_START  = 2_402e6
FREQUENCY_STEP   = 1e6
N_CHANNELS_TOTAL = 75
CHANNELS_TO_SKIP = (21, 22, 23)
USED_CHANNELS    = [c for c in range(N_CHANNELS_TOTAL) if c not in CHANNELS_TO_SKIP]
N_USED_CHANNELS  = len(USED_CHANNELS)
CHAN_TO_POS      = {c: pos for pos, c in enumerate(USED_CHANNELS)}

CHANNEL_FREQS = np.array(
    [FREQUENCY_START + c * FREQUENCY_STEP for c in USED_CHANNELS],
    dtype=np.float64,
)

DIST_MIN = 0.5
DIST_MAX = 30.0


class DistanceCNN(nn.Module):
    def __init__(self, in_channels=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, 3, padding=1), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.30),
            nn.Linear(128, 64), nn.ReLU(inplace=True),
            nn.Dropout(0.30),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.head(self.features(x))


def normalise_iq(iq_complex):
    if iq_complex.ndim != 1:
        raise ValueError("normalise_iq verwacht een 1-D complex input")
    peak = float(np.max(np.abs(iq_complex)))
    if peak < 1e-12:
        peak = 1.0
    z = iq_complex / peak
    return np.stack([z.real, z.imag], axis=0).astype(np.float32)


def filter_outlier_tones(iq, mag_threshold_ratio=0.05):
    mag = np.abs(iq)
    if mag.max() < 1e-12:
        return iq
    median_mag = np.median(mag[mag > 1e-12])
    bad = mag < (mag_threshold_ratio * median_mag)
    if not bad.any():
        return iq
    out = iq.copy()
    n = len(iq)
    for i in np.where(bad)[0]:
        left = i - 1
        while left >= 0 and bad[left]:
            left -= 1
        right = i + 1
        while right < n and bad[right]:
            right += 1
        if left >= 0 and right < n:
            out[i] = 0.5 * (iq[left] + iq[right])
        elif left >= 0:
            out[i] = iq[left]
        elif right < n:
            out[i] = iq[right]
    return out


def aggregate_to_72(channels, i_arr, q_arr):
    out = np.zeros(N_USED_CHANNELS, dtype=np.complex128)
    iq_full = i_arr + 1j * q_arr
    for ch_id in np.unique(channels):
        ch = int(ch_id)
        if ch in CHAN_TO_POS:
            out[CHAN_TO_POS[ch]] = iq_full[channels == ch_id].mean()
    return out.astype(np.complex64)


def dist_to_target(d):
    return (np.log(d) - math.log(DIST_MIN)) / (math.log(DIST_MAX) - math.log(DIST_MIN))


def target_to_dist(t):
    if isinstance(t, torch.Tensor):
        return torch.exp(t * (math.log(DIST_MAX) - math.log(DIST_MIN)) + math.log(DIST_MIN))
    return np.exp(t * (math.log(DIST_MAX) - math.log(DIST_MIN)) + math.log(DIST_MIN))


def enable_mc_dropout(model):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def mc_dropout_predict(model, x, n_passes=15):
    model.eval()
    enable_mc_dropout(model)
    predictions = []
    with torch.no_grad():
        for _ in range(n_passes):
            predictions.append(model(x).cpu().numpy().ravel())
    model.eval()
    return np.median(np.stack(predictions, axis=0), axis=0)


def save_distance_boxplot(true_d, pred_d, title, out_path, figsize=(11, 6)):
    import matplotlib.pyplot as plt

    unique_distances = sorted(set(np.round(true_d, 2)))
    data_per_d = [pred_d[np.isclose(true_d, d, atol=0.05)] for d in unique_distances]

    fig, ax = plt.subplots(figsize=figsize)
    ax.boxplot(
        data_per_d,
        positions=unique_distances,
        widths=0.5,
        showfliers=True,
        flierprops=dict(marker="o", markersize=3, alpha=0.4),
    )

    lim_lo = 0.0
    lim_hi = max(max(unique_distances) + 2, float(pred_d.max()) + 1)
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", lw=1.2, label="ideaal (y = x)")
    ax.set_xticks(unique_distances)
    ax.set_xticklabels([f"{d:g}" for d in unique_distances])
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_xlabel("werkelijke afstand (m)")
    ax.set_ylabel("voorspelde afstand (m)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)