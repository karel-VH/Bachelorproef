import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

from common import (
    DistanceCNN,
    normalise_iq, filter_outlier_tones, target_to_dist,
    mc_dropout_predict, save_distance_boxplot,
    aggregate_to_72,
    N_USED_CHANNELS,
)

ROOT             = Path(__file__).resolve().parent
MODEL_PATH       = ROOT / "outputs" / "cnn_distance_model.pt"
OUT_DIR          = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)
DEFAULT_DATA_DIR = ROOT / "real_data" / "2x2_72"

_DIST_RE = re.compile(r"_rd_(\d+)[._](\d+)[._]iq", re.IGNORECASE)


def distance_from_filename(path):
    m = _DIST_RE.search(path.name)
    return float(f"{m.group(1)}.{m.group(2)}") if m else None


def trimmed_mean_4(values):
    #drop min en max van de 4 paden, gemiddelde van de twee middelste
    s = np.sort(values, axis=-1)
    return s[..., 1:3].mean(axis=-1)


def parse_jsonl_2x2_72(path):
    with open(path, "r") as f:
        meta = json.loads(f.readline())
        if meta.get("type") != "meta":
            raise ValueError(f"{path.name}: first line is not metadata")

        cfg = meta.get("config", {})
        if cfg.get("antenna_configuration_string") != "2x2":
            print(f"[warn] {path.name}: antenna configuration is "
                  f"{cfg.get('antenna_configuration_string')!r}, expected '2x2'")

        d_meta = meta.get("real_distance")
        d_file = distance_from_filename(path)
        d_true = float(d_meta if d_meta is not None else d_file)

        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("type") == "meta":
                continue
            if "i" not in obj or "q" not in obj:
                continue

            i_lst, q_lst = obj["i"], obj["q"]
            channels = np.asarray(obj["channels"], dtype=np.int64)
            if not isinstance(i_lst, list) or len(i_lst) != 4:
                continue

            iq_paths = np.empty((4, N_USED_CHANNELS), dtype=np.complex64)
            for p in range(4):
                i_arr = np.asarray(i_lst[p], dtype=np.float64)
                q_arr = np.asarray(q_lst[p], dtype=np.float64)
                iq_paths[p] = aggregate_to_72(channels, i_arr, q_arr)
            yield d_true, iq_paths


def main():
    data_dir = Path(sys.argv[1]) if len(sys.argv) >= 2 else DEFAULT_DATA_DIR
    if not data_dir.is_dir():
        raise SystemExit(f"Data folder not found: {data_dir}")
    if not MODEL_PATH.exists():
        raise SystemExit(f"Model not found: {MODEL_PATH}\nRun 01_train_synthetic.py first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device = {device}")

    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model = DistanceCNN().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    files = sorted(p for p in data_dir.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"No .jsonl files found in {data_dir}")

    chosen = []
    for p in files:
        try:
            with open(p, "r") as f:
                meta  = json.loads(f.readline())
                line2 = f.readline().strip()
                if not line2:
                    continue
                obj = json.loads(line2)
            if meta.get("config", {}).get("antenna_configuration_string") != "2x2":
                continue
            i_lst = obj["i"]
            if (isinstance(i_lst, list) and len(i_lst) == 4
                    and len(i_lst[0]) == N_USED_CHANNELS):
                chosen.append(p)
        except Exception as e:
            print(f"[warn] could not read {p.name}: {e}")
    if not chosen:
        raise SystemExit(f"No 2x2 / 72-tone .jsonl files in {data_dir}")
    print(f"[info] found {len(chosen)} 2x2 / 72-tone files in {data_dir}")

    N_MC  = 15
    BATCH = 256
    rows  = []

    for f_path in chosen:
        measurements = list(parse_jsonl_2x2_72(f_path))
        n = len(measurements)
        if n == 0:
            print(f"[warn] {f_path.name}: 0 measurements")
            continue

        true_d  = np.array([d for d, _ in measurements], dtype=np.float64)
        iq_all  = np.stack([iq for _, iq in measurements], axis=0)
        flat    = iq_all.reshape(n * 4, N_USED_CHANNELS)
        Xflat   = np.stack([normalise_iq(filter_outlier_tones(z)) for z in flat], axis=0)

        pred_flat = np.empty(n * 4, dtype=np.float64)
        for s in range(0, Xflat.shape[0], BATCH):
            xb = torch.from_numpy(Xflat[s:s + BATCH]).to(device)
            t  = mc_dropout_predict(model, xb, n_passes=N_MC)
            pred_flat[s:s + BATCH] = target_to_dist(t)

        pred_per_path = pred_flat.reshape(n, 4)
        pred_final    = trimmed_mean_4(pred_per_path)

        abs_err = np.abs(pred_final - true_d)
        print(f"[file] {f_path.name}  d_true={true_d[0]:.1f} m  N={n}  "
              f"pred_med={np.median(pred_final):.2f} m  "
              f"raw bias={(pred_final - true_d).mean():+.2f} m  "
              f"MedAE={np.median(abs_err):.2f} m")

        for i in range(n):
            rows.append((f_path.name, i, float(true_d[i]),
                         float(pred_final[i]),
                         *pred_per_path[i].tolist()))

    if not rows:
        raise SystemExit("No measurements processed.")

    all_true = np.array([r[2] for r in rows])
    all_pred = np.array([r[3] for r in rows])

    ref_distance = float(np.min(all_true))
    ref_mask     = np.isclose(all_true, ref_distance, atol=0.05)
    bias_calib   = float(np.median(all_pred[ref_mask] - all_true[ref_mask]))
    all_pred_cal = all_pred - bias_calib

    print()
    print(f"[calib] reference distance       = {ref_distance:.1f} m")
    print(f"[calib] one-point bias correction = {bias_calib:+.2f} m")

    print()
    print(f"{'d_true':>8} | {'N':>4} | {'raw pred':>10} {'raw bias':>10} {'raw MAE':>10}"
          f" | {'cal pred':>10} {'cal bias':>10} {'cal MAE':>10}")
    print('-' * 100)
    for d in sorted(set(np.round(all_true, 2))):
        m = np.isclose(all_true, d, atol=0.05)
        if not m.any():
            continue
        p, p_c = all_pred[m], all_pred_cal[m]
        print(f"{d:>8.2f} | {m.sum():>4d} | "
              f"{np.median(p):>9.2f}m {(p - d).mean():>+9.2f}m "
              f"{np.mean(np.abs(p - d)):>9.2f}m | "
              f"{np.median(p_c):>9.2f}m {(p_c - d).mean():>+9.2f}m "
              f"{np.mean(np.abs(p_c - d)):>9.2f}m")
    print('-' * 100)
    print(f"{'OVERALL':>8} | {len(rows):>4d} | "
          f"{'-':>10} {(all_pred - all_true).mean():>+9.2f}m "
          f"{np.mean(np.abs(all_pred - all_true)):>9.2f}m | "
          f"{'-':>10} {(all_pred_cal - all_true).mean():>+9.2f}m "
          f"{np.mean(np.abs(all_pred_cal - all_true)):>9.2f}m")

    csv_path = OUT_DIR / "test_real_2x2_72.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "meas_idx", "true_distance_m",
                    "predicted_distance_m_trimmed_mean_raw",
                    "predicted_distance_m_calibrated",
                    "pred_path0_m", "pred_path1_m",
                    "pred_path2_m", "pred_path3_m",
                    "abs_error_raw_m", "abs_error_calibrated_m",
                    "calibration_bias_subtracted_m"])
        for (fname, idx, td, pd_, p0, p1, p2, p3), pdc in zip(rows, all_pred_cal):
            w.writerow([fname, idx, f"{td:.4f}",
                        f"{pd_:.4f}", f"{pdc:.4f}",
                        f"{p0:.4f}", f"{p1:.4f}",
                        f"{p2:.4f}", f"{p3:.4f}",
                        f"{abs(pd_ - td):.4f}", f"{abs(pdc - td):.4f}",
                        f"{bias_calib:.4f}"])
    print(f"[info] CSV  -> {csv_path}")

    raw_mae   = float(np.mean(  np.abs(all_pred     - all_true)))
    raw_medae = float(np.median(np.abs(all_pred     - all_true)))
    raw_bias  = float(np.mean(       all_pred     - all_true ))
    cal_mae   = float(np.mean(  np.abs(all_pred_cal - all_true)))
    cal_medae = float(np.median(np.abs(all_pred_cal - all_true)))

    save_distance_boxplot(
        all_true, all_pred,
        f"Echt 2x2 / 72 tonen, trimmed mean van 4 paden -- ruwe voorspellingen\n"
        f"N = {len(rows)}, {len(chosen)} bestanden, MC-dropout passes = {N_MC}\n"
        f"ruwe MAE = {raw_mae:.2f} m, MedAE = {raw_medae:.2f} m, bias = {raw_bias:+.2f} m",
        OUT_DIR / "test_real_2x2_72_raw.png",
    )
    save_distance_boxplot(
        all_true, all_pred_cal,
        f"Echt 2x2 / 72 tonen, trimmed mean van 4 paden -- na eenpuntskalibratie\n"
        f"N = {len(rows)}, kalibratiereferentie = {ref_distance:.1f} m, "
        f"bias afgetrokken = {bias_calib:+.2f} m\n"
        f"gekalibreerde MAE = {cal_mae:.2f} m, MedAE = {cal_medae:.2f} m",
        OUT_DIR / "test_real_2x2_72_calibrated.png",
    )
    print(f"[info] PNG  -> {OUT_DIR / 'test_real_2x2_72_raw.png'}")
    print(f"[info] PNG  -> {OUT_DIR / 'test_real_2x2_72_calibrated.png'}")
    print("[done]")


if __name__ == "__main__":
    main()