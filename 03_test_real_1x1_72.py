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
)

ROOT             = Path(__file__).resolve().parent
MODEL_PATH       = ROOT / "outputs" / "cnn_distance_model.pt"
OUT_DIR          = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)
DEFAULT_DATA_DIR = ROOT / "real_data" / "1x1_72"

_DIST_RE = re.compile(r"_rd_(\d+)[._](\d+)[._]iq", re.IGNORECASE)


def distance_from_filename(path):
    m = _DIST_RE.search(path.name)
    return float(f"{m.group(1)}.{m.group(2)}") if m else None


def parse_jsonl_1x1(path):
    with open(path, "r") as f:
        meta = json.loads(f.readline())
        if meta.get("type") != "meta":
            raise ValueError(f"{path.name}: first line is not metadata")

        cfg = meta.get("config", {})
        if cfg.get("antenna_configuration_string") != "1x1":
            print(f"[warn] {path.name}: antenna configuration is "
                  f"{cfg.get('antenna_configuration_string')!r}, expected '1x1'")

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

            i_lst, q_lst, channels = obj["i"], obj["q"], obj["channels"]
            if not isinstance(i_lst, list) or not i_lst:
                continue

            i_arr = np.asarray(i_lst[0], dtype=np.float64)
            q_arr = np.asarray(q_lst[0], dtype=np.float64)
            chans = np.asarray(channels, dtype=np.int64)

            iq = aggregate_to_72(chans, i_arr, q_arr)
            yield d_true, iq


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

    one_one_files = []
    for p in files:
        try:
            with open(p, "r") as f:
                meta = json.loads(f.readline())
            if meta.get("config", {}).get("antenna_configuration_string") == "1x1":
                one_one_files.append(p)
        except Exception as e:
            print(f"[warn] could not read metadata of {p.name}: {e}")
    if not one_one_files:
        raise SystemExit(f"No 1x1 .jsonl files in {data_dir}")
    print(f"[info] found {len(one_one_files)} 1x1 files in {data_dir}")

    N_MC  = 15
    BATCH = 256
    rows  = []

    for f_path in one_one_files:
        measurements = list(parse_jsonl_1x1(f_path))
        n = len(measurements)
        if n == 0:
            print(f"[warn] {f_path.name}: 0 measurements")
            continue

        true_d = np.array([d for d, _ in measurements], dtype=np.float64)
        iq_all = np.stack([iq for _, iq in measurements], axis=0)
        X = np.stack([normalise_iq(filter_outlier_tones(iq)) for iq in iq_all], axis=0)

        pred_d = np.empty(n, dtype=np.float64)
        for s in range(0, n, BATCH):
            xb = torch.from_numpy(X[s:s + BATCH]).to(device)
            t  = mc_dropout_predict(model, xb, n_passes=N_MC)
            pred_d[s:s + BATCH] = target_to_dist(t)

        abs_err = np.abs(pred_d - true_d)
        print(f"[file] {f_path.name}  d_true={true_d[0]:.1f} m  N={n}  "
              f"pred_med={np.median(pred_d):.2f} m  "
              f"raw bias={(pred_d - true_d).mean():+.2f} m  "
              f"MedAE={np.median(abs_err):.2f} m")

        for i in range(n):
            rows.append((f_path.name, i, true_d[i], pred_d[i]))

    if not rows:
        raise SystemExit("No measurements processed.")

    all_true = np.array([r[2] for r in rows])
    all_pred = np.array([r[3] for r in rows])

    #eenpuntskalibratie
    ref_distance = float(np.min(all_true))
    ref_mask     = np.isclose(all_true, ref_distance, atol=0.05)
    bias_calib   = float(np.median(all_pred[ref_mask] - all_true[ref_mask]))
    all_pred_cal = all_pred - bias_calib

    print()
    print(f"[calib] reference distance       = {ref_distance:.1f} m  "
          f"({ref_mask.sum()} measurements)")
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

    csv_path = OUT_DIR / "test_real_1x1_72.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "meas_idx", "true_distance_m",
                    "predicted_distance_m_raw",
                    "predicted_distance_m_calibrated",
                    "abs_error_raw_m", "abs_error_calibrated_m",
                    "calibration_bias_subtracted_m"])
        for (fname, idx, td, pd_), pdc in zip(rows, all_pred_cal):
            w.writerow([fname, idx, f"{td:.4f}",
                        f"{pd_:.4f}", f"{pdc:.4f}",
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
        f"Echt 1x1 / 72 tonen -- ruwe voorspellingen\n"
        f"N = {len(rows)}, {len(one_one_files)} bestanden, MC-dropout passes = {N_MC}\n"
        f"ruwe MAE = {raw_mae:.2f} m, MedAE = {raw_medae:.2f} m, bias = {raw_bias:+.2f} m",
        OUT_DIR / "test_real_1x1_72_raw.png",
    )
    save_distance_boxplot(
        all_true, all_pred_cal,
        f"Echt 1x1 / 72 tonen -- na eenpuntskalibratie\n"
        f"N = {len(rows)}, kalibratiereferentie = {ref_distance:.1f} m, "
        f"bias afgetrokken = {bias_calib:+.2f} m\n"
        f"gekalibreerde MAE = {cal_mae:.2f} m, MedAE = {cal_medae:.2f} m",
        OUT_DIR / "test_real_1x1_72_calibrated.png",
    )
    print(f"[info] PNG  -> {OUT_DIR / 'test_real_1x1_72_raw.png'}")
    print(f"[info] PNG  -> {OUT_DIR / 'test_real_1x1_72_calibrated.png'}")
    print("[done]")


if __name__ == "__main__":
    main()