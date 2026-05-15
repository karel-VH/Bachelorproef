import csv
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from common import (
    DistanceCNN,
    normalise_iq, filter_outlier_tones, target_to_dist,
    mc_dropout_predict, save_distance_boxplot,
    N_USED_CHANNELS, CHAN_TO_POS,
)


#strategie die gebruikt wordt voor de finale output
FINAL_STRATEGY = "iq_first"


ROOT             = Path(__file__).resolve().parent
MODEL_PATH       = ROOT / "outputs" / "cnn_distance_model.pt"
OUT_DIR          = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)
DEFAULT_DATA_DIR = ROOT / "real_data" / "2x2_256"

_DIST_RE = re.compile(r"_rd_(\d+)[._](\d+)[._]iq", re.IGNORECASE)


def collapse_first(channels, i_arr, q_arr):
    out  = np.zeros(N_USED_CHANNELS, dtype=np.complex64)
    seen = set()
    for k in range(len(channels)):
        ch = int(channels[k])
        if ch in seen or ch not in CHAN_TO_POS:
            continue
        out[CHAN_TO_POS[ch]] = i_arr[k] + 1j * q_arr[k]
        seen.add(ch)
    return out


def collapse_mean(channels, i_arr, q_arr):
    out = np.zeros(N_USED_CHANNELS, dtype=np.complex64)
    iq_full = i_arr + 1j * q_arr
    for ch_id in np.unique(channels):
        ch = int(ch_id)
        if ch not in CHAN_TO_POS:
            continue
        out[CHAN_TO_POS[ch]] = iq_full[channels == ch_id].mean()
    return out


def collapse_median_re_im(channels, i_arr, q_arr):
    out = np.zeros(N_USED_CHANNELS, dtype=np.complex64)
    for ch_id in np.unique(channels):
        ch = int(ch_id)
        if ch not in CHAN_TO_POS:
            continue
        mask = channels == ch_id
        out[CHAN_TO_POS[ch]] = (float(np.median(i_arr[mask]))
                                + 1j * float(np.median(q_arr[mask])))
    return out


def collapse_median_complex(channels, i_arr, q_arr):
    out = np.zeros(N_USED_CHANNELS, dtype=np.complex64)
    iq_full = i_arr + 1j * q_arr
    for ch_id in np.unique(channels):
        ch = int(ch_id)
        if ch not in CHAN_TO_POS:
            continue
        samples = iq_full[channels == ch_id]
        if len(samples) == 1:
            out[CHAN_TO_POS[ch]] = samples[0]
        else:
            dists = np.abs(samples[:, None] - samples[None, :]).sum(axis=1)
            out[CHAN_TO_POS[ch]] = samples[int(np.argmin(dists))]
    return out


def collapse_trimmed_mean(channels, i_arr, q_arr):
    out = np.zeros(N_USED_CHANNELS, dtype=np.complex64)
    iq_full = i_arr + 1j * q_arr
    for ch_id in np.unique(channels):
        ch = int(ch_id)
        if ch not in CHAN_TO_POS:
            continue
        samples = iq_full[channels == ch_id]
        if len(samples) < 3:
            out[CHAN_TO_POS[ch]] = samples.mean()
        else:
            mags = np.abs(samples)
            keep = np.ones(len(samples), dtype=bool)
            keep[int(np.argmin(mags))] = False
            keep[int(np.argmax(mags))] = False
            out[CHAN_TO_POS[ch]] = samples[keep].mean()
    return out


def collapse_magweighted(channels, i_arr, q_arr):
    out = np.zeros(N_USED_CHANNELS, dtype=np.complex64)
    iq_full = i_arr + 1j * q_arr
    for ch_id in np.unique(channels):
        ch = int(ch_id)
        if ch not in CHAN_TO_POS:
            continue
        samples = iq_full[channels == ch_id]
        weights = np.abs(samples)
        s = weights.sum()
        out[CHAN_TO_POS[ch]] = (samples * weights).sum() / s if s > 1e-12 else samples.mean()
    return out


STRATEGIES = {
    "iq_first":          collapse_first,
    "iq_mean":           collapse_mean,
    "iq_median_re_im":   collapse_median_re_im,
    "iq_median_complex": collapse_median_complex,
    "iq_trimmed_mean":   collapse_trimmed_mean,
    "iq_magweighted":    collapse_magweighted,
}


def trimmed_mean_4_paths(values):
    s = np.sort(values, axis=-1)
    return s[..., 1:3].mean(axis=-1)


def distance_from_filename(path):
    m = _DIST_RE.search(path.name)
    return float(f"{m.group(1)}.{m.group(2)}") if m else None


def parse_jsonl_2x2_256(path):
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

            i_paths = np.stack([np.asarray(i_lst[p], dtype=np.float64) for p in range(4)])
            q_paths = np.stack([np.asarray(q_lst[p], dtype=np.float64) for p in range(4)])
            yield d_true, channels, i_paths, q_paths


def evaluate_strategy(collapse_fn, preloaded_data, model, device,
                      n_mc=15, batch=256):
    file_names   = []
    true_d       = []
    iq_72_buffer = []

    for f_path, lines in preloaded_data:
        for d, channels, i_paths, q_paths in lines:
            file_names.append(f_path.name)
            true_d.append(d)
            for p in range(4):
                iq_72_buffer.append(collapse_fn(channels, i_paths[p], q_paths[p]))

    n_meas = len(true_d)
    iq_arr = np.stack(iq_72_buffer, axis=0)
    Xflat  = np.stack([normalise_iq(filter_outlier_tones(z)) for z in iq_arr], axis=0)

    pred_flat = np.empty(Xflat.shape[0], dtype=np.float64)
    for s in range(0, Xflat.shape[0], batch):
        xb = torch.from_numpy(Xflat[s:s + batch]).to(device)
        t  = mc_dropout_predict(model, xb, n_passes=n_mc)
        pred_flat[s:s + batch] = target_to_dist(t)

    per_path = pred_flat.reshape(n_meas, 4)
    final    = trimmed_mean_4_paths(per_path)
    return (np.array(file_names),
            np.array(true_d, dtype=np.float64),
            per_path,
            final)


def make_summary(strat_name, true_d, pred_final, ref_distance):
    ref_mask   = np.isclose(true_d, ref_distance, atol=0.05)
    bias_calib = float(np.median(pred_final[ref_mask] - true_d[ref_mask]))
    pred_cal   = pred_final - bias_calib
    raw_err    = pred_final - true_d
    cal_err    = pred_cal   - true_d
    return dict(
        strategy   = strat_name,
        bias_calib = bias_calib,
        raw_bias   = float(np.mean(raw_err)),
        raw_mae    = float(np.mean(np.abs(raw_err))),
        raw_medae  = float(np.median(np.abs(raw_err))),
        raw_p90    = float(np.percentile(np.abs(raw_err), 90)),
        cal_bias   = float(np.mean(cal_err)),
        cal_mae    = float(np.mean(np.abs(cal_err))),
        cal_medae  = float(np.median(np.abs(cal_err))),
        cal_p90    = float(np.percentile(np.abs(cal_err), 90)),
        pred_cal   = pred_cal,
    )


def main():
    data_dir = Path(sys.argv[1]) if len(sys.argv) >= 2 else DEFAULT_DATA_DIR
    if not data_dir.is_dir():
        raise SystemExit(f"Data folder not found: {data_dir}")
    if not MODEL_PATH.exists():
        raise SystemExit(f"Model not found: {MODEL_PATH}\nRun 01_train_synthetic.py first.")

    if FINAL_STRATEGY is not None and FINAL_STRATEGY not in STRATEGIES:
        raise SystemExit(f"Unknown FINAL_STRATEGY = {FINAL_STRATEGY!r}. "
                         f"Valid options: {list(STRATEGIES)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device = {device}")
    print(f"[info] FINAL_STRATEGY = "
          f"{FINAL_STRATEGY if FINAL_STRATEGY else 'auto (best calibrated MAE)'}")

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
                    and len(i_lst[0]) > N_USED_CHANNELS):
                chosen.append(p)
        except Exception as e:
            print(f"[warn] could not read {p.name}: {e}")
    if not chosen:
        raise SystemExit(f"No 2x2 / extended .jsonl files in {data_dir}")
    print(f"[info] found {len(chosen)} 2x2 / extended files in {data_dir}")

    preloaded = []
    for f_path in chosen:
        lines = list(parse_jsonl_2x2_256(f_path))
        preloaded.append((f_path, lines))
        print(f"  {f_path.name}: {len(lines)} measurements")

    results = []
    print()
    for strat_name, fn in STRATEGIES.items():
        print(f"[strategy] {strat_name}")
        names, true_d, per_path, pred_final = evaluate_strategy(fn, preloaded, model, device)
        ref_distance = float(np.min(true_d))
        summary = make_summary(strat_name, true_d, pred_final, ref_distance)
        results.append(dict(summary,
                            names=names, true_d=true_d,
                            per_path=per_path, pred_final=pred_final,
                            ref_distance=ref_distance))
        print(f"           raw : bias={summary['raw_bias']:+.2f} m  "
              f"MAE={summary['raw_mae']:.2f} m  MedAE={summary['raw_medae']:.2f} m")
        print(f"           cal : bias={summary['cal_bias']:+.2f} m  "
              f"MAE={summary['cal_mae']:.2f} m  MedAE={summary['cal_medae']:.2f} m  "
              f"(calib offset = {summary['bias_calib']:+.2f} m at d = {ref_distance:.1f} m)")

    print()
    print("=" * 100)
    print("STRATEGY COMPARISON  (lower is better)")
    print("=" * 100)
    print(f"{'strategy':>20} | {'raw MAE':>8} {'raw MedAE':>10} {'raw P90':>8}"
          f" | {'cal MAE':>8} {'cal MedAE':>10} {'cal P90':>8} | {'calib bias':>10}")
    print("-" * 100)
    results_sorted = sorted(results, key=lambda r: r["cal_mae"])
    auto_best = results_sorted[0]
    for r in results_sorted:
        marker = " <- best (auto)" if r is auto_best else ""
        print(f"{r['strategy']:>20} | "
              f"{r['raw_mae']:>7.2f}m {r['raw_medae']:>9.2f}m {r['raw_p90']:>7.2f}m | "
              f"{r['cal_mae']:>7.2f}m {r['cal_medae']:>9.2f}m {r['cal_p90']:>7.2f}m | "
              f"{r['bias_calib']:>+9.2f}m{marker}")
    print("=" * 100)

    if FINAL_STRATEGY is None:
        chosen_strat = auto_best
        print(f"selected strategy (auto): {chosen_strat['strategy']}  "
              f"(calibrated MAE = {chosen_strat['cal_mae']:.2f} m)")
    else:
        chosen_strat = next(r for r in results if r["strategy"] == FINAL_STRATEGY)
        print(f"selected strategy (configured): {chosen_strat['strategy']}  "
              f"(calibrated MAE = {chosen_strat['cal_mae']:.2f} m)")

    summary_csv = OUT_DIR / "test_real_2x2_256_strategies.csv"
    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["strategy",
                    "raw_bias_m", "raw_mae_m", "raw_medae_m", "raw_p90_m",
                    "cal_bias_m", "cal_mae_m", "cal_medae_m", "cal_p90_m",
                    "calibration_bias_subtracted_m"])
        for r in results:
            w.writerow([r["strategy"],
                        f"{r['raw_bias']:.4f}", f"{r['raw_mae']:.4f}",
                        f"{r['raw_medae']:.4f}", f"{r['raw_p90']:.4f}",
                        f"{r['cal_bias']:.4f}", f"{r['cal_mae']:.4f}",
                        f"{r['cal_medae']:.4f}", f"{r['cal_p90']:.4f}",
                        f"{r['bias_calib']:.4f}"])
    print(f"[info] CSV  -> {summary_csv}")

    csv_path = OUT_DIR / "test_real_2x2_256.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "meas_idx", "true_distance_m",
                    "predicted_distance_m_raw",
                    "predicted_distance_m_calibrated",
                    "pred_path0_m", "pred_path1_m",
                    "pred_path2_m", "pred_path3_m",
                    "abs_error_raw_m", "abs_error_calibrated_m",
                    "calibration_bias_subtracted_m",
                    "collapse_strategy"])
        for i in range(len(chosen_strat["true_d"])):
            td  = float(chosen_strat["true_d"][i])
            pd_ = float(chosen_strat["pred_final"][i])
            pdc = float(chosen_strat["pred_cal"][i])
            paths = chosen_strat["per_path"][i]
            w.writerow([chosen_strat["names"][i], i, f"{td:.4f}",
                        f"{pd_:.4f}", f"{pdc:.4f}",
                        f"{paths[0]:.4f}", f"{paths[1]:.4f}",
                        f"{paths[2]:.4f}", f"{paths[3]:.4f}",
                        f"{abs(pd_ - td):.4f}", f"{abs(pdc - td):.4f}",
                        f"{chosen_strat['bias_calib']:.4f}",
                        chosen_strat["strategy"]])
    print(f"[info] CSV  -> {csv_path}")

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(results))
    raw_mae = [r["raw_mae"] for r in results]
    cal_mae = [r["cal_mae"] for r in results]
    width = 0.35
    ax.bar(x - width / 2, raw_mae, width, label="ruw",          color="tab:red",   alpha=0.75)
    ax.bar(x + width / 2, cal_mae, width, label="gekalibreerd", color="tab:green", alpha=0.75)
    ax.set_xticks(x)
    ax.set_xticklabels([r["strategy"] for r in results], rotation=20, ha="right")
    ax.set_ylabel("MAE (m)")
    ax.set_title("2x2 / uitgebreid -- vergelijking collapse-strategieën\n"
                 "lager is beter; gekalibreerd = na eenpuntskalibratie")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "test_real_2x2_256_strategy_comparison.png", dpi=140)
    plt.close(fig)
    print(f"[info] PNG  -> {OUT_DIR / 'test_real_2x2_256_strategy_comparison.png'}")

    save_distance_boxplot(
        chosen_strat["true_d"], chosen_strat["pred_final"],
        f"Echt 2x2 / uitgebreid, collapse = {chosen_strat['strategy']}, "
        f"trimmed mean van 4 paden -- ruwe voorspellingen\n"
        f"N = {len(chosen_strat['true_d'])}, {len(chosen)} bestanden, MC-dropout passes = 15\n"
        f"ruwe MAE = {chosen_strat['raw_mae']:.2f} m, "
        f"MedAE = {chosen_strat['raw_medae']:.2f} m, "
        f"bias = {chosen_strat['raw_bias']:+.2f} m",
        OUT_DIR / "test_real_2x2_256_raw.png",
    )
    print(f"[info] PNG  -> {OUT_DIR / 'test_real_2x2_256_raw.png'}")

    save_distance_boxplot(
        chosen_strat["true_d"], chosen_strat["pred_cal"],
        f"Echt 2x2 / uitgebreid, collapse = {chosen_strat['strategy']}, "
        f"trimmed mean van 4 paden -- na eenpuntskalibratie\n"
        f"N = {len(chosen_strat['true_d'])}, kalibratiereferentie = "
        f"{chosen_strat['ref_distance']:.1f} m, "
        f"bias afgetrokken = {chosen_strat['bias_calib']:+.2f} m\n"
        f"gekalibreerde MAE = {chosen_strat['cal_mae']:.2f} m, "
        f"MedAE = {chosen_strat['cal_medae']:.2f} m",
        OUT_DIR / "test_real_2x2_256_calibrated.png",
    )
    print(f"[info] PNG  -> {OUT_DIR / 'test_real_2x2_256_calibrated.png'}")
    print("[done]")


if __name__ == "__main__":
    main()