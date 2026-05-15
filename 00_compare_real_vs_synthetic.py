import csv
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import (
    N_USED_CHANNELS,
    aggregate_to_72,
)
from synthetic import generate_synthetic_iq


SEED          = 42
N_SYNTH_REPS  = 6
FALLBACK_HW_M = 1.9

ROOT             = Path(__file__).resolve().parent
OUT_DIR          = ROOT / "outputs"
COMPARE_OUT_DIR  = OUT_DIR / "compare"
COMPARE_OUT_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_DATA_DIR = ROOT / "real_data"

DATASETS = ["1x1_72", "2x2_72", "2x2_256"]

CALIB_CSV = {
    "1x1_72":  OUT_DIR / "test_real_1x1_72.csv",
    "2x2_72":  OUT_DIR / "test_real_2x2_72.csv",
    "2x2_256": OUT_DIR / "test_real_2x2_256.csv",
}

N_PATHS_PER_DATASET = {
    "1x1_72":  1,
    "2x2_72":  4,
    "2x2_256": 4,
}

_DIST_RE = re.compile(r"_rd_(\d+)[._](\d+)[._]iq", re.IGNORECASE)


def distance_from_filename(path):
    m = _DIST_RE.search(path.name)
    return float(f"{m.group(1)}.{m.group(2)}") if m else None


def normalise_magnitude(z):
    mag = np.abs(z).astype(np.float64)
    peak = float(mag.max())
    if peak > 0:
        mag = mag / peak
    return mag


def unwrap_phase(z):
    return np.unwrap(np.angle(z).astype(np.float64))


def aggregate_paths(paths_complex):
    paths_complex = np.asarray(paths_complex, dtype=np.complex128)
    if paths_complex.ndim == 1:
        return paths_complex
    if paths_complex.shape[0] == 1:
        return paths_complex[0]

    mag = np.median(np.abs(paths_complex), axis=0)

    unit  = np.zeros_like(paths_complex, dtype=np.complex128)
    denom = np.abs(paths_complex)
    mask  = denom > 1e-12
    unit[mask] = paths_complex[mask] / denom[mask]

    mean_unit = np.mean(unit, axis=0)
    phase     = np.angle(mean_unit)
    return mag * np.exp(1j * phase)


def representative_complex_median(arr, axis=0):
    arr = np.asarray(arr, dtype=np.complex128)
    return np.median(arr.real, axis=axis) + 1j * np.median(arr.imag, axis=axis)


def get_calibration_bias_for_dataset(dataset_name):
    csv_path = CALIB_CSV.get(dataset_name)
    if csv_path is not None and csv_path.exists():
        try:
            with open(csv_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                first  = next(reader, None)
            if first and "calibration_bias_subtracted_m" in first:
                bias = float(first["calibration_bias_subtracted_m"])
                print(f"[info] {dataset_name}: calibration bias = {bias:+.2f} m "
                      f"(from {csv_path.name})")
                return bias
        except Exception as e:
            print(f"[warn] could not read {csv_path.name}: {e}")

    print(f"[warn] {dataset_name}: no calibration CSV found, "
          f"using fallback {FALLBACK_HW_M:+.2f} m. "
          f"Run 03/04/05 first for an automatic value.")
    return FALLBACK_HW_M


def parse_jsonl_for_compare(path):
    with open(path, "r", encoding="utf-8") as f:
        meta_line = f.readline()
        try:
            meta = json.loads(meta_line)
        except Exception:
            meta = {}
        if meta.get("type") != "meta":
            f.seek(0)

        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("type") == "meta":
                continue
            if "i" not in obj or "q" not in obj:
                continue

            i_lst    = obj["i"]
            q_lst    = obj["q"]
            channels = np.asarray(obj.get("channels", []), dtype=np.int64)

            if not isinstance(i_lst, list) or not i_lst:
                continue

            n_paths = len(i_lst)
            i_paths = np.stack([np.asarray(i_lst[p], dtype=np.float64) for p in range(n_paths)])
            q_paths = np.stack([np.asarray(q_lst[p], dtype=np.float64) for p in range(n_paths)])
            yield channels, i_paths, q_paths


def build_representative_real(file_path):
    measurements = []
    for channels, i_paths, q_paths in parse_jsonl_for_compare(file_path):
        n_paths = i_paths.shape[0]
        per_path_72 = np.zeros((n_paths, N_USED_CHANNELS), dtype=np.complex64)
        for p in range(n_paths):
            per_path_72[p] = aggregate_to_72(channels, i_paths[p], q_paths[p])
        measurements.append(per_path_72)

    if not measurements:
        raise RuntimeError(f"No valid measurements in {file_path}")

    arr = np.stack(measurements, axis=0)
    rep_per_path = representative_complex_median(arr, axis=0)
    rep_track    = aggregate_paths(rep_per_path)
    return rep_track, len(measurements)


def build_representative_synthetic(distance_m, n_paths, seed):
    rng = np.random.default_rng(seed)
    tracks = []
    for _ in range(N_SYNTH_REPS):
        if n_paths == 1:
            iq = generate_synthetic_iq(distance_m, rng)
            track = iq.astype(np.complex128)
        else:
            paths = np.stack([generate_synthetic_iq(distance_m, rng)
                              for _ in range(n_paths)], axis=0)
            track = aggregate_paths(paths.astype(np.complex128))
        tracks.append(track)

    mags, phases = [], []
    for z in tracks:
        mags.append(normalise_magnitude(z))
        ph = unwrap_phase(z)
        phases.append(ph - ph[0])

    mag_rep   = np.median(np.asarray(mags),   axis=0)
    phase_rep = np.median(np.asarray(phases), axis=0)
    return mag_rep * np.exp(1j * phase_rep)


def plot_compare(real_track, synth_track, dataset_name, distance, out_png):
    mag_real  = normalise_magnitude(real_track)
    mag_synth = normalise_magnitude(synth_track)

    phase_real  = unwrap_phase(real_track);   phase_real  -= phase_real[0]
    phase_synth = unwrap_phase(synth_track);  phase_synth -= phase_synth[0]

    x = np.arange(len(mag_real))

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(x, mag_real,  marker="o", linewidth=1.5, label="Echte meting")
    axes[0].plot(x, mag_synth, marker="s", linestyle="--", linewidth=1.5,
                 label="Synthetische data")
    axes[0].set_title(f"Echte vs synthetische data op {distance:.1f} m -- {dataset_name}")
    axes[0].set_ylabel("Magnitude (genorm.)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(x, phase_real,  marker="o", linewidth=1.5, label="Echte meting")
    axes[1].plot(x, phase_synth, marker="s", linestyle="--", linewidth=1.5,
                 label="Synthetische data")
    axes[1].set_xlabel("Kanaalindex")
    axes[1].set_ylabel("Fase (rad), baseline = 0")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def collect_dataset_files(folder):
    if not folder.is_dir():
        return []
    files = []
    for p in folder.glob("*.jsonl"):
        d = distance_from_filename(p)
        if d is not None:
            files.append((d, p))
    files.sort(key=lambda x: x[0])
    return files


def main():
    data_dir = Path(sys.argv[1]) if len(sys.argv) >= 2 else DEFAULT_DATA_DIR
    if not data_dir.is_dir():
        raise SystemExit(f"Data folder not found: {data_dir}")

    print(f"[info] data_dir   = {data_dir}")
    print(f"[info] output_dir = {COMPARE_OUT_DIR}")

    summary_rows = []
    total_plots  = 0

    for dataset_name in DATASETS:
        folder = data_dir / dataset_name
        files  = collect_dataset_files(folder)
        if not files:
            print(f"[skip] {dataset_name}: no .jsonl files in {folder}")
            continue

        bias_m  = get_calibration_bias_for_dataset(dataset_name)
        n_paths = N_PATHS_PER_DATASET[dataset_name]
        ds_out  = COMPARE_OUT_DIR / dataset_name
        ds_out.mkdir(exist_ok=True)

        print(f"[dataset] {dataset_name}: {len(files)} files, "
              f"n_paths={n_paths}, bias={bias_m:+.2f} m")

        for file_idx, (distance, file_path) in enumerate(files):
            try:
                real_track, n_meas = build_representative_real(file_path)

                synth_seed = (SEED
                              + 1_000_003 * file_idx
                              + int(round(distance * 100))
                              + (0 if dataset_name == "1x1_72"
                                 else 100_000 if dataset_name == "2x2_72"
                                 else 200_000))

                synth_track = build_representative_synthetic(
                    distance_m=distance + bias_m,
                    n_paths=n_paths,
                    seed=synth_seed,
                )

                out_png = ds_out / f"compare_{dataset_name}_{distance:.1f}m.png"
                plot_compare(real_track, synth_track, dataset_name, distance, out_png)

                summary_rows.append({
                    "dataset":            dataset_name,
                    "distance_m":         f"{distance:.1f}",
                    "input_file":         str(file_path),
                    "num_measurements":   n_meas,
                    "calibration_bias_m": f"{bias_m:.4f}",
                    "output_png":         str(out_png),
                })
                total_plots += 1
                print(f"  [save] {out_png.name}  (d={distance:.1f} m, n_meas={n_meas})")

            except Exception as e:
                print(f"  [error] {file_path.name} -> {e}")

    summary_csv = COMPARE_OUT_DIR / "compare_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "distance_m", "input_file",
                        "num_measurements", "calibration_bias_m",
                        "output_png"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print()
    print(f"[done] generated {total_plots} comparison plots")
    print(f"[info] summary CSV -> {summary_csv}")


if __name__ == "__main__":
    main()