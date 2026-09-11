#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from process_height_timeseries import (
    DATA_DIR_FINGERPRINT_KEY,
    parse_path,
    parse_data_dir,
    group_key,
    read_yts,
    output_group_dir_from_data_dir,
    directory_stat_fingerprint,
)


FIELDS = [
    "type_perc",
    "num_colors",
    "dim",
    "L",
    "f_T",
    "c",
    "control_param",
    "epsilon",
    "rho",
    "color",
    "t",
    "n",
    "y_width_mean",
    "y_width_std",
    "y_width_sem",
    "y_mean_mean",
    "y_mean_std",
    "y_mean_sem",
    "y_max_mean",
    "y_max_std",
    "y_max_sem",
    "y_front_count_mean",
    "y_front_count_std",
    "y_front_count_sem",
]

HEIGHT_ENSEMBLE_PROCESSING_VERSION = 1
HEIGHT_ENSEMBLE_MANIFEST = "height_ensemble_manifest.json"
HEIGHT_ENSEMBLE_FILE = "height_ensemble_timeseries.csv.gz"


class RunningSeries:
    DENSE_MAX_SIZE = 5_000_000

    def __init__(self) -> None:
        self.n = np.zeros(0, dtype=np.int32)
        self.sum = np.zeros(0, dtype=np.float64)
        self.sumsq = np.zeros(0, dtype=np.float64)
        self.sparse: dict[int, list[float]] | None = None

    def _convert_to_sparse(self) -> None:
        if self.sparse is not None:
            return
        sparse: dict[int, list[float]] = {}
        for idx in np.flatnonzero(self.n):
            count = int(self.n[idx])
            sparse[int(idx)] = [count, float(self.sum[idx]), float(self.sumsq[idx])]
        self.n = np.zeros(0, dtype=np.int32)
        self.sum = np.zeros(0, dtype=np.float64)
        self.sumsq = np.zeros(0, dtype=np.float64)
        self.sparse = sparse

    def _ensure_size(self, size: int) -> None:
        if self.sparse is not None:
            return
        if size > self.DENSE_MAX_SIZE:
            self._convert_to_sparse()
            return
        if size <= self.n.size:
            return
        extra = size - self.n.size
        self.n = np.pad(self.n, (0, extra), constant_values=0)
        self.sum = np.pad(self.sum, (0, extra), constant_values=0.0)
        self.sumsq = np.pad(self.sumsq, (0, extra), constant_values=0.0)

    def add(self, time: np.ndarray, values: np.ndarray) -> None:
        keep = np.isfinite(time) & np.isfinite(values)
        if not np.any(keep):
            return
        idx = time[keep].astype(np.int64, copy=False)
        vals = values[keep]
        if idx.size == 0:
            return
        if np.any(idx < 0):
            raise ValueError("negative time values are not supported")
        self._ensure_size(int(idx.max()) + 1)
        if self.sparse is not None:
            for t, val in zip(idx, vals):
                bucket = self.sparse.setdefault(int(t), [0, 0.0, 0.0])
                bucket[0] += 1
                bucket[1] += float(val)
                bucket[2] += float(val) * float(val)
            return
        np.add.at(self.n, idx, 1)
        np.add.at(self.sum, idx, vals)
        np.add.at(self.sumsq, idx, vals * vals)

    def add_stats(self, t: int, count: int, mean: float, std: float) -> None:
        if count <= 0 or not math.isfinite(mean):
            return
        if not math.isfinite(std):
            std = 0.0
        if t < 0:
            raise ValueError("negative time values are not supported")
        self._ensure_size(t + 1)
        total = mean * count
        sumsq = mean * mean if count == 1 else std * std * (count - 1) + total * total / count
        if self.sparse is not None:
            bucket = self.sparse.setdefault(int(t), [0, 0.0, 0.0])
            bucket[0] += int(count)
            bucket[1] += float(total)
            bucket[2] += float(sumsq)
            return
        self.n[t] += int(count)
        self.sum[t] += float(total)
        self.sumsq[t] += float(sumsq)

    def mean_std_sem(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.sparse is not None:
            size = max(self.sparse, default=-1) + 1
            if size > self.DENSE_MAX_SIZE:
                raise ValueError(
                    f"sparse time grid extends to {size - 1}; use rows() instead of dense arrays"
                )
            self.n = np.zeros(size, dtype=np.int32)
            self.sum = np.zeros(size, dtype=np.float64)
            self.sumsq = np.zeros(size, dtype=np.float64)
            for idx, (count, total, sumsq) in self.sparse.items():
                self.n[idx] = int(count)
                self.sum[idx] = float(total)
                self.sumsq[idx] = float(sumsq)
            self.sparse = None
        mean = np.full(self.n.size, math.nan, dtype=np.float64)
        std = np.full(self.n.size, math.nan, dtype=np.float64)
        sem = np.full(self.n.size, math.nan, dtype=np.float64)
        ok = self.n > 0
        mean[ok] = self.sum[ok] / self.n[ok]
        many = self.n > 1
        var = np.zeros(self.n.size, dtype=np.float64)
        var[many] = (self.sumsq[many] - (self.sum[many] * self.sum[many]) / self.n[many]) / (self.n[many] - 1)
        var[many] = np.maximum(var[many], 0.0)
        std[many] = np.sqrt(var[many])
        std[self.n == 1] = 0.0
        sem[ok] = std[ok] / np.sqrt(self.n[ok])
        return mean, std, sem

    def stats_at(self, t: int) -> tuple[int, float, float, float]:
        if self.sparse is not None:
            count, total, sumsq = self.sparse.get(int(t), [0, 0.0, 0.0])
        elif 0 <= int(t) < self.n.size:
            count = int(self.n[int(t)])
            total = float(self.sum[int(t)])
            sumsq = float(self.sumsq[int(t)])
        else:
            return 0, math.nan, math.nan, math.nan
        if count <= 0:
            return 0, math.nan, math.nan, math.nan
        mean = total / count
        if count == 1:
            return count, mean, 0.0, 0.0
        var = max((sumsq - total * total / count) / (count - 1), 0.0)
        std = math.sqrt(var)
        return count, mean, std, std / math.sqrt(count)

    def times_with_min_count(self, min_count: int) -> list[int]:
        if self.sparse is not None:
            return sorted(t for t, stats in self.sparse.items() if int(stats[0]) >= min_count)
        return [int(t) for t in np.flatnonzero(self.n >= min_count)]


def collect_data_dirs(root: Path, args: argparse.Namespace) -> list[tuple]:
    data_dirs = []
    for path in sorted(root.rglob("data")):
        if not path.is_dir():
            continue
        try:
            meta = parse_data_dir(path)
        except ValueError:
            continue
        if args.type_perc and meta.type_perc != args.type_perc:
            continue
        if args.lengths and meta.L not in args.lengths:
            continue
        if args.f_T and not any(math.isclose(meta.f_T, value, rel_tol=1e-12, abs_tol=1e-12) for value in args.f_T):
            continue
        data_dirs.append((meta, path))
    return data_dirs


def rows_for_color(meta, color: int, acc: dict[str, RunningSeries], min_count: int) -> list[dict]:
    rows = []
    for t in acc["y_width"].times_with_min_count(min_count):
        n_t, width_mean, width_std, width_sem = acc["y_width"].stats_at(t)
        _, y_mean_mean, y_mean_std, y_mean_sem = acc["y_mean"].stats_at(t)
        _, y_max_mean, y_max_std, y_max_sem = acc["y_max"].stats_at(t)
        _, front_mean, front_std, front_sem = acc["y_front_count"].stats_at(t)
        rows.append(
            {
                "type_perc": meta.type_perc,
                "num_colors": meta.num_colors,
                "dim": meta.dim,
                "L": meta.L,
                "f_T": f"{meta.f_T:.17g}",
                "c": f"{meta.c:.17g}",
                "control_param": "" if meta.control_param is None else f"{meta.control_param:.17g}",
                "epsilon": "" if meta.epsilon is None else f"{meta.epsilon:.17g}",
                "rho": f"{meta.rho:.17g}",
                "color": color,
                "t": int(t),
                "n": n_t,
                "y_width_mean": width_mean,
                "y_width_std": width_std,
                "y_width_sem": width_sem,
                "y_mean_mean": y_mean_mean,
                "y_mean_std": y_mean_std,
                "y_mean_sem": y_mean_sem,
                "y_max_mean": y_max_mean,
                "y_max_std": y_max_std,
                "y_max_sem": y_max_sem,
                "y_front_count_mean": front_mean,
                "y_front_count_std": front_std,
                "y_front_count_sem": front_sem,
            }
        )
    return rows


def write_csv_gz(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_gz(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with gzip.open(path, "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != FIELDS:
            raise ValueError(f"unexpected header in {path}: {reader.fieldnames}")
        return [dict(row) for row in reader]


def load_manifest(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def finite_float(value) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def restore_accumulators(rows: list[dict], colors: list[int]) -> dict[int, dict[str, RunningSeries]]:
    acc_by_color = {
        color: {
            "y_width": RunningSeries(),
            "y_mean": RunningSeries(),
            "y_max": RunningSeries(),
            "y_front_count": RunningSeries(),
        }
        for color in colors
    }
    for row in rows:
        try:
            color = int(row["color"])
            t = int(float(row["t"]))
            count = int(float(row["n"]))
        except Exception:
            continue
        if color not in acc_by_color:
            continue
        for prefix, key in (
            ("y_width", "y_width"),
            ("y_mean", "y_mean"),
            ("y_max", "y_max"),
            ("y_front_count", "y_front_count"),
        ):
            mean = finite_float(row.get(f"{prefix}_mean"))
            std = finite_float(row.get(f"{prefix}_std"))
            if mean is not None:
                acc_by_color[color][key].add_stats(t, count, mean, 0.0 if std is None else std)
    return acc_by_color


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Average .yts height time series over samples, keeping n(t)."
    )
    parser.add_argument("--root", type=Path, default=Path("SOP_data/raw_growth_test_dynamic"))
    parser.add_argument("--out-root", type=Path, default=Path("SOP_data/processed_height_timeseries"))
    parser.add_argument("--type-perc", choices=["bond", "node"], default=None)
    parser.add_argument("--lengths", type=int, nargs="*", default=None)
    parser.add_argument("--f-T", type=float, nargs="*", default=None)
    parser.add_argument("--colors", type=int, nargs="*", default=None)
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument("--max-samples-per-group", type=int, default=None)
    args = parser.parse_args()

    if args.min_count < 1:
        raise SystemExit("--min-count must be >= 1.")

    data_dirs = collect_data_dirs(args.root, args)
    existing_group_paths = sorted(args.out_root.rglob(HEIGHT_ENSEMBLE_FILE))
    if not data_dirs and not existing_group_paths:
        print(f"[info] No .yts files selected in {args.root}.")
        return 0

    total_samples = 0
    all_path = args.out_root / "height_ensemble_timeseries_all.csv.gz"
    all_path.parent.mkdir(parents=True, exist_ok=True)

    with gzip.open(all_path, "wt", newline="") as all_handle:
        all_writer = csv.DictWriter(all_handle, fieldnames=FIELDS)
        all_writer.writeheader()
        seen_group_paths: set[Path] = set()

        for first_meta, data_dir in data_dirs:
            key = group_key(first_meta)
            colors = args.colors if args.colors is not None else list(range(first_meta.num_colors))
            out_dir = output_group_dir_from_data_dir(args.root, args.out_root, data_dir)
            group_path = out_dir / HEIGHT_ENSEMBLE_FILE
            seen_group_paths.add(group_path.resolve())
            manifest_path = out_dir / HEIGHT_ENSEMBLE_MANIFEST
            manifest = load_manifest(manifest_path)
            manifest_ok = (
                int(manifest.get("height_ensemble_processing_version", 0) or 0)
                == HEIGHT_ENSEMBLE_PROCESSING_VERSION
            )
            processed_files = set(map(str, manifest.get("processed_yts_files", []))) if manifest_ok else set()
            current_data_dir_fingerprint = directory_stat_fingerprint(data_dir)

            if (
                manifest_ok
                and group_path.exists()
                and manifest.get(DATA_DIR_FINGERPRINT_KEY) == current_data_dir_fingerprint
            ):
                try:
                    group_rows = read_csv_gz(group_path)
                except Exception as exc:
                    print(f"[warn] Could not fast-skip unreadable published height ensemble ({exc}): {group_path}")
                else:
                    if group_rows:
                        all_writer.writerows(group_rows)
                        total_samples += len(processed_files)
                        print(f"[skip-fast] {key}: {len(processed_files)} samples already in {group_path}")
                        continue

            items = []
            for path in sorted(data_dir.glob("*.yts")):
                try:
                    meta = parse_path(path)
                except ValueError:
                    continue
                items.append((meta, path))
            if args.max_samples_per_group:
                items = items[: args.max_samples_per_group]
            if not items:
                continue
            existing_rows = read_csv_gz(group_path) if manifest_ok else []
            new_items = [(meta, path) for meta, path in items if path.name not in processed_files]

            if not new_items and existing_rows:
                group_rows = existing_rows
                all_writer.writerows(group_rows)
                total_samples += len(processed_files)
                manifest[DATA_DIR_FINGERPRINT_KEY] = current_data_dir_fingerprint
                save_manifest(manifest_path, manifest)
                print(f"[skip] {key}: {len(processed_files)} samples already in {group_path}")
                continue

            acc_by_color = restore_accumulators(existing_rows, colors)
            valid_samples_count = 0
            processed_new_files: list[str] = []
            for _, path in new_items:
                try:
                    data = read_yts(path, colors)
                except Exception as exc:
                    print(f"[warn] Skipping corrupt/unreadable .yts ({exc}): {path}")
                    continue
                valid_samples_count += 1
                for idx, color in enumerate(colors):
                    acc_by_color[color]["y_width"].add(data.time, data.y_width[idx])
                    acc_by_color[color]["y_mean"].add(data.time, data.y_mean[idx])
                    acc_by_color[color]["y_max"].add(data.time, data.y_max[idx])
                    acc_by_color[color]["y_front_count"].add(data.time, data.y_front_count[idx])
                processed_new_files.append(path.name)

            if valid_samples_count == 0 and not existing_rows:
                print(f"[warn] {key}: no valid .yts samples found.")
                continue

            group_rows: list[dict] = []
            for color, acc in acc_by_color.items():
                group_rows.extend(rows_for_color(first_meta, color, acc, args.min_count))

            write_csv_gz(group_path, group_rows)
            all_writer.writerows(group_rows)
            processed_files.update(processed_new_files)
            manifest.update({
                "height_ensemble_processing_version": HEIGHT_ENSEMBLE_PROCESSING_VERSION,
                "processed_yts_files": sorted(processed_files),
                "n_processed_yts_files": len(processed_files),
                DATA_DIR_FINGERPRINT_KEY: current_data_dir_fingerprint,
                "last_update": datetime.now(timezone.utc).isoformat(),
            })
            save_manifest(manifest_path, manifest)
            total_samples += len(processed_files)
            print(f"[ok] {key}: +{valid_samples_count}/{len(new_items)} new samples, {len(processed_files)} total -> {group_path}")

            del group_rows
            del acc_by_color

        for group_path in existing_group_paths:
            if group_path.resolve() in seen_group_paths:
                continue
            try:
                group_rows = read_csv_gz(group_path)
            except Exception as exc:
                print(f"[warn] Skipping unreadable published height ensemble ({exc}): {group_path}")
                continue
            if not group_rows:
                continue
            all_writer.writerows(group_rows)
            print(f"[keep] published-only group -> {group_path}")

    print(f"[done] averaged {total_samples} samples in {len(data_dirs)} groups")
    print(f"[done] series -> {all_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
