#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from process_height_timeseries import (
    PATH_RE,
    parse_path,
    group_key,
    read_yts,
    output_group_dir,
)


FIELDS = [
    "type_perc",
    "num_colors",
    "dim",
    "L",
    "f_T",
    "c",
    "f0",
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


class RunningSeries:
    def __init__(self) -> None:
        self.n = np.zeros(0, dtype=np.int32)
        self.sum = np.zeros(0, dtype=np.float64)
        self.sumsq = np.zeros(0, dtype=np.float64)

    def _ensure_size(self, size: int) -> None:
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
        np.add.at(self.n, idx, 1)
        np.add.at(self.sum, idx, vals)
        np.add.at(self.sumsq, idx, vals * vals)

    def mean_std_sem(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def collect_groups(root: Path, args: argparse.Namespace) -> dict[tuple, list[tuple]]:
    groups: dict[tuple, list[tuple]] = defaultdict(list)
    for path in sorted(root.rglob("*.yts")):
        if not PATH_RE.search(str(path)):
            continue
        meta = parse_path(path)
        if args.type_perc and meta.type_perc != args.type_perc:
            continue
        if args.lengths and meta.L not in args.lengths:
            continue
        if args.f_T and not any(math.isclose(meta.f_T, value, rel_tol=1e-12, abs_tol=1e-12) for value in args.f_T):
            continue
        groups[group_key(meta)].append((meta, path))
    if args.max_samples_per_group:
        groups = {key: values[: args.max_samples_per_group] for key, values in groups.items()}
    return groups


def rows_for_color(meta, color: int, acc: dict[str, RunningSeries], min_count: int) -> list[dict]:
    width_mean, width_std, width_sem = acc["y_width"].mean_std_sem()
    y_mean_mean, y_mean_std, y_mean_sem = acc["y_mean"].mean_std_sem()
    y_max_mean, y_max_std, y_max_sem = acc["y_max"].mean_std_sem()
    front_mean, front_std, front_sem = acc["y_front_count"].mean_std_sem()
    n = acc["y_width"].n
    rows = []
    for t in np.flatnonzero(n >= min_count):
        rows.append(
            {
                "type_perc": meta.type_perc,
                "num_colors": meta.num_colors,
                "dim": meta.dim,
                "L": meta.L,
                "f_T": f"{meta.f_T:.17g}",
                "c": f"{meta.c:.17g}",
                "f0": "" if meta.f0 is None else f"{meta.f0:.17g}",
                "epsilon": "" if meta.epsilon is None else f"{meta.epsilon:.17g}",
                "rho": f"{meta.rho:.17g}",
                "color": color,
                "t": int(t),
                "n": int(n[t]),
                "y_width_mean": width_mean[t],
                "y_width_std": width_std[t],
                "y_width_sem": width_sem[t],
                "y_mean_mean": y_mean_mean[t],
                "y_mean_std": y_mean_std[t],
                "y_mean_sem": y_mean_sem[t],
                "y_max_mean": y_max_mean[t],
                "y_max_std": y_max_std[t],
                "y_max_sem": y_max_sem[t],
                "y_front_count_mean": front_mean[t],
                "y_front_count_std": front_std[t],
                "y_front_count_sem": front_sem[t],
            }
        )
    return rows


def write_csv_gz(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


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

    groups = collect_groups(args.root, args)
    if not groups:
        raise SystemExit("No .yts files selected.")

    all_rows: list[dict] = []
    total_samples = 0
    for key, items in sorted(groups.items()):
        first_meta = items[0][0]
        colors = args.colors if args.colors is not None else list(range(first_meta.num_colors))
        acc_by_color = {
            color: {
                "y_width": RunningSeries(),
                "y_mean": RunningSeries(),
                "y_max": RunningSeries(),
                "y_front_count": RunningSeries(),
            }
            for color in colors
        }
        for _, path in items:
            data = read_yts(path, colors)
            for idx, color in enumerate(colors):
                acc_by_color[color]["y_width"].add(data.time, data.y_width[idx])
                acc_by_color[color]["y_mean"].add(data.time, data.y_mean[idx])
                acc_by_color[color]["y_max"].add(data.time, data.y_max[idx])
                acc_by_color[color]["y_front_count"].add(data.time, data.y_front_count[idx])

        group_rows: list[dict] = []
        for color, acc in acc_by_color.items():
            group_rows.extend(rows_for_color(first_meta, color, acc, args.min_count))

        out_dir = output_group_dir(args.root, args.out_root, items[0][1])
        write_csv_gz(out_dir / "height_ensemble_timeseries.csv.gz", group_rows)
        all_rows.extend(group_rows)
        total_samples += len(items)
        print(f"[ok] {key}: {len(items)} samples -> {out_dir / 'height_ensemble_timeseries.csv.gz'}")

    write_csv_gz(args.out_root / "height_ensemble_timeseries_all.csv.gz", all_rows)
    print(f"[done] averaged {total_samples} samples in {len(groups)} groups")
    print(f"[done] series -> {args.out_root / 'height_ensemble_timeseries_all.csv.gz'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
