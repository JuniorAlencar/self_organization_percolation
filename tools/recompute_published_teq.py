#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

THIS_FILE = Path(__file__).resolve()
TOOLS_DIR = THIS_FILE.parent

import sys

if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from process_data import _tail_stats_from_mean_series, weighted_mean_and_sem


def default_published_root() -> Path:
    return (TOOLS_DIR / ".." / "SOP_data" / "published").resolve()


def rolling_mean(y: np.ndarray, window: int, center: bool, min_periods: int) -> np.ndarray:
    return (
        pd.Series(np.asarray(y, dtype=float))
        .rolling(window=window, center=center, min_periods=min_periods)
        .mean()
        .to_numpy()
    )


def block_mean_regular_time(
    t: np.ndarray,
    y: np.ndarray,
    window_block: int,
    drop_last: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if window_block < 1:
        raise ValueError("window_block deve ser >= 1.")

    n = min(t.size, y.size)
    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    t = t[:n]
    y = y[:n]
    n_blocks = n // window_block
    if n_blocks == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    n_use = n_blocks * window_block if drop_last else n
    centers: list[float] = []
    means: list[float] = []

    for i0 in range(0, n_use, window_block):
        i1 = min(i0 + window_block, n)
        if drop_last and (i1 - i0) < window_block:
            continue

        t_block = t[i0:i1]
        y_block = y[i0:i1]
        mask = np.isfinite(t_block) & np.isfinite(y_block)
        if not np.any(mask):
            continue

        centers.append(float(np.mean(t_block[mask])))
        means.append(float(np.mean(y_block[mask])))

    return np.asarray(centers, dtype=float), np.asarray(means, dtype=float)


def first_teq_from_s_prime(
    t: Any,
    pt_mean: Any,
    *,
    threshold: float,
    window_roll: int,
    window_block: int,
    center: bool,
    drop_last: bool,
    use_rolling_for_blocks: bool,
    abs_s_prime: bool,
) -> float | None:
    t_arr = np.asarray(t, dtype=float)
    p_arr = np.asarray(pt_mean, dtype=float)
    n = min(t_arr.size, p_arr.size)
    if n == 0:
        return None

    t_arr = t_arr[:n]
    p_arr = p_arr[:n]
    finite = np.isfinite(t_arr) & np.isfinite(p_arr)
    if np.count_nonzero(finite) < max(2, window_block):
        return None

    t_arr = t_arr[finite]
    p_arr = p_arr[finite]

    if use_rolling_for_blocks:
        y_for_blocks = rolling_mean(
            p_arr,
            window=window_roll,
            center=center,
            min_periods=1,
        )
    else:
        y_for_blocks = p_arr

    t_j, j_w = block_mean_regular_time(
        t_arr,
        y_for_blocks,
        window_block=window_block,
        drop_last=drop_last,
    )
    if j_w.size < 3:
        return None

    s = np.abs(np.diff(j_w))
    t_s = 0.5 * (t_j[:-1] + t_j[1:])
    if s.size < 2:
        return None

    s_prime = np.gradient(s, t_s)
    test_values = np.abs(s_prime) if abs_s_prime else s_prime
    matches = np.flatnonzero(np.isfinite(test_values) & (test_values < threshold))
    if matches.size == 0:
        return None

    return float(t_s[int(matches[0])])


def null_order_pc_sop(existing: Any, n_boot: int) -> dict[str, Any]:
    old = existing if isinstance(existing, dict) else {}
    return {
        "mean": None,
        "std_boot": None,
        "n_tail_points": 0,
        "n_boot": int(old.get("n_boot", n_boot) or n_boot),
        "t0": None,
        "pc_method": "null: no s_prime below stability threshold",
    }


def null_group_pc_sop(existing: Any, n_boot: int) -> dict[str, Any]:
    old = existing if isinstance(existing, dict) else {}
    return {
        "mean": None,
        "std_boot": None,
        "n_seeds": int(old.get("n_seeds", 0) or 0),
        "n_boot": int(old.get("n_boot", n_boot) or n_boot),
        "t0_global": None,
        "pc_method": "null: no finite t_eq from s_prime threshold",
    }


def update_bundle(
    bundle: dict[str, Any],
    *,
    threshold: float,
    window_roll: int,
    window_block: int,
    center: bool,
    drop_last: bool,
    use_rolling_for_blocks: bool,
    abs_s_prime: bool,
) -> tuple[bool, dict[str, int]]:
    meta = bundle.setdefault("meta", {})
    bootstrap = meta.get("bootstrap", {}) if isinstance(meta.get("bootstrap", {}), dict) else {}
    n_boot = int(bootstrap.get("n_boot", 20000) or 20000)

    stats = {
        "groups": 0,
        "orders": 0,
        "orders_with_teq": 0,
        "orders_without_teq": 0,
    }
    changed = False

    meta["t_eq_recompute"] = {
        "method": "first t_s where s_prime < threshold from block means of pt_mean",
        "threshold": float(threshold),
        "window_roll": int(window_roll),
        "window_block": int(window_block),
        "center": bool(center),
        "drop_last": bool(drop_last),
        "use_rolling_for_blocks": bool(use_rolling_for_blocks),
        "abs_s_prime": bool(abs_s_prime),
    }

    for group in bundle.get("p0_groups", []):
        if not isinstance(group, dict):
            continue

        stats["groups"] += 1
        order_pc_values: list[tuple[float, float]] = []
        order_t_eq_values: list[float] = []

        for order in group.get("orders", []):
            if not isinstance(order, dict):
                continue
            data = order.get("data", {})
            if not isinstance(data, dict):
                continue

            stats["orders"] += 1
            t = data.get("time", [])
            pt = data.get("pt_mean", [])
            pt_sem = data.get("pt_sem", [])

            t_eq = first_teq_from_s_prime(
                t,
                pt,
                threshold=threshold,
                window_roll=window_roll,
                window_block=window_block,
                center=center,
                drop_last=drop_last,
                use_rolling_for_blocks=use_rolling_for_blocks,
                abs_s_prime=abs_s_prime,
            )

            if t_eq is None:
                data["t_eq"] = None
                data["t_eq_source"] = "none: no s_prime below stability threshold"
                data["pc_sop"] = null_order_pc_sop(data.get("pc_sop"), n_boot)
                stats["orders_without_teq"] += 1
                changed = True
                continue

            t_arr = np.asarray(t, dtype=float)
            pt_arr = np.asarray(pt, dtype=float)
            sem_arr = np.asarray(pt_sem, dtype=float)
            pc_mean, pc_sem, n_tail = _tail_stats_from_mean_series(
                t_arr,
                pt_arr,
                sem_arr,
                t_eq,
            )

            data["t_eq"] = float(t_eq)
            data["t_eq_source"] = (
                "recomputed from first t_s where s_prime < stability threshold"
            )
            data["pc_sop"] = {
                "mean": float(pc_mean) if math.isfinite(pc_mean) else None,
                "std_boot": float(pc_sem) if math.isfinite(pc_sem) else None,
                "n_tail_points": int(n_tail),
                "n_boot": n_boot,
                "t0": float(t_eq),
                "pc_method": (
                    "ensemble-mean tail after t_eq from s_prime stability threshold"
                ),
            }

            if math.isfinite(pc_mean) and math.isfinite(pc_sem) and pc_sem > 0:
                order_pc_values.append((float(pc_mean), float(pc_sem)))
            order_t_eq_values.append(float(t_eq))
            stats["orders_with_teq"] += 1
            changed = True

        if order_pc_values:
            means = [m for m, _ in order_pc_values]
            sems = [s for _, s in order_pc_values]
            pc_mean, pc_sem = weighted_mean_and_sem(means, sems)
            t0_global = float(max(order_t_eq_values)) if order_t_eq_values else None
            group["pc_sop"] = {
                "mean": float(pc_mean),
                "std_boot": float(pc_sem),
                "n_seeds": int(group.get("num_seeds", 0) or 0),
                "n_boot": n_boot,
                "t0_global": t0_global,
                "pc_method": (
                    "combine orders of ensemble-mean tails after order t_eq from "
                    "s_prime stability threshold"
                ),
            }
        else:
            group["pc_sop"] = null_group_pc_sop(group.get("pc_sop"), n_boot)
        changed = True

    return changed, stats


def iter_bundle_paths(root: Path) -> list[Path]:
    return sorted(root.rglob("properties_mean_bundle.json"))


def render_json(bundle: dict[str, Any]) -> str:
    return json.dumps(bundle, indent=2, allow_nan=True) + "\n"


def atomic_write_text(path: Path, text: str) -> None:
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=str(path.parent),
            delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(text)

        os.replace(tmp_path, path)
    except Exception:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Recalcula t_eq e pc_sop nos properties_mean_bundle.json publicados "
            "a partir do primeiro s_prime abaixo de um limiar."
        )
    )
    parser.add_argument("--published-root", default=str(default_published_root()))
    parser.add_argument("--threshold", type=float, default=1e-6)
    parser.add_argument("--window-roll", type=int, default=15)
    parser.add_argument("--window-block", type=int, default=20)
    parser.add_argument("--no-center", action="store_true")
    parser.add_argument("--keep-last-block", action="store_true")
    parser.add_argument("--raw-blocks", action="store_true")
    parser.add_argument(
        "--abs-s-prime",
        action="store_true",
        help="Usa abs(s_prime) < threshold em vez de s_prime < threshold.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    root = Path(args.published_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"published-root não encontrado: {root}")

    paths = iter_bundle_paths(root)
    if args.limit is not None:
        paths = paths[: args.limit]

    total = {
        "files": 0,
        "changed_files": 0,
        "groups": 0,
        "orders": 0,
        "orders_with_teq": 0,
        "orders_without_teq": 0,
    }

    for idx, path in enumerate(paths, start=1):
        old_text = path.read_text(encoding="utf-8")
        bundle = json.loads(old_text)

        changed, stats = update_bundle(
            bundle,
            threshold=args.threshold,
            window_roll=args.window_roll,
            window_block=args.window_block,
            center=not args.no_center,
            drop_last=not args.keep_last_block,
            use_rolling_for_blocks=not args.raw_blocks,
            abs_s_prime=args.abs_s_prime,
        )

        total["files"] += 1
        for key, value in stats.items():
            total[key] += value

        if changed:
            new_text = render_json(bundle)
            changed = new_text != old_text
            if changed:
                total["changed_files"] += 1
                if not args.dry_run:
                    atomic_write_text(path, new_text)

        if not args.quiet and (idx == 1 or idx % 100 == 0 or idx == len(paths)):
            action = "dry-run" if args.dry_run else "write"
            print(
                f"[{action}] {idx}/{len(paths)} files | "
                f"orders_with_teq={total['orders_with_teq']} | "
                f"orders_without_teq={total['orders_without_teq']}"
            )

    print(
        "Resumo: "
        f"files={total['files']} changed_files={total['changed_files']} "
        f"groups={total['groups']} orders={total['orders']} "
        f"orders_with_teq={total['orders_with_teq']} "
        f"orders_without_teq={total['orders_without_teq']} "
        f"dry_run={args.dry_run}"
    )


if __name__ == "__main__":
    main()
