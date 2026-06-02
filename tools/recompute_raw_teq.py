#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from recompute_published_teq import estimate_teq_block_derivative


TOOLS_DIR = Path(__file__).resolve().parent


def default_root() -> Path:
    return (TOOLS_DIR / ".." / "SOP_data").resolve()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    text = json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n"
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


def iter_raw_json_files(root: Path):
    for path in root.rglob("*.json"):
        rel_parts = path.relative_to(root).parts
        if not rel_parts:
            continue
        if rel_parts[0].startswith("raw"):
            yield path


def count_raw_json_files(root: Path, limit: int | None = None) -> int:
    count = 0
    for _ in iter_raw_json_files(root):
        count += 1
        if limit is not None and count >= limit:
            break
    return count


def format_duration(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "--:--"
    seconds_i = int(round(seconds))
    hours, rem = divmod(seconds_i, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def print_progress(done: int, total: int, start_time: float) -> None:
    width = 32
    frac = 0.0 if total <= 0 else min(1.0, done / total)
    filled = int(round(width * frac))
    bar = "#" * filled + "-" * (width - filled)
    elapsed = time.monotonic() - start_time
    eta = (elapsed / done) * (total - done) if done > 0 and total > 0 else float("nan")
    print(
        f"\r[{bar}] {done}/{total} ({100.0 * frac:5.1f}%) "
        f"elapsed={format_duration(elapsed)} eta={format_duration(eta)}",
        end="",
        flush=True,
    )


def _as_float_array(value: Any) -> np.ndarray | None:
    try:
        arr = np.asarray(value, dtype=float)
    except Exception:
        return None
    if arr.ndim != 1 or arr.size == 0:
        return None
    return arr


def extract_mean_pt(js: dict[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    results = js.get("results", {})
    if not isinstance(results, dict):
        return None

    rows: list[np.ndarray] = []
    t_ref: np.ndarray | None = None
    min_len: int | None = None

    for block in results.values():
        if not isinstance(block, dict):
            continue
        data = block.get("data", {})
        if not isinstance(data, dict):
            continue

        t = _as_float_array(data.get("time", data.get("t")))
        pt = _as_float_array(data.get("pt"))
        if t is None or pt is None:
            continue

        n = min(t.size, pt.size)
        if n <= 0:
            continue
        t = t[:n]
        pt = pt[:n]

        if t_ref is None:
            t_ref = t
            min_len = n
        else:
            min_len = min(min_len or n, n, t_ref.size)

        rows.append(pt)

    if t_ref is None or min_len is None or not rows:
        return None

    t_out = t_ref[:min_len]
    mat = np.vstack([row[:min_len] for row in rows])
    pt_mean = np.nanmean(mat, axis=0)
    finite = np.isfinite(t_out) & np.isfinite(pt_mean)
    if np.count_nonzero(finite) == 0:
        return None
    return t_out[finite], pt_mean[finite]


def recompute_file(
    path: Path,
    *,
    threshold: float,
    window_roll: int,
    window_block: int,
    min_stable_steps: int,
    rel_tol: float,
    abs_tol: float,
    center: bool,
    drop_last: bool,
    use_rolling_for_blocks: bool,
) -> tuple[bool, float | None, float | None, dict[str, Any] | None]:
    try:
        js = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False, None, None, None

    if not isinstance(js, dict):
        return False, None, None, None

    series = extract_mean_pt(js)
    if series is None:
        return False, None, None, None

    t, pt_mean = series
    t_eq = estimate_teq_block_derivative(
        t,
        pt_mean,
        s_prime_threshold=threshold,
        window_roll=window_roll,
        window_block=window_block,
        min_stable_steps=min_stable_steps,
        rel_tol=rel_tol,
        abs_tol=abs_tol,
        center=center,
        drop_last=drop_last,
        use_rolling_for_blocks=use_rolling_for_blocks,
    )

    meta = js.setdefault("meta", {})
    if not isinstance(meta, dict):
        js["meta"] = {}
        meta = js["meta"]

    old_raw = meta.get("t_eq")
    try:
        old = float(old_raw)
        if not math.isfinite(old):
            old = None
    except Exception:
        old = None

    if t_eq is None or not math.isfinite(float(t_eq)):
        meta["t_eq"] = None
    else:
        meta["t_eq"] = float(t_eq)

    meta["t_eq_recompute"] = {
        "method": (
            "first persistent block sequence with small |delta p_block| "
            "and small |d(delta p_block)/dt|"
        ),
        "s_prime_threshold": float(threshold),
        "window_roll": int(window_roll),
        "window_block": int(window_block),
        "min_stable_steps": int(min_stable_steps),
        "rel_tol": float(rel_tol),
        "abs_tol": float(abs_tol),
        "center": bool(center),
        "drop_last": bool(drop_last),
        "use_rolling_for_blocks": bool(use_rolling_for_blocks),
    }

    return True, old, None if t_eq is None else float(t_eq), js


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recalcula meta.t_eq de todos os JSONs brutos em SOP_data/raw*."
    )
    parser.add_argument("--root", default=str(default_root()))
    parser.add_argument("--threshold", type=float, default=5.0e-4)
    parser.add_argument("--window-roll", type=int, default=15)
    parser.add_argument("--window-block", type=int, default=10)
    parser.add_argument("--min-stable-steps", type=int, default=15)
    parser.add_argument("--rel-tol", type=float, default=2.5e-2)
    parser.add_argument("--abs-tol", type=float, default=1.0e-6)
    parser.add_argument("--no-center", action="store_true")
    parser.add_argument("--keep-last-block", action="store_true")
    parser.add_argument("--raw-blocks", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"root nao encontrado: {root}")

    scanned = 0
    updated = 0
    skipped = 0
    if not args.quiet:
        print("Counting raw JSON files...", flush=True)
    total_files = count_raw_json_files(root, args.limit)
    start_time = time.monotonic()

    if not args.quiet:
        print_progress(0, total_files, start_time)

    for path in iter_raw_json_files(root):
        if args.limit is not None and scanned >= args.limit:
            break
        scanned += 1
        old_text = path.read_text(encoding="utf-8")
        ok, old, new, js = recompute_file(
            path,
            threshold=args.threshold,
            window_roll=args.window_roll,
            window_block=args.window_block,
            min_stable_steps=args.min_stable_steps,
            rel_tol=args.rel_tol,
            abs_tol=args.abs_tol,
            center=not args.no_center,
            drop_last=not args.keep_last_block,
            use_rolling_for_blocks=not args.raw_blocks,
        )
        if not ok:
            skipped += 1
            if not args.quiet and (
                scanned == total_files or scanned % max(1, args.progress_every) == 0
            ):
                print_progress(scanned, total_files, start_time)
            continue

        new_text = json.dumps(js, separators=(",", ":"), allow_nan=False) + "\n"
        changed = new_text != old_text
        if changed:
            updated += 1
            if not args.quiet:
                print(f"\n{path}: t_eq {old} -> {new}")

        if changed and not args.dry_run:
            atomic_write_json(path, js)

        if not args.quiet and (
            scanned == total_files or scanned % max(1, args.progress_every) == 0
        ):
            print_progress(scanned, total_files, start_time)

    if not args.quiet:
        print()
    print(
        f"scanned={scanned} updated={updated} skipped={skipped} "
        f"dry_run={bool(args.dry_run)}"
    )


if __name__ == "__main__":
    main()
