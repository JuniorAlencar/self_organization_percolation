#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def hole_value_to_counts(value: Any) -> dict[str, int]:
    if isinstance(value, dict):
        out: dict[str, int] = {}
        for key, count in value.items():
            try:
                size = str(int(key))
                n = int(count)
            except (TypeError, ValueError):
                continue
            if n > 0:
                out[size] = out.get(size, 0) + n
        return dict(sorted(out.items(), key=lambda item: int(item[0])))

    if not isinstance(value, list):
        return {}

    counts: Counter[int] = Counter()
    for item in value:
        try:
            counts[int(item)] += 1
        except (TypeError, ValueError):
            continue
    return {str(size): counts[size] for size in sorted(counts)}


def migrate_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    data = payload.get("data")
    if not isinstance(data, dict):
        return payload, False

    changed = False
    if "hole_sizes" in data:
        raw_holes = data.pop("hole_sizes")
        if "hole_size_counts" not in data:
            if isinstance(raw_holes, list):
                data["hole_size_counts"] = [hole_value_to_counts(row) for row in raw_holes]
            else:
                data["hole_size_counts"] = []
        changed = True

    hull = data.get("hull_length")
    external = data.get("external_perimeter_length")
    if isinstance(hull, list) and isinstance(external, list):
        if "full_boundary_length" not in data:
            data["full_boundary_length"] = hull
            changed = True
        if data.get("hull_length") != external:
            data["hull_length"] = external
            changed = True

    meta = payload.get("meta")
    if isinstance(meta, dict):
        try:
            dim = int(meta.get("dim"))
            L = int(meta.get("L"))
        except (TypeError, ValueError):
            dim = 0
            L = 0
        if L > 0 and dim in {2, 3}:
            expected_n_total = L * L if dim == 2 else L * L * L
            expected_e_total = (2 * L * L - L) if dim == 2 else (3 * L * L * L - L * L)
            if meta.get("N_total") != expected_n_total:
                meta["N_total"] = expected_n_total
                changed = True
            if meta.get("E_total") != expected_e_total:
                meta["E_total"] = expected_e_total
                changed = True

        convention = meta.get("hull_convention")
        new_convention = (
            "hull_length counts the exterior hull of the giant component: edges in "
            "2D or faces in 3D adjacent to the exterior complement, excluding "
            "enclosed holes/cavities; full_boundary_length counts all "
            "component-complement boundary elements including internal "
            "holes/cavities/fjords; external_perimeter_length is kept as an alias "
            "of hull_length for backward comparison; hole_size_counts stores "
            "enclosed void area/volume histograms per sample as {size: count}"
        )
        if convention != new_convention:
            meta["hull_convention"] = new_convention
            changed = True

    return payload, changed


def migrate_file(path: Path, dry_run: bool) -> bool:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        return False

    migrated, changed = migrate_payload(payload)
    if not changed:
        return False
    if dry_run:
        return True

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(migrated, fh, ensure_ascii=False, indent=2, allow_nan=False)
        fh.write("\n")
    tmp_path.replace(path)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert raw_fractions hole_sizes lists to compact hole_size_counts maps."
    )
    parser.add_argument("--raw-root", type=Path, default=Path("SOP_data/raw_fractions"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    paths = sorted(args.raw_root.rglob("*.json"))
    changed = 0
    failed = 0
    for path in paths:
        try:
            if migrate_file(path, args.dry_run):
                changed += 1
                if not args.quiet:
                    action = "would migrate" if args.dry_run else "migrated"
                    print(f"[{action}] {path}")
        except Exception as exc:
            failed += 1
            print(f"[error] {path}: {exc}")

    print(
        f"[migrate_raw_fraction_holes] scanned={len(paths)} "
        f"changed={changed} failed={failed} dry_run={args.dry_run}"
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
