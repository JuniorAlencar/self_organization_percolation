#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

FLOAT_PATTERN = r"(?:[0-9]*\.[0-9]+|[0-9]+(?:\.[0-9]*)?)(?:[eE][+-]?[0-9]+)?"
RE_P0 = re.compile(rf"(?:^|_)P0_(?P<P0>{FLOAT_PATTERN})(?:_|\.json$)")
RE_p0 = re.compile(rf"(?:^|_)p0_(?P<p0>{FLOAT_PATTERN})(?:_|\.json$)")
RE_SAMPLE_LAYOUT = re.compile(r"(?:^|_)base_(?P<layout>[a-zA-Z0-9_-]+?)(?:\.json|\.yts|$)")


def extract_layout_from_filename(filename: str) -> str:
    """Extract initial layout from sample filename using regex, defaulting to 'random'."""
    fname = filename.split("::")[-1]
    m = RE_SAMPLE_LAYOUT.search(fname)
    if m:
        return m.group("layout")
    return "random"


def parse_p0_p0_from_filename(filename: str) -> tuple[float, float] | None:
    fname = filename.split("::")[-1]
    m_P0 = RE_P0.search(fname)
    m_p0 = RE_p0.search(fname)
    if m_P0 and m_p0:
        return round(float(m_P0.group("P0")), 6), round(float(m_p0.group("p0")), 6)
    return None


def read_bundle(path: Path) -> dict[str, Any]:
    if path.suffix == ".xz":
        proc = subprocess.run(["xz", "-dc", str(path)], stdout=subprocess.PIPE, check=True)
        return json.loads(proc.stdout.decode("utf-8"))
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_bundle_atomic(path: Path, data: dict[str, Any]) -> None:
    tmp_path = path.with_name(f"{path.stem}.tmp{path.suffix}")
    json_bytes = json.dumps(data, separators=(",", ":")).encode("utf-8")
    if path.suffix == ".xz":
        with tmp_path.open("wb") as out_f:
            subprocess.run(
                ["xz", "-1", "-T1", "-c"],
                input=json_bytes,
                stdout=out_f,
                check=True,
            )
    else:
        with tmp_path.open("wb") as out_f:
            out_f.write(json_bytes)
            out_f.write(b"\n")
    tmp_path.replace(path)


def load_manifest_files_by_p0(manifest_path: Path) -> dict[tuple[float, float], list[str]]:
    if not manifest_path.exists():
        return {}
    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        files_by_p0: dict[tuple[float, float], list[str]] = {}
        for s in manifest.get("processed_json_files", []):
            parsed = parse_p0_p0_from_filename(s)
            if parsed:
                files_by_p0.setdefault(parsed, []).append(s)
        return files_by_p0
    except Exception:
        return {}


def migrate_single_bundle(
    bundle_path: Path,
    manifests_root: Path,
    published_root: Path,
    dry_run: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    try:
        bundle = read_bundle(bundle_path)
    except Exception as exc:
        return {"status": "error", "path": str(bundle_path), "error": str(exc)}

    p0_groups = bundle.get("p0_groups", [])
    if not isinstance(p0_groups, list):
        return {"status": "skipped", "path": str(bundle_path), "reason": "no p0_groups"}

    # Check if already migrated
    needs_update = force
    if not needs_update:
        for pg in p0_groups:
            if not isinstance(pg, dict):
                continue
            if "initial_layout" not in pg or "initial_base_layout" not in pg:
                needs_update = True
                break
        if "initial_layout" not in bundle.get("meta", {}):
            needs_update = True

    if not needs_update:
        return {"status": "skipped", "path": str(bundle_path), "reason": "already migrated"}

    # Look up manifest if any group needs fallback
    rel_group = bundle_path.parent.relative_to(published_root)
    manifest_path = manifests_root / rel_group / "manifest.json"
    manifest_files_by_p0: dict[tuple[float, float], list[str]] | None = None

    group_layouts_summary = []
    for pg in p0_groups:
        if not isinstance(pg, dict):
            continue

        P0 = round(float(pg.get("P0_value", 0.0) or 0.0), 6)
        p0 = round(float(pg.get("p0_value", 0.0) or 0.0), 6)

        sample_layouts: list[str] = []
        orders = pg.get("orders", [])
        if isinstance(orders, list):
            for order in orders:
                if not isinstance(order, dict):
                    continue
                samples = order.get("samples", [])
                if isinstance(samples, list):
                    for sample in samples:
                        if not isinstance(sample, dict):
                            continue
                        fn = sample.get("filename", "")
                        layout = extract_layout_from_filename(fn)
                        sample["initial_layout"] = layout
                        sample["initial_base_layout"] = layout
                        sample_layouts.append(layout)

        # Fallback to manifest if no samples were in orders (e.g. no species stabilized)
        if not sample_layouts:
            if manifest_files_by_p0 is None:
                manifest_files_by_p0 = load_manifest_files_by_p0(manifest_path)
            matched_files = manifest_files_by_p0.get((P0, p0), [])
            if not matched_files:
                # Try matching by P0 only or p0 only if exact float rounding differed slightly
                for (mP0, mp0), mfiles in manifest_files_by_p0.items():
                    if abs(mP0 - P0) < 1e-4 and abs(mp0 - p0) < 1e-4:
                        matched_files = mfiles
                        break
            for fn in matched_files:
                sample_layouts.append(extract_layout_from_filename(fn))

        if not sample_layouts:
            sample_layouts = ["random"]

        unique_layouts = sorted(set(sample_layouts))
        layout_val = unique_layouts[0] if len(unique_layouts) == 1 else unique_layouts

        pg["initial_layout"] = layout_val
        pg["initial_base_layout"] = layout_val
        if len(unique_layouts) > 1:
            pg["initial_layouts"] = unique_layouts
            pg["initial_layout_counts"] = dict(Counter(sample_layouts))

        group_layouts_summary.append((P0, p0, layout_val))

    # Update bundle meta
    meta = bundle.setdefault("meta", {})
    if isinstance(meta, dict):
        all_meta_layouts = sorted({
            l
            for pg in p0_groups
            if isinstance(pg, dict) and "initial_layout" in pg
            for l in (
                pg["initial_layout"]
                if isinstance(pg["initial_layout"], list)
                else [pg["initial_layout"]]
            )
        })
        meta_layout_val = (
            all_meta_layouts[0] if len(all_meta_layouts) == 1 else all_meta_layouts
        ) if all_meta_layouts else "random"
        meta["initial_layout"] = meta_layout_val
        meta["initial_base_layout"] = meta_layout_val

    if not dry_run:
        write_bundle_atomic(bundle_path, bundle)

    return {
        "status": "updated",
        "path": str(bundle_path),
        "groups": group_layouts_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract InitialLayout from dynamic sample names and store in p0_groups."
    )
    parser.add_argument(
        "--sop-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "SOP_data",
        help="SOP data root directory (default: SOP_data)",
    )
    parser.add_argument(
        "--published-dir",
        type=str,
        default="published_dynamic",
        help="Published dynamic directory name",
    )
    parser.add_argument(
        "--manifests-dir",
        type=str,
        default="manifests_dynamic",
        help="Manifests dynamic directory name",
    )
    parser.add_argument(
        "-j", "--jobs",
        type=int,
        default=min(10, os.cpu_count() or 4),
        help="Number of worker processes",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run without writing changes",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force migration even if initial_layout is already present",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of bundles to process (for testing)",
    )
    args = parser.parse_args()

    sop_root = args.sop_root.resolve()
    published_root = sop_root / args.published_dir
    manifests_root = sop_root / args.manifests_dir

    if not published_root.exists():
        print(f"[ERROR] Published directory not found: {published_root}", file=sys.stderr)
        sys.exit(1)

    bundle_paths = sorted(
        set(published_root.rglob("properties_dynamic_bundle.json.xz"))
        | set(published_root.rglob("properties_dynamic_bundle.json.gz"))
        | set(published_root.rglob("properties_dynamic_bundle.json"))
    )

    if args.limit > 0:
        bundle_paths = bundle_paths[:args.limit]

    total_bundles = len(bundle_paths)
    print(f"[migrate_layout] Found {total_bundles} bundles to inspect.")
    print(f"[migrate_layout] Config: sop_root={sop_root} jobs={args.jobs} dry_run={args.dry_run} force={args.force}")

    t0 = time.time()
    updated_count = 0
    skipped_count = 0
    error_count = 0
    layout_counter: Counter[str] = Counter()

    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(
                migrate_single_bundle,
                b_path,
                manifests_root,
                published_root,
                dry_run=args.dry_run,
                force=args.force,
            ): b_path
            for b_path in bundle_paths
        }

        completed = 0
        for future in as_completed(futures):
            res = future.result()
            completed += 1
            status = res.get("status")
            if status == "updated":
                updated_count += 1
                for _, _, layout in res.get("groups", []):
                    if isinstance(layout, list):
                        for l in layout:
                            layout_counter[f"mixed_{l}"] += 1
                    else:
                        layout_counter[layout] += 1
            elif status == "skipped":
                skipped_count += 1
            elif status == "error":
                error_count += 1
                print(f"[ERROR] {res.get('path')}: {res.get('error')}", file=sys.stderr)

            if completed % 200 == 0 or completed == total_bundles:
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                print(
                    f"[migrate_layout] Progress: {completed}/{total_bundles} "
                    f"({completed*100.0/total_bundles:.1f}%) | "
                    f"updated={updated_count} skipped={skipped_count} errors={error_count} "
                    f"({rate:.1f} bundles/s)"
                )

    total_time = time.time() - t0
    print(f"\n[migrate_layout] Finished in {total_time:.2f}s.")
    print(f"[migrate_layout] Total bundles: {total_bundles}")
    print(f"[migrate_layout] Updated: {updated_count}")
    print(f"[migrate_layout] Skipped: {skipped_count}")
    print(f"[migrate_layout] Errors: {error_count}")
    print(f"[migrate_layout] Layout distribution across p0_groups: {dict(layout_counter)}")


if __name__ == "__main__":
    main()
