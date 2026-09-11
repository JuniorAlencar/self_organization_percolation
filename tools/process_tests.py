#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


DEFAULT_TESTS = ("relative",)


def run_command(cmd: list[str], *, cwd: Path) -> None:
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def process_dynamic_test(
    *,
    project_root: Path,
    sop_root: Path,
    test_name: str,
    jobs: int,
    fingerprint_mode: str,
    series_mode: str,
    detect_replaced_files: bool,
    include_laterals: bool,
    write_all_data: bool,
    migrate_published: bool,
    clear: bool,
) -> None:
    script = project_root / "tools" / "process_dynamic_growth.py"
    cmd = [
        "python3",
        str(script),
        "--sop-root",
        str(sop_root),
        "--raw-dir",
        f"tests_data/{test_name}",
        "--published-dir",
        f"processed_tests/{test_name}",
        "--manifests-dir",
        f"manifests_tests/{test_name}",
        "--all-data-name",
        f"all_data_tests_{test_name}.dat",
        "--all-colors-name",
        f"all_colors_tests_{test_name}.dat",
        "--fingerprint-mode",
        fingerprint_mode,
        "--series-mode",
        series_mode,
        "-j",
        str(jobs),
    ]
    cmd.append("--detect-replaced-files" if detect_replaced_files else "--no-detect-replaced-files")
    cmd.append("--include-laterals" if include_laterals else "--no-laterals")
    cmd.append("--write-all-data" if write_all_data else "--skip-all-data")
    cmd.append("--migrate-published" if migrate_published else "--no-migrate-published")
    if clear:
        cmd.append("--clear")
    run_command(cmd, cwd=project_root)


def process_height_test_samples(
    *,
    project_root: Path,
    sop_root: Path,
    test_name: str,
    max_samples_per_group: int | None,
) -> None:
    script = project_root / "tools" / "process_height_timeseries.py"
    cmd = [
        "python3",
        str(script),
        "--root",
        str(sop_root / "tests_data" / test_name),
        "--out-root",
        str(sop_root / "height_tests" / test_name),
    ]
    if max_samples_per_group is not None:
        cmd.extend(["--max-samples-per-group", str(max_samples_per_group)])
    run_command(cmd, cwd=project_root)


def process_height_test_ensemble(
    *,
    project_root: Path,
    sop_root: Path,
    test_name: str,
    min_count: int,
    max_samples_per_group: int | None,
) -> None:
    script = project_root / "tools" / "process_height_ensemble_series.py"
    cmd = [
        "python3",
        str(script),
        "--root",
        str(sop_root / "tests_data" / test_name),
        "--out-root",
        str(sop_root / "height_tests" / test_name),
        "--min-count",
        str(min_count),
    ]
    if max_samples_per_group is not None:
        cmd.extend(["--max-samples-per-group", str(max_samples_per_group)])
    run_command(cmd, cwd=project_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process SOP tests_data for dynamic summaries and .yts height series."
    )
    project_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--sop-root", default=str(project_root / "SOP_data"))
    parser.add_argument("--tests", nargs="+", default=list(DEFAULT_TESTS))
    parser.add_argument("-j", "--jobs", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--fingerprint-mode", choices=("stat", "hash"), default="stat")
    parser.add_argument("--series-mode", choices=("full", "profiles", "scalars"), default="full")
    parser.add_argument("--detect-replaced-files", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--laterals", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--write-all-data", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--migrate-published", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--height-min-count", type=int, default=1)
    parser.add_argument("--height-max-samples-per-group", type=int, default=None)
    parser.add_argument("--skip-dynamic", action="store_true")
    parser.add_argument("--skip-height-samples", action="store_true")
    parser.add_argument("--skip-height-ensemble", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    sop_root = Path(args.sop_root).expanduser().resolve()
    tests = [str(test).strip() for test in args.tests if str(test).strip()]

    if not tests:
        raise SystemExit("No tests selected.")

    for test_name in tests:
        raw_root = sop_root / "tests_data" / test_name
        if not raw_root.exists():
            print(f"[skip] {test_name}: {raw_root} does not exist", flush=True)
            continue

        print(f"[test] {test_name}", flush=True)
        if not args.skip_dynamic:
            process_dynamic_test(
                project_root=project_root,
                sop_root=sop_root,
                test_name=test_name,
                jobs=args.jobs,
                fingerprint_mode=args.fingerprint_mode,
                series_mode=args.series_mode,
                detect_replaced_files=args.detect_replaced_files,
                include_laterals=args.laterals,
                write_all_data=args.write_all_data,
                migrate_published=args.migrate_published,
                clear=args.clear,
            )
        if not args.skip_height_samples:
            process_height_test_samples(
                project_root=project_root,
                sop_root=sop_root,
                test_name=test_name,
                max_samples_per_group=args.height_max_samples_per_group,
            )
        if not args.skip_height_ensemble:
            process_height_test_ensemble(
                project_root=project_root,
                sop_root=sop_root,
                test_name=test_name,
                min_count=args.height_min_count,
                max_samples_per_group=args.height_max_samples_per_group,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
