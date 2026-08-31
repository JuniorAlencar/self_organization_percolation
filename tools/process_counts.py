#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


FILENAME_TAG_RE = re.compile(
    r"_seed_(?P<seed>-?\d+)_.*_P0_(?P<P0>[^_]+)_p0_(?P<p0>[^_.]+(?:\.[^_.]+)?)"
)


BOX_OBSERVABLES = {
    "d_f": ("box_counting_component", "counts"),
    "d_hull": ("hull_complete", "box_counts"),
    "d_hull_ext": ("hull_external", "box_counts"),
}


PARAM_COLUMNS = [
    "type_percolation",
    "num_colors",
    "dim",
    "L",
    "f_T",
    "c",
    "rho",
    "gap_over_L",
    "P0",
    "p0",
]


GROUP_DIR_COLUMNS = ["type_percolation", "dim", "L", "f_T", "c", "rho"]


def parse_prefixed_int(text: str, prefix: str) -> int:
    if not text.startswith(prefix):
        raise ValueError(f"expected {prefix!r} in {text!r}")
    return int(text.removeprefix(prefix))


def parse_prefixed_float(text: str, prefix: str) -> float:
    if not text.startswith(prefix):
        raise ValueError(f"expected {prefix!r} in {text!r}")
    value = text.removeprefix(prefix)
    if value.endswith("L"):
        value = value[:-1]
    return float(value)


def parse_params_from_path(path: Path, raw_root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    rel = path.relative_to(raw_root)
    parts = rel.parts
    if len(parts) < 11:
        raise ValueError(f"unexpected counts path: {path}")

    filename_match = FILENAME_TAG_RE.search(path.name)
    meta = payload.get("meta", {}) if isinstance(payload.get("meta"), dict) else {}
    return {
        "type_percolation": parts[0].removesuffix("_percolation"),
        "num_colors": parse_prefixed_int(parts[1], "num_colors_"),
        "dim": parse_prefixed_int(parts[2], "dim_"),
        "L": parse_prefixed_int(parts[3], "L_"),
        "f_T": parse_prefixed_float(parts[5], "fT_"),
        "c": parse_prefixed_float(parts[6], "c_"),
        "rho": parse_prefixed_float(parts[7], "rho_"),
        "gap_over_L": parse_prefixed_float(parts[8], "gap_"),
        "P0": float(filename_match.group("P0")) if filename_match else None,
        "p0": float(filename_match.group("p0")) if filename_match else None,
        "file_seed": int(filename_match.group("seed")) if filename_match else meta.get("seed"),
    }


def finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def group_key(params: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(params.get(col) for col in PARAM_COLUMNS)


def key_to_params(key: tuple[Any, ...]) -> dict[str, Any]:
    return dict(zip(PARAM_COLUMNS, key, strict=True))


def compact_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def param_token(name: str, value: Any) -> str:
    return f"{name}_{compact_value(value)}"


def group_output_dir(out_root: Path, params: dict[str, Any]) -> Path:
    return out_root.joinpath(
        f"{params['type_percolation']}_percolation",
        param_token("dim", params["dim"]),
        param_token("L", params["L"]),
        param_token("fT", params["f_T"]),
        param_token("c", params["c"]),
        param_token("rho", params["rho"]),
    )


def processed_counts_name(params: dict[str, Any]) -> str:
    return f"counts_P0_{compact_value(params['P0'])}_p0_{compact_value(params['p0'])}.json"


def sample_key(sample: dict[str, Any]) -> tuple[Any, Any]:
    return (
        sample.get("source_file", sample.get("sample_id")),
        sample.get("sample_index"),
    )


def load_existing_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        return {}
    return payload


def params_from_meta(meta: dict[str, Any]) -> dict[str, Any] | None:
    params = {col: meta.get(col) for col in PARAM_COLUMNS}
    if any(params[col] is None for col in PARAM_COLUMNS):
        return None
    return params


def discover_existing_outputs(out_root: Path) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    existing: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(out_root.glob("**/counts_P0_*_p0_*.json")):
        payload = load_existing_payload(path)
        meta = payload.get("meta", {})
        if not isinstance(meta, dict):
            continue
        params = params_from_meta(meta)
        if params is None:
            continue
        existing[group_key(params)].append({
            "path": path,
            "params": params,
            "samples": payload.get("samples", []) if isinstance(payload.get("samples"), list) else [],
        })
    return existing


def normalize_existing_processed_sample(sample: dict[str, Any]) -> dict[str, Any]:
    properties = sample.get("properties")
    if not isinstance(properties, dict):
        return sample
    d_min = properties.get("d_min")
    if not isinstance(d_min, dict):
        return sample
    if "ell_min" not in d_min and "ell" in d_min:
        d_min["ell_mean"] = d_min.get("ell_mean", d_min["ell"])
        d_min["ell_min"] = d_min["ell"]
        d_min["legacy_d_min_fallback"] = True
        d_min["legacy_d_min_fallback_reason"] = (
            "Processed sample did not contain ell_min and raw counts were not available; "
            "ell_min was set to the existing ell series for backward compatibility."
        )
    d_min["y_name"] = "ell_min"
    d_min["default_y"] = "ell_min"
    d_min["ell"] = d_min["ell_min"]
    return sample


def merge_samples(
    existing_samples: list[dict[str, Any]],
    new_samples: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int, int]:
    merged_by_key: dict[tuple[Any, Any], dict[str, Any]] = {}
    order: list[tuple[Any, Any]] = []

    for sample in existing_samples:
        sample = normalize_existing_processed_sample(sample)
        key = sample_key(sample)
        if key not in merged_by_key:
            order.append(key)
        merged_by_key[key] = sample

    appended = 0
    updated = 0
    for sample in new_samples:
        key = sample_key(sample)
        if key in merged_by_key:
            updated += 1
        else:
            appended += 1
            order.append(key)
        merged_by_key[key] = sample

    return [merged_by_key[key] for key in order], appended, updated


def box_curve_for_sample(
    sample: dict[str, Any],
    observable: str,
    L: int,
    epsilon_min: float | None,
    epsilon_max: float | None,
) -> list[dict[str, Any]]:
    section_key, rows_key = BOX_OBSERVABLES[observable]
    section = sample.get(section_key)
    if not isinstance(section, dict):
        return []
    if section_key.startswith("hull") and not section.get("defined", False):
        return []
    rows = section.get(rows_key, [])
    by_eps: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        eps = finite_float(row.get("epsilon"))
        count = finite_float(row.get("num_boxes"))
        if eps is None or count is None or eps <= 0 or count <= 0:
            continue
        if epsilon_min is not None and eps < epsilon_min:
            continue
        if epsilon_max is not None and eps > epsilon_max:
            continue
        by_eps[int(eps)].append(count)

    curve = []
    for eps in sorted(by_eps):
        counts = by_eps[eps]
        avg_count = mean(counts)
        curve.append({
            "observable": observable,
            "epsilon": eps,
            "L_over_epsilon": float(L) / float(eps),
            "mean_count": avg_count,
            "n_offsets": len(counts),
        })
    return curve


def box_series(curve: list[dict[str, Any]], y_name: str) -> dict[str, Any]:
    return {
        "x_name": "L_over_epsilon",
        "y_name": y_name,
        "epsilon": [row["epsilon"] for row in curve],
        "L_over_epsilon": [row["L_over_epsilon"] for row in curve],
        y_name: [row["mean_count"] for row in curve],
        "n_offsets": [row["n_offsets"] for row in curve],
    }


def chemical_curve_for_sample(
    sample: dict[str, Any],
    r_min: float | None,
    r_max: float | None,
) -> list[dict[str, Any]]:
    chemical = sample.get("chemical_distance")
    if not isinstance(chemical, dict):
        return []
    radial_min_path = chemical.get("radial_min_path")
    if isinstance(radial_min_path, list):
        curve = []
        for fallback_idx, row in enumerate(radial_min_path):
            if not isinstance(row, dict):
                continue
            r_at_min_ell = finite_float(row.get("r_at_min_ell"))
            r_center = finite_float(row.get("r_center"))
            r_mean = finite_float(row.get("r_mean"))
            min_ell = finite_float(row.get("ell_min"))
            max_ell = finite_float(row.get("ell_max"))
            pairs = finite_float(row.get("pairs"))
            r_value = r_at_min_ell if r_at_min_ell is not None else r_center if r_center is not None else r_mean
            if r_value is None or min_ell is None or r_value <= 0 or min_ell <= 0:
                continue
            if r_min is not None and r_value < r_min:
                continue
            if r_max is not None and r_value > r_max:
                continue
            curve.append({
                "observable": "d_min",
                "bin_index": int(row.get("bin_index", fallback_idx)),
                "mean_r": r_mean if r_mean is not None else r_value,
                "r_at_min_ell": r_value,
                "mean_ell": finite_float(row.get("ell_mean")) or min_ell,
                "ell_std": finite_float(row.get("ell_std")) or 0.0,
                "min_ell": min_ell,
                "max_ell": max_ell if max_ell is not None else min_ell,
                "truncated": bool(row.get("truncated", False)),
                "pairs": int(pairs) if pairs is not None else 0,
            })
        if curve:
            return curve
    bins: dict[int, dict[str, float]] = defaultdict(
        lambda: {
            "count": 0.0,
            "sum_r": 0.0,
            "sum_ell": 0.0,
            "sum_ell2": 0.0,
            "min_ell": math.inf,
            "max_ell": 0.0,
        }
    )
    max_chemical_distance = finite_float(chemical.get("max_chemical_distance"))
    for origin in chemical.get("origins", []):
        for idx, row in enumerate(origin.get("bins", [])):
            count = finite_float(row.get("count"))
            sum_r = finite_float(row.get("sum_r"))
            sum_ell = finite_float(row.get("sum_ell"))
            sum_ell2 = finite_float(row.get("sum_ell2"))
            if count is None or sum_r is None or sum_ell is None or count <= 0:
                continue
            mean_r = sum_r / count
            mean_ell = sum_ell / count
            if mean_r <= 0 or mean_ell <= 0:
                continue
            if r_min is not None and mean_r < r_min:
                continue
            if r_max is not None and mean_r > r_max:
                continue
            bins[idx]["count"] += count
            bins[idx]["sum_r"] += sum_r
            bins[idx]["sum_ell"] += sum_ell
            bins[idx]["sum_ell2"] += sum_ell2 if sum_ell2 is not None else 0.0
            min_ell = finite_float(row.get("min_ell"))
            max_ell = finite_float(row.get("max_ell"))
            r_at_min_ell = finite_float(row.get("r_at_min_ell"))
            r_center = finite_float(row.get("r_center"))
            if min_ell is not None:
                current_min_ell = bins[idx]["min_ell"]
                current_r_at_min = bins[idx].get("r_at_min_ell", math.inf)
                candidate_r_at_min = (
                    r_at_min_ell
                    if r_at_min_ell is not None
                    else r_center if r_center is not None
                    else mean_r
                )
                if (
                    min_ell < current_min_ell or
                    (
                        min_ell == current_min_ell and
                        candidate_r_at_min < current_r_at_min
                    )
                ):
                    bins[idx]["min_ell"] = min_ell
                    bins[idx]["r_at_min_ell"] = candidate_r_at_min
            if max_ell is not None:
                bins[idx]["max_ell"] = max(bins[idx]["max_ell"], max_ell)

    curve = []
    for idx in sorted(bins):
        row = bins[idx]
        count = row["count"]
        if count <= 0:
            continue
        mean_ell = row["sum_ell"] / count
        ell_var = max(0.0, row["sum_ell2"] / count - mean_ell * mean_ell)
        max_ell = row["max_ell"]
        curve.append({
            "observable": "d_min",
            "bin_index": idx,
            "mean_r": row["sum_r"] / count,
            "r_at_min_ell": row.get("r_at_min_ell"),
            "mean_ell": mean_ell,
            "ell_std": math.sqrt(ell_var),
            "min_ell": None if math.isinf(row["min_ell"]) else row["min_ell"],
            "max_ell": max_ell,
            "truncated": (
                max_chemical_distance is not None and
                max_chemical_distance > 0 and
                max_ell >= max_chemical_distance
            ),
            "pairs": int(count),
        })
    return curve


def chemical_series(curve: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "x_name": "r",
        "y_name": "ell_min",
        "default_y": "ell_min",
        "bin_index": [row["bin_index"] for row in curve],
        "r": [
            row["r_at_min_ell"] if row.get("r_at_min_ell") is not None else row["mean_r"]
            for row in curve
        ],
        "r_mean": [row["mean_r"] for row in curve],
        "r_at_min_ell": [row.get("r_at_min_ell") for row in curve],
        "ell": [row["min_ell"] for row in curve],
        "ell_mean": [row["mean_ell"] for row in curve],
        "ell_std": [row["ell_std"] for row in curve],
        "ell_min": [row["min_ell"] for row in curve],
        "ell_max": [row["max_ell"] for row in curve],
        "truncated": [row["truncated"] for row in curve],
        "pairs": [row["pairs"] for row in curve],
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def remove_legacy_flat_outputs(out_root: Path) -> None:
    for filename in [
        "fractal_dimension_samples.csv",
        "fractal_dimensions.csv",
        "fractal_dimensions_long.csv",
        "box_counting_curves.csv",
        "chemical_distance_curves.csv",
    ]:
        path = out_root / filename
        if path.exists():
            path.unlink()


def remove_legacy_group_outputs(group_dir: Path) -> None:
    for filename in [
        "fractal_dimension_samples.csv",
        "fractal_dimensions.csv",
        "fractal_dimensions_long.csv",
        "box_counting_curves.csv",
        "chemical_distance_curves.csv",
    ]:
        path = group_dir / filename
        if path.exists():
            path.unlink()


def process_counts(args: argparse.Namespace) -> dict[str, Any]:
    sop_root = args.sop_root.resolve()
    raw_root = sop_root / args.raw_dir
    out_root = sop_root / args.out_dir
    count_paths = sorted(raw_root.glob("**/counts/*_counts.json"))
    remove_legacy_flat_outputs(out_root)
    existing_outputs = discover_existing_outputs(out_root)

    files_by_group: dict[tuple[Any, ...], set[str]] = defaultdict(set)
    samples_by_group: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)

    for path in count_paths:
        payload = json.loads(path.read_text())
        params = parse_params_from_path(path, raw_root, payload)
        gkey = group_key(params)
        files_by_group[gkey].add(str(path.relative_to(raw_root)))

        for sample_pos, sample in enumerate(payload.get("samples", [])):
            if not isinstance(sample, dict) or not sample.get("eligible", True):
                continue
            sample_index = sample.get("sample_index", sample_pos)
            sample_id = f"{path.stem}:sample_{sample_index}"
            L = int(sample.get("L", params["L"]))
            properties: dict[str, Any] = {}

            for observable, y_name in [
                ("d_f", "N_bulk"),
                ("d_hull", "N_hull"),
                ("d_hull_ext", "N_hull_ext"),
            ]:
                curve = box_curve_for_sample(
                    sample,
                    observable,
                    L,
                    args.box_epsilon_min,
                    args.box_epsilon_max,
                )
                properties[observable] = box_series(curve, y_name)

            curve = chemical_curve_for_sample(sample, args.chemical_r_min, args.chemical_r_max)
            properties["d_min"] = chemical_series(curve)

            samples_by_group[gkey].append({
                "sample_id": sample_id,
                "source_file": str(path.relative_to(raw_root)),
                "sample_index": sample_index,
                "seed": params.get("file_seed"),
                "properties": properties,
            })

    split_outputs = []
    all_groups = sorted(set(existing_outputs) | set(samples_by_group))
    for gkey in all_groups:
        params = (
            existing_outputs[gkey][0]["params"]
            if gkey in existing_outputs and existing_outputs[gkey]
            else key_to_params(gkey)
        )
        group_dir = group_output_dir(out_root, params)
        remove_legacy_group_outputs(group_dir)
        group_dir.mkdir(parents=True, exist_ok=True)
        output_path = group_dir / processed_counts_name(params)
        existing_samples = []
        existing_paths = []
        for item in existing_outputs.get(gkey, []):
            existing_paths.append(item["path"])
            existing_samples.extend([
                sample for sample in item["samples"]
                if isinstance(sample, dict)
            ])
        merged_samples, appended_samples, updated_samples = merge_samples(
            existing_samples,
            samples_by_group.get(gkey, []),
        )
        merged_source_files = {
            sample["source_file"]
            for sample in merged_samples
            if isinstance(sample.get("source_file"), str)
        }
        legacy_d_min_fallback_samples = sum(
            1
            for sample in merged_samples
            if sample.get("properties", {}).get("d_min", {}).get("legacy_d_min_fallback") is True
        )
        output_payload = {
            "meta": {
                **{col: params.get(col) for col in PARAM_COLUMNS},
                "relations": {
                    "d_f": "N_bulk ~ (L_over_epsilon)^d_f",
                    "d_hull": "N_hull ~ (L_over_epsilon)^d_hull",
                    "d_hull_ext": "N_hull_ext ~ (L_over_epsilon)^d_hull_ext",
                    "d_min": "ell ~ r^d_min",
                },
                "d_min_note": (
                    "No L rescaling is applied to d_min. The default processed series "
                    "uses ell_min over all bins. ell_mean, ell_max and truncated flags "
                    "are kept only as diagnostics."
                ),
                "n_count_files": len(merged_source_files),
                "n_samples": len(merged_samples),
                "n_legacy_d_min_fallback_samples": legacy_d_min_fallback_samples,
                "last_run_new_samples": appended_samples,
                "last_run_updated_samples": updated_samples,
            },
            "samples": merged_samples,
        }
        output_path.write_text(json.dumps(output_payload, separators=(",", ":")) + "\n")
        split_outputs.append({
            **params,
            "path": str(output_path),
            "n_count_files": len(merged_source_files),
            "n_samples": len(merged_samples),
            "n_legacy_d_min_fallback_samples": legacy_d_min_fallback_samples,
            "last_run_new_samples": appended_samples,
            "last_run_updated_samples": updated_samples,
            "raw_count_files_this_run": len(files_by_group.get(gkey, set())),
            "existing_processed_files": len(existing_paths),
        })

    write_csv(
        out_root / "split_outputs.csv",
        split_outputs,
        [
            *PARAM_COLUMNS,
            "path",
            "n_count_files",
            "n_samples",
            "n_legacy_d_min_fallback_samples",
            "last_run_new_samples",
            "last_run_updated_samples",
            "raw_count_files_this_run",
            "existing_processed_files",
        ],
    )

    manifest = {
        "processing": "counts",
        "n_raw_count_files_this_run": len(count_paths),
        "n_existing_processed_files": sum(len(items) for items in existing_outputs.values()),
        "n_parameter_sets": len(all_groups),
        "grouping": GROUP_DIR_COLUMNS,
        "state_source": "published_counts plus raw_fractions updates",
        "outputs": [
            str(out_root / "split_outputs.csv"),
        ],
        "split_outputs": split_outputs,
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Process SOP raw_fractions/counts JSON files into fractal dimensions and plot-ready curves."
    )
    parser.add_argument("--sop-root", type=Path, default=Path("SOP_data"))
    parser.add_argument("--raw-dir", default="raw_fractions")
    parser.add_argument("--out-dir", default="published_counts")
    parser.add_argument("--box-epsilon-min", type=float, default=None)
    parser.add_argument("--box-epsilon-max", type=float, default=None)
    parser.add_argument("--chemical-r-min", type=float, default=None)
    parser.add_argument("--chemical-r-max", type=float, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = process_counts(args)
    print(
        "[process_counts] "
        f"raw_count_files_this_run={manifest['n_raw_count_files_this_run']} "
        f"parameter_sets={manifest['n_parameter_sets']} "
        f"out={args.sop_root.resolve() / args.out_dir}"
    )


if __name__ == "__main__":
    main()
