#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FRACTIONS_PROCESSING_VERSION = 2

FILENAME_TAG_RE = re.compile(
    r"_P0_(?P<P0>[^_]+)_p0_(?P<p0>[^_.]+(?:\.[^_.]+)?)"
    r"(?:_base_(?P<base>.+?))?\.json$"
)


PARAM_META_KEYS = (
    "dim",
    "L",
    "num_colors",
    "type_percolation",
    "N_total",
    "E_total",
    "sample_gap_layers",
    "sample_gap_over_L",
    "rho",
    "window_convention",
    "memory_convention",
)

FRACTION_DATA_KEYS = (
    "anchor_z",
    "t_inst",
    "t_stab",
    "p_inst_bond",
    "p_inst_node",
    "S_inst",
    "E_inst",
    "SP_inst",
    "p_stab_bond",
    "p_stab_node",
    "S_stab",
    "E_stab",
    "SP_stab",
)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json_xz(path: Path, payload: dict[str, Any], pretty: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with lzma.open(tmp_path, "wt", encoding="utf-8", preset=6) as fh:
        if pretty:
            json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=True)
            fh.write("\n")
        else:
            json.dump(payload, fh, ensure_ascii=False, separators=(",", ":"))
    tmp_path.replace(path)


def manifest_path(manifests_root: Path, rel_parent: Path, bundle_name: str) -> Path:
    return manifests_root / rel_parent / bundle_name.removesuffix(".json.xz") / "manifest.json"


def load_manifest(manifests_root: Path, rel_parent: Path, bundle_name: str) -> dict[str, Any]:
    path = manifest_path(manifests_root, rel_parent, bundle_name)
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {
        "processed_json_files": [],
        "n_processed_json_files": 0,
        "summary_file": None,
        "last_update": None,
    }


def save_manifest(
    manifests_root: Path,
    rel_parent: Path,
    bundle_name: str,
    manifest: dict[str, Any],
) -> Path:
    path = manifest_path(manifests_root, rel_parent, bundle_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2, allow_nan=False)
        fh.write("\n")
    return path


def file_hash_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_stat_fingerprint(path: Path) -> str:
    stat = path.stat()
    return f"stat:{stat.st_size}:{stat.st_mtime_ns}"


def file_fingerprint_for_mode(path: Path, mode: str) -> str:
    if mode == "hash":
        return file_hash_fingerprint(path)
    if mode == "stat":
        return file_stat_fingerprint(path)
    raise ValueError(f"Unknown fingerprint mode: {mode}")


def fingerprint_matches_mode(fingerprint: str | None, mode: str) -> bool:
    if not fingerprint:
        return False
    if mode == "stat":
        return fingerprint.startswith("stat:")
    if mode == "hash":
        return not fingerprint.startswith("stat:")
    return False


def parse_filename_tags(path: Path) -> tuple[str, str, str | None]:
    match = FILENAME_TAG_RE.search(path.name)
    if not match:
        return "unknown", "unknown", None
    return match.group("P0"), match.group("p0"), match.group("base")


def bundle_filename(P0: str, p0: str, base: str | None) -> str:
    name = f"fractions_bundle_P0_{P0}_p0_{p0}"
    if base:
        name += f"_base_{base}"
    return name + ".json.xz"


def group_raw_fraction_files(raw_root: Path) -> dict[tuple[Path, str, str, str | None], list[Path]]:
    groups: dict[tuple[Path, str, str, str | None], list[Path]] = defaultdict(list)
    for path in sorted(raw_root.rglob("*.json")):
        if not path.is_file():
            continue
        P0, p0, base = parse_filename_tags(path)
        rel_parent = path.parent.relative_to(raw_root)
        groups[(rel_parent, P0, p0, base)].append(path)
    return dict(groups)


def parameter_meta(meta: dict[str, Any]) -> dict[str, Any]:
    return {key: meta.get(key) for key in PARAM_META_KEYS if key in meta}


def validate_parameter_meta(
    first_meta: dict[str, Any],
    next_meta: dict[str, Any],
    path: Path,
    strict: bool,
) -> list[str]:
    mismatches = []
    for key in PARAM_META_KEYS:
        if first_meta.get(key) != next_meta.get(key):
            mismatches.append(key)
    if mismatches and strict:
        joined = ", ".join(mismatches)
        raise ValueError(f"{path}: inconsistent parameter metadata: {joined}")
    return mismatches


def normalize_sample_list(value: list[Any], n_samples: int) -> list[Any]:
    if len(value) == n_samples:
        return value
    if len(value) > n_samples:
        return value[:n_samples]
    return value + [None] * (n_samples - len(value))


def concatenate_fraction_data(
    data_out: dict[str, list[Any]],
    data_in: dict[str, Any],
    path: Path,
    strict: bool,
    total_samples_so_far: int,
) -> int:
    list_values: dict[str, list[Any]] = {}
    for key, value in data_in.items():
        if not isinstance(value, list):
            if strict:
                raise ValueError(f"{path}: data.{key} is not a list")
            continue
        list_values[key] = value

    if not list_values:
        return 0

    sample_len = len(list_values.get("anchor_z", next(iter(list_values.values()))))
    for key, value in list_values.items():
        if len(value) != sample_len and strict:
            raise ValueError(
                f"{path}: data.{key} has length {len(value)}, expected {sample_len}"
            )

    all_keys = set(FRACTION_DATA_KEYS) | set(data_out) | set(list_values)
    for key in sorted(all_keys):
        if key not in data_out and total_samples_so_far > 0:
            data_out[key].extend([None] * total_samples_so_far)
        if key in list_values:
            values = list_values[key]
            data_out[key].extend(values if strict else normalize_sample_list(values, sample_len))
        else:
            data_out[key].extend([None] * sample_len)

    return sample_len


def build_published_fractions_bundle(
    files: list[Path],
    raw_root: Path,
    rel_parent: Path,
    P0: str,
    p0: str,
    base: str | None,
    strict: bool = True,
) -> dict[str, Any]:
    if not files:
        raise ValueError("cannot build a bundle without input files")

    data_out: dict[str, list[Any]] = defaultdict(list)
    source_files = []
    seeds = []
    sample_counts = []
    requested_counts = []
    stop_reasons: Counter[str] = Counter()
    inconsistent_meta: Counter[str] = Counter()

    first_payload = read_json(files[0])
    first_meta = first_payload.get("meta", {})
    first_param_meta = parameter_meta(first_meta)

    for idx, path in enumerate(files):
        payload = first_payload if idx == 0 else read_json(path)
        meta = payload.get("meta", {})
        data = payload.get("data", {})
        for key in validate_parameter_meta(first_meta, meta, path, strict):
            inconsistent_meta[key] += 1

        n_samples = concatenate_fraction_data(
            data_out=data_out,
            data_in=data,
            path=path,
            strict=strict,
            total_samples_so_far=sum(sample_counts),
        )
        seed = meta.get("seed")
        source_files.append(str(path.relative_to(raw_root)))
        seeds.append(seed)
        sample_counts.append(n_samples)
        requested_counts.append(meta.get("requested_samples"))
        stop_reasons[str(meta.get("stop_reason", "unknown"))] += 1

    total_samples = sum(sample_counts)
    data_out["source_seed"] = [
        seed for seed, n_samples in zip(seeds, sample_counts) for _ in range(n_samples)
    ]
    data_out["source_file_index"] = [
        idx for idx, n_samples in enumerate(sample_counts) for _ in range(n_samples)
    ]

    return {
        "meta": {
            **first_param_meta,
            "mode": "published_fractions",
            "source_mode": "raw_fractions",
            "schema_version": 1,
            "processed_at_utc": datetime.now(timezone.utc).isoformat(),
            "relative_parameter_dir": str(rel_parent),
            "P0": None if P0 == "unknown" else float(P0),
            "p0": None if p0 == "unknown" else float(p0),
            "initial_layout": base or "random",
            "num_source_files": len(files),
            "total_samples": total_samples,
            "source_sample_counts": sample_counts,
            "source_requested_samples": requested_counts,
            "source_seeds": seeds,
            "source_files": source_files,
            "stop_reasons": dict(sorted(stop_reasons.items())),
            "inconsistent_parameter_meta": dict(sorted(inconsistent_meta.items())),
        },
        "data": dict(data_out),
    }


def process_raw_fractions(
    sop_root: Path,
    raw_dir: str = "raw_fractions",
    published_dir: str = "published_fractions",
    manifests_dir: str = "manifests_fractions",
    pretty_json: bool = False,
    dry_run: bool = False,
    clear: bool = False,
    fingerprint_mode: str = "stat",
    detect_replaced_files: bool = False,
    strict: bool = True,
) -> list[Path]:
    raw_root = sop_root / raw_dir
    published_root = sop_root / published_dir
    manifests_root = sop_root / manifests_dir
    groups = group_raw_fraction_files(raw_root)
    written_paths = []

    for (rel_parent, P0, p0, base), files in sorted(groups.items()):
        bundle_name = bundle_filename(P0, p0, base)
        out_path = published_root / rel_parent / bundle_name
        current_json_files = sorted(path.name for path in files)
        files_by_name = {path.name: path for path in files}
        manifest = load_manifest(manifests_root, rel_parent, bundle_name)
        manifest_files = set(map(str, manifest.get("processed_json_files", [])))
        manifest_fingerprints_raw = manifest.get("processed_json_file_fingerprints", {})
        manifest_fingerprints = (
            {str(k): str(v) for k, v in manifest_fingerprints_raw.items()}
            if isinstance(manifest_fingerprints_raw, dict)
            else {}
        )
        manifest_version = int(manifest.get("fractions_processing_version", 0) or 0)
        names_new_to_manifest = sorted(set(current_json_files) - manifest_files)
        current_file_fingerprints: dict[str, str] = {}

        if manifest_fingerprints:
            names_to_fingerprint = set(names_new_to_manifest)
            if detect_replaced_files:
                names_to_fingerprint.update(
                    name for name in current_json_files if name in manifest_fingerprints
                )
            current_file_fingerprints = {
                name: file_fingerprint_for_mode(files_by_name[name], fingerprint_mode)
                for name in sorted(names_to_fingerprint)
            }
            replaced_files = (
                [
                    name for name in current_json_files
                    if name in current_file_fingerprints
                    and fingerprint_matches_mode(manifest_fingerprints.get(name), fingerprint_mode)
                    and manifest_fingerprints.get(name) != current_file_fingerprints.get(name)
                ]
                if detect_replaced_files
                else []
            )
            changed_files = sorted(set(names_new_to_manifest) | set(replaced_files))
        else:
            missing_manifest_files = bool(manifest_files - set(current_json_files))
            changed_files = current_json_files if missing_manifest_files else names_new_to_manifest
            current_file_fingerprints = {
                name: file_fingerprint_for_mode(files_by_name[name], fingerprint_mode)
                for name in sorted(changed_files)
            }

        should_rebuild = (
            clear
            or not out_path.exists()
            or manifest_version != FRACTIONS_PROCESSING_VERSION
            or manifest.get("fingerprint_mode") != fingerprint_mode
            or bool(changed_files)
        )

        if dry_run:
            action = "rebuild" if should_rebuild else "skip"
            print(f"[dry-run:{action}] {len(files)} file(s) -> {out_path}")
            written_paths.append(out_path)
            continue
        if not should_rebuild:
            print(f"[skip] {out_path}")
            written_paths.append(out_path)
            continue

        if changed_files:
            preview = ", ".join(changed_files[:5])
            suffix = "" if len(changed_files) <= 5 else f", ... (+{len(changed_files) - 5} more)"
            print(f"[update] detected {len(changed_files)} changed file(s) for {rel_parent}: {preview}{suffix}")

        bundle = build_published_fractions_bundle(
            files=files,
            raw_root=raw_root,
            rel_parent=rel_parent,
            P0=P0,
            p0=p0,
            base=base,
            strict=strict,
        )
        write_json_xz(out_path, bundle, pretty=pretty_json)
        fingerprints_out = {
            name: file_fingerprint_for_mode(files_by_name[name], fingerprint_mode)
            for name in current_json_files
        }
        manifest.update({
            "group_relpath": rel_parent.as_posix(),
            "processed_json_files": current_json_files,
            "n_processed_json_files": len(current_json_files),
            "processed_json_file_fingerprints": dict(sorted(fingerprints_out.items())),
            "fingerprint_mode": fingerprint_mode,
            "summary_file": out_path.as_posix(),
            "fractions_processing_version": FRACTIONS_PROCESSING_VERSION,
            "last_update": datetime.now(timezone.utc).isoformat(),
        })
        save_manifest(manifests_root, rel_parent, bundle_name, manifest)
        print(f"[published_fractions] {len(files)} file(s) -> {out_path}")
        written_paths.append(out_path)

    return written_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Concatenate SOP_data/raw_fractions runs into published_fractions "
            "bundles, preserving the raw parameter directory structure."
        )
    )
    parser.add_argument("--sop-root", type=Path, default=Path("SOP_data"))
    parser.add_argument("--raw-dir", default="raw_fractions")
    parser.add_argument("--published-dir", default="published_fractions")
    parser.add_argument("--manifests-dir", default="manifests_fractions")
    parser.add_argument("--pretty-json", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--clear", action="store_true", help="Ignore manifest cache and rebuild all bundles.")
    parser.add_argument(
        "--fingerprint-mode",
        choices=("stat", "hash"),
        default="stat",
        help="How to fingerprint input files in the manifest. stat is much faster; hash is stricter.",
    )
    parser.add_argument(
        "--detect-replaced-files",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also fingerprint already-known filenames to detect files recreated with the same name.",
    )
    parser.add_argument("--no-strict", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    paths = process_raw_fractions(
        sop_root=args.sop_root,
        raw_dir=args.raw_dir,
        published_dir=args.published_dir,
        manifests_dir=args.manifests_dir,
        pretty_json=args.pretty_json,
        dry_run=args.dry_run,
        clear=args.clear,
        fingerprint_mode=args.fingerprint_mode,
        detect_replaced_files=args.detect_replaced_files,
        strict=not args.no_strict,
    )
    print(f"[published_fractions] processed {len(paths)} bundle(s)")


if __name__ == "__main__":
    main()
