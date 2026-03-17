from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Set, Any
import json
from datetime import datetime
import shutil

from sop_paths import load_config, get_data_dirs, parse_group_relpath, group_dict_to_key, ensure_dir, get_group_dirs


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def list_group_sample_jsons(data_dir: Path) -> List[Path]:
    if not data_dir.is_dir():
        return []
    return sorted([p for p in data_dir.glob("*.json") if p.is_file()])


def list_group_network_npys(network_dir: Path) -> List[Path]:
    if not network_dir.is_dir():
        return []
    return sorted([p for p in network_dir.glob("*.npy") if p.is_file()])


def get_manifest_path(manifests_dir: Path, group_relpath: Path) -> Path:
    return manifests_dir / group_relpath / "manifest.json"


def get_summary_path(reduced_local_dir: Path, group_relpath: Path) -> Path:
    return reduced_local_dir / group_relpath / "summary.json"


def get_published_path(published_dir: Path, group_relpath: Path) -> Path:
    return published_dir / group_relpath / "summary.json"


def load_manifest_or_default(
    manifest_path: Path,
    group_key: str,
    group_relpath: Path,
    machine_name: str,
    data_dir: Path,
    network_dir: Path,
) -> Dict:
    if manifest_path.exists():
        return load_json(manifest_path)

    return {
        "group_key": group_key,
        "group_relpath": str(group_relpath),
        "data_dir": str(data_dir),
        "network_dir": str(network_dir),
        "processed_json_files": [],
        "n_processed_json_files": 0,
        "last_update": None,
        "machine": machine_name,
        "summary_file": None,
    }


def load_summary_or_default(summary_path: Path, group_dict: Dict[str, str]) -> Dict:
    if summary_path.exists():
        return load_json(summary_path)

    return {
        "meta": {
            **group_dict,
            "created_at": now_iso(),
            "updated_at": None,
        },
        "results": {
            "num_samples": 0,
            "sample_files": [],
            "data": []
        }
    }


def parse_sample_json(sample_path: Path) -> Dict:
    return load_json(sample_path)


def merge_sample_into_summary(summary: Dict, sample_obj: Dict, sample_filename: str) -> None:
    summary["results"]["sample_files"].append(sample_filename)
    summary["results"]["data"].append(sample_obj)
    summary["results"]["num_samples"] += 1
    summary["meta"]["updated_at"] = now_iso()


def process_one_group(cfg: Dict, dirs: Dict[str, Path], group_dir: Path) -> None:
    group_relpath = group_dir.relative_to(dirs["raw"])
    group_dict = parse_group_relpath(group_relpath)
    group_key = group_dict_to_key(group_dict)

    data_dir = group_dir / "data"
    network_dir = group_dir / "network"

    manifest_path = get_manifest_path(dirs["manifests"], group_relpath)
    summary_path = get_summary_path(dirs["reduced_local"], group_relpath)

    manifest = load_manifest_or_default(
        manifest_path=manifest_path,
        group_key=group_key,
        group_relpath=group_relpath,
        machine_name=cfg["machine_name"],
        data_dir=data_dir,
        network_dir=network_dir,
    )

    summary = load_summary_or_default(summary_path, group_dict)

    processed_files: Set[str] = set(manifest["processed_json_files"])
    sample_jsons = list_group_sample_jsons(data_dir)
    new_files = [p for p in sample_jsons if p.name not in processed_files]

    if not new_files:
        print(f"[SKIP] Nenhum JSON novo em {group_relpath}")
        return

    print(f"[PROC] {group_relpath} | novos JSONs: {len(new_files)}")

    for sample_path in new_files:
        try:
            sample_obj = parse_sample_json(sample_path)
            merge_sample_into_summary(summary, sample_obj, sample_path.name)
            processed_files.add(sample_path.name)
        except Exception as e:
            print(f"[ERRO] Falha ao processar {sample_path}: {e}")

    manifest["processed_json_files"] = sorted(processed_files)
    manifest["n_processed_json_files"] = len(processed_files)
    manifest["last_update"] = now_iso()
    manifest["summary_file"] = str(summary_path)

    save_json(summary_path, summary)
    save_json(manifest_path, manifest)

    if cfg.get("publish_enabled", False) and cfg.get("machine_name") == cfg.get("publish_machine"):
        published_path = get_published_path(dirs["published"], group_relpath)
        ensure_dir(published_path.parent)
        shutil.copy2(summary_path, published_path)
        print(f"[PUB] Publicado em {published_path}")


def main(cfg_path: str = "config/sop_data_config.yaml") -> None:
    cfg = load_config(cfg_path)
    dirs = get_data_dirs(cfg)

    group_dirs = get_group_dirs(dirs["raw"])
    print(f"[INFO] Grupos encontrados: {len(group_dirs)}")

    for group_dir in group_dirs:
        process_one_group(cfg, dirs, group_dir)


if __name__ == "__main__":
    main()