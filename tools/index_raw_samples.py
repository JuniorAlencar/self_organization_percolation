from __future__ import annotations

from pathlib import Path
from typing import Dict
import json

from sop_paths import load_config, get_data_dirs, parse_group_relpath, group_dict_to_key, get_group_dirs


def build_index(cfg_path: str = "config/sop_data_config.yaml") -> None:
    cfg = load_config(cfg_path)
    dirs = get_data_dirs(cfg)

    groups = {}

    for group_dir in get_group_dirs(dirs["raw"]):
        group_relpath = group_dir.relative_to(dirs["raw"])
        group_dict = parse_group_relpath(group_relpath)

        data_dir = group_dir / "data"
        network_dir = group_dir / "network"

        json_files = sorted([p.name for p in data_dir.glob("*.json")]) if data_dir.is_dir() else []
        npy_files = sorted([p.name for p in network_dir.glob("*.npy")]) if network_dir.is_dir() else []

        groups[str(group_relpath)] = {
            "group_relpath": str(group_relpath),
            "group_key": group_dict_to_key(group_dict),
            "group_dict": group_dict,
            "data_dir": str(data_dir),
            "network_dir": str(network_dir),
            "n_json_files": len(json_files),
            "n_npy_files": len(npy_files),
            "json_files": json_files,
            "npy_files": npy_files,
        }

    out_file = dirs["tmp"] / f"raw_groups_index_{cfg['machine_name']}.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(groups, f, indent=2, ensure_ascii=False)

    print(f"[OK] Índice salvo em: {out_file}")
    print(f"[OK] Total de grupos encontrados: {len(groups)}")


if __name__ == "__main__":
    build_index()