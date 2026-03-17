from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import yaml


def load_config(config_path: str | Path) -> Dict:
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_data_dirs(cfg: Dict) -> Dict[str, Path]:
    root = Path(cfg["data_root"]).expanduser()

    dirs = {
        "root": root,
        "raw": root / cfg["raw_dir"],
        "manifests": root / cfg["manifest_dir"],
        "reduced_local": root / cfg["reduced_local_dir"],
        "published": root / cfg["published_dir"],
        "logs": root / cfg["logs_dir"],
        "tmp": root / cfg["tmp_dir"],
    }

    for path in dirs.values():
        ensure_dir(path)

    return dirs


def parse_group_relpath(group_relpath: Path) -> Dict[str, str]:
    """
    Espera paths no formato:
    bond_percolation/num_colors_4/dim_3/L_256/NT_constant/NT_3000/k_1.0e-06/rho_2.5000e-01
    """
    parts = group_relpath.parts
    if len(parts) != 8:
        raise ValueError(f"group_relpath inválido: {group_relpath}")

    out = {}

    # 1) type_perc
    first = parts[0]
    if not first.endswith("_percolation"):
        raise ValueError(f"Primeiro nível inesperado: {first}")
    out["type"] = first.replace("_percolation", "")

    # 2..8
    for item in parts[1:]:
        key, value = item.split("_", 1)
        out[key] = value

    return out


def group_dict_to_key(group: Dict[str, str]) -> str:
    keys = ["type", "num_colors", "dim", "L", "NT", "k", "rho"]
    return "|".join(f"{k}={group[k]}" for k in keys)


def get_group_dirs(raw_dir: Path) -> List[Path]:
    """
    Retorna diretórios de grupo-base, isto é, diretórios rho_* que possuem data/ e/ou network/.
    """
    groups = []
    for rho_dir in raw_dir.rglob("rho_*"):
        if not rho_dir.is_dir():
            continue
        has_data = (rho_dir / "data").is_dir()
        has_network = (rho_dir / "network").is_dir()
        if has_data or has_network:
            groups.append(rho_dir)

    return sorted(set(groups))