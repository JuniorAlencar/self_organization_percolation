from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import yaml
import os
import socket


def load_config(config_path: str | Path) -> dict:
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not cfg.get("machine_name"):
        # primeiro tenta variável de ambiente, depois usuário, depois hostname
        cfg["machine_name"] = (
            os.getenv("SOP_MACHINE")
            or os.getenv("USER")
            or os.getenv("USERNAME")
            or socket.gethostname()
        )

    return cfg

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

    out: Dict[str, str] = {}

    # 1) tipo de percolação
    first = parts[0]
    if not first.endswith("_percolation"):
        raise ValueError(f"Primeiro nível inesperado: {first}")
    out["type"] = first.replace("_percolation", "")

    # 2) num_colors_4  -> key='num_colors', value='4'
    second = parts[1]
    if not second.startswith("num_colors_"):
        raise ValueError(f"Nível num_colors inesperado: {second}")
    out["num_colors"] = second[len("num_colors_"):]

    # 3) dim_3
    third = parts[2]
    if not third.startswith("dim_"):
        raise ValueError(f"Nível dim inesperado: {third}")
    out["dim"] = third[len("dim_"):]

    # 4) L_256
    fourth = parts[3]
    if not fourth.startswith("L_"):
        raise ValueError(f"Nível L inesperado: {fourth}")
    out["L"] = fourth[len("L_"):]

    # 5) NT_constant (fixo)
    fifth = parts[4]
    if fifth != "NT_constant":
        raise ValueError(f"Nível NT_constant inesperado: {fifth}")

    # 6) NT_3000
    sixth = parts[5]
    if not sixth.startswith("NT_"):
        raise ValueError(f"Nível NT inesperado: {sixth}")
    out["NT"] = sixth[len("NT_"):]

    # 7) k_1.0e-06
    seventh = parts[6]
    if not seventh.startswith("k_"):
        raise ValueError(f"Nível k inesperado: {seventh}")
    out["k"] = seventh[len("k_"):]

    # 8) rho_2.5000e-01
    eighth = parts[7]
    if not eighth.startswith("rho_"):
        raise ValueError(f"Nível rho inesperado: {eighth}")
    out["rho"] = eighth[len("rho_"):]

    return out


def group_dict_to_key(group: Dict[str, str]) -> str:
    keys = ["type", "num_colors", "dim", "L", "NT", "k", "rho"]
    return "|".join(f"{k}={group[k]}" for k in keys)


def get_group_dirs(raw_dir: Path) -> List[Path]:
    """
    Retorna diretórios de grupo-base válidos, isto é, diretórios rho_* que possuem data/
    e cujo caminho relativo siga exatamente o padrão esperado pelo pipeline:

    bond_percolation/num_colors_4/dim_3/L_256/NT_constant/NT_3000/k_1.0e-06/rho_2.5000e-01

    Diretórios auxiliares, como bond_percolation_equilibration, são ignorados.
    """
    groups = []

    for rho_dir in raw_dir.rglob("rho_*"):
        if not rho_dir.is_dir():
            continue

        if not (rho_dir / "data").is_dir():
            continue

        try:
            relpath = rho_dir.relative_to(raw_dir)
            parse_group_relpath(relpath)
        except Exception:
            continue

        groups.append(rho_dir)

    return sorted(set(groups))
