from __future__ import annotations

from pathlib import Path
import json
import re
from typing import Any

from sop_paths import load_config, get_data_dirs, parse_group_relpath, group_dict_to_key, get_group_dirs

FLOAT = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"

CF_T_GROUP_RE = re.compile(
    rf"""
    ^(?P<type>[A-Za-z0-9]+)_percolation
    /num_colors_(?P<num_colors>\d+)
    /dim_(?P<dim>\d+)
    /L_(?P<L>\d+)
    /fT_constant
    /fT_(?P<f_T>{FLOAT})
    /c_(?P<c>{FLOAT})
    /rho_(?P<rho>{FLOAT})$
    """,
    re.X,
)


def parse_group_relpath_c_fT(group_relpath: Path) -> dict[str, Any]:
    rel = group_relpath.as_posix().rstrip("/")
    m = CF_T_GROUP_RE.match(rel)
    if not m:
        raise ValueError(f"Primeiro nível inesperado ou padrão antigo: {group_relpath}")

    g = m.groupdict()
    return {
        "type_perc": g["type"],
        "num_colors": int(g["num_colors"]),
        "dim": int(g["dim"]),
        "L": int(g["L"]),
        "f_T": float(g["f_T"]),
        "c": float(g["c"]),
        "rho": float(g["rho"]),
    }


def group_dict_to_key_c_fT(group_dict: dict[str, Any]) -> str:
    return (
        f"{group_dict['type_perc']}"
        f"_nc{int(group_dict['num_colors'])}"
        f"_dim{int(group_dict['dim'])}"
        f"_L{int(group_dict['L'])}"
        f"_fT{float(group_dict['f_T']):.4e}"
        f"_c{float(group_dict['c']):.4e}"
        f"_rho{float(group_dict['rho']):.4e}"
    )


def parse_group_relpath_compat(group_relpath: Path) -> tuple[dict[str, Any], str]:
    """
    Tenta primeiro o parser do projeto. Se ele ainda estiver no padrão antigo
    N_T/k, usa o parser novo para a estrutura f_T/c.
    """
    try:
        group_dict = parse_group_relpath(group_relpath)
        group_key = group_dict_to_key(group_dict)
        return group_dict, group_key
    except ValueError:
        group_dict = parse_group_relpath_c_fT(group_relpath)
        group_key = group_dict_to_key_c_fT(group_dict)
        return group_dict, group_key


def build_index(cfg_path: str = "../config/sop_data_config.yaml") -> None:
    cfg = load_config(cfg_path)
    dirs = get_data_dirs(cfg)

    groups = {}

    for group_dir in get_group_dirs(dirs["raw"]):
        group_relpath = group_dir.relative_to(dirs["raw"])

        try:
            group_dict, group_key = parse_group_relpath_compat(group_relpath)
        except ValueError:
            print(f"[skip] caminho ignorado: {group_relpath}")
            continue

        data_dir = group_dir / "data"
        json_files = sorted([p.name for p in data_dir.glob("*.json")]) if data_dir.is_dir() else []

        groups[str(group_relpath)] = {
            "group_relpath": str(group_relpath),
            "group_key": group_key,
            "group_dict": group_dict,
            "data_dir": str(data_dir),
            "n_json_files": len(json_files),
            "json_files": json_files,
        }

    out_file = dirs["tmp"] / f"raw_groups_index_{cfg['machine_name']}.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(groups, f, indent=2, ensure_ascii=False)

    print(f"[OK] Índice salvo em: {out_file}")
    print(f"[OK] Total de grupos encontrados: {len(groups)}")


if __name__ == "__main__":
    build_index()
