#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
TOOLS_DIR = THIS_FILE.parent
REPO_DIR = TOOLS_DIR.parent

if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

try:
    from process_data import process_all_data
except ImportError as e:
    raise ImportError(
        "Não foi possível importar process_all_data de tools/process_data.py"
    ) from e


def default_sop_root() -> Path:
    # Estrutura correta no seu projeto:
    # repo/tools/update_published_means.py
    # repo/SOP_data/
    return (TOOLS_DIR / ".." / "SOP_data").resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Atualiza os jsons médios em published/, gera colors_percolation.dat e reconstrói all_data.dat/all_colors.dat a partir de SOP_data/raw."
    )
    parser.add_argument(
        "--sop-root",
        default=str(default_sop_root()),
        help="Raiz do SOP_data. Padrão: ../SOP_data relativo à pasta tools/.",
    )
    parser.add_argument(
        "--clear-data",
        action="store_true",
        help="Força reprocessamento completo, ignorando o controle incremental por manifest.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduz mensagens de log.",
    )
    parser.add_argument(
        "--p0",
        nargs="*",
        type=float,
        default=[1.0],
        help="Lista de valores de p0 a considerar.",
    )

    args = parser.parse_args()

    sop_root = Path(args.sop_root).expanduser().resolve()

    process_all_data(
        clear_data=args.clear_data,
        sop_root=str(sop_root),
        p0_lst=args.p0,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()