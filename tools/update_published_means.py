#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
TOOLS_DIR = THIS_FILE.parent

if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from process_data import process_all_data


def default_sop_root() -> Path:
    return (TOOLS_DIR / ".." / "SOP_data").resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Atualiza published/, colors_percolation.dat, all_data.dat e all_colors.dat a partir de SOP_data/raw."
    )
    parser.add_argument("--sop-root", default=str(default_sop_root()))
    parser.add_argument("--clear-data", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--p0", nargs="*", type=float, default=None,
                        help="Lista explícita de p0 a processar. Se omitido, detecta automaticamente todos os p0 presentes em cada pasta.")

    args = parser.parse_args()
    sop_root = Path(args.sop_root).expanduser().resolve()
    if not sop_root.exists():
        raise FileNotFoundError(f"SOP_data não encontrado em: {sop_root}")

    process_all_data(
        clear_data=args.clear_data,
        sop_root=str(sop_root),
        p0_lst=args.p0,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()