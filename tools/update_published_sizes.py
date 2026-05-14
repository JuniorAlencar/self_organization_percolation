#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from process_data import process_all_data_sizes


def default_sop_root() -> Path:
    return (Path(__file__).resolve().parent / ".." / "SOP_data").resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Atualiza published/properties_sizes_bundle.json e all_data_sizes.dat a partir de SOP_data/raw."
    )
    parser.add_argument("--sop-root", default=str(default_sop_root()))
    parser.add_argument("--raw-dir", default="raw")
    parser.add_argument("--published-dir", default="published")
    parser.add_argument("--manifests-dir", default="manifests_sizes")
    parser.add_argument("--output-suffix", default="")
    parser.add_argument("--clear-data", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--p0",
        nargs="*",
        type=float,
        default=None,
        help="Lista explícita de p0 a processar. Se omitido, detecta automaticamente todos os p0 presentes em cada pasta."
    )

    args = parser.parse_args()
    sop_root = Path(args.sop_root).expanduser().resolve()
    if not sop_root.exists():
        raise FileNotFoundError(f"SOP_data não encontrado em: {sop_root}")

    process_all_data_sizes(
        clear_data=args.clear_data,
        sop_root=str(sop_root),
        raw_dir=args.raw_dir,
        published_dir=args.published_dir,
        manifests_dir=args.manifests_dir,
        output_suffix=args.output_suffix,
        p0_lst=args.p0,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
