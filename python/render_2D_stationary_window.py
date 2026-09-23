#!/usr/bin/env python3
import argparse
import json
import struct
from pathlib import Path

import numpy as np
from PIL import Image


MAGIC_NETG = 0x4E455447
HEADER_BYTES = 16


def parse_hex_color(value):
    s = value.strip()
    if s.startswith("#"):
        s = s[1:]
    if len(s) != 6:
        raise argparse.ArgumentTypeError("cor deve estar no formato #RRGGBB")
    try:
        return tuple(int(s[i : i + 2], 16) for i in (0, 2, 4))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("cor deve estar no formato #RRGGBB") from exc


def even_dimension(value):
    value = int(value)
    return value if value % 2 == 0 else value - 1


def read_compact_bin_header(path):
    with path.open("rb") as f:
        magic = struct.unpack("<I", f.read(4))[0]
        if magic != MAGIC_NETG:
            raise ValueError(f"Arquivo nao parece ser NetworkCompact .bin: magic={hex(magic)}")
        n_sites = struct.unpack("<I", f.read(4))[0]
        n_edges = struct.unpack("<Q", f.read(8))[0]
    return int(n_sites), int(n_edges)


def read_z_stat(json_path, z_stat_index=0):
    with json_path.open() as f:
        data = json.load(f)
    z_stats = data.get("meta", {}).get("z_stat")
    if not isinstance(z_stats, list):
        raise ValueError(f"{json_path} nao possui meta.z_stat em formato de lista")
    try:
        z_stat = z_stats[int(z_stat_index)]
    except IndexError as exc:
        raise ValueError(f"z_stat_index={z_stat_index} nao existe em {json_path}") from exc
    if z_stat is None:
        raise ValueError(f"z_stat_index={z_stat_index} esta null em {json_path}")
    return int(z_stat)


def choose_one(folder, suffix):
    files = sorted(folder.glob(f"*{suffix}"))
    if len(files) != 1:
        raise FileNotFoundError(
            f"Esperava exatamente um arquivo {suffix} em {folder}; encontrei {len(files)}"
        )
    return files[0]


def render_stationary_window(
    bin_path,
    json_path,
    L,
    output_path,
    output_width=1600,
    output_height=None,
    active_color=(201, 37, 45),
    background_color=(255, 247, 239),
    z_stat_index=0,
    inclusive_top=True,
):
    n_sites, n_edges = read_compact_bin_header(bin_path)
    if n_sites % int(L) != 0:
        raise ValueError(f"N={n_sites} nao e multiplo de L={L}")

    full_height = n_sites // int(L)
    z_stat = read_z_stat(json_path, z_stat_index=z_stat_index)
    top = z_stat + int(L)
    window_height = top - z_stat + (1 if inclusive_top else 0)
    if z_stat < 0 or top >= full_height:
        raise ValueError(
            f"Janela [{z_stat}, {top}] excede a altura H={full_height} de {bin_path}"
        )

    out_w = even_dimension(output_width)
    if output_height is None:
        out_h = even_dimension(round(out_w * window_height / int(L)))
    else:
        out_h = even_dimension(output_height)
    out_h = max(2, out_h)

    pos = np.memmap(bin_path, dtype=np.uint32, mode="r", offset=HEADER_BYTES, shape=(n_sites,))
    species_offset = HEADER_BYTES + 4 * n_sites
    species = np.memmap(bin_path, dtype=np.uint8, mode="r", offset=species_offset, shape=(n_sites,))

    active = species > 0
    y = pos // np.uint32(L)
    if inclusive_top:
        in_window = active & (y >= z_stat) & (y <= top)
    else:
        in_window = active & (y >= z_stat) & (y < top)

    idx = np.flatnonzero(in_window)
    pos_window = np.asarray(pos[idx], dtype=np.uint64)
    x = pos_window % np.uint64(L)
    y_local = (pos_window // np.uint64(L)) - np.uint64(z_stat)

    px = ((x * out_w) // np.uint64(L)).astype(np.int32, copy=False)
    py = ((y_local * out_h) // np.uint64(window_height)).astype(np.int32, copy=False)
    py = np.clip(py, 0, out_h - 1)

    canvas = np.empty((out_h, out_w, 3), dtype=np.uint8)
    canvas[:, :] = background_color
    canvas[py, px] = np.array(active_color, dtype=np.uint8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.flipud(canvas), mode="RGB").save(output_path, optimize=False)

    return {
        "bin": str(bin_path),
        "json": str(json_path),
        "output": str(output_path),
        "L": int(L),
        "H": int(full_height),
        "z_stat": int(z_stat),
        "top": int(top),
        "window_height": int(window_height),
        "n_edges": int(n_edges),
        "n_points": int(idx.size),
        "output_size": [int(out_w), int(out_h)],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Renderiza a regiao 2D z_stat..z_stat+L com o mesmo design da animacao."
    )
    parser.add_argument(
        "folders",
        nargs="*",
        default=None,
        help="pastas animation_2D_growth_...; default usa animation_2D_growth_L8192_seed44_fT*",
    )
    parser.add_argument("--L", type=int, default=8192)
    parser.add_argument("--output-width", type=int, default=1600)
    parser.add_argument("--output-height", type=int, default=None)
    parser.add_argument("--z-stat-index", type=int, default=0)
    parser.add_argument("--active-color", type=parse_hex_color, default=parse_hex_color("#c9252d"))
    parser.add_argument("--background-color", type=parse_hex_color, default=parse_hex_color("#fff7ef"))
    parser.add_argument("--exclusive-top", action="store_true")
    args = parser.parse_args()

    if args.folders:
        folders = [Path(folder) for folder in args.folders]
    else:
        folders = sorted(Path(".").glob("animation_2D_growth_L8192_seed44_fT*"))

    if not folders:
        raise FileNotFoundError("Nenhuma pasta animation_2D_growth_L8192_seed44_fT* encontrada")

    for folder in folders:
        bin_path = choose_one(folder, ".bin")
        json_path = choose_one(folder, ".json")
        output_path = folder / "stationary_window_zstat_to_zstat_plus_L.png"
        result = render_stationary_window(
            bin_path=bin_path,
            json_path=json_path,
            L=args.L,
            output_path=output_path,
            output_width=args.output_width,
            output_height=args.output_height,
            active_color=args.active_color,
            background_color=args.background_color,
            z_stat_index=args.z_stat_index,
            inclusive_top=not args.exclusive_top,
        )
        print(
            f"[done] {folder}: z_stat={result['z_stat']} top={result['top']} "
            f"points={result['n_points']} size={result['output_size']} -> {result['output']}"
        )


if __name__ == "__main__":
    main()
