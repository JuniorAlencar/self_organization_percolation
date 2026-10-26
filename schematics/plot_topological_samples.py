#!/usr/bin/env python3
"""Plot two stabilized subgraphs and the real 2D network from its base."""
from pathlib import Path
import argparse
import base64
import gzip
import io
import json
import re
import shutil
import struct
import subprocess

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

BG = "#fff7ef"
BLUE = "#1f77b4"
INK = "#263238"
PRE_STABILIZATION = "#f28e2b"  # warm orange, distinct from stabilized purple
FULL_NETWORK = "#8a5cf6"  # stabilized network
GAP_GRAY = "#9e9e9e"  # gap region gray
ACCENT = "#d1495b"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot two stabilized subgraphs and the real 2D network from its base."
    )
    parser.add_argument(
        "snapshot_dir",
        type=Path,
        nargs="?",
        default=Path("component_snapshots_2samples"),
        help="Directory with component snapshots (default: component_snapshots_2samples)",
    )
    parser.add_argument(
        "sample_data",
        type=Path,
        nargs="?",
        default=Path("sample_data_2samples.json"),
        help="JSON sample metadata (default: sample_data_2samples.json)",
    )
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=None,
        help="Output PNG path (default auto-derived from metadata)",
    )
    parser.add_argument(
        "--svg-output",
        type=Path,
        default=None,
        help="Also save an SVG with vector text, annotations and axes",
    )
    parser.add_argument(
        "--base-snapshot",
        type=Path,
        default=None,
        help="SOPBASE1 mask snapshot for z=0 through z_stat (default: auto-detected)",
    )
    parser.add_argument(
        "--gap-snapshot",
        type=Path,
        default=None,
        help="SOPGAP1 mask snapshot for gap slab (default: auto-detected)",
    )
    parser.add_argument(
        "--f-T",
        type=float,
        default=None,
        help="Target f_T parameter (default: 0.1309299)",
    )
    parser.add_argument(
        "--template-svg",
        type=Path,
        default=None,
        help="Template SVG to preserve publication layout/connectors (default: auto-detected)",
    )
    parser.add_argument(
        "--no-template",
        action="store_true",
        help="Do not use SVG template even if found, render pure matplotlib figure",
    )
    return parser.parse_args()


def component_rgb(labels, giant_label):
    labels = np.asarray(labels, dtype=np.uint32)
    colors = np.empty(labels.shape + (3,), dtype=np.uint8)
    empty = labels == 0
    giant = labels == giant_label
    other = ~(empty | giant)
    colors[empty] = np.asarray([255, 247, 239], dtype=np.uint8)
    colors[giant] = np.asarray([31, 119, 180], dtype=np.uint8)  # animation blue
    if np.any(other):
        ids = labels[other].astype(np.uint64) - 1  # animation labels start at zero
        x = (ids + 1) * np.uint64(0x9E3779B1)
        x ^= x >> np.uint64(16)
        x = ((x & np.uint64(0xFFFFFFFF)) * np.uint64(0x85EBCA6B)) & np.uint64(0xFFFFFFFF)
        hue = (x % np.uint64(360)).astype(np.float64) / 360.0
        sat = 0.86 + 0.12 * ((x >> np.uint64(9)) & np.uint64(255)).astype(np.float64) / 255.0
        val = 0.74 + 0.20 * ((x >> np.uint64(17)) & np.uint64(255)).astype(np.float64) / 255.0
        h6 = hue * 6.0
        sector = np.floor(h6).astype(np.int8) % 6
        frac = h6 - np.floor(h6)
        p = val * (1.0 - sat)
        q = val * (1.0 - frac * sat)
        t = val * (1.0 - (1.0 - frac) * sat)
        rgb = np.empty((hue.size, 3), dtype=np.float64)
        choices = (
            (val, t, p), (q, val, p), (p, val, t),
            (p, q, val), (t, p, val), (val, p, q),
        )
        for n, triple in enumerate(choices):
            mask = sector == n
            for channel in range(3):
                rgb[mask, channel] = triple[channel][mask]
        colors[other] = np.rint(rgb * 255).astype(np.uint8)
    return colors


def load_component_image(path, expected_L):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as stream:
        if stream.read(8) != b"SOPCOMP1":
            raise SystemExit(f"Invalid component snapshot: {path}")
        width, height, largest_label = struct.unpack("<III", stream.read(12))
        if width != expected_L or height != expected_L:
            raise SystemExit(f"Unexpected dimensions in {path}: {width} x {height}")
        if path.suffix == ".gz":
            labels = np.frombuffer(stream.read(), dtype="<u4").reshape(height, width)
        else:
            labels = np.memmap(path, dtype="<u4", mode="r", offset=20,
                               shape=(height, width))
    sample_idx = np.linspace(0, expected_L - 1, min(expected_L, 4096), dtype=np.int64)
    preview = np.asarray(labels[np.ix_(sample_idx, sample_idx)])
    return component_rgb(preview, largest_label)


def resolve_existing(paths):
    for p in paths:
        if p is not None and p.exists():
            return p
    return None


def snapshot_path(snapshot_dir, i):
    compressed = snapshot_dir / f"sample_{i}.components.gz"
    plain = snapshot_dir / f"sample_{i}.components"
    if compressed.exists():
        return compressed
    if plain.exists():
        return plain
    raise FileNotFoundError(f"Neither {compressed} nor {plain} found")


def load_base_image(base_path, expected_L, expected_zstat):
    base_opener = gzip.open if base_path.suffix == ".gz" else open
    with base_opener(base_path, "rb") as stream:
        if stream.read(8) != b"SOPBASE1":
            raise SystemExit(f"Invalid base snapshot: {base_path}")
        base_L, base_height = struct.unpack("<II", stream.read(8))
        if base_L != expected_L or base_height != expected_zstat:
            raise SystemExit(
                f"Base snapshot dimensions {base_L} x {base_height} do not match L={expected_L}, z_stat={expected_zstat}")
        base_mask = np.frombuffer(stream.read(), dtype=np.uint8).reshape(base_height, base_L)
    base_idx = np.linspace(0, base_L - 1, min(base_L, 4096), dtype=np.int64)
    base_view = base_mask[np.ix_(base_idx, base_idx)]
    base_rgb = np.empty(base_view.shape + (3,), dtype=np.uint8)
    base_rgb[:] = np.asarray([255, 247, 239], dtype=np.uint8)
    base_rgb[base_view > 0] = np.asarray([242, 142, 43], dtype=np.uint8)
    return base_rgb


def load_gap_image(gap_path, expected_L):
    if gap_path is None or not gap_path.exists():
        return None
    gap_opener = gzip.open if gap_path.suffix == ".gz" else open
    with gap_opener(gap_path, "rb") as stream:
        magic = stream.read(8)
        if magic not in (b"SOPGAP1\0", b"SOPGAP1"):
            raise SystemExit(f"Invalid gap snapshot: {gap_path}")
        gw, gh = struct.unpack("<II", stream.read(8))
        if gw != expected_L or gh != expected_L:
            raise SystemExit(f"Unexpected dimensions in {gap_path}: {gw} x {gh}")
        gmask = np.frombuffer(stream.read(), dtype=np.uint8).reshape(gh, gw)
    g_idx = np.linspace(0, gw - 1, min(gw, 4096), dtype=np.int64)
    g_view = gmask[np.ix_(g_idx, g_idx)]
    gap_rgb = np.empty(g_view.shape + (3,), dtype=np.uint8)
    gap_rgb[:] = np.asarray([255, 247, 239], dtype=np.uint8)
    gap_rgb[g_view > 0] = np.asarray([158, 158, 158], dtype=np.uint8)
    return gap_rgb


def main():
    args = parse_args()

    metadata = json.loads(args.sample_data.read_text())
    L = int(metadata["meta"]["L"])
    type_perc = metadata["meta"].get("type_percolation", "bond")
    sample_count = int(metadata["meta"].get("collected_samples", 0))
    if sample_count != 2:
        raise SystemExit(f"Expected 2 collected samples, found {sample_count}")
    anchors = metadata["data"].get("anchor_z", [])
    if len(anchors) != 2:
        raise SystemExit(f"Expected 2 sample anchors, found {len(anchors)}")
    z_stat = int(metadata["meta"].get("z_stab", metadata["meta"]["z_stat_by_species"][0]))
    if z_stat <= 0:
        raise SystemExit(f"Invalid z_stat in metadata: {z_stat}")

    # Determine default output names if not specified
    ft_val = args.f_T or metadata["meta"].get("f_T")
    if ft_val is None:
        ft_val = 0.1545 if z_stat == 4096 else 0.1309299
    ft_str = f"{ft_val:.7g}"

    if args.output is None:
        args.output = Path(f"subgraphs_{type_perc}_L{L}_ft{ft_str}.png")
    if args.svg_output is None:
        args.svg_output = args.output.with_suffix(".svg")

    # Resolve snapshots
    base_snapshot = args.base_snapshot or resolve_existing([
        Path("base_network_z0_to_zstat.sopbase.gz"),
        Path("base_network_z0_to_zstat.sopbase"),
    ])
    if base_snapshot is None or not base_snapshot.exists():
        raise FileNotFoundError("Base snapshot (SOPBASE1) not found")

    gap_snapshot = args.gap_snapshot or resolve_existing([
        Path("gap_region_snapshot.sopgap.gz"),
        Path("gap_region_snapshot.sopgap"),
    ])

    template_svg = None
    if not args.no_template:
        template_svg = args.template_svg or resolve_existing([
            args.svg_output,
            Path(f"subgraphs_{type_perc}_L{L}_ft{ft_str}.svg"),
            Path("subgraphs_bond_L4096_ft0.1309299.svg"),
            Path("subgraphs_bond_L4096_ft0.1545.svg"),
        ])

    images = [load_component_image(snapshot_path(args.snapshot_dir, i), L) for i in range(1, 3)]
    base_rgb = load_base_image(base_snapshot, L, z_stat)
    gap_rgb = load_gap_image(gap_snapshot, L)

    fig = plt.figure(figsize=(8.5, 8.2), facecolor="white")

    overview_bottom = 0.12
    overview_height = 0.76
    ax = fig.add_axes([0.035, overview_bottom, 0.25, overview_height], facecolor=BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 4)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.add_patch(Rectangle((0, 0), 1, 4, fill=False, edgecolor=INK, linewidth=1.1))

    background_rgb = np.asarray([255, 247, 239], dtype=np.uint8)
    network_rgb = np.asarray([138, 92, 246], dtype=np.uint8)

    ax.imshow(base_rgb, interpolation="nearest", extent=(0, 1, 0, 1),
              origin="upper", rasterized=True, aspect="auto")

    # Sample 1
    occ1 = np.any(images[0] != background_rgb, axis=2)
    view1 = np.empty_like(images[0])
    view1[:] = background_rgb
    view1[occ1] = network_rgb
    ax.imshow(view1, interpolation="nearest", extent=(0, 1, 1, 2),
              origin="upper", rasterized=True, aspect="auto")
    ax.add_patch(Rectangle((0, 1), 1, 1, fill=False, edgecolor=ACCENT, linewidth=1.5))

    # Gap region
    if gap_rgb is not None:
        ax.imshow(gap_rgb, interpolation="nearest", extent=(0, 1, 2, 3),
                  origin="upper", rasterized=True, aspect="auto")

    # Sample 2
    occ2 = np.any(images[1] != background_rgb, axis=2)
    view2 = np.empty_like(images[1])
    view2[:] = background_rgb
    view2[occ2] = network_rgb
    ax.imshow(view2, interpolation="nearest", extent=(0, 1, 3, 4),
              origin="upper", rasterized=True, aspect="auto")
    ax.add_patch(Rectangle((0, 3), 1, 1, fill=False, edgecolor=ACCENT, linewidth=1.5))

    ax.text(0.5, 0.06, "y = 0", ha="center", va="center", fontsize=8.5,
            color=INK, bbox=dict(facecolor="white", alpha=0.82, edgecolor="none", pad=1.5))
    ax.axhline(1, color=ACCENT, linewidth=1.1, zorder=5)
    ax.text(0.5, 1.06, rf"$y_{{\mathrm{{stat}}}} = {z_stat}$", ha="center", va="bottom",
            fontsize=8.5, color=ACCENT,
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="none", pad=1.5), zorder=6)
    if gap_rgb is None:
        ax.text(0.5, 2.5, "Gap = L", ha="center", va="center", fontsize=9,
                color=ACCENT,
                bbox=dict(facecolor="white", alpha=0.82, edgecolor="none", pad=1.5))

    # Scale bar for full network transverse width
    ax.annotate("", xy=(1, -0.10), xytext=(0, -0.10), annotation_clip=False,
                arrowprops=dict(arrowstyle="<->", color=INK, linewidth=1.1))
    ax.text(0.5, -0.24, f"L = {L}", ha="center", va="top", fontsize=10,
            color=INK, clip_on=False)

    # Enlarged samples
    panel_left = 0.30
    cell_height = overview_height / 4.0
    scale_band = 0.04
    panel_size = overview_height / 3.0 - scale_band
    panel_width = panel_size * 8.2 / 8.5
    sample_bands = (1, 3)
    panel_bottoms = [overview_bottom + (band + 0.5) * cell_height - panel_size / 2
                     for band in sample_bands]
    panels = []
    for i, (image, panel_bottom) in enumerate(zip(images, panel_bottoms)):
        panel = fig.add_axes([panel_left, panel_bottom, panel_width, panel_size], facecolor=BG)
        panel.imshow(image, interpolation="nearest", origin="upper",
                     extent=(0, L, 0, L), rasterized=True)
        panel.set_xlim(0, L)
        panel.set_ylim(0, L)
        panel.set_xticks([])
        panel.set_yticks([])
        for spine in panel.spines.values():
            spine.set_edgecolor(INK)
            spine.set_linewidth(1.0)
        panel.text(0.5, 0.975, f"Sample {i+1}", transform=panel.transAxes,
                   ha="center", va="top", fontsize=10, color=INK, weight="bold",
                   bbox=dict(facecolor="white", alpha=0.78, edgecolor="none", pad=2))

        # Horizontal L bar
        panel.annotate("", xy=(L, -0.02 * L), xytext=(0, -0.02 * L),
                       annotation_clip=False,
                       arrowprops=dict(arrowstyle="<->", color=INK, linewidth=1.0))
        panel.text(L / 2, -0.06 * L, "L", ha="center", va="top",
                   fontsize=9, color=INK, clip_on=False)
        # Vertical L bar
        panel.annotate("", xy=(1.08 * L, L), xytext=(1.08 * L, 0),
                       annotation_clip=False,
                       arrowprops=dict(arrowstyle="<->", color=INK, linewidth=1.0))
        panel.text(1.15 * L, L / 2, "L", ha="left", va="center", rotation=90,
                   fontsize=9, color=INK, clip_on=False)
        panels.append(panel)

    fig.canvas.draw()
    for i, panel in enumerate(panels):
        source_px = ax.transData.transform((1.0, sample_bands[i] + 0.5))
        source = fig.transFigure.inverted().transform(source_px)
        target_px = panel.transAxes.transform((0.0, 0.5))
        target = fig.transFigure.inverted().transform(target_px)
        fig.add_artist(FancyArrowPatch(source, target, transform=fig.transFigure,
                        arrowstyle="-|>", mutation_scale=12, linewidth=1.1,
                        color=ACCENT, connectionstyle="arc3,rad=0"))

    # Legend categories
    legend_y = 0.025
    fig.text(0.035, legend_y, "●", color=PRE_STABILIZATION, fontsize=13, ha="left", va="center")
    fig.text(0.058, legend_y, "Before stabilization", fontsize=8.5, ha="left", va="center", color=INK)
    fig.text(0.245, legend_y, "●", color=FULL_NETWORK, fontsize=13, ha="left", va="center")
    fig.text(0.268, legend_y, "After stabilization", fontsize=8.5, ha="left", va="center", color=INK)
    fig.text(0.455, legend_y, "●", color=BLUE, fontsize=13, ha="left", va="center")
    fig.text(0.478, legend_y, "Largest component", fontsize=8.5, ha="left", va="center", color=INK)
    for j in range(4):
        label = j + 1
        ids = np.asarray([label], dtype=np.uint32)
        marker_rgb = component_rgb(ids, 0)[0]
        fig.text(0.72 + j * 0.012, legend_y, "●", color=marker_rgb / 255.0,
                 fontsize=13, ha="center", va="center")
    fig.text(0.775, legend_y, "Other clusters", fontsize=8.5,
             ha="left", va="center", color=INK)

    # Render intermediate raw SVG to extract high-res raster data URIs
    raw_svg_io = io.BytesIO()
    plt.rcParams["svg.fonttype"] = "none"
    fig.savefig(raw_svg_io, format="svg", dpi=600, bbox_inches="tight", facecolor="white")

    # If template SVG is available, update publication master
    if template_svg is not None and template_svg.exists():
        import xml.etree.ElementTree as ET
        raw_svg_text = raw_svg_io.getvalue().decode("utf-8")
        r_root = ET.fromstring(raw_svg_text)
        r_images = r_root.findall(".//{http://www.w3.org/2000/svg}image")
        hrefs = [im.attrib.get("{http://www.w3.org/1999/xlink}href") for im in r_images]

        master_text = template_svg.read_text(encoding="utf-8")
        id_order = ["image21584fd0b5", "image47d07ac6d0", "image4f3d814183"]
        for img_id, new_href in zip(id_order, hrefs):
            pattern = re.compile(
                r'(<image\b[^>]*?\b(?:xlink:)?href=\")[^\"]*(\"[^>]*?\bid=\"'
                + re.escape(img_id)
                + r'\")',
                re.S,
            )
            master_text, n = pattern.subn(r"\g<1>" + new_href + r"\g<2>", master_text)

        # Update coordinate variable (y vs z) and stat value
        coord_label = "y" if any("y" in t.get_text() for t in ax.texts) else "z"
        master_text = re.sub(
            r'id="text1055">[zy]\s*=\s*0<', lambda m: f'id="text1055">{coord_label} = 0<', master_text
        )
        master_text = re.sub(
            r'id="tspan1080">[zy]<', lambda m: f'id="tspan1080">{coord_label}<', master_text
        )
        master_text = re.sub(
            r'id="tspan1084">=[0-9]+<', lambda m: f'id="tspan1084">={z_stat}<', master_text
        )
        master_text = re.sub(
            r'[zy]_\\{\\\\mathrm\\{stat\\}\\}\s*=\s*[0-9]+',
            lambda m: f"{coord_label}_{{{{\\mathrm{{stat}}}}}} = {z_stat}",
            master_text,
        )
        master_text = re.sub(
            r'<!--\s*\$[zy]_\\mathrm\{stat\}\s*=\s*[0-9]+\s*\$-->',
            lambda m: f"<!-- ${coord_label}_\\mathrm{{stat}} = {z_stat} $-->",
            master_text,
        )

        # Update docname and export filename
        out_base = args.output.stem
        master_text = re.sub(
            r'subgraphs_bond_L[0-9]+_ft[0-9.]+', lambda m: out_base, master_text
        )
        master_text = re.sub(
            r'(sodipodi:docname=")[^"]*(")', lambda m: f'{m.group(1)}{args.svg_output.name}{m.group(2)}', master_text
        )
        master_text = re.sub(
            r'(inkscape:export-filename=")[^"]*(")',
            lambda m: f'{m.group(1)}{args.output.name}{m.group(2)}',
            master_text,
        )

        args.svg_output.parent.mkdir(parents=True, exist_ok=True)
        args.svg_output.write_text(master_text, encoding="utf-8")

        # Export high-res PNG with Inkscape if available
        if shutil.which("inkscape"):
            args.output.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["inkscape", str(args.svg_output), "-d", "300", "-o", str(args.output)],
                check=True,
            )
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(args.output, dpi=500, bbox_inches="tight", facecolor="white")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=500, bbox_inches="tight", facecolor="white")
        if args.svg_output:
            args.svg_output.parent.mkdir(parents=True, exist_ok=True)
            args.svg_output.write_bytes(raw_svg_io.getvalue())

    plt.close(fig)
    print(args.output)
    if args.svg_output:
        print(args.svg_output)


if __name__ == "__main__":
    main()
