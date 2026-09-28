#!/usr/bin/env python3
"""Plot two stabilized subgraphs and the real 2D network from its base."""
from pathlib import Path
import argparse
import json
import struct
import gzip

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from PIL import Image

BG = "#fff7ef"
BLUE = "#1f77b4"
INK = "#263238"
PRE_STABILIZATION = "#f28e2b"  # warm orange, distinct from stabilized purple
FULL_NETWORK = "#8a5cf6"  # stabilized network
ACCENT = "#d1495b"

parser = argparse.ArgumentParser()
parser.add_argument("snapshot_dir", type=Path)
parser.add_argument("sample_data", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--svg-output", type=Path,
                    help="Also save an SVG with vector text, annotations and axes")
parser.add_argument("--base-snapshot", type=Path, required=True,
                    help="SOPBASE1 mask snapshot for z=0 through z_stat")
args = parser.parse_args()

metadata = json.loads(args.sample_data.read_text())
L = int(metadata["meta"]["L"])
sample_count = int(metadata["meta"].get("collected_samples", 0))
if sample_count != 2:
    raise SystemExit(f"Expected 2 collected samples, found {sample_count}")
anchors = metadata["data"].get("anchor_z", [])
if len(anchors) != 2:
    raise SystemExit(f"Expected 2 sample anchors, found {len(anchors)}")
z_stat = int(metadata["meta"].get("z_stab", metadata["meta"]["z_stat_by_species"][0]))
if z_stat <= 0:
    raise SystemExit(f"Invalid z_stat in metadata: {z_stat}")

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


def snapshot_path(i):
    plain = args.snapshot_dir / f"sample_{i}.components"
    compressed = args.snapshot_dir / f"sample_{i}.components.gz"
    if plain.exists():
        return plain
    return compressed


images = [load_component_image(snapshot_path(i), L) for i in range(1, 3)]

base_opener = gzip.open if args.base_snapshot.suffix == ".gz" else open
with base_opener(args.base_snapshot, "rb") as stream:
    if stream.read(8) != b"SOPBASE1":
        raise SystemExit(f"Invalid base snapshot: {args.base_snapshot}")
    base_L, base_height = struct.unpack("<II", stream.read(8))
    if base_L != L or base_height != z_stat:
        raise SystemExit(
            f"Base snapshot dimensions {base_L} x {base_height} do not match L={L}, z_stat={z_stat}")
    base_mask = np.frombuffer(stream.read(), dtype=np.uint8).reshape(base_height, base_L)
base_idx = np.linspace(0, base_L - 1, min(base_L, 4096), dtype=np.int64)
base_view = base_mask[np.ix_(base_idx, base_idx)]
base_rgb = np.empty(base_view.shape + (3,), dtype=np.uint8)
base_rgb[:] = np.asarray([255, 247, 239], dtype=np.uint8)
base_rgb[base_view > 0] = np.asarray([242, 142, 43], dtype=np.uint8)

fig = plt.figure(figsize=(8.5, 8.2), facecolor="white")

# The left column preserves the absolute z origin: the base is [0,z_stat),
# sample 1 is [z_stat,z_stat+L), then one L-sized gap and sample 2.
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
for i, image in enumerate(images):
    low = 1 + 2 * i
    occupied = np.any(image != background_rgb, axis=2)
    full_network_view = np.empty_like(image)
    full_network_view[:] = background_rgb
    full_network_view[occupied] = network_rgb
    ax.imshow(full_network_view, interpolation="nearest", extent=(0, 1, low, low + 1),
              origin="upper", rasterized=True, aspect="auto")
    ax.add_patch(Rectangle((0, low), 1, 1, fill=False,
                           edgecolor=ACCENT, linewidth=1.5))
ax.text(0.5, 0.06, "z = 0", ha="center", va="center", fontsize=8.5,
        color=INK, bbox=dict(facecolor="white", alpha=0.82, edgecolor="none", pad=1.5))
ax.axhline(1, color=ACCENT, linewidth=1.1, zorder=5)
ax.text(0.5, 1.06, rf"$z_{{\mathrm{{stat}}}} = {z_stat}$", ha="center", va="bottom",
        fontsize=8.5, color=ACCENT,
        bbox=dict(facecolor="white", alpha=0.9, edgecolor="none", pad=1.5), zorder=6)
ax.text(0.5, 2.5, "Gap = L", ha="center", va="center", fontsize=9,
        color=ACCENT,
        bbox=dict(facecolor="white", alpha=0.82, edgecolor="none", pad=1.5))
# Scale bar for the full network's transverse width.
ax.annotate("", xy=(1, -0.10), xytext=(0, -0.10), annotation_clip=False,
            arrowprops=dict(arrowstyle="<->", color=INK, linewidth=1.1))
ax.text(0.5, -0.24, f"L = {L}", ha="center", va="top", fontsize=10,
        color=INK, clip_on=False)

# Enlarged samples with horizontal and vertical L scale bars.
panel_left = 0.30
cell_height = overview_height / 4.0
scale_band = 0.04
panel_size = overview_height / 3.0 - scale_band  # retain the original panel size
panel_width = panel_size * 8.2 / 8.5  # square panels in physical units
sample_bands = (1, 3)
panel_bottoms = [overview_bottom + (band + 0.5) * cell_height - panel_size / 2
                 for band in sample_bands]
panels = []
for i, (image, panel_bottom) in enumerate(zip(images, panel_bottoms)):
    center = panel_bottom + panel_size / 2
    panel = fig.add_axes([panel_left, panel_bottom,
                          panel_width, panel_size], facecolor=BG)
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

    # Horizontal L bar underneath each square.
    panel.annotate("", xy=(L, -0.02 * L), xytext=(0, -0.02 * L),
                   annotation_clip=False,
                   arrowprops=dict(arrowstyle="<->", color=INK, linewidth=1.0))
    panel.text(L / 2, -0.06 * L, "L", ha="center", va="top",
               fontsize=9, color=INK, clip_on=False)
    # Vertical L bar to the right of each square.
    panel.annotate("", xy=(1.08 * L, L), xytext=(1.08 * L, 0),
                   annotation_clip=False,
                   arrowprops=dict(arrowstyle="<->", color=INK, linewidth=1.0))
    panel.text(1.15 * L, L / 2, "L", ha="left", va="center", rotation=90,
               fontsize=9, color=INK, clip_on=False)
    panels.append(panel)

# Connect each selected window to its corresponding enlargement.
fig.canvas.draw()
for i, panel in enumerate(panels):
    source_px = ax.transData.transform((1.0, sample_bands[i] + 0.5))
    source = fig.transFigure.inverted().transform(source_px)
    target_px = panel.transAxes.transform((0.0, 0.5))
    target = fig.transFigure.inverted().transform(target_px)
    fig.add_artist(FancyArrowPatch(source, target, transform=fig.transFigure,
                    arrowstyle="-|>", mutation_scale=12, linewidth=1.1,
                    color=ACCENT, connectionstyle="arc3,rad=0"))

# Four legend categories separate pre- and post-stabilization network regions.
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

args.output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.output, dpi=500, bbox_inches="tight", facecolor="white")
if args.svg_output:
    args.svg_output.parent.mkdir(parents=True, exist_ok=True)
    # Keep labels editable as text in vector editors. Dense network pixels
    # remain embedded images so the SVG stays reasonably small.
    plt.rcParams["svg.fonttype"] = "none"
    fig.savefig(args.svg_output, format="svg", dpi=600,
                bbox_inches="tight", facecolor="white")
plt.close(fig)
print(args.output)
if args.svg_output:
    print(args.svg_output)
