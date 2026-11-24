#!/usr/bin/env python3
"""Create 3-D schematics of the initial base-layer seeding layouts.

The placement follows src/network.cpp for num_colors=2 and dim=3.  P0=1 is
used deliberately, so rho is the visible occupied fraction of the base for
each color.  The simulation itself activates only the z=0 layer initially.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


L = 12
P0 = 1.0
COLORS = ("#e76f51", "#277da1")
LAYOUTS = ("blocks", "alternating", "random", "clustered")


def quota(rho: float) -> int:
    return round(P0 * rho * L * L)


def evenly_spaced(candidates: list[tuple[int, int]], n: int) -> list[tuple[int, int]]:
    """Same midpoint selection as activate_deterministic_base_sites."""
    n = min(n, len(candidates))
    selected: list[tuple[int, int]] = []
    previous = -1
    for k in range(n):
        pos = min(((2 * k + 1) * len(candidates)) // (2 * n), len(candidates) - 1)
        pos = max(pos, previous + 1)
        previous = pos
        selected.append(candidates[pos])
    return selected


def clustered(color: int, n: int, occupied: set[tuple[int, int]]) -> list[tuple[int, int]]:
    """Periodic breadth-first expansion used by activate_clustered_base_sites."""
    x0 = color * L // 2 + L // 4
    y0 = L // 2
    queue = deque([(x0, y0)])
    visited = {(x0, y0)}
    selected: list[tuple[int, int]] = []
    while queue and len(selected) < n:
        x, y = queue.popleft()
        if (x, y) not in occupied:
            occupied.add((x, y))
            selected.append((x, y))
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbor = ((x + dx) % L, (y + dy) % L)
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return selected


def positions(layout: str, rho: float) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    n = quota(rho)
    # The sparse two-colour rules in src/network.cpp enumerate a full y column
    # before advancing in x, producing left / gray centre / right blocks and a
    # centred (colour 1, colour 2, gray) sequence.
    def from_base_order(order: int) -> tuple[int, int]:
        return order // L, order % L

    if rho < .5 and layout == "blocks":
        return ([from_base_order(k) for k in range(n)],
                [from_base_order(L * L - 1 - k) for k in range(n)])
    if layout == "random":
        rng = np.random.default_rng(20261124 + int(rho * 100))
        available = [(x, y) for y in range(L) for x in range(L)]
        rng.shuffle(available)
        return available[:n], available[n:2 * n]
    if layout == "clustered":
        occupied: set[tuple[int, int]] = set()
        return clustered(0, n, occupied), clustered(1, n, occupied)

    groups: list[list[tuple[int, int]]] = [[], []]
    for y in range(L):
        for x in range(L):
            color = (x // (L // 2)) if layout == "blocks" else ((x + y) % 2)
            groups[color].append((x, y))
    return evenly_spaced(groups[0], n), evenly_spaced(groups[1], n)


def draw_cell_grid(ax: plt.Axes) -> None:
    for i in range(L + 1):
        ax.plot([i - .5, i - .5], [-.5, L - .5], color="#d6d6d6", lw=.45, zorder=0)
        ax.plot([-.5, L - .5], [i - .5, i - .5], color="#d6d6d6", lw=.45, zorder=0)


def draw_panel(ax: plt.Axes, layout: str, rho: float) -> None:
    first, second = positions(layout, rho)
    draw_cell_grid(ax)
    for points, color, label in ((first, COLORS[0], "cor 1"), (second, COLORS[1], "cor 2")):
        x, y = zip(*points) if points else ([], [])
        ax.scatter(x, y, s=76, marker="s", c=color, edgecolors="white", linewidths=.35, label=label, zorder=2)
    # Dashed perspective guides: the occupied cells are on z=0, not through the volume.
    ax.plot([-.5, -.5, L - .5, L - .5, -.5], [-.5, L - .5, L - .5, -.5, -.5],
            color="#505050", lw=.8, zorder=3)
    ax.text(.02, .95, "base z = 0", transform=ax.transAxes, fontsize=8, color="#555")
    ax.set_aspect("equal")
    ax.set(xlim=(-.8, L - .2), ylim=(-.8, L - .2), xticks=(), yticks=())
    ax.set_title(f"{layout}\n$\\rho={rho:.1f}$  →  {len(first)} sítios/cor", fontsize=10, weight="bold")


def main() -> None:
    fig, axes = plt.subplots(2, 4, figsize=(13.4, 7.5), constrained_layout=True)
    for row, rho in enumerate((0.5, 0.4)):
        for col, layout in enumerate(LAYOUTS):
            draw_panel(axes[row, col], layout, rho)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="lower center", bbox_to_anchor=(.5, -.035), frameon=False)
    fig.suptitle(
        "Esquema de condições iniciais — dim=3, nc=2, P0=1\n"
        "A semeadura ocupa apenas a camada-base z=0; o crescimento posterior segue a direção z.",
        fontsize=14, weight="bold",
    )
    out = Path("figures") / "initial_layout_schematic_dim3_nc2.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
    print(out)


if __name__ == "__main__":
    main()
