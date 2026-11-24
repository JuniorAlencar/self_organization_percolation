#!/usr/bin/env python3
"""Render the dim=2 initial-base layouts implemented in src/network.cpp."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


L = 48
COLORS = ("#e76f51", "#277da1")
LAYOUTS = ("blocks", "alternating", "random", "clustered")


def evenly_spaced(candidates: list[int], n: int) -> list[int]:
    return [min(((2 * k + 1) * len(candidates)) // (2 * n), len(candidates) - 1) for k in range(n)]


def sites(layout: str, rho: float) -> tuple[list[int], list[int]]:
    n = round(rho * L)  # P0=1, matching q = round(P0 * rho * base_size)
    if rho < .5 and layout == "blocks":
        return list(range(n)), list(range(L - n, L))
    if rho < .5 and layout == "alternating":
        gray = L - 2 * n
        first, second = [], []
        position = 0
        for pair in range(n + 1):
            gray_before = ((pair + 1) * gray) // (n + 1) - (pair * gray) // (n + 1)
            position += gray_before
            if pair < n:
                first.append(position)
                second.append(position + 1)
                position += 2
        return first, second
    if layout == "blocks":
        candidates = [list(range(0, L // 2)), list(range(L // 2, L))]
        return tuple([candidates[c][i] for i in evenly_spaced(candidates[c], n)] for c in range(2))
    if layout == "alternating":
        candidates = [list(range(c, L, 2)) for c in range(2)]
        return tuple([candidates[c][i] for i in evenly_spaced(candidates[c], n)] for c in range(2))
    if layout == "random":
        rng = np.random.default_rng(20261124 + int(rho * 100))
        available = rng.permutation(L).tolist()
        return available[:n], available[n:2 * n]

    # The 2-D clustered implementation alternates right/left around each
    # color's sector centre, with periodic wrapping.
    occupied: set[int] = set()
    out: list[list[int]] = []
    for color in range(2):
        center = color * (L // 2) + L // 4
        chosen: list[int] = []
        step = 0
        while len(chosen) < n:
            if step == 0:
                x = center
            elif step % 2:
                x = (center + (step + 1) // 2) % L
            else:
                x = (center - step // 2) % L
            step += 1
            if x not in occupied:
                occupied.add(x)
                chosen.append(x)
        out.append(chosen)
    return out[0], out[1]


def draw(ax: plt.Axes, layout: str, rho: float) -> None:
    first, second = sites(layout, rho)
    for x in range(L):
        ax.add_patch(plt.Rectangle((x, 0), 1, 1, facecolor="white", edgecolor="#d6d6d6", lw=.45))
    for selected, color in ((first, COLORS[0]), (second, COLORS[1])):
        for x in selected:
            ax.add_patch(plt.Rectangle((x + .09, .09), .82, .82, facecolor=color, edgecolor="white", lw=.35))
    ax.set(xlim=(0, L), ylim=(0, 1), aspect="equal", xticks=(), yticks=())
    ax.set_title(f"{layout}\n$\\rho={rho:.1f}$ → {len(first)} sítios/cor", fontsize=11, weight="bold")


def main() -> None:
    fig, axes = plt.subplots(2, 4, figsize=(15.5, 3.7), constrained_layout=True)
    for row, rho in enumerate((.5, .4)):
        for col, layout in enumerate(LAYOUTS):
            draw(axes[row, col], layout, rho)
    fig.legend(
        [plt.Rectangle((0, 0), 1, 1, color=color) for color in COLORS],
        ["cor 1", "cor 2"], ncol=2, loc="lower center", bbox_to_anchor=(.5, -.12), frameon=False,
    )
    fig.suptitle("Esquema de condições iniciais — dim=2, nc=2, P0=1\nA semeadura ocorre na única camada y=0.", fontsize=14, weight="bold")
    out = Path("figures") / "initial_layout_schematic_dim2_nc2.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
    print(out)


if __name__ == "__main__":
    main()
