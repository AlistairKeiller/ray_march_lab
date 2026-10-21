"""Figure for Puzzle 3: the 16 x 16 room, its distance field, and the eight sphere-marched
rays with the step circles that give the method its name. Writes media/03.png and 03d.png.

    uv run --extra viz python book/src/puzzle_03/puzzle_03_viz.py
"""

from pathlib import Path

import matplotlib
import numpy as np
from scipy.ndimage import distance_transform_edt

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

SIZE, NUM_RAYS, MAX_PX = 16, 8, 8.0
OUT = Path(__file__).parent / "media"


def room(size):
    free = np.zeros((size, size), bool)
    free[1:-1, 1:-1] = True
    free[4:6, 10:12] = False
    return distance_transform_edt(free)


def march(edt, p, d, max_px):
    """The samples (point, radius) the march visits, and the distance it reports."""
    samples, dist = [], 0.0
    while dist < max_px:
        col, row = int(p[0]), int(p[1])
        if not (0 <= col < edt.shape[1] and 0 <= row < edt.shape[0]):
            break
        step = edt[row, col]
        samples.append((p.copy(), step))
        if step == 0.0:
            break
        p = p + d * step
        dist += step
    return samples, min(dist, max_px)


def draw(dark):
    plt.style.use("dark_background" if dark else "default")
    fig, ax = plt.subplots(figsize=(6.4, 6.4), dpi=150)
    edt = room(SIZE)
    ax.imshow(edt, cmap="magma" if dark else "Blues", origin="upper", extent=(0, SIZE, SIZE, 0), vmin=0, vmax=edt.max() * 1.3)
    ax.set_xticks(range(0, SIZE + 1, 2))
    ax.set_yticks(range(0, SIZE + 1, 2))
    ax.grid(color="0.6" if dark else "0.8", lw=0.5)
    for (r, c), v in np.ndenumerate(edt):
        if v > 0:
            ax.text(c + 0.5, r + 0.5, f"{v:.0f}" if v == int(v) else f"{v:.1f}", ha="center", va="center", fontsize=6, color="0.9" if dark else "0.3")

    start = np.array([SIZE / 2 + 0.5, SIZE / 2 + 0.5])
    colors = plt.cm.tab10(np.linspace(0, 1, NUM_RAYS))
    for k, color in enumerate(colors):
        a = k * 2 * np.pi / NUM_RAYS
        d = np.array([np.cos(a), np.sin(a)])
        samples, dist = march(edt, start, d, MAX_PX)
        end = start + d * dist
        ax.plot([start[0], end[0]], [start[1], end[1]], color=color, lw=1.6)
        for p, r in samples:
            if r > 0:
                ax.add_patch(Circle(p, r, fill=False, color=color, lw=0.9, alpha=0.6))
            ax.plot(*p, "o", color=color, ms=3)
        ax.plot(*end, "x", color=color, ms=6, mew=1.5)
    ax.plot(*start, "o", color="white" if dark else "black", ms=5)
    ax.set_xlabel("col")
    ax.set_ylabel("row")
    ax.set_title("Puzzle 3: sphere march. Each circle is one step of radius edt[row, col]", fontsize=9)
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / ("03d.png" if dark else "03.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    draw(dark=False)
    draw(dark=True)
    print(f"wrote {OUT / '03.png'} and {OUT / '03d.png'}")
