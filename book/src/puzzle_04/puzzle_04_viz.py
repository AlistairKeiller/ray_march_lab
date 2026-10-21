"""Figure for Puzzle 4: four cars on the Spielberg track with their 108-beam scans. Uses the
same seed as the puzzle, so these are the cars the puzzle checks. Writes media/04.png and 04d.png.

    uv run --extra viz python book/src/puzzle_04/puzzle_04_viz.py
"""

from pathlib import Path

import matplotlib
import numpy as np

from harness.track import Track

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).parents[3]
OUT = Path(__file__).parent / "media"
NUM_CARS, NUM_LIDAR, LIDAR_FOV, LIDAR_RANGE, LIDAR_MOUNT_X = 4, 108, np.radians(270.0), 20.0, 0.2733


def scan(track, car, dirs):
    """Hit points in pixel coordinates for one car, marching in float64 (this is a picture)."""
    x, y, psi = car
    ch, sh = np.cos(psi), np.sin(psi)
    p0 = np.array(track.world_to_px(x + LIDAR_MOUNT_X * ch, y + LIDAR_MOUNT_X * sh))
    hits = []
    for bx, by in dirs:
        d = np.array([ch * bx - sh * by, -(sh * bx + ch * by)])
        p, dist, max_px = p0.copy(), 0.0, LIDAR_RANGE / track.res
        while dist < max_px:
            col, row = int(p[0]), int(p[1])
            if not (0 <= col < track.w and 0 <= row < track.h):
                break
            step = track.edt[row, col]
            if step == 0.0:
                break
            p, dist = p + d * step, dist + step
        hits.append(p0 + d * min(dist, max_px))
    return p0, np.array(hits)


def draw(dark):
    plt.style.use("dark_background" if dark else "default")
    track = Track(ROOT / "maps" / "Spielberg_map.yaml")
    rng = np.random.default_rng(0)
    cars = track.sample_poses(NUM_CARS, rng)
    beams = np.linspace(-LIDAR_FOV / 2, LIDAR_FOV / 2, NUM_LIDAR)
    dirs = np.column_stack([np.cos(beams), np.sin(beams)])

    fig, ax = plt.subplots(figsize=(9, 9 * track.h / track.w), dpi=150)
    ax.imshow(track.edt > 0, cmap="gray", origin="upper", interpolation="nearest", vmin=-0.3 if dark else 0, vmax=1 if dark else 1.4)
    colors = plt.cm.tab10(np.arange(NUM_CARS))
    for car, color in zip(cars, colors):
        p0, hits = scan(track, car, dirs)
        for h in hits:
            ax.plot([p0[0], h[0]], [p0[1], h[1]], color=color, lw=0.4, alpha=0.7)
        ax.plot(hits[:, 0], hits[:, 1], ".", color=color, ms=1.5)
        ax.plot(*p0, "o", color=color, ms=5, mec="white" if dark else "black")
    ax.set_xlim(0, track.w)
    ax.set_ylim(track.h, 0)
    ax.set_axis_off()
    ax.set_title(f"Puzzle 4: {NUM_CARS} cars x {NUM_LIDAR} beams on {track.name} ({track.w} x {track.h} px, {track.res} m/px)", fontsize=9)
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / ("04d.png" if dark else "04.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    draw(dark=False)
    draw(dark=True)
    print(f"wrote {OUT / '04.png'} and {OUT / '04d.png'}")
