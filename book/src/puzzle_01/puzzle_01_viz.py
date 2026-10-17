"""Render the map for puzzle 1 with car 0's seven beams sphere-traced through it.

Writes media/01.png (light) and media/01d.png (dark). The marching code here is
the same algorithm as the kernel, in numpy, so the picture matches the numbers.

    pixi run viz01
"""

import shutil
from pathlib import Path

import numpy as np
from manim import *
from scipy.ndimage import distance_transform_edt

MAP = """\
########################################
#......................................#
#......................................#
#......................####............#
#......................####..###.......#
#..........####..............###.......#
#..........####..............###.......#
#..........####........................#
#......................................#
#......................................#
#......................................#
#......................................#
#..........................#######.....#
#......................................#
#......................................#
########################################"""
NUM_BEAMS = 7
FOV = np.radians(270.0)
MAX_RANGE = np.float32(16.0)
CARS = [(19.5, 8.5, 0.0), (6.5, 12.5, np.pi / 2), (34.5, 3.5, -3 * np.pi / 4)]

rows = MAP.split("\n")
H, W = len(rows), len(rows[0])
free = np.array([[ch != "#" for ch in r] for r in rows])
edt = distance_transform_edt(free).astype(np.float32)
thetas = np.linspace(-FOV / 2, FOV / 2, NUM_BEAMS)
dirs = np.column_stack([np.cos(thetas), np.sin(thetas)]).astype(np.float32)


def raycast(p, d):
    """Sphere-march; returns the list of (position, step) visited."""
    p = np.array(p, np.float32)
    dist = np.float32(0)
    visits = []
    while dist < MAX_RANGE:
        col, row = int(p[0]), int(p[1])
        if not (0 <= col < W and 0 <= row < H):
            break
        step = edt[row, col]
        if step == 0:
            break
        visits.append((p.copy(), step))
        p = (p + d * step).astype(np.float32)
        dist += step
    return visits, p


class Puzzle01Visualization(Scene):
    dark = False

    def construct(self):
        s = 0.3  # scene units per px
        ink = WHITE if self.dark else BLACK
        wall = "#3b4252" if self.dark else "#2e3440"
        floor = "#1b1f27" if self.dark else "#f4f6fa"

        def at(col, row):
            return np.array([(col - W / 2) * s, (H / 2 - row) * s, 0])

        grid = VGroup()
        for r in range(H):
            for c in range(W):
                cell = Square(side_length=s, stroke_width=0.4, stroke_color=ink, stroke_opacity=0.25)
                cell.set_fill(wall if rows[r][c] == "#" else floor, opacity=1)
                cell.move_to(at(c + 0.5, r + 0.5))
                grid.add(cell)
        self.add(grid)

        # Car 0's beams: the step circles are the whole idea of sphere tracing.
        col, row, psi = CARS[0]
        for j in range(NUM_BEAMS):
            d = np.array([dirs[j][0], -dirs[j][1]], np.float32)
            visits, end = raycast((col, row), d)
            for p, step in visits:
                self.add(Circle(radius=step * s, stroke_width=1, stroke_color=BLUE_C, stroke_opacity=0.55).move_to(at(*p)))
                self.add(Dot(at(*p), radius=0.025, color=BLUE_C))
            self.add(Line(at(col, row), at(*end), stroke_width=2, color=ORANGE))
            self.add(Dot(at(*end), radius=0.04, color=RED))

        for k, (col, row, psi) in enumerate(CARS):
            tip = (col + 0.9 * np.cos(psi), row - 0.9 * np.sin(psi))
            self.add(Arrow(at(col, row), at(*tip), buff=0, stroke_width=3, color=GREEN_C, max_tip_length_to_length_ratio=0.5))
            self.add(Dot(at(col, row), radius=0.07, color=GREEN_C))
            self.add(Text(f"car {k}", font_size=14, color=ink).next_to(at(col, row), DOWN, buff=0.08))

        title = Text("edt[p] is the radius of an empty circle around p: step that far", font_size=16, color=ink)
        title.next_to(grid, UP, buff=0.2)
        self.add(title)


if __name__ == "__main__":
    here = Path(__file__).parent
    for dark, name in ((False, "01"), (True, "01d")):
        with tempconfig(
            {
                "format": "png",
                "save_last_frame": True,
                "preview": False,
                "media_dir": str(here / "media"),
                "output_file": name,
                "pixel_width": 1600,
                "pixel_height": 760,
                "frame_width": 14.0,
                "frame_height": 6.65,
                "background_color": "#0f1419" if dark else WHITE,
            }
        ):
            Puzzle01Visualization.dark = dark
            Puzzle01Visualization().render()
        shutil.move(here / "media" / "images" / f"{name}.png", here / "media" / f"{name}.png")
