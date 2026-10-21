"""Wall-distance field of a ROS-style map (yaml + image): the raster the sim ray-marches.

A trimmed-down `warporacer.track.Track`: no centerline or waypoint LUT, just the Euclidean
distance transform of the drivable pixels, in pixels. The Mojo lab drives this same file
through Python interop, so both labs load maps identically."""

from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from yaml import safe_load

OCC_THRESH = 230  # image value at/above which a pixel is drivable


class Track:
    """edt (h, w) float32 px to nearest wall; res m/px; (ox, oy) world origin of the
    bottom-left pixel; h, w. Image row 0 is the top of the map, so rows run against y."""

    def __init__(self, yaml_path):
        yaml_path = Path(yaml_path)
        meta = safe_load(yaml_path.read_text())
        self.name = yaml_path.stem
        image = np.asarray(Image.open(yaml_path.parent / meta["image"]).convert("L"))
        self.res = float(meta["resolution"])
        self.ox, self.oy = float(meta["origin"][0]), float(meta["origin"][1])
        self.h, self.w = image.shape
        self.edt = distance_transform_edt(image >= OCC_THRESH).astype(np.float32)

    def world_to_px(self, x, y):
        return (x - self.ox) / self.res, self.h - 1 - (y - self.oy) / self.res

    def px_to_world(self, col, row):
        return self.ox + col * self.res, self.oy + (self.h - 1 - row) * self.res

    def sample_poses(self, n, rng, clearance=0.5):
        """n random (x, y, psi) at pixel centres at least `clearance` m from a wall."""
        rows, cols = np.nonzero(self.edt * self.res > clearance)
        k = rng.integers(0, len(rows), n)
        x, y = self.px_to_world(cols[k] + 0.5, rows[k] + 0.5)
        return np.column_stack([x, y, rng.uniform(-np.pi, np.pi, n)]).astype(np.float32)


def stack(tracks):
    """Pad tracks into one (n_maps, H, W) raster. Padding is zero, i.e. wall, so a ray that
    leaves a small map's true bounds stops there instead of wandering into the padding."""
    h, w = max(t.h for t in tracks), max(t.w for t in tracks)
    edt = np.zeros((len(tracks), h, w), np.float32)
    for k, t in enumerate(tracks):
        edt[k, : t.h, : t.w] = t.edt
    return edt


def load_tracks(maps, k, rng):
    """Up to k maps from a yaml or a directory of them, sampled like warporacer does."""
    maps = Path(maps)
    paths = sorted(maps.glob("*.yaml")) if maps.is_dir() else [maps]
    if not paths:
        raise FileNotFoundError(f"no map yamls under {maps}")
    return [Track(p) for p in rng.choice(paths, size=min(k, len(paths)), replace=False)]


def spawn(tracks, num_envs, rng):
    """A map id per car, uniform over `tracks`, and a random drivable pose on that map, the way
    warporacer's Env spawns its fleet. Returns (cars (n, 3) float32, map_id (n,) int32)."""
    map_id = rng.integers(0, len(tracks), num_envs).astype(np.int32)
    cars = np.zeros((num_envs, 3), np.float32)
    for k, t in enumerate(tracks):
        mask = map_id == k
        cars[mask] = t.sample_poses(int(mask.sum()), rng)
    return cars, map_id
