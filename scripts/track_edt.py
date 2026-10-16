#!/usr/bin/env python3
"""Turn a ROS-style map into the wall-distance field the lidar kernel marches on.

Reads the map yaml (image, resolution), thresholds the image the way
warporacer's `track.py` does (pixels at or above 230 are drivable), runs scipy's
Euclidean distance transform, and writes `<name>.edt` next to the yaml:

    int32 H, int32 W, float32 resolution (m/px), then H*W float32, row-major

`p01.mojo --benchmark` reads this file. It is a few MB and gitignored, so run
`pixi run edt01` once after cloning (the `p01` task does it for you).

Usage:
    python scripts/track_edt.py maps/Spielberg_map.yaml
"""

import struct
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from scipy.ndimage import distance_transform_edt

OCC_THRESH = 230  # image value at/above which a pixel is drivable


def main(yaml_path: Path) -> None:
    meta = yaml.safe_load(yaml_path.read_text())
    image = np.asarray(Image.open(yaml_path.parent / meta["image"]).convert("L"))
    free = image >= OCC_THRESH
    edt = distance_transform_edt(free).astype(np.float32)
    res = float(meta["resolution"])

    out = yaml_path.with_suffix(".edt")
    with open(out, "wb") as f:
        f.write(struct.pack("<iif", edt.shape[0], edt.shape[1], res))
        f.write(edt.tobytes())
    print(
        f"{out}: {edt.shape[0]} x {edt.shape[1]} px at {res} m/px, "
        f"{free.mean() * 100:.1f}% drivable, "
        f"median clearance on track {np.median(edt[free]) * res:.2f} m"
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(Path(sys.argv[1]))
