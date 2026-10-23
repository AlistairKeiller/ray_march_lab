# Puzzle 4: Lidar

## Overview

Implement the lidar kernel: one thread per (car, beam) that shoots a ray from
the car's laser mount, in the beam's direction rotated by the car's heading, and
writes the range in meters into `scan[i, j]`. This is warporacer's
`lidar_kernel` on a single map, run on the Spielberg track with 4 cars and 108
beams.

**Note:** _You have a 2D launch: one block per car, one thread per beam._

<img src="./media/04.png" alt="Four lidar scans on the Spielberg track" class="light-mode-img">
<img src="./media/04d.png" alt="Four lidar scans on the Spielberg track" class="dark-mode-img">

## Key concepts

In this puzzle, you'll learn about:

- A 2D grid: one block per car (`block_idx.y`) and one thread per beam
  (`thread_idx.x`), launched with `grid_dim=(1, NUM_CARS)` and
  `block_dim=(TPB, 1)`
- A guard: 128 threads per block for 108 beams, so `if j < NUM_LIDAR` keeps
  the spare twenty from writing past the row
- Tensors whose shape is only known at run time: `RowMajorLayout[Int, Int]`,
  built with `row_major(Coord(h, w))`
- Rotating a direction into the heading frame, and the sign flip the row axis
  introduces
- Composing what you have: the pixel transform from Puzzle 2 and `raycast` from
  Puzzle 3

For car \\(i\\) at \\((x_i, y_i)\\) with heading \\(\psi_i\\), and beam \\(j\\)
with unit direction \\((b_x, b_y)\\) in the car frame, the ray starts at the
pixel

\\[\Large p = \begin{pmatrix} (x_i + m \cos\psi_i - o_x) / res \\\\ (H - 1) - (y_i + m \sin\psi_i - o_y) / res \end{pmatrix}\\]

and points along

\\[\Large d = \begin{pmatrix} \cos\psi_i \cdot b_x - \sin\psi_i \cdot b_y \\\\ -(\sin\psi_i \cdot b_x + \cos\psi_i \cdot b_y) \end{pmatrix}\\]

\\[\Large scan[i, j] = raycast(edt, p, d, R / res) \times res\\]

The first is Puzzle 2's world-to-pixel transform without the floor and the
clamp, applied to the laser mount instead of the car's centre; the second is
the beam rotated by the heading, with its row component negated.

| in the formula | in the code | meaning |
|---|---|---|
| \\(x_i, y_i, \psi_i\\) | `cars[i, 0]`, `cars[i, 1]`, `cars[i, 2]` | the car's world position and heading |
| \\(\cos\psi_i, \sin\psi_i\\) | `ch`, `sh` | computed once per thread |
| \\(m\\) | `LIDAR_MOUNT_X` | the laser mount, 0.2733 m ahead of the car's centre |
| \\(b_x, b_y\\) | `lidar_dirs[j, 0]`, `lidar_dirs[j, 1]` | beam \\(j\\)'s unit direction in the car frame |
| \\(o_x, o_y\\), \\(res\\), \\(H\\) | `ox`, `oy`, `res`, `edt.dim[0]()` | the map, as in Puzzle 2 |
| \\(R\\) | `LIDAR_RANGE` | the lidar range, 20 m |
| \\(p, d\\) | `p`, `d` | `raycast`'s start pixel and unit direction from Puzzle 3 |

### The two frames

The beam directions are unit vectors in the car frame (\\(x\\) forward,
\\(y\\) left), spread over a 270° fan centred on the car's nose: beam
\\(j\\) points at \\(\theta_j = -135^\circ + 270^\circ \cdot j / 107\\), so
\\((b_x, b_y) = (\cos\theta_j, \sin\theta_j)\\), with beam 0 pointing to the
car's right-rear and beam 107 to its left-rear. Rotating by \\(\psi_i\\) takes
a beam into the world frame. The raster's row axis points the opposite way to
world \\(y\\), so the row component of the direction is negated: that is the
minus sign on the second row of \\(d\\). And the ray starts at the laser, not
the car centre: `LIDAR_MOUNT_X` ahead of `base_link` along the heading.

### Launch order

In a block, consecutive threads are consecutive beams of the same car. Their
rays start at the same pixel and fan out, so early in the march they read the
same cache lines. That matters in Puzzle 6.

## Configuration

- Map: `maps/Spielberg_map.yaml`, 1130 × 1769 px at 0.05796 m/px (an F1TENTH
  racetrack)
- Cars: `NUM_CARS = 4`, spawned at random drivable pixels at least 0.5 m from a
  wall, with random headings
- Beams: `NUM_LIDAR = 108` over `LIDAR_FOV = 270°`, `LIDAR_RANGE = 20` m, mount
  offset `LIDAR_MOUNT_X = 0.2733` m
- Grid: `NUM_CARS` blocks of `TPB = 128` threads
- Output: `scan` of shape `(NUM_CARS, NUM_LIDAR)`, in meters
- Check: within one pixel of a CPU float32 reference

Notes:

- **Map loading** is `harness.track.Track`: the image thresholded at 230,
  through SciPy's `distance_transform_edt`, copied to a device buffer. The
  loader itself is `harness/track.py`, a few lines of Python called through
  interop; it is warporacer's loader minus the centerline extraction, and the
  same file the Warp twin uses
- **Beam directions** are built on the host into a `(NUM_LIDAR, 2)` tensor
  of `(cos, sin)` pairs, `NUM_LIDAR` angles evenly spaced over `LIDAR_FOV`
- `raycast` is Puzzle 3's solution, unchanged; it's in the listing so you can
  see its signature. The map's layout is `EdtLayout = RowMajorLayout[Int, Int]`
  rather than a `row_major[H, W]()` constant, because the size comes from the
  file

## Code to complete

```mojo
{{#include ../../../problems/p04/p04.mojo:lidar}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p04/p04.mojo" class="filename">View full file: problems/p04/p04.mojo</a>

<details>
<summary><strong>Tips</strong></summary>

<div class="solution-tips">

1. Start point: add `LIDAR_MOUNT_X * ch` to `x` and `LIDAR_MOUNT_X * sh` to
   `y`, then apply Puzzle 2's world-to-pixel transform, with `edt.dim[0]()`
   for the height. No clamp this time; `raycast` handles rays that leave the
   map
2. Direction: rotate `lidar_dirs[j, 0]`, `lidar_dirs[j, 1]` by `(ch, sh)` and
   negate the row component
3. Range: `raycast(edt, p, d, LIDAR_RANGE / res)` returns pixels; multiply by
   `res` for meters

</div>
</details>

## Running the code

To test your solution, run the following command in your terminal:

```bash
uv run mojo -I . problems/p04/p04.mojo
```

Your output will look like this if the puzzle isn't solved yet:

```txt
map: Spielberg_map (1130 x 1769 px, 0.05796 m/px)
car 0 pose (x, y, psi): -3.9124591 -24.971817 -2.8841484
out (car 0, every 18th beam): [0.0 0.0 0.0 0.0 0.0 0.0]
expected: [1.8487979 1.2861104 1.7497904 20.0 1.8376681 1.3440704]
max |out - expected|: 20.0 m
```

and like this once it is:

```txt
map: Spielberg_map (1130 x 1769 px, 0.05796 m/px)
car 0 pose (x, y, psi): -3.9124591 -24.971817 -2.8841484
out (car 0, every 18th beam): [1.8487979 1.2861104 1.7497904 20.0 1.8376681 1.3440704]
expected: [1.8487979 1.2861104 1.7497904 20.0 1.8376681 1.3440704]
max |out - expected|: 0.0 m
Puzzle 04 complete ✅
```

## Solution

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../solutions/p04/p04.mojo:lidar_solution}}
```

<div class="solution-explanation">

This solution, per thread:

1. **Start point** `p`: the laser mount in world coordinates,
   `(x + LIDAR_MOUNT_X * ch, y + LIDAR_MOUNT_X * sh)`, pushed through the
   Puzzle 2 transform. The row uses `Scalar[dtype](Int(edt.dim[0]()) - 1) - ...` for the flip.
   There is no clamp because `raycast` breaks out as soon as a sample leaves
   the raster
2. **Direction** `d`: the beam `(bx, by) = lidar_dirs[j]` rotated by the
   heading, `(ch * bx - sh * by, sh * bx + ch * by)`, with the second component
   negated so that "up" in the world is "toward row 0" in the raster
3. **Range**: `raycast` works in pixels, so the maximum is `LIDAR_RANGE / res`
   and the result is multiplied back by `res`

Car 0's fourth sampled beam reports exactly `20.0`: it ran the full range down
a straight without touching anything, which is `min(dist, max_px)` doing its
job from Puzzle 3.

The check is one pixel, `atol = track.res`. On the Apple GPU this book was
written on, the kernel and the CPU reference agree exactly, as the `0.0`
shows. A GPU that fuses multiply-adds can move a sample by an ulp, and a sample
an ulp from a pixel boundary can land in the neighbouring pixel. That changes
the reported range by at most one step, and the last step before a wall is
about one pixel.

</div>
</details>
