# Puzzle 5: Many Maps

## Overview

Implement `lidar_multi`: the lidar kernel for cars spread across several tracks
at once. The maps are stacked into one zero-padded `(n_maps, H, W)` raster,
every car carries a `map_id`, and each map's resolution, origin and height come
from per-map tensors. This is warporacer's `lidar_kernel` exactly, and it's
what lets one `Env` train on a rotating pool of eight tracks.

**Note:** _One block per car and one thread per beam, as in Puzzle 4. The new
work is looking up the car's map._

## Key concepts

In this puzzle, you'll learn about:

- 3D tensors: `RowMajorLayout[Int, Int, Int]`, indexed `edt[mid, row, col]`
- Per-car indirection: `mid = Int(map_id[i])`, then `res[mid]`,
  `origin[mid, 0]`, `map_h[mid]`
- Integer tensors: `TileTensor[mut=False, DType.int32, ...]` for the map ids
  and heights
- Padding as geometry: zero padding reads as wall, so rays stop at each map's
  true edge without a per-map bounds check
- Why the row flip must use the map's own height, not the padded one

### Padding

Maps come in different sizes (here 600 × 600, 602 × 431 and 1130 × 1769), and a
kernel argument has one shape. `harness.track.stack` pads every map into the
top-left corner of an `(n, H, W)` array filled with zeros, on the Python side,
and the stack is copied to the device once. Since `edt == 0` means wall, a ray
that runs off a small map's real area hits padding and stops. The bounds test
inside `raycast` only has to guard the stack's outer edges, and it does so with
`edt.dim[1]()` and `edt.dim[2]()`.

```txt
edt[0]  berlin   600 x 600  |  edt[1]  skirk   602 x 431  |  edt[2]  Spielberg 1130 x 1769
+----------+..............  |  +------+..................  |  +------------------------+
| map      |  zeros (wall)  |  | map  |  zeros (wall)      |  |                        |
+----------+..............  |  |      |                    |  |      whole raster      |
............................  |  +------+..................  |  |                        |
............................  |  ..........................  |  +------------------------+
```

The one place padding bites is the row flip. World \\(y = o_y\\) is the map's
**own** bottom row, `map_h[mid] - 1`, not row \\(H - 1\\) of the padded stack.
Use `map_h[mid]`. With `edt.dim[1]()` instead, berlin's cars would be placed
530 rows, 26.5 meters, below their track, in the padding.

## Configuration

- Maps: `berlin` (600 × 600, 0.05 m/px), `skirk` (602 × 431, 0.05 m/px),
  `Spielberg_map` (1130 × 1769, 0.05796 m/px)
- Stack: `edt` of shape `(3, 1130, 1769)`, layout `row_major(Coord(3, 1130, 1769))`
- Cars: `CARS_PER_MAP = 2` per map, so `map_id = [0, 0, 1, 1, 2, 2]`
- Per-map tensors: `origin` of shape `(NUM_MAPS, 2)`, `res` and `map_h` of
  shape `(NUM_MAPS,)`, the last one `int32`
- Beams, range, mount offset and grid as in Puzzle 4

Notes:

- `map_id` and `map_h` are `DType.int32` tensors; `Int(map_id[i])` turns an
  element into an index
- The per-map tensors are filled on the host from the `Track`s:
  `o[k * 2] = tracks[k].ox`, `r[k] = tracks[k].res`, and so on
- `raycast` now takes the map index as its second argument and reads
  `edt[mid, row, col]`; it is given

## Code to complete

```mojo
{{#include ../../../problems/p05/p05.mojo:lidar_multi}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p05/p05.mojo" class="filename">View full file: problems/p05/p05.mojo</a>

<details>
<summary><strong>Tips</strong></summary>

<div class="solution-tips">

1. `var mid = Int(map_id[i])`, then pull `r = res[mid]` and `org` out once
2. Same start point and direction as Puzzle 4, with
   `org = Vec2(origin[mid, 0], origin[mid, 1])` in place of `ox`, `oy` and
   `Scalar[dtype](Int(map_h[mid]) - 1)` in the row flip
3. `scan[i, j] = raycast(edt, mid, p, d, LIDAR_RANGE / r) * r`

</div>
</details>

## Running the code

To test your solution, run the following command in your terminal:

```bash
uv run mojo -I . problems/p05/p05.mojo
```

Your output will look like this if the puzzle isn't solved yet:

```txt
edt stack: (3, 1130, 1769) from berlin (600 x 600), skirk (602 x 431), Spielberg_map (1130 x 1769)
berlin car 0
  out (every 18th beam): [0.0 0.0 0.0 0.0 0.0 0.0]
  expected:              [2.5235968 1.5508063 1.967034 5.533931 3.5355096 2.3413758]
skirk car 2
  out (every 18th beam): [0.0 0.0 0.0 0.0 0.0 0.0]
  expected:              [1.9987246 1.0361542 1.1664318 1.7951711 3.9266014 5.3179855]
Spielberg_map car 4
  out (every 18th beam): [0.0 0.0 0.0 0.0 0.0 0.0]
  expected:              [1.62288 7.9735894 2.14452 1.27512 1.56492 10.828722]
max |out - expected|: 20.0 m
```

## Solution

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../solutions/p05/p05.mojo:lidar_multi_solution}}
```

<div class="solution-explanation">

This solution is Puzzle 4's kernel with three lookups in front of it:

- `mid = Int(map_id[i])` picks the car's map
- `r = res[mid]` and `org = Vec2(origin[mid, 0], origin[mid, 1])` replace the
  scalar `res`, `ox` and `oy` arguments
- `map_h[mid]` replaces `edt.dim[0]()` in the row flip, because the stack's
  height is the largest map's, not this one's

Everything after that is unchanged, except that `raycast` gets `mid` and reads
from the 3D stack.

This is warporacer's `lidar_kernel`, argument for argument. In the sim the
scan goes into an observation row after the steering angle and speed, so the
write is `obs[i, 2 + j]` instead of `scan[i, j]`; that's the whole difference.
The sim also keeps the per-map arrays you built here (`origin`, `res`,
`map_h`, plus `map_w` and a centerline per map) on the `Env` and rebuilds them
whenever `rotate()` swaps in a new pool of tracks.

> **Why pad rather than loop over maps?** One launch over all cars keeps the
> GPU full. Eight launches of 1,024 cars each would each be too small to hide
> memory latency, and the per-launch overhead would add up. The price is memory
> (`(8, 1772, 1769)` floats is about 100 MB for warporacer's pool), which the
> GPU has plenty of, and the `map_h` subtlety above.

</div>
</details>
