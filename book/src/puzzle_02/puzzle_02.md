# Puzzle 2: World to Pixel

## Overview

Implement a kernel that, for each car, looks up the wall distance under the car
and converts it to the car's **clearance** in meters: the distance from the
car's centre to the nearest wall, minus half the car's diagonal. A negative
clearance is a crash.

**Note:** _You have 1 thread per car. Cars can be anywhere, including off the
map, so the pixel indices need clamping._

## Key concepts

In this puzzle, you'll learn about:

- 2D tensors: `row_major[SIZE, SIZE]()`, indexed `edt[row, col]`
- The map origin as two scalar arguments, `ox` and `oy`
- Float to int conversion with `Int()`, which truncates toward zero
- Clamping indices with `Int.clamp()` instead of branching
- Reading a tensor's shape inside a kernel: `edt.dim[0]()`

For each car \\(i\\) at world position \\((x_i, y_i)\\), on a map of \\(H\\)
rows by \\(W\\) columns:

\\[\Large col = \operatorname{clamp}\left(\left\lfloor \frac{x_i - o_x}{res} \right\rfloor, 0, W - 1\right)\\]

\\[\Large row = \operatorname{clamp}\left(\left\lfloor (H - 1) - \frac{y_i - o_y}{res} \right\rfloor, 0, H - 1\right)\\]

\\[\Large output[i] = edt[row, col] \times res - \frac{D}{2} \qquad D = \sqrt{0.42^2 + 0.58^2} \text{ m}\\]

| in the formula | in the code | meaning |
|---|---|---|
| \\(x_i, y_i\\) | `cars[i, 0]`, `cars[i, 1]` | the car's world position, in meters |
| \\(o_x, o_y\\) | `ox`, `oy` | the world position of the map's bottom-left pixel, `ORIGIN` |
| \\(res\\) | `res` | the map's resolution, in meters per pixel |
| \\(H, W\\) | `edt.dim[0]()`, `edt.dim[1]()` | the map's height and width, in pixels |
| \\(D / 2\\) | `HALF_DIAG` | half the car's diagonal, 0.358 m: the furthest any corner of the 0.42 m by 0.58 m F1TENTH reaches from its centre |

The floors are what `Int()` does for every point inside the map: it truncates
toward zero, which only differs from the floor for negative coordinates, a car
left of or below the map, and the clamp sends those to 0 either way.

### The row flip

A ROS map's `origin` is the world position of the **bottom-left** pixel, and
world \\(y\\) increases upward. Image rows increase downward from the top. So
\\(x\\) maps straight onto columns, but \\(y\\) maps onto rows from the bottom:
row \\(H - 1\\) is \\(y = o_y\\). Get this flip wrong and every car drives a
mirrored track, and nothing complains until the lidar looks wrong.

### Why clamp

A car that drives off the edge of the map has a pixel index outside the raster.
Reading `edt[-3, 12]` is an out-of-bounds memory access, and nothing on the GPU
checks for it. Clamping to the border pixel reads a wall (`edt` is 0 there),
which is also the right answer: off the map counts as crashed.

## Configuration

- Cars: `NUM_CARS = 4`, as rows of `[x, y, psi]`, layout `row_major[NUM_CARS, 3]()`
- Map: an `8 x 8` px room with a 1 px wall all round, `RES = 0.5` m/px,
  `ORIGIN = (-2, -2)`
- Car size: `CAR_W = 0.42` m by `CAR_L = 0.58` m, the F1TENTH's width and length;
  `HALF_DIAG = hypot(CAR_W, CAR_L) / 2` is half its diagonal
- One thread per car: one block of `NUM_CARS` threads

Notes:

- **Shape in kernel**: a tensor knows its own shape, `edt.dim[0]()` rows by
  `edt.dim[1]()` columns; wrap one in `Int(...)` to do arithmetic with it. The
  room is `SIZE` pixels square and `SIZE` is a compile-time constant, so `SIZE`
  would work here too, but Puzzle 4 reads the size from a file, so ask the
  tensor
- **Origin**: `ORIGIN` is a `Vec2` on the host and arrives in the kernel as the
  two scalars `ox` and `oy`
- **Module constants**: `HALF_DIAG` is a `comptime` constant, and is baked into
  the kernel. It is a literal because `sqrt` and `hypot` are not folded at
  compile time and do not lower inside a Metal kernel; use it rather than
  recomputing the diagonal
- **Truncation**: `Int(x)` drops the fraction toward zero, which is the floor
  for every point inside the map, and is the conversion warporacer's own
  `wp.int32()` does. If you would rather write what the formula says,
  `Int(floor(x))` with `from std.math import floor` works in the kernel too
- **Integer indices**: `edt[row, col]` needs `Int` indices. `//` on a `Scalar`
  returns a `Scalar`, so if the compiler fails deep inside `tile_tensor.mojo`
  with `CoordLike requires integral types`, one of the two is still a float:
  wrap it in `Int(...)`

## Code to complete

```mojo
{{#include ../../../problems/p02/p02.mojo:clearance}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p02/p02.mojo" class="filename">View full file: problems/p02/p02.mojo</a>

<details>
<summary><strong>Tips</strong></summary>

<div class="solution-tips">

1. Column: `(cars[i, 0] - ox) / res`, truncated with `Int(...)`
2. Row: `Scalar[dtype](Int(edt.dim[0]()) - 1) - (cars[i, 1] - oy) / res`,
   truncated the same way
3. Clamp both into the raster with `Int`'s method:
   `.clamp(0, Int(edt.dim[1]()) - 1)` for the column and `edt.dim[0]()` for
   the row
4. `output[i] = edt[row, col] * res - HALF_DIAG`

</div>
</details>

## Running the code

To test your solution, run the following command in your terminal:

```bash
uv run mojo -I . problems/p02/p02.mojo
```

Your output will look like this if the puzzle isn't solved yet:

```txt
room (px to wall):
  0 0 0 0 0 0 0 0
  0 1 1 1 1 1 1 0
  0 1 2 2 2 2 1 0
  0 1 2 3 3 2 1 0
  0 1 2 3 3 2 1 0
  0 1 2 2 2 2 1 0
  0 1 1 1 1 1 1 0
  0 0 0 0 0 0 0 0
out: HostBuffer([0.0, 0.0, 0.0, 0.0])
expected: HostBuffer([1.1419497, 0.14194971, 0.6419497, -0.3580503])
```

## Solution

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../solutions/p02/p02.mojo:clearance_solution}}
```

<div class="solution-explanation">

This solution:

- Converts world \\(x\\) to a column and world \\(y\\) to a row, flipping the
  row axis with `Scalar[dtype](Int(edt.dim[0]()) - 1) - ...`
- Truncates with `Int()` and clamps both indices into the raster with
  `.clamp(0, Int(edt.dim[k]()) - 1)`
- Reads the wall distance, scales it to meters, and subtracts `HALF_DIAG`

Following the four cars through:

```txt
Car 0 at (0.0, 0.0):    col = (0 + 2) / 0.5 = 4,   row = 7 - 4   = 3 -> edt[3, 4] = 3 px -> 1.5 - 0.358 =  1.142 m
Car 1 at (1.2, -1.4):   col = 3.2 / 0.5 = 6.4 -> 6, row = 7 - 1.2 = 5.8 -> 5 -> edt[5, 6] = 1 px -> 0.142 m
Car 2 at (-0.9, 0.3):   col = 1.1 / 0.5 = 2.2 -> 2, row = 7 - 4.6 = 2.4 -> 2 -> edt[2, 2] = 2 px -> 0.642 m
Car 3 at (5.0, 9.0):    col = 14 -> clamped to 7,   row = -15 -> clamped to 0 -> edt[0, 7] = 0 px -> -0.358 m (crash)
```

`Int()` truncates toward zero, so a car a fraction of a pixel left of the map
(say `(x - o_x) / res = -0.3`) lands in column 0, the wall, before the clamp
even runs. Either way it reads 0 and reports a crash, which is what the sim
wants.

This is the lookup in warporacer's `step_kernel`, where the result feeds the
crash test and a wall-proximity penalty. The only difference is that the sim
reads the map's height and width from per-map arrays, because it runs several
maps at once. That is Puzzle 5.

</div>
</details>
