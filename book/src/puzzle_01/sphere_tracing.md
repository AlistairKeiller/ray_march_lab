# Sphere Tracing

Implement `raycast`, which marches one ray through the distance field `edt`
until it hits a wall, and `ray_march`, which fires one ray per thread from car
0 and stores each ray's length in 1D TileTensor `output`.

**Note:** _You have 1 thread per beam, and one thread to spare. Each thread
loops, and the loop runs a different number of times for every beam._

## Key concepts

In this puzzle, you'll learn about:

- A `while` loop inside a kernel whose trip count depends on the data
- Turning a floating point position into a `TileTensor` index with `Int()`
- Leaving the loop early with `break` when a ray leaves the map or lands in a
  wall
- Using `SIMD[dtype, 2]` as a 2D vector so one `+=` moves both coordinates

The key insight is what the distance field buys you. `edt[row, col]` is the
distance from that pixel to the nearest wall pixel, in any direction. Nothing
can be closer than that, so the ray can advance by exactly that much without
passing through a wall. Where the field reads 0 you are standing in a wall:
that is the hit. In the middle of the track the field is large and one step
covers a lot of ground; only near walls do the steps shrink.

- **Start**: car 0 sits at `(CAR_COL, CAR_ROW) = (19.5, 8.5)`, the centre of
  pixel (row 8, col 19), heading right along +col
- **Directions**: `dirs[j]` is \\((\cos\theta_j, \sin\theta_j)\\) in the car's
  frame, x forward and y to the left, for \\(\theta\\) from −135° to +135° in
  45° steps
- **The flip**: image rows grow downwards, so +y in the car's frame is the −row
  direction. With heading 0 the pixel-space direction is
  `(dirs[j, 0], -dirs[j, 1])`
- **Range**: the loop stops once `dist` reaches `MAX_RANGE = 16` px and the
  result is clamped there
- **Thread bounds**: 8 threads for 7 beams

## Configuration

- Map: `W = 40` x `H = 16` px, walls one pixel thick around the outside
- Beams: `NUM_BEAMS = 7` over a `FOV` of 270°
- Range: `MAX_RANGE = 16.0` px
- Threads per block: `TPB = 8`, one block
- Layouts: `edt` is `row_major[H, W]`, `dirs` is `row_major[NUM_BEAMS, 2]`,
  `output` is `row_major[NUM_BEAMS]`

Notes:

- **Index order**: the field is `edt[row, col]` but a position is
  `Vec2(col, row)`, so `pos[0]` is the column
- **Truncation**: `Int(pos[0])` rounds towards zero, which is the pixel the
  point is in for anything inside the map. Check `0 <= col < W` and
  `0 <= row < H` before indexing, using `edt.dim[1]()` and `edt.dim[0]()`:
  `raycast` is generic over the map size because the benchmark reuses it on a
  real track
- **Reading the field**: `rebind[Scalar[dtype]](edt[row, col])`, the same
  cast as the shared-memory loads in puzzle 11

## Code to complete

```mojo
{{#include ../../../problems/p01/p01.mojo:ray_march}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p01/p01.mojo" class="filename">View full file: problems/p01/p01.mojo</a>

> **Note**: The skeleton declares `var pos = p` and `var beam = thread_idx.x`
> for you, and nothing uses them until you fill in the gaps, so the compiler
> warns `assignment to 'pos' was never used; assign to '_' instead?`. That
> clears as soon as your loop reads `pos`.

<details>
<summary><strong>Tips</strong></summary>

<div class="solution-tips">

1. `while dist < max_px:` is the whole loop; everything else goes inside it
2. `var col = Int(pos[0])` and `var row = Int(pos[1])`, then `break` if either
   is outside the map
3. Read the field at `[row, col]`; if it is 0 you are in a wall, `break`
4. `pos += d * step` and `dist += step`
5. In `ray_march`, guard with `beam < NUM_BEAMS`, build `d` with the y sign
   flipped, and store `raycast(edt, Vec2(CAR_COL, CAR_ROW), d, MAX_RANGE)` in
   `output[beam]`

</div>
</details>

### Running the code

To test your solution, run the following command in your terminal:

<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">pixi NVIDIA (default)</button>
    <button class="tab-button">pixi AMD</button>
    <button class="tab-button">pixi Apple</button>
    <button class="tab-button">uv</button>
  </div>
  <div class="tab-content">

```bash
pixi run p01 --simple
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p01 --simple
```

  </div>
  <div class="tab-content">

```bash
pixi run -e apple p01 --simple
```

  </div>
  <div class="tab-content">

```bash
uv run poe p01 --simple
```

  </div>
</div>

Your output will look like this if the puzzle isn't solved yet:

```txt
out: HostBuffer([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
expected: HostBuffer([10.09902, 7.0990195, 10.09902, 16.0, 5.0990195, 8.09902, 11.513233])
```

## Solution

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../solutions/p01/p01.mojo:ray_march_solution}}
```

<div class="solution-explanation">

### Following one ray

Beam 3 is \\(\theta = 0\\), straight ahead along +col from (19.5, 8.5). The
field at the car's pixel is 5.099: the nearest wall is the corner of the left
block at (row 7, col 14), which is behind the ray and off to the side, but a
step of 5.099 is safe in every direction, so the ray takes it.

```txt
step  pixel (row, col)  edt     position after      dist
1     (8, 19)           5.0990  (24.599, 8.5)        5.099
2     (8, 24)           4.0     (28.599, 8.5)        9.099
3     (8, 28)           2.2361  (30.835, 8.5)       11.335   passing under the pillar
4     (8, 30)           2.0     (32.835, 8.5)       13.335
5     (8, 32)           2.2361  (35.071, 8.5)       15.571
6     (8, 35)           4.0     (39.071, 8.5)       19.571   >= MAX_RANGE, stop
```

The right wall is 19.5 px away, past `MAX_RANGE`, so the loop exits on the
range check and the result is clamped to 16.0. Six lookups for a 16 px ray,
with the steps shrinking as the ray passes beneath the pillar at rows 4 to 6
and growing again once it is clear.

Beam 4 (\\(\theta = 45°\\), up and to the right on the image) is the short
one: a single step of 5.099 lands at (23.106, 4.894), pixel (row 4, col 23),
which is the top block. Beam 1 (\\(\theta = -90°\\), straight down) takes
5.099 then 2.0 and lands in row 15, the bottom wall, for 7.099.

### Where the 0.099 comes from

The bottom wall begins 6.5 px below the car, yet beam 1 reports 7.099. The
field is measured between pixel centres, so a step of exactly the field value
taken from a pixel centre lands on the centre of the nearest wall pixel, half
a pixel past its edge, and from anywhere else in the pixel it can land up to a
pixel past. The ray always stops inside the wall pixel, never on its boundary.
warporacer lives with this: a pixel is 5 cm on a real track, the car itself is
42 x 58 cm, and the training randomises far bigger things than that every
episode.

### Why not step 1 px at a time

A 1 px stepper reads the map once per pixel of range. Sphere tracing reads it
once per step:

```txt
beam  theta   sphere steps  1 px steps
0     -135    4             10
1      -90    2              7
2      -45    4             10
3        0    6             16
4       45    1              5
5       90    2              8
6      135    5             11
```

About 3x fewer loop iterations on this toy map, where nothing is more than a
few pixels from a wall. On a real track with a 345 px range the gap is much
wider; the [benchmark](./benchmark.md) measures it.

### Thread mapping

- One thread per beam, 8 threads for 7 beams. Thread 7 has nothing to do, and
  the guard keeps it from writing `output[7]`, which is in the canary's margin.
- The loop count is different for every thread. Threads in a warp issue the
  loop body in lockstep until the last one breaks, so a fan of beams takes as
  long as its longest ray. Part 2 lays the grid out so that a warp holds one
  car's neighbouring beams, whose lengths tend to be similar.
- `raycast` is an ordinary function called from the kernel; Mojo compiles it
  for the GPU along with its caller. It is generic over `EdtLayoutT` so the
  same code marches the 40 x 16 map here and the 1130 x 1769 track in part 3.

</div>
</details>
