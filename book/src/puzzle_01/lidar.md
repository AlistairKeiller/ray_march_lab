# Lidar Sweep

Implement `beam_dir`, which turns a beam from the car's frame into a
pixel-space direction for a car with heading `psi`, and `lidar`, which runs
one thread per (car, beam) and stores every car's scan in 2D TileTensor `obs`.

**Note:** _You have 1 block per car and 1 thread per beam, with one spare
thread per block. `raycast` is the one you wrote in part 1._

## Key concepts

In this puzzle, you'll learn about:

- A 2D grid where `block_idx.y` picks the car and `thread_idx.x` the beam, as
  in puzzle 7
- Rotating a direction vector, and why the rotation turns the other way on an
  image
- Reading a per-row record `(col, row, heading)` out of a `TileTensor`
- Putting one car's beams in one block, which is how warporacer launches it:
  `wp.launch(lidar_kernel, dim=(num_cars, NUM_LIDAR))`

The key insight is the order of the two frame changes. A beam at angle
\\(\theta\\) in the car's frame is \\((\cos\theta, \sin\theta)\\). Rotating it
by the heading \\(\psi\\) gives the world direction, and then the image's
downward row axis flips the y component:

\\[\Large d_{col} = \cos\psi\, b_x - \sin\psi\, b_y, \qquad d_{row} = -(\sin\psi\, b_x + \cos\psi\, b_y)\\]

`beam_dir` starts from the heading-0 answer of part 1, `d = (b_x, -b_y)`, and
rotates that in place. Written in terms of the flipped `d`, the same formula is
`(ch * d[0] + sh * d[1], ch * d[1] - sh * d[0])`: a clockwise rotation on the
image, because flipping y flips the sense of rotation too.

- **Cars**: `cars[car, 0]` and `cars[car, 1]` are the start pixel, `cars[car, 2]`
  the heading in radians, counter-clockwise with 0 pointing along +col
- **Car 0** is the car from part 1, heading 0. `--simple` is the special case
  of this part where no rotation happens
- **Thread bounds**: the guard is still `beam < NUM_BEAMS`; there are exactly
  `NUM_CARS` blocks

## Configuration

- Cars: `NUM_CARS = 3`, layout `row_major[NUM_CARS, 3]`
  - car 0: (19.5, 8.5), heading 0, facing right
  - car 1: (6.5, 12.5), heading \\(\pi/2\\), facing up the image
  - car 2: (34.5, 3.5), heading \\(-3\pi/4\\), facing down and left
- Grid: `BLOCKS_PER_GRID_2 = (1, NUM_CARS)` blocks of
  `THREADS_PER_BLOCK_2 = (TPB, 1)` threads
- Output: `obs` is `row_major[NUM_CARS, NUM_BEAMS]`, one row per car

## Code to complete

```mojo
{{#include ../../../problems/p01/p01.mojo:lidar}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p01/p01.mojo" class="filename">View full file: problems/p01/p01.mojo</a>

<details>
<summary><strong>Tips</strong></summary>

<div class="solution-tips">

1. In `beam_dir` the rotation is one assignment to `d`. Check it with
   \\(\psi = \pi/2\\): the forward beam `(1, 0)` must come out as `(0, -1)`,
   straight up the image
2. `car = block_idx.y`, `beam = thread_idx.x`, guard `beam < NUM_BEAMS`
3. `p = Vec2(cars[car, 0], cars[car, 1])` with the same `rebind` as before
4. `d = beam_dir(Vec2(dirs[beam, 0], dirs[beam, 1]), cars[car, 2])`
5. `obs[car, beam] = raycast(edt, p, d, MAX_RANGE)`

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
pixi run p01 --lidar
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p01 --lidar
```

  </div>
  <div class="tab-content">

```bash
pixi run -e apple p01 --lidar
```

  </div>
  <div class="tab-content">

```bash
uv run poe p01 --lidar
```

  </div>
</div>

Your output will look like this if the puzzle isn't solved yet:

```txt
out: HostBuffer([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
expected: HostBuffer([10.09902, 7.0990195, 10.09902, 16.0, 5.0990195, 8.09902, 11.513233, 4.0, 16.0, 7.2426405, 12.385164, 8.0, 6.0, 4.0, 3.0, 4.0, 8.414213, 4.0, 11.57649, 7.0, 5.0])
```

## Solution

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../solutions/p01/p01.mojo:lidar_solution}}
```

<div class="solution-explanation">

### Reading the scans

`obs` is one row per car, one column per beam from −135° (right-back) to
+135° (left-back):

```txt
car  heading   -135    -90     -45      0      45     90    135
0      0       10.10   7.10   10.10  16.00   5.10   8.10  11.51
1     90        4.00  16.00    7.24  12.39   8.00   6.00   4.00
2   -135        3.00   4.00    8.41   4.00  11.58   7.00   5.00
```

Row 0 is exactly the `--simple` output, since heading 0 makes `beam_dir` the
plain flip. Car 1 faces up the image, so its forward beam (column 0°) runs up
column 6 and stops in the top wall 12.39 px away, and its −90° beam, which
points right along row 12, runs the length of the open stretch and saturates
at 16. Car 2 faces down-left into the corner between the top block and the
pillar, so everything in front of it is short.

### The rotation

`beam_dir` does the rotation on the already-flipped vector. Expanding
`Vec2(ch * d[0] + sh * d[1], ch * d[1] - sh * d[0])` with
`d = (b_x, -b_y)` gives \\((\cos\psi\, b_x - \sin\psi\, b_y,\;
-(\sin\psi\, b_x + \cos\psi\, b_y))\\), which is warporacer's

```python
d = wp.vec2(ch * lidar_dirs[j][0] - sh * lidar_dirs[j][1],
            -(sh * lidar_dirs[j][0] + ch * lidar_dirs[j][1]))
```

Either form is fine; what matters is that the sign on the row component flips
before anything is compared with `0 <= row < H`.

### Thread mapping

- `grid_dim=(1, NUM_CARS)` with `block_dim=(TPB, 1)`: block `y` is a car and
  thread `x` is a beam. Every thread in a block reads the same `cars[car, :]`
  record, and adjacent threads read adjacent `dirs` rows and write adjacent
  `obs` entries, so the stores coalesce.
- Neighbouring beams usually see similar distances, so threads in a warp exit
  the march at about the same time. The other layout, a warp of different cars
  looking along the same beam, puts a 2 step ray next to a 40 step one and the
  whole warp waits for the long one. The benchmark times both.

### What the real kernel adds

`lidar_kernel` in `warporacer/sim.py` is this kernel with units attached. Each
map has a world origin and a resolution in metres per pixel, so the start
point is `((x - ox) / res, (H - 1) - (y - oy) / res)` rather than a pixel
coordinate; the laser is mounted 0.2733 m ahead of the car's reference point,
so `LIDAR_MOUNT_X * (ch, sh)` is added first; and the result is multiplied by
`res` to give metres, which is what the policy network sees. It also looks up
a per-car map id so one launch can serve cars on eight different tracks.

</div>
</details>
