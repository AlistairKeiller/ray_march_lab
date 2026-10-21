# Thread Layout: One Thread per Car vs One per Beam

In this puzzle you'll implement the second of four launches that produce the
same scan, then benchmark all four.

## One thread per (car, beam)

This is the Puzzle 5 kernel, `lidar_kernel`, and it's given. The launch is
`grid_dim=(1, num_envs)` blocks of `block_dim=(128, 1)` threads: one block per
car, one thread per beam. Each thread computes its car's start point and
heading, rotates one beam and marches one ray. Consecutive threads are
consecutive beams of one car.

**Thread organisation:**

- **Launch**: 8,192 blocks of 128 threads, 884,736 threads with 108 of every
  128 doing work; warporacer's `block_dim=128`
- **Per thread**: a `cos`/`sin`, a start point, one rotation, one march
- **Memory pattern**: a block holds one car's 108 beams, fanning out from one
  point

## One thread per car

The kernel you write. The launch is 64 blocks of 128 threads, one thread per
car; each thread computes its start point once and loops over all 108 beams.
108× fewer threads, each doing 108× the marching.

**Thread organisation:**

- **Launch**: `grid_dim=64`, `block_dim=128`, 8,192 threads at full scale,
  with `car = block_dim.x * block_idx.x + thread_idx.x`
- **Per thread**: one `cos`/`sin` and start point, then
  `for j in range(NUM_LIDAR)` with a rotation and a march per iteration
- **Memory pattern**: 32 adjacent threads are 32 different cars, on up to 8
  different maps, reading unrelated parts of a 100 MB raster

## The two launches that are given

- **`lidar_fixed_step_kernel`** is `lidar_kernel` with `raycast_fixed_step`
  in place of `raycast`: advance one pixel, test for wall, repeat. It is the
  march you would write without a distance field, and it is what the field
  is measured against.
- **`lidar_transposed_kernel`** is `lidar_kernel` with the grid turned round:
  `grid_dim=(64, NUM_LIDAR)` blocks of `(128, 1)` threads, so a block holds
  128 consecutive cars looking along one beam (`block_idx.y`) instead of one
  car's consecutive beams. Same work, different neighbours.

The start point and the beam rotation from Puzzle 5 are factored into two
device functions, `laser_px` and `beam_px`, so that the four kernels in the
file share them. The fleet's sizes are only known at run time, so the kernel
argument types (`Scan`, `Cars`, `Edt`, ...) are `TileTensor`s over
`RowMajorLayout[Int, ...]` layouts, which carry their shape with them.

### Code to complete

```mojo
{{#include ../../../problems/p06/p06.mojo:lidar_car_kernel}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p06/p06.mojo" class="filename">View full file: problems/p06/p06.mojo</a>

<details>
<summary><strong>Tips</strong></summary>

<div class="solution-tips">

- The start point depends only on the car, so look up `mid`, `r`, `ch`, `sh`
  and `p = laser_px(...)` once, before the loop, exactly as `lidar_kernel`
  does
- Loop `for j in range(NUM_LIDAR):`, build the direction with
  `beam_px(Vec2(lidar_dirs[j, 0], lidar_dirs[j, 1]), ch, sh)` and write
  `scan[i, j]`
- The result has to match `lidar_kernel` to the bit; the benchmark checks that
  before it times anything

</div>
</details>

## The benchmark harness

The rest of the file is the lidar half of warporacer's `Env`, and it's worth
reading:

- **`harness.track.load_tracks`** picks up to `MAX_ACTIVE_MAPS = 8` yamls at
  random from the directory you pass (or just the one yaml you pass), like
  warporacer's `load_tracks()`, and **`harness.track.spawn`** gives each car a
  random map and a random drivable pose on it. Both are the Python loader from
  Puzzle 4, called through interop, with the results copied to the device
- **`Fleet.__init__`** stacks the maps with `harness.track.stack`, builds the
  per-map `origin`/`res`/`map_h` buffers and the beam directions, and
  allocates the `(num_envs, 108)` scan
- **`launch[variant]`** wraps the fleet's buffers in run-time-shaped
  `TileTensor`s and enqueues one of the four kernels with its grid; the
  `comptime if` on `variant` picks the kernel at compile time
- **`scan_of`** runs one launch and copies the scan to the host for the
  correctness check
- **`bench`** warms up (the first launch compiles the kernel), then times
  `steps` launches between two `ctx.synchronize()` calls. Without the sync a
  launch returns as soon as it is queued and you'd time the queueing, not the
  kernel

## Run the code

```bash
uv run mojo -I . problems/p06/p06.mojo                               # 8 random maps, 8192 cars, 500 steps
uv run mojo -I . problems/p06/p06.mojo maps/ --num-envs 8192 --steps 500
uv run mojo -I . problems/p06/p06.mojo maps/Spielberg_map.yaml --num-envs 1024
```

Until the kernel is filled in, the correctness check fails and nothing is timed:

```text
Testing Correctness...
   per car                      max difference vs per (car, beam): 20.0 m
   per (beam, car)              max difference vs per (car, beam): 0.0 m
   per (car, beam), 1 px steps  within 2 px of per (car, beam) on 99.1% of rays
   ❌ Layouts disagree
```

When successful, you should see output similar to:

```text
Puzzle 06: Lidar Benchmark
======================================================================

Configuration: 8192 envs x 108 beams, 20.0 m range, device Apple M5 Pro
maps: Sepang_map, BrandsHatch_map, skirk, MoscowRaceway_map, Melbourne_map, IMS_map, Austin_map, Oschersleben_map
edt stack: (8, 1772, 1759) float32, 100 MB
------------------------------------------------------------
Testing Correctness...
   per car                      max difference vs per (car, beam): 0.0 m
   per (beam, car)              max difference vs per (car, beam): 0.0 m
   per (car, beam), 1 px steps  within 2 px of per (car, beam) on 99.1% of rays
   ✅ All layouts agree

Benchmarking (500 steps)...

Performance Results:
   per (car, beam)                 0.508 ms/step    1740.7 M rays/s   16.12 M env-steps/s
   per car                         1.232 ms/step     717.9 M rays/s    6.65 M env-steps/s
   per (beam, car)                 0.664 ms/step    1332.4 M rays/s   12.34 M env-steps/s
   per (car, beam), 1 px steps     3.044 ms/step     290.6 M rays/s    2.69 M env-steps/s
   sphere tracing is 5.99x faster than 1 px steps
   per (car, beam) is 2.42x faster than per car
   per (car, beam) is 1.31x faster than per (beam, car)

Key Learning Points:
• The distance field turns a loop over every pixel of range into a loop over a handful of lookups
• Same rays, same results, two thread layouts: 108x more threads or 108x more work per thread
• A GPU needs far more than 8k threads in flight to hide memory latency; a CPU core does not
• Which 32 rays share a warp decides how long it idles for its slowest lane; only a GPU shows this
• The timed loop launches the kernel `steps` times and syncs once, so it measures the kernel, not the host
• warporacer's whole step (physics + lidar + PPO bookkeeping) runs ~7M env-steps/s at 8k envs on one GPU
Puzzle 06 complete ✅
```

> These numbers are from an Apple M5 Pro with MAX 26.6 / Mojo 1.1. The
> [next page](./performance.md) explains them, next to the CPU numbers from the
> Warp twin on the same laptop, where every ratio comes out the other way.

## Solution

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../solutions/p06/p06.mojo:lidar_car_kernel_solution}}
```

<div class="solution-explanation">

The per-car kernel is `lidar_kernel`'s body with the beam loop moved inside the
thread. Everything that depends only on the car (`mid`, `r`, the heading's
`cos` and `sin`, the start point `p`) is computed once; the loop over `j`
rotates each beam and marches it. The guard `i < scan.dim[0]()` is new: 8,192
cars fill 64 blocks of 128 exactly, but `--num-envs` need not be a multiple of
128, and the spare threads of the last block must not write.

Because the two kernels run the same arithmetic on the same values in the same
order, the scans agree to the bit, and the benchmark requires that before
timing anything. A layout change that altered the answer would not be a layout
change.

</div>
</details>
