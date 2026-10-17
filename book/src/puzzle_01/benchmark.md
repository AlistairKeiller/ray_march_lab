# Benchmark: 8k Cars on Spielberg

## Overview

The two parts above run 7 and 21 rays. warporacer runs 8192 cars x 108 beams,
close to 900k rays, every step, and its headline number is about 7 million
env-steps per second on one GPU with the lidar being most of the work. This
page runs _your_ `raycast` and `beam_dir` at that scale on a real track and
puts two other ways of doing it next to them.

> **Key insight:** _The distance field turns a loop over every pixel of range
> into a loop over a handful of lookups. How large the win is depends on the
> track, so measure it on a track._

## What runs

`--benchmark` mode in `p01.mojo` does the following:

- Loads `maps/Spielberg_map.edt`, the wall-distance field of the Spielberg
  F1TENTH track: 1130 x 1769 px at 0.058 m/px, built from the map image by
  `scripts/track_edt.py` the same way `warporacer/track.py` does it (threshold
  the image, `scipy.ndimage.distance_transform_edt`). The `p01` task builds it
  for you the first time; it is a few MB and not checked in.
- Parks `BENCH_CARS = 8192` cars at random drivable pixels at least half a car
  diagonal from any wall, with random headings, like a fleet respawn.
- Fans `BENCH_BEAMS = 108` beams over 270° with a 20 m range, which is 345 px.
- Checks that sphere tracing and 1 px stepping stop at the same walls (within
  the pixel each one overshoots by) before timing anything.
- Times three kernels with the puzzles' `Bench` harness, 50 sweeps each after
  5 warm-up sweeps. The clock is wall time around a batch of launches with a
  `synchronize()` on each side, since `max.benchmark`'s device timer only
  times the enqueue on Metal:

| name                         | what it is                                                                  |
|------------------------------|-----------------------------------------------------------------------------|
| `sphere_trace`               | `sweep[SPHERE_TRACE]`: your `raycast`, one block per car, one thread per beam |
| `fixed_step_1px`             | same launch, `raycast_fixed_step`: advance 1 px and test for wall             |
| `sphere_trace_cars_adjacent` | your `raycast` with the grid transposed: a block holds 128 cars on one beam   |

The sweep kernel is `lidar` from part 2 with the fleet's layouts and a
`variant` parameter; `comptime if` picks the marcher and the index mapping at
compile time so all three run the same code otherwise.

```mojo
{{#include ../../../solutions/p01/p01.mojo:lidar_benchmark}}
```

## Running the benchmark

<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">pixi NVIDIA (default)</button>
    <button class="tab-button">pixi AMD</button>
    <button class="tab-button">pixi Apple</button>
    <button class="tab-button">uv</button>
  </div>
  <div class="tab-content">

```bash
pixi run p01 --benchmark
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p01 --benchmark
```

  </div>
  <div class="tab-content">

```bash
pixi run -e apple p01 --benchmark
```

  </div>
  <div class="tab-content">

```bash
uv run poe edt01 && uv run poe p01 --benchmark
```

  </div>
</div>

The run below is an Apple M5 Pro with MAX 26.6.0 / Mojo 1.1.0; read the
ratios, not the absolute times:

```txt
Spielberg: 1130 x 1769 px at 0.05796 m/px, lidar range 20.0 m = 345.06555 px
fleet: 8192 cars x 108 beams = 884736 rays per sweep
mean range 3.30695 m, 17168 rays at max range
sphere trace and 1 px stepping agree within 2 px on 98.1 % of rays; worst is car 2134 beam 12 at ( 1640.5184 , 844.9422 ) heading -1.028159 : 42.47331 vs 345.06555 px
--------------------------------------------------------------------------------
Running sphere_trace
Running fixed_step_1px
Running sphere_trace_cars_adjacent
| name                       | met (ms) | iters |
| -------------------------- | -------- | ----- |
| sphere_trace               | 0.43546  | 50    |
| fixed_step_1px             | 2.53596  | 50    |
| sphere_trace_cars_adjacent | 0.56     | 50    |
sphere_trace: 0.435 ms per sweep -> 2.03 G rays/s, 18.8 M lidar-only env-steps/s
```

## Reading the numbers

- **Sphere tracing vs 1 px steps: 5.8x.** A ray on Spielberg travels 3.3 m on
  average, 57 px, so the 1 px stepper does 57 lookups per ray and 346 for the
  2% that reach max range. Sphere tracing does 7 on average (median 4, 90th
  percentile 17, worst 163 for a ray skimming a wall), 7.8x fewer. The time
  ratio is smaller than the lookup ratio for two reasons: a sphere-trace
  lookup lands somewhere new every step, so the field reads miss the cache
  more often than a stepper creeping through consecutive pixels, and a warp
  runs its loop until the longest of its 32 rays finishes.
- **Beams-adjacent vs cars-adjacent: 29%.** Both layouts do exactly the same
  work; what differs is which 32 rays share a warp. For 32 neighbouring beams
  of one car the longest ray is about 3x the mean; for 32 cars on the same
  beam it is about 5x, so the warp spends more of its time with most lanes
  masked off waiting for one ray. One car's beams also read the same patch of
  the field, which the cache likes. warporacer launches on `(car, beam)` with
  beam fastest for this reason, and its CUDA path goes further and binds the
  field as a texture so the scattered reads hit a cache built for them.
- **Against the real sim.** 0.43 ms per sweep is 2 G rays/s, or about 19 M
  fleet lidar reads per second on a laptop GPU. warporacer's ~7 M env-steps/s
  on one GPU includes the physics kernel (RK4 with six substeps per car), the
  reward, respawns and the PPO rollout on top of this, so the lidar being the
  single largest kernel is consistent with that budget. The launch size is
  what makes it work at all: 885k threads per launch amortise everything.
  With 32 cars the same kernel is launch overhead.
- **98.1% agreement.** The two marchers disagree by more than 2 px on 1.9% of
  rays, and the worst by 300 px. Those are grazing rays. The worst one, car
  2134 beam 12, runs almost parallel to a wall whose rasterised edge is a
  staircase of pixels: the sphere tracer, taking sub-pixel steps that close to
  a wall, lands inside the corner pixel at (row 856, col 1599) and stops at
  42 px, while the 1 px stepper samples just before and just after that pixel
  and carries on to max range. Neither answer is wrong at the map's
  resolution. warporacer's lidar has the same property and policies train
  fine on it, but it is why the check above counts rays within 2 px instead
  of comparing to `1e-5` the way the first two parts do.

## What the real sim does on top

warporacer's lidar is this kernel plus a few things that each bought a
measurable amount of throughput, in case you want to keep going:

- **Texture memory for the field.** The sphere-march reads are scattered: each
  step lands somewhere new and neighbouring threads diverge as their rays
  spread out. On CUDA the field is bound as a 3D texture (`lidar_tex_kernel`),
  whose cache is built for exactly this access pattern, for +14% sim throughput.
  Border addressing also returns 0 outside the map, so the bounds check goes
  away.
- **`fast_math`.** The `cos` / `sin` per thread and the physics kernel's `tan`
  use the SFU approximations, +22%. With ±15% domain randomisation on friction
  and wheelbase every episode, the lost precision is noise.
- **One launch, many tracks.** Fields for up to eight tracks are stacked into a
  padded `(n_maps, H, W)` buffer and each car carries a map id, so a fleet
  spread over different circuits is still one kernel launch.
- **No host round trips.** Observations land in a Warp buffer that PyTorch
  reads zero-copy on the same CUDA stream, so the policy runs on the scan
  without a synchronisation. At ~7M env-steps/s a single sync per step would
  cost more than the lidar does.
