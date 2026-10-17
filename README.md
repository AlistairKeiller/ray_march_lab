<p align="center">
  <h1 align="center">Ray Marching Lab</h1>
</p>

<p align="center">
  <h3 align="center">A Mojo🔥 GPU puzzle built from warporacer's lidar kernel</h3>
</p>

## Overview

[warporacer](https://github.com/uci-f1tenth/warporacer) trains racing policies
by stepping thousands of F1TENTH cars at once on the GPU. The most expensive
thing each car does per step is read its lidar: 108 rays, each marched through
the track's occupancy grid until it hits a wall. This lab pulls that one kernel
out and turns it into a puzzle in the style of
[Mojo GPU Puzzles](https://puzzles.modular.com/), so you can write it yourself
and then see how fast it runs at the scale the real sim uses.

The puzzle has three parts:

1. **Sphere tracing**: march seven rays from one car through a 40 x 16 map
   using a precomputed wall-distance field.
2. **Lidar sweep**: one thread per (car, beam), with each car's heading rotating
   its fan of rays.
3. **Benchmark**: 8192 cars x 108 beams on the Spielberg track at 20 m range,
   against a one-pixel-at-a-time baseline.

It assumes you have been through at least puzzles 1 to 8 of Mojo GPU Puzzles
(thread indexing, guards, 2D blocks, `TileTensor`).

## Getting Started

You'll need a [compatible GPU](https://max.modular.com/packages/#gpu-compatibility)
and [pixi](https://pixi.sh/latest/#installation). On macOS you also need the
Metal toolchain: if `xcrun -sdk macosx metal` complains that it is missing, run
`xcodebuild -downloadComponent MetalToolchain`.

```bash
git clone https://github.com/AlistairKeiller/ray_march_lab
cd ray_march_lab

pixi run p01 --simple        # NVIDIA (default); add `-e amd` or `-e apple` for other GPUs
pixi run p01 --lidar
pixi run p01 --benchmark     # builds maps/Spielberg_map.edt first (pixi run edt01)

pixi run book                # the puzzle text, with hints and the solution
```

Your work goes in `problems/p01/p01.mojo`; the reference is
`solutions/p01/p01.mojo`. The benchmark runs whatever is in the file, so once
your kernels pass the two checks you can time your own code.

## Development

```bash
pixi run tests               # run the solution in every mode
pixi run check-skeletons     # problem == solution outside the fill-in regions
pixi run compile-problems    # the unfilled skeleton still compiles
pixi run check-book-refs     # book links and images resolve
pixi run format
```

## Layout

| path | what it is |
|---|---|
| `problems/p01/p01.mojo` | the puzzle, with `# FILL ME IN` gaps |
| `solutions/p01/p01.mojo` | reference solution and the `--benchmark` mode |
| `book/src/puzzle_01/` | the chapter: overview, both parts, benchmark write-up |
| `maps/Spielberg_map.{png,yaml}` | the track the benchmark marches on, from the F1TENTH map set |
| `scripts/track_edt.py` | map image -> wall-distance field, the way `warporacer/track.py` does it |
| `harness/` | the guarded output buffers from Mojo GPU Puzzles |

## Acknowledgments

- The puzzle format, test harness, book theme and most of `scripts/` come from
  Modular's [Mojo GPU Puzzles](https://github.com/modular/mojo-gpu-puzzles).
- The kernel is `lidar_kernel` / `raycast` from
  [warporacer](https://github.com/uci-f1tenth/warporacer), written for NVIDIA
  Warp.

## License

Apache License v2.0 with LLVM Exceptions, same as Mojo GPU Puzzles. See
[LICENSE](LICENSE).
