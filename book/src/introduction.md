# Ray Marching Lab

> _"Give it a ROS-style track image and it PPO-trains a lidar-driven racing
> policy at ~7M env-steps/s at 8k envs."_ — the warporacer README

Welcome to a one-puzzle lab in the style of
[Mojo🔥 GPU Puzzles](https://puzzles.modular.com/). The puzzle is the lidar
from [warporacer](https://github.com/uci-f1tenth/warporacer), a GPU racing
simulator for F1TENTH cars: every step, every car fires 108 rays into the
track and learns from how far each one got. The sim is written in NVIDIA Warp;
here you write the same kernel in Mojo and then run it at the sim's scale.

## What you need

- A GPU that [MAX supports](https://max.modular.com/packages/#gpu-compatibility)
  and [pixi](https://pixi.sh/latest/#installation). Setup is the same as the
  puzzles: `pixi run p01 --simple` compiles and runs the first part (add
  `-e apple` or `-e amd` to every `pixi run` on those GPUs).
- Puzzles 1 to 8 of Mojo GPU Puzzles, or the equivalent: thread and block
  indices, guards, and reading and writing a 2D `TileTensor`.

## How it is organised

The puzzle lives in `problems/p01/p01.mojo` and has the usual shape: a
`# FILL ME IN` region per part, a `main()` that builds the inputs, runs your
kernel and compares against a CPU reference, and the guarded output buffer
from the puzzles that fails the run if you write outside `output`.

- [🔰 Sphere Tracing](./puzzle_01/sphere_tracing.md): the marching loop, one
  car, seven beams.
- [⭐ Lidar Sweep](./puzzle_01/lidar.md): several cars with headings, one
  thread per (car, beam).
- [📊 Benchmark](./puzzle_01/benchmark.md): 8k cars x 108 beams on a real
  track, timed against the naive marcher.

The solution is behind a "Reveal Solution" button on each page, with an
explanation that walks through the numbers.

## Acknowledgements

The format, harness and theme are Modular's, from
[mojo-gpu-puzzles](https://github.com/modular/mojo-gpu-puzzles), under the
Apache 2.0 license with LLVM exceptions. The Spielberg map comes from the
F1TENTH racetrack set that ships with warporacer.
