# Puzzle 6: Lidar Benchmark

> ## Four launches of one kernel
>
> Part III turns the kernel you built into a measurement. Puzzles 4 and 5
> launched one thread per (car, beam), a car's beams adjacent, which is the
> layout warporacer ships. Here you write the obvious alternative, one thread
> per car that loops over its 108 beams, and the benchmark times both at
> warporacer's scale next to two more launches that are given: the same grid
> with a one-pixel stepper in place of the sphere march, and the same kernel
> with the grid turned round so that adjacent threads are adjacent cars. All
> four produce the same scan. Which is fastest, and by how much, is different
> on a CPU and a GPU, and understanding why is the point.

## Overview

You'll implement `lidar_car_kernel`, a per-car version of the Puzzle 5 kernel,
and run a benchmark that reproduces the lidar half of warporacer's `Env`:

- loads up to 8 random maps from `maps/` and pads them into one `(8, H, W)`
  raster, as `Env.rotate()` does
- spawns 8,192 cars (4,096 on Warp's CPU backend) at random drivable poses with
  random headings, spread across the maps
- launches each kernel `steps` times and synchronizes once at each end, the
  way the sim's `step()` does

and reports ms per step, rays per second and env-steps per second for each
launch, plus three ratios: sphere tracing against one-pixel steps, per car
against per (car, beam), and beams adjacent against cars adjacent.

## Background: what the sim pays for lidar

warporacer's step is two kernels: physics/reward/respawn (an RK4 kinematic
bicycle at six substeps, a wall check, centerline progress) and the lidar. Per
car the physics is a few hundred flops on values that live in registers. The
lidar is 108 rays, each taking somewhere between 2 and 30 data-dependent reads
from a raster that does not fit in cache. The README quotes ~7M env-steps/s
for the whole training loop at 8k envs on one GPU, policy forward pass and PPO
bookkeeping included, so the lidar kernel on its own has to run a good deal
faster than that. The benchmark tells you how much faster, on your hardware,
and whether the layout matters.

## Learning path

This puzzle is in two parts:

### **[Thread layout](./thread_layout.md)**

Write the per-car kernel and run the benchmark.

**What you'll do:**

- Complete `lidar_car_kernel`: one thread per car, a `for` loop over beams
- Run all four launches on your device and see which wins
- Scale the run with `--num-envs`, `--steps` and `--max-active-maps`

### **[Performance comparison](./performance.md)**

Why the four launches land where they do, on a CPU and on a GPU, and what the
sim does on top.

**What you'll learn:**

- Why the distance field is worth 6x on a GPU and nothing on a CPU
- Why 8k threads can't keep a GPU busy and 884k can
- How divergence in a `while` loop interacts with which rays share a warp
- What the Warp sim adds on top: recorded launches, CPU sharding, texture sampling

## Getting started

Start with **[Thread layout](./thread_layout.md)**, then read
**[Performance comparison](./performance.md)** with your own numbers in front of
you.
