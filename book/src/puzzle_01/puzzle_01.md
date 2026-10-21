# Puzzle 1: Map

## Overview

Implement a kernel that converts each wall distance in `edt`, measured in
pixels, to meters by multiplying it by the map resolution `res`, storing the
result in `output`.

**Note:** _You have 1 thread per position._

## Key concepts

In this puzzle, you'll learn about:

- Writing a GPU kernel: an ordinary `def` whose arguments are tensors and
  scalars
- Thread indexing with `thread_idx.x`
- Typed tensor arguments: `TileTensor[mut=False, dtype, Layout, ImmutAnyOrigin]`
- Scalar kernel arguments: `res: Scalar[dtype]`
- Launching with
  `ctx.enqueue_function[px_to_m](..., grid_dim=1, block_dim=SIZE)`

For each position \\(i\\):
\\[\Large output[i] = edt[i] \times res\\]

### Where this sits in the sim

Every track in warporacer is a ROS map: an image whose white pixels are
drivable, plus a yaml with the map's `resolution` in meters per pixel. The sim
precomputes the distance from every pixel to the nearest wall, in pixels, and
converts to meters whenever it needs a physical quantity, such as how far a car
is from the wall. This kernel is that conversion.

## Configuration

- Tensor size: `SIZE = 4` elements, layout `row_major[SIZE]()`
- Resolution: `RES = 0.05` m/px, the F1TENTH default
- One thread per element: one block of `SIZE` threads

Notes:

- **Kernel**: any `def` whose arguments can be handed to the device can be
  enqueued. `enqueue_function[px_to_m]` compiles it for your GPU the first
  time it runs and caches the result
- **Thread index**: `thread_idx.x` is this thread's index within its block.
  With one block of `SIZE` threads it runs from `0` to `SIZE - 1`
- **Tensors**: a `TileTensor` is a pointer plus a layout. `edt[i]` reads and
  `output[i] = ...` writes; `mut=True` and `MutAnyOrigin` mark the one the
  kernel writes to. `dtype` is `DType.float32` throughout the lab, so an
  element is a `Float32`
- **Launch**: `enqueue_function` queues the kernel and returns;
  `ctx.synchronize()` waits for it. The output buffer comes from the harness,
  `mem.output(SIZE)`, which zeroes it and guards its margins

## Code to complete

```mojo
{{#include ../../../problems/p01/p01.mojo:px_to_m}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p01/p01.mojo" class="filename">View full file: problems/p01/p01.mojo</a>

> **Note**: The skeleton declares `var i = thread_idx.x` for you, and nothing
> uses `i` until you fill in the kernel, so the compiler warns
> `assignment to 'i' was never used`. It still compiles and runs; it just
> leaves `output` as the zeros it was handed. Later puzzles scaffold their
> variables the same way.

<details>
<summary><strong>Tips</strong></summary>

<div class="solution-tips">

1. `var i = thread_idx.x` is already there
2. Multiply `edt[i]` by `res`
3. Store the result in `output[i]`

</div>
</details>

## Running the code

To test your solution, run the following command in your terminal:

```bash
uv run mojo -I . problems/p01/p01.mojo
```

Your output will look like this if the puzzle isn't solved yet:

```txt
out: HostBuffer([0.0, 0.0, 0.0, 0.0])
expected: HostBuffer([0.0, 0.05, 0.1, 0.15])
```

## Solution

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../solutions/p01/p01.mojo:px_to_m_solution}}
```

<div class="solution-explanation">

This solution:

- Gets the thread index with `var i = thread_idx.x`
- Scales the pixel distance to meters: `output[i] = edt[i] * res`

Two things worth noticing. `Scalar[dtype]` is a 32-bit float, so
`edt[i] * res` is a single-precision multiply and matches the CPU reference
(also computed in `Float32`) exactly. And there is no bounds check, because
the block has exactly `SIZE` threads and the tensor exactly `SIZE` elements.
Puzzle 2 is where indices start needing clamps, and Puzzle 4 is where launches
start needing guards.

</div>
</details>
