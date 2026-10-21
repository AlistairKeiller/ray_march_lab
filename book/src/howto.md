# How to Use This Book

Each puzzle keeps the same structure:

- **Overview**: what the kernel has to do and where it sits in the sim
- **Key concepts**: the Mojo features and the geometry the puzzle introduces
- **Configuration**: sizes, units, and the layout of the tensors involved
- **Code to complete**: the file in `problems/pXX/` with a `# FILL ME IN` gap
- **Tips**: hints you can open without seeing the answer
- **Running the code**: the command, and what the output looks like before and
  after you solve it
- **Solution**: the reference implementation with an explanation

The puzzles build on each other, so work through them in order. Puzzle 4 uses
the function you write in Puzzle 3, and Puzzle 6 benchmarks the kernel from
Puzzle 5.

This lab has a twin written in NVIDIA Warp,
[ray_march_lab_warp](https://github.com/AlistairKeiller/ray_march_lab_warp),
with the same six puzzles, the same map loader and the same book. Where this
page says `uv run mojo -I . problems/pXX/pXX.mojo`, that one says
`uv run python problems/pXX/pXX.py`.

## Running the code

Every puzzle is a Mojo program that launches your kernel, compares the result
with a reference computed on the CPU, and prints `Puzzle XX complete ✅` when
they agree. The reference runs the same arithmetic in float32. A GPU that fuses
multiply-adds can move a sample by an ulp, so the lidar puzzles compare within
one pixel; on the Apple GPU this book was written on the two agree exactly.

## Prerequisites

### System requirements

- **Python 3.14** and [uv](https://docs.astral.sh/uv/), which installs the
  pinned interpreter, Mojo, MAX and the dependencies for you
- **A GPU that [MAX supports](https://max.modular.com/packages/#gpu-compatibility)**:
  an NVIDIA GPU with a CUDA 12 driver, an AMD GPU with ROCm, or Apple silicon
  on macOS 15 or later. On a Mac you also need the Metal toolchain: if
  `xcrun -sdk macosx metal` says it is missing, run
  `xcodebuild -downloadComponent MetalToolchain`
- **[mdbook](https://rust-lang.github.io/mdBook/)** to serve this book locally
  (`brew install mdbook` or `cargo install mdbook`)

### Programming knowledge

Puzzles 1 to 8 of [Mojo GPU Puzzles](https://puzzles.modular.com/), or the
equivalent: thread and block indices, guards, and reading and writing a
`TileTensor`. Each puzzle here introduces the rest as it needs it. The host side
of every puzzle is Mojo too, with a little Python interop for the map loader.

## Setting up your environment

1. Clone the repository:

   ```bash
   git clone https://github.com/AlistairKeiller/ray_march_lab
   cd ray_march_lab
   ```

2. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you
   do not have it:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. Install everything and run your first puzzle:

   ```bash
   uv sync

   # This fails waiting for your implementation. Follow the book.
   uv run mojo -I . problems/p01/p01.mojo
   ```

   `uv sync` creates `.venv` with Python 3.14, Mojo, MAX, NumPy, SciPy and the
   image/yaml readers the map loader needs. `-I .` puts the repo root on Mojo's
   import path, where the `harness` package lives. The first run of each puzzle
   compiles its kernel for your GPU (a few seconds); later runs hit the compile
   cache.

## Working with puzzles

### Project structure

- **`problems/`**: where you implement your solutions (this is where you work!)
- **`solutions/`**: the reference solutions, which the book quotes
- **`harness/`**: the guarded output buffers and the map loader every puzzle
  imports, with a self-test of the guard under `harness/test/`
- **`maps/`**: the track maps from warporacer, as ROS map yaml + image pairs

### Workflow

1. Open `problems/pXX/pXX.mojo` and find the `# FILL ME IN` comment
2. Write the kernel body
3. Run `uv run mojo -I . problems/pXX/pXX.mojo` until it prints the ✅
4. Compare with `solutions/pXX/pXX.mojo` and read the explanation in the book

### Guarded output buffers

Every output buffer a puzzle hands your kernel sits inside a larger allocation
whose margins hold NaN. After the kernel runs the puzzle checks those margins,
and fails if your kernel wrote into one:

```text
❌ Write detected outside of output buffer
```

A kernel can write every expected value into `output` and still write past the
end of it, which comparing values alone cannot detect. This message means your
indexing ran outside the output region. Mojo does not bounds-check `TileTensor`
accesses on the GPU, so without the margins such a write would corrupt whatever
happened to be next in memory, silently.

The margins extend 256 bytes past each end, so a write further out than that
goes unreported. A clean check is good evidence, not proof. Puzzle 6 drives
its kernels from a benchmark loop rather than the harness and has no guarded
buffer; it checks its four launches against each other instead.

### Essential commands

```bash
# Run puzzles
uv run mojo -I . problems/pXX/pXX.mojo    # your implementation
uv run mojo -I . solutions/pXX/pXX.mojo   # the reference

# Test solutions
bash solutions/run.sh                     # all of them, plus the harness self-test
bash solutions/run.sh p03                 # one

# The benchmark at full scale (defaults: 8192 envs, 500 steps, 8 random maps)
uv run mojo -I . problems/p06/p06.mojo maps/ --num-envs 8192 --steps 500
uv run mojo -I . problems/p06/p06.mojo maps/Spielberg_map.yaml

# Checks the book relies on
bash scripts/check_problems.sh            # every unfilled skeleton compiles and fails at its own check
uv run python scripts/check_skeletons.py  # problems == solutions outside the FILL regions
uv run python scripts/check_canaries.py   # every learner-written kernel uses the guarded buffer
uv run python scripts/check_book_refs.py  # every include, link and image in the book resolves
uv run mojo format harness problems solutions

# Figures (needs the viz extra)
uv sync --extra viz
uv run python book/src/puzzle_03/puzzle_03_viz.py
```

## Device support matrix

| Puzzle                 | NVIDIA | AMD | Apple | Notes                                      |
|------------------------|--------|-----|-------|--------------------------------------------|
| 1 - Map                | ✅     | ✅  | ✅    |                                            |
| 2 - World to Pixel     | ✅     | ✅  | ✅    |                                            |
| 3 - Sphere March       | ✅     | ✅  | ✅    |                                            |
| 4 - Lidar              | ✅     | ✅  | ✅    | Loads one real track map                   |
| 5 - Many Maps          | ✅     | ✅  | ✅    | Three maps, padded into one raster         |
| 6 - Lidar Benchmark    | ✅     | ✅  | ✅    | Four launches of one kernel, timed         |

The kernels use nothing vendor-specific: plain loads, stores, `cos` and `sin`.
The numbers in this book are from an Apple M5 Pro; the other columns run the
same code through MAX's CUDA and ROCm backends.
