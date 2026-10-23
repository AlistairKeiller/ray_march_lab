# Ray March Lab

GPU ray marching puzzles in [Mojo](https://docs.modular.com/mojo/), in the format of
[Mojo GPU Puzzles](https://puzzles.modular.com/). Six puzzles build the lidar kernel from
[warporacer](https://github.com/uci-f1tenth/warporacer), a GPU simulator that trains F1TENTH
racing policies by stepping thousands of cars at once, one piece at a time: pixels to meters,
world to pixel, one sphere-marched ray, a 108-beam scan on a real track, a padded multi-map
stack, and a benchmark at warporacer scale (8k cars x 108 beams x 20 m on 8 maps) that times
four ways of launching the same kernel. Every puzzle runs on any GPU that
[MAX supports](https://max.modular.com/packages/#gpu-compatibility): NVIDIA, AMD or Apple
silicon.

The same lab exists in NVIDIA Warp, the framework warporacer is written in:
[ray_march_lab_warp](https://github.com/AlistairKeiller/ray_march_lab_warp). The two share
their book, map loader, checks and puzzle numbering, so you can read a puzzle in one language
and write it in the other.

## Getting started

You need [uv](https://docs.astral.sh/uv/) and a supported GPU. On macOS you also need the Metal
toolchain: if `xcrun -sdk macosx metal` says it is missing, run
`xcodebuild -downloadComponent MetalToolchain`.

```bash
git clone https://github.com/AlistairKeiller/ray_march_lab
cd ray_march_lab
uv sync

uv run mojo -I . problems/p01/p01.mojo     # work a puzzle: fill in the FILL ME IN gap until it passes
uv run mojo -I . solutions/p01/p01.mojo    # the reference
uv run mojo -I . problems/p06/p06.mojo     # the benchmark, 8k cars on 8 maps
mdbook serve book --open                   # the book (brew install mdbook, or cargo install mdbook)
```

`-I .` puts the repo root on Mojo's import path, where the `harness` package lives. Your work
goes in `problems/pNN/pNN.mojo`; the reference is `solutions/pNN/pNN.mojo`. The book has one
page per puzzle with the code to complete, tips, the expected output and the solution.

## Development

```bash
bash solutions/run.sh                      # every solution and the harness test report ✅ (one: bash solutions/run.sh p03)
bash scripts/check_problems.sh             # every unfilled skeleton still compiles, and fails at its own check
uv run python scripts/check_skeletons.py   # problems == solutions outside the FILL ME IN regions
uv run python scripts/check_canaries.py    # every learner-written kernel writes into a guarded buffer
uv run python scripts/check_book_refs.py   # every book include, link and image resolves; no math Markdown would mangle
uv run mojo format harness problems solutions
uv run --extra viz python book/src/puzzle_03/puzzle_03_viz.py   # regenerate a figure
```

## Layout

| path | what it is |
|---|---|
| `problems/pNN/pNN.mojo` | the puzzle: a kernel with a `# FILL ME IN` gap and a check that fails until you fill it |
| `solutions/pNN/pNN.mojo` | the same file with the gap filled; the book includes both via `# ANCHOR` markers |
| `harness/canary.mojo` | guarded output buffers: NaN margins either side of the output catch out-of-bounds writes |
| `harness/track.py` | map yaml + image -> wall-distance field in px, random drivable poses, zero-padded multi-map stacks |
| `harness/track.mojo` | the same loader driven from Mojo through Python interop, with the field copied to the device |
| `maps/` | ROS-style track maps from warporacer: the F1TENTH racetracks plus two hallway maps |
| `book/` | the mdbook: one page per puzzle with the code to complete, tips, expected output and the solution |
| `scripts/` | the checks above |

## Acknowledgments

- The puzzle format, the guarded-buffer harness and the shape of the book come from Modular's
  [Mojo GPU Puzzles](https://github.com/modular/mojo-gpu-puzzles).
- The kernels are `lidar_kernel` / `raycast` from
  [warporacer](https://github.com/uci-f1tenth/warporacer), written for NVIDIA Warp, and the
  maps are the F1TENTH racetrack set that ships with it.

## License

Apache License v2.0 with LLVM Exceptions, same as Mojo GPU Puzzles. See [LICENSE](LICENSE).
