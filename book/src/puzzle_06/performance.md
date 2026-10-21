# Performance Comparison

Same rays, same results, four launches. The numbers below are from one laptop,
an Apple M5 Pro, running both twins of this lab: the Warp lab on its CPU
backend (18 shards, 4,096 cars, 100 steps) and the Mojo lab on the laptop's GPU
(8,192 cars, 500 steps). The CPU and the GPU disagree about almost everything,
and this page is about why.

| launch                      | CPU (Warp, 18 cores)   | GPU (Mojo, Metal)        |
|-----------------------------|------------------------|--------------------------|
| per (car, beam)             | 3.14 ms, 141 M rays/s  | 0.51 ms, 1,741 M rays/s  |
| per car                     | 2.54 ms, 174 M rays/s  | 1.23 ms, 718 M rays/s    |
| per (beam, car)             | 6.86 ms, 65 M rays/s   | 0.66 ms, 1,332 M rays/s  |
| per (car, beam), 1 px steps | 2.99 ms, 148 M rays/s  | 3.04 ms, 291 M rays/s    |

The CPU rows are 4,096 cars and the GPU rows 8,192, so compare rays per second,
not milliseconds.

## What the distance field is worth

On the GPU, sphere tracing is 6x faster than stepping one pixel at a time. On
the CPU it is not faster at all.

A ray on these tracks travels about 3.3 m on average, 57 px at 5 cm/px, so the
1 px stepper does 57 lookups per ray and 346 for the 1% that reach the 20 m
range. The sphere trace does about 7 (median 4, 90th percentile 17, and over a
hundred for a ray skimming along a wall): 8x fewer loop iterations, each a
little more expensive.

**Why the CPU does not care.** A sphere-trace lookup lands somewhere new every
step, in a 100 MB raster that fits in no cache, so most of its 7 reads are
misses at around 100 ns each. The stepper's 57 reads walk consecutive pixels:
the first miss brings in a 64-byte line and the next 15 pixels along a row come
for free (a vertical ray misses more often, a diagonal one in between). Seven
misses cost about what fifty-seven mostly-hits cost, and the stepper's loop
body is cheaper besides: no multiply, and a branch the predictor gets right.
The two come out even.

**Why the GPU cares.** A GPU hides memory latency by keeping far more threads
resident than it has execution units, 884,736 in this launch, and switching
between them while loads are outstanding. Once latency is hidden, what is left
is the number of instructions and the number of memory transactions, and there
the 8x fewer iterations show up almost in full.

**The 99.1%.** The two marchers disagree by more than 2 px on about 1% of rays,
and the worst of those by the whole 20 m. Those are grazing rays: a ray running
almost parallel to a wall whose rasterised edge is a staircase of pixels. The
sphere tracer, taking sub-pixel steps that close to a wall, lands inside a
corner pixel and stops; the stepper samples just before and just after that
pixel and carries on. Neither answer is wrong at the map's resolution.
warporacer's lidar has the same property and policies train fine on it, but it
is why the check counts rays within 2 px instead of demanding a bit-for-bit
match the way the other three launches do.

## Per car or per (car, beam)

**The CPU prefers per car (1.24x).** A Warp CPU kernel is a compiled C++ loop
over `tid`, running on one core per launch. "Threads" cost nothing to create,
but each one redoes its setup: in the per-(car, beam) layout every one of a
car's 108 threads recomputes `cos`, `sin` and the start point, and re-reads the
car's row and its map's parameters. The per-car layout does that once and
spends the rest of its time marching. The marching is identical in both, so the
difference is the setup. Cache locality favours per-car too: one core walks one
car's 108 rays, which all start at the same pixel and sweep a 20 m disc of the
map; by the time it moves to the next car the relevant part of the raster is in
its L2.

Sharding is what makes CPU numbers respectable at all. One shard, meaning one
core, manages around 20 M rays/s; 18 shards reach 140 to 170 M. The pool costs
a few hundred microseconds per step to dispatch, which is why the benchmark's
CPU default is 4,096 envs: at 1,024 the dispatch is about a third of the step.

**The GPU prefers per (car, beam) (2.4x).** Three reasons, in order of
importance.

*Occupancy.* A GPU hides memory latency by keeping many more threads resident
than it has execution units, and switching between them while loads are in
flight. An A100 holds 2,048 resident threads per SM across 108 SMs, about
221,000. 8,192 per-car threads fill 4% of that; 884,736 per-(car, beam) threads
fill it four times over. The scattered `edt` reads in the march are exactly the
latency this mechanism exists for: each read's address depends on the previous
read's value, so the only way to keep the memory system busy is more
independent rays in flight.

*Divergence.* Threads run in warps of 32 that execute in lockstep, and a
`while` loop runs until the *slowest* lane exits. In the per-(car, beam) layout
a warp is 32 adjacent beams of one car. Their rays start at the same point and
point in similar directions, so they take similar numbers of steps. In the
per-car layout a warp is 32 different cars, each looping over 108 beams; the
warp finishes when the car with the longest total march does, and every other
lane idles for the difference. Across eight maps with different corridor widths
that difference is large.

*Access pattern.* Adjacent lanes in a per-(car, beam) warp read neighbouring
pixels early in the march and share cache lines. Adjacent lanes in a per-car
warp read from 32 unrelated places in a 100 MB raster from the first step.

None of this is specific to lidar. It's the usual GPU rule of mapping the
finest independent unit of work to a thread, and it's why warporacer
parallelises over (car, beam).

## Beams adjacent or cars adjacent

The transposed launch does the same work with the same per-thread setup; only
the neighbours change. On the GPU it costs 31%: a warp of 32 adjacent beams of
one car finishes its `while` loop when the longest of 32 similar rays does, and
its lanes read the same patch of the field; a warp of 32 cars on the same beam
waits for the longest of 32 unrelated rays, and its lanes read 32 unrelated
places from the first step. On the CPU it costs 2.2x, for the cache reason
above turned on its head: a shard's loop runs beams of one car back to back in
the normal layout, so a car's 108 rays sweep one disc of the raster while it is
in L2; transposed, consecutive iterations are consecutive cars, each somewhere
else on one of eight maps.

warporacer launches on (car, beam) with beam fastest for this reason, and its
CUDA path goes further and binds the field as a texture so that the scattered
reads hit a cache built for them.

## What else warporacer does

The Warp lab's benchmark reproduces three things from `warporacer/sim.py` that
are worth knowing about even if you never change them. The Mojo lab launches
its kernels directly, and MAX has no texture API, so its numbers are plain
global-memory loads.

### Recorded launches

`wp.launch(..., record_cmd=True)` returns a `Launch` with the kernel and its
arguments already packed, and `launch.launch()` replays it. The sim steps at
60 Hz over 8,192 envs for millions of steps, and the Python side of a normal
`wp.launch` (type checks, argument packing, slicing the per-shard arrays) is
paid once instead of every step.

### Texture sampling (CUDA)

On CUDA the Warp benchmark also times `lidar_tex_kernel`, which reads the
stack through a `wp.Texture3D` with nearest filtering and border addressing.
The texture cache is built for exactly this kind of spatially scattered,
read-only access, and border addressing returns 0 (wall) outside the volume so
the bounds check disappears from the loop. warporacer measured +14% sim
throughput from it. The sample coordinate is `(col, row, mid + 0.5)` so that
nearest sampling picks `edt[mid, int(row), int(col)]` exactly, which is why the
correctness check expects the texture row to match the others to the bit.

### Streams

Not in the benchmark, but in the sim: the kernels launch on PyTorch's current
CUDA stream (`wp.stream_from_torch`), so the policy network reads `obs`
straight out of the Warp buffer with no synchronisation between the frameworks.
The observation, reward and done arrays are zero-copy torch views of Warp
arrays for the same reason.

## Putting the number in context

warporacer's README quotes ~7M env-steps/s at 8k envs on a single GPU for the
full loop: physics kernel, lidar kernel, policy forward pass, and the PPO update
amortised over the rollout. The per-(car, beam) row on the laptop GPU above is
16 M lidar-only env-steps/s, and a datacenter GPU has several times its memory
bandwidth, so the lidar being the single largest kernel is consistent with that
budget. Whatever `M env-steps/s` the per-(car, beam) row shows on your device
bounds the lidar's share of it. If you want to move the number, the levers are
the ones above: more rays in flight, less divergence, cheaper reads. Not fewer
flops.

## Key concepts

| Concept                   | per car                          | per (car, beam)             | per (beam, car)                 | 1 px steps                      |
|---------------------------|----------------------------------|-----------------------------|---------------------------------|---------------------------------|
| **Threads at 8k envs**    | 8,192                            | 884,736                     | 884,736                         | 884,736                         |
| **Work per thread**       | 108 marches                      | 1 march                     | 1 march                         | 1 march, 57 reads on average    |
| **Setup per car**         | once                             | 108 times                   | 108 times                       | 108 times                       |
| **Warp contents (GPU)**   | 32 cars, 32 unrelated addresses  | 32 beams of one car         | 32 cars on one beam             | 32 beams of one car             |
| **Loop divergence**       | slowest car × 108 beams          | slowest of 32 similar rays  | slowest of 32 unrelated rays    | slowest of 32 similar rays      |
| **CPU, 18 shards**        | 174 M rays/s                     | 141 M rays/s                | 65 M rays/s                     | 148 M rays/s                    |
| **GPU (M5 Pro)**          | 718 M rays/s                     | 1,741 M rays/s              | 1,332 M rays/s                  | 291 M rays/s                    |

The core lesson: a thread layout is a decision about **how many independent
memory accesses the hardware can have in flight**. On a CPU core the answer is
"a few", so setup cost and cache locality decide. On a GPU the answer is "tens
of thousands", and a layout that can't supply them leaves the machine idle.
