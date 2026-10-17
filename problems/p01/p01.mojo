from max.gpu import thread_idx, block_idx, block_dim
from max.gpu.host import DeviceBuffer, DeviceContext
from layout import TileTensor, TensorEngine, TensorLayout
from layout.tile_layout import row_major
from std.benchmark import Bench, BenchConfig, Bencher, BenchId, keep
from std.math import cos, sin, sqrt
from std.math.constants import pi
from std.pathlib import Path
from std.random import random_float64, seed
from std.sys import argv
from std.testing import assert_almost_equal
from std.time import perf_counter_ns

from harness.canary import PuzzleMemory

comptime dtype = DType.float32
comptime Vec2 = SIMD[dtype, 2]  # (col, row) in px

# The track, 40 x 16 px. `#` is wall, `.` is drivable. Row 0 is the top of the
# image and y points up, so moving in +y means moving to a smaller row.
comptime MAP: StaticString = """\
########################################
#......................................#
#......................................#
#......................####............#
#......................####..###.......#
#..........####..............###.......#
#..........####..............###.......#
#..........####........................#
#......................................#
#......................................#
#......................................#
#......................................#
#..........................#######.....#
#......................................#
#......................................#
########################################"""
comptime H = 16
comptime W = 40

# ANCHOR: ray_march
comptime NUM_BEAMS = 7
comptime FOV = 3 * pi / 2  # 270 degrees, the Hokuyo on an F1TENTH car
comptime MAX_RANGE: Scalar[dtype] = 16.0  # px
comptime CAR_COL: Scalar[dtype] = 19.5
comptime CAR_ROW: Scalar[dtype] = 8.5
comptime TPB = 8
comptime BLOCKS_PER_GRID = 1
comptime THREADS_PER_BLOCK = TPB
comptime edt_layout = row_major[H, W]()
comptime dirs_layout = row_major[NUM_BEAMS, 2]()
comptime out_layout = row_major[NUM_BEAMS]()
comptime EdtLayout = type_of(edt_layout)
comptime DirsLayout = type_of(dirs_layout)
comptime OutLayout = type_of(out_layout)


def raycast[
    EdtLayoutT: TensorLayout,
    Engine: TensorEngine,
](
    edt: TileTensor[
        mut=False, dtype, EdtLayoutT, ImmutAnyOrigin, Engine=Engine
    ],
    p: Vec2,
    d: Vec2,
    max_px: Scalar[dtype],
) -> Scalar[dtype] where (Engine.element_size == 1):
    """Sphere-march from pixel `p` along unit direction `d` (col, row).
    Returns the distance travelled in px, capped at `max_px`."""
    var pos = p
    var dist: Scalar[dtype] = 0
    # FILL ME IN (roughly 10 lines)
    return min(dist, max_px)


def ray_march[
    Engine: TensorEngine,
](
    output: TileTensor[mut=True, dtype, OutLayout, MutAnyOrigin, Engine=Engine],
    edt: TileTensor[mut=False, dtype, EdtLayout, ImmutAnyOrigin, Engine=Engine],
    dirs: TileTensor[
        mut=False, dtype, DirsLayout, ImmutAnyOrigin, Engine=Engine
    ],
) where (Engine.element_size == 1):
    var beam = thread_idx.x
    # FILL ME IN (roughly 6 lines)


# ANCHOR_END: ray_march

comptime NUM_CARS = 3
comptime BLOCKS_PER_GRID_2 = (1, NUM_CARS)
comptime THREADS_PER_BLOCK_2 = (TPB, 1)
comptime cars_layout = row_major[NUM_CARS, 3]()
comptime obs_layout = row_major[NUM_CARS, NUM_BEAMS]()
comptime CarsLayout = type_of(cars_layout)
comptime ObsLayout = type_of(obs_layout)


# ANCHOR: lidar
def beam_dir(dir: Vec2, psi: Scalar[dtype]) -> Vec2:
    """Car-frame beam direction (x forward, y left) to pixel space for a car
    with heading `psi`. Rows grow downwards, so y flips sign and a
    counter-clockwise heading turns the beam clockwise on the image."""
    var ch = cos(psi)
    var sh = sin(psi)
    var d = Vec2(dir[0], -dir[1])  # heading 0 is just the row flip from part 1
    # FILL ME IN (roughly 1 line)
    return d


def lidar[
    Engine: TensorEngine,
](
    obs: TileTensor[mut=True, dtype, ObsLayout, MutAnyOrigin, Engine=Engine],
    cars: TileTensor[
        mut=False, dtype, CarsLayout, ImmutAnyOrigin, Engine=Engine
    ],
    edt: TileTensor[mut=False, dtype, EdtLayout, ImmutAnyOrigin, Engine=Engine],
    dirs: TileTensor[
        mut=False, dtype, DirsLayout, ImmutAnyOrigin, Engine=Engine
    ],
) where (Engine.element_size == 1):
    var car = block_idx.y
    var beam = thread_idx.x
    # FILL ME IN (roughly 13 lines)


# ANCHOR_END: lidar


# ANCHOR: lidar_benchmark
# warporacer's defaults: 8k cars, a 108-beam scan, 20 m range, F1TENTH tracks
# at ~5 cm/px. Spielberg is 1130 x 1769 px; `scripts/track_edt.py` writes its
# distance field to maps/Spielberg_map.edt.
comptime BENCH_MAP = "Spielberg_map.edt"
comptime BENCH_H = 1130
comptime BENCH_W = 1769
comptime BENCH_CARS = 8192
comptime BENCH_BEAMS = 108
comptime BENCH_TPB = 128
comptime LIDAR_RANGE = 20.0  # m
comptime HALF_DIAG = 0.358  # m, half the car's diagonal: spawn no closer than this
comptime bench_edt_layout = row_major[BENCH_H, BENCH_W]()
comptime bench_cars_layout = row_major[BENCH_CARS, 3]()
comptime bench_dirs_layout = row_major[BENCH_BEAMS, 2]()
comptime bench_obs_layout = row_major[BENCH_CARS, BENCH_BEAMS]()
comptime BenchEdtLayout = type_of(bench_edt_layout)
comptime BenchCarsLayout = type_of(bench_cars_layout)
comptime BenchDirsLayout = type_of(bench_dirs_layout)
comptime BenchObsLayout = type_of(bench_obs_layout)

# The three things `--benchmark` times.
comptime SPHERE_TRACE = 0  # `lidar` as written: one car's beams per block
comptime FIXED_STEP = 1  # same launch, 1 px steps instead of the field value
comptime CARS_ADJACENT = 2  # sphere tracing, grid transposed to (car, beam)


def raycast_fixed_step[
    EdtLayoutT: TensorLayout,
    Engine: TensorEngine,
](
    edt: TileTensor[
        mut=False, dtype, EdtLayoutT, ImmutAnyOrigin, Engine=Engine
    ],
    p: Vec2,
    d: Vec2,
    max_px: Scalar[dtype],
) -> Scalar[dtype] where (Engine.element_size == 1):
    """The baseline you would write without a distance field: advance 1 px
    at a time and look up whether the pixel is wall."""
    var pos = p
    var dist: Scalar[dtype] = 0
    while dist < max_px:
        var col = Int(pos[0])
        var row = Int(pos[1])
        if (
            col < 0
            or col >= Int(edt.dim[1]())
            or row < 0
            or row >= Int(edt.dim[0]())
        ):
            break
        if rebind[Scalar[dtype]](edt[row, col]) == 0:
            break
        pos += d
        dist += 1
    return min(dist, max_px)


def sweep[
    variant: Int,
    Engine: TensorEngine,
](
    obs: TileTensor[
        mut=True, dtype, BenchObsLayout, MutAnyOrigin, Engine=Engine
    ],
    cars: TileTensor[
        mut=False, dtype, BenchCarsLayout, ImmutAnyOrigin, Engine=Engine
    ],
    edt: TileTensor[
        mut=False, dtype, BenchEdtLayout, ImmutAnyOrigin, Engine=Engine
    ],
    dirs: TileTensor[
        mut=False, dtype, BenchDirsLayout, ImmutAnyOrigin, Engine=Engine
    ],
    max_px: Scalar[dtype],
) where (Engine.element_size == 1):
    """`lidar` at fleet scale. With CARS_ADJACENT, neighbouring threads are
    neighbouring cars looking along the same beam instead of one car's
    neighbouring beams."""
    var car: Int
    var beam: Int
    comptime if variant == CARS_ADJACENT:
        car = block_dim.x * block_idx.x + thread_idx.x
        beam = block_idx.y
    else:
        car = block_idx.y
        beam = thread_idx.x
    if car < BENCH_CARS and beam < BENCH_BEAMS:
        var p = Vec2(
            rebind[Scalar[dtype]](cars[car, 0]),
            rebind[Scalar[dtype]](cars[car, 1]),
        )
        var d = beam_dir(
            Vec2(
                rebind[Scalar[dtype]](dirs[beam, 0]),
                rebind[Scalar[dtype]](dirs[beam, 1]),
            ),
            rebind[Scalar[dtype]](cars[car, 2]),
        )
        comptime if variant == FIXED_STEP:
            obs[car, beam] = raycast_fixed_step(edt, p, d, max_px)
        else:
            obs[car, beam] = raycast(edt, p, d, max_px)


struct Fleet:
    """The Spielberg distance field plus 8k cars parked at random drivable
    pixels with random headings, the way warporacer respawns them."""

    var edt: DeviceBuffer[dtype]
    var cars: DeviceBuffer[dtype]
    var dirs: DeviceBuffer[dtype]
    var res: Scalar[dtype]  # m/px
    var max_px: Scalar[dtype]

    def __init__(out self, ctx: DeviceContext) raises:
        # `pixi run tests` runs from solutions/p01, everything else from the root.
        var path = Path("maps") / BENCH_MAP
        if not path.exists():
            path = Path("../../maps") / BENCH_MAP
        if not path.exists():
            raise Error(
                "maps/"
                + BENCH_MAP
                + " not found. Build it with "
                "`python scripts/track_edt.py maps/Spielberg_map.yaml` "
                "(`pixi run edt01`)."
            )
        var data: List[UInt8]
        with open(path, "r") as f:
            data = f.read_bytes()
        var header = data.unsafe_ptr().unsafe_bitcast[Int32]()
        if (
            Int(header[unsafe_offset=0]) != BENCH_H
            or Int(header[unsafe_offset=1]) != BENCH_W
        ):
            raise Error(
                "expected a "
                + String(BENCH_H)
                + " x "
                + String(BENCH_W)
                + " map"
            )
        # Header is 3 words: H, W, then the resolution; the field follows.
        var field = data.unsafe_ptr().unsafe_bitcast[Scalar[dtype]]()
        self.res = field[unsafe_offset=2]
        self.max_px = Scalar[dtype](LIDAR_RANGE) / self.res

        self.edt = ctx.enqueue_create_buffer[dtype](BENCH_H * BENCH_W)
        with self.edt.map_to_host() as edt_host:
            for i in range(BENCH_H * BENCH_W):
                edt_host[i] = field[unsafe_offset=3 + i]

        # Rejection-sample spawn points: drivable, and far enough from a wall
        # that the car body fits. Same seed every run so the variants see the
        # same fleet.
        seed(2024)
        var clearance = Scalar[dtype](HALF_DIAG) / self.res
        self.cars = ctx.enqueue_create_buffer[dtype](BENCH_CARS * 3)
        with self.cars.map_to_host() as cars_host:
            for i in range(BENCH_CARS):
                while True:
                    var col = Scalar[dtype](random_float64(0, BENCH_W))
                    var row = Scalar[dtype](random_float64(0, BENCH_H))
                    if (
                        field[unsafe_offset=3 + Int(row) * BENCH_W + Int(col)]
                        > clearance
                    ):
                        cars_host[i * 3] = col
                        cars_host[i * 3 + 1] = row
                        break
                cars_host[i * 3 + 2] = Scalar[dtype](random_float64(-pi, pi))

        self.dirs = ctx.enqueue_create_buffer[dtype](BENCH_BEAMS * 2)
        with self.dirs.map_to_host() as dirs_host:
            for j in range(BENCH_BEAMS):
                var theta = -FOV / 2 + FOV * Float64(j) / Float64(
                    BENCH_BEAMS - 1
                )
                dirs_host[j * 2] = Scalar[dtype](cos(theta))
                dirs_host[j * 2 + 1] = Scalar[dtype](sin(theta))
        ctx.synchronize()


def launch_sweep[
    variant: Int,
    Engine: TensorEngine,
](
    ctx: DeviceContext,
    obs: TileTensor[
        mut=True, dtype, BenchObsLayout, MutAnyOrigin, Engine=Engine
    ],
    cars: TileTensor[
        mut=False, dtype, BenchCarsLayout, ImmutAnyOrigin, Engine=Engine
    ],
    edt: TileTensor[
        mut=False, dtype, BenchEdtLayout, ImmutAnyOrigin, Engine=Engine
    ],
    dirs: TileTensor[
        mut=False, dtype, BenchDirsLayout, ImmutAnyOrigin, Engine=Engine
    ],
    max_px: Scalar[dtype],
) raises where (Engine.element_size == 1):
    """One sweep of the whole fleet. The transposed variant needs a transposed
    grid: blocks of 128 cars along x, one block row per beam."""
    comptime if variant == CARS_ADJACENT:
        ctx.enqueue_function[sweep[variant, Engine]](
            obs,
            cars,
            edt,
            dirs,
            max_px,
            grid_dim=(BENCH_CARS // BENCH_TPB, BENCH_BEAMS),
            block_dim=(BENCH_TPB, 1),
        )
    else:
        ctx.enqueue_function[sweep[variant, Engine]](
            obs,
            cars,
            edt,
            dirs,
            max_px,
            grid_dim=(1, BENCH_CARS),
            block_dim=(BENCH_TPB, 1),
        )


def benchmark_sweep[variant: Int](mut b: Bencher) raises:
    var ctx = DeviceContext()
    var fleet = Fleet(ctx)
    var obs = ctx.enqueue_create_buffer[dtype](BENCH_CARS * BENCH_BEAMS)
    obs.enqueue_fill(0)
    var obs_tensor = TileTensor(obs, bench_obs_layout)
    var cars_tensor = (
        TileTensor(fleet.cars, bench_cars_layout)
        .as_unsafe_any_origin()
        .as_immut()
    )
    var edt_tensor = (
        TileTensor(fleet.edt, bench_edt_layout)
        .as_unsafe_any_origin()
        .as_immut()
    )
    var dirs_tensor = (
        TileTensor(fleet.dirs, bench_dirs_layout)
        .as_unsafe_any_origin()
        .as_immut()
    )

    # Wall clock around a batch of launches with a sync on each side. The
    # puzzles use max.benchmark's bencher_iter_custom for this, but on Metal
    # it only times the enqueue (~0.05 ms whatever the kernel), so do it by
    # hand; the number includes launch overhead, which the real sim pays too.
    @always_inline
    def timed(num_iters: Int) {imm} -> Int:
        try:
            ctx.synchronize()
            var start = perf_counter_ns()
            for _ in range(num_iters):
                launch_sweep[variant](
                    ctx,
                    obs_tensor,
                    cars_tensor,
                    edt_tensor,
                    dirs_tensor,
                    fleet.max_px,
                )
            ctx.synchronize()
            return Int(perf_counter_ns() - start)
        except e:
            print("sweep failed:", e)
            return 0

    b.iter_custom(timed)
    keep(obs.unsafe_ptr())


def run_benchmark() raises:
    with DeviceContext() as ctx:
        var fleet = Fleet(ctx)
        print(
            "Spielberg:",
            BENCH_H,
            "x",
            BENCH_W,
            "px at",
            fleet.res,
            "m/px,",
            "lidar range",
            LIDAR_RANGE,
            "m =",
            fleet.max_px,
            "px",
        )
        print(
            "fleet:",
            BENCH_CARS,
            "cars x",
            BENCH_BEAMS,
            "beams =",
            BENCH_CARS * BENCH_BEAMS,
            "rays per sweep",
        )

        # Sanity check before timing anything: both marchers stop at the same
        # walls, give or take the pixel each one overshoots by.
        var obs_a = ctx.enqueue_create_buffer[dtype](BENCH_CARS * BENCH_BEAMS)
        var obs_b = ctx.enqueue_create_buffer[dtype](BENCH_CARS * BENCH_BEAMS)
        obs_a.enqueue_fill(0)
        obs_b.enqueue_fill(0)
        var obs_a_tensor = TileTensor(obs_a, bench_obs_layout)
        var obs_b_tensor = TileTensor(obs_b, bench_obs_layout)
        var cars_tensor = (
            TileTensor(fleet.cars, bench_cars_layout)
            .as_unsafe_any_origin()
            .as_immut()
        )
        var edt_tensor = (
            TileTensor(fleet.edt, bench_edt_layout)
            .as_unsafe_any_origin()
            .as_immut()
        )
        var dirs_tensor = (
            TileTensor(fleet.dirs, bench_dirs_layout)
            .as_unsafe_any_origin()
            .as_immut()
        )
        launch_sweep[SPHERE_TRACE](
            ctx,
            obs_a_tensor,
            cars_tensor,
            edt_tensor,
            dirs_tensor,
            fleet.max_px,
        )
        launch_sweep[FIXED_STEP](
            ctx,
            obs_b_tensor,
            cars_tensor,
            edt_tensor,
            dirs_tensor,
            fleet.max_px,
        )
        ctx.synchronize()
        var max_diff: Scalar[dtype] = 0
        var worst = 0
        var agree = 0
        var mean_range: Float64 = 0
        var saturated = 0
        with obs_a.map_to_host() as a_host, obs_b.map_to_host() as b_host:
            for i in range(BENCH_CARS * BENCH_BEAMS):
                var diff = abs(a_host[i] - b_host[i])
                if diff <= 2.0:
                    agree += 1
                if diff > max_diff:
                    max_diff = diff
                    worst = i
                mean_range += Float64(a_host[i])
                if a_host[i] >= fleet.max_px:
                    saturated += 1
        mean_range /= Float64(BENCH_CARS * BENCH_BEAMS)
        print(
            "mean range",
            Scalar[dtype](mean_range) * fleet.res,
            "m,",
            saturated,
            "rays at max range",
        )
        # Grazing rays are the exception: a ray skimming a wall at a shallow
        # angle can clip the corner of one wall pixel, and whether a marcher
        # notices depends on exactly where its steps land.
        with obs_a.map_to_host() as a_host, obs_b.map_to_host() as b_host, fleet.cars.map_to_host() as cars_host:
            var car = worst // BENCH_BEAMS
            print(
                "sphere trace and 1 px stepping agree within 2 px on",
                round(
                    Float64(agree) * 100.0 / Float64(BENCH_CARS * BENCH_BEAMS),
                    1,
                ),
                "% of rays;",
                "worst is car",
                car,
                "beam",
                worst % BENCH_BEAMS,
                "at (",
                cars_host[car * 3],
                ",",
                cars_host[car * 3 + 1],
                ") heading",
                cars_host[car * 3 + 2],
                ":",
                a_host[worst],
                "vs",
                b_host[worst],
                "px",
            )
        if agree < BENCH_CARS * BENCH_BEAMS * 95 // 100:
            raise Error(
                "the two marchers disagree: is raycast stepping by the field"
                " value?"
            )

    print("-" * 80)
    var bench_config = BenchConfig(max_iters=50, num_warmup_iters=5)
    var bench = Bench(bench_config.copy())
    bench.bench_function(
        lambda (mut b: Bencher) raises: benchmark_sweep[SPHERE_TRACE](b),
        BenchId("sphere_trace"),
    )
    bench.bench_function(
        lambda (mut b: Bencher) raises: benchmark_sweep[FIXED_STEP](b),
        BenchId("fixed_step_1px"),
    )
    bench.bench_function(
        lambda (mut b: Bencher) raises: benchmark_sweep[CARS_ADJACENT](b),
        BenchId("sphere_trace_cars_adjacent"),
    )
    bench.dump_report()

    # warporacer reports env-steps/s; one sweep is one lidar read for the
    # whole fleet.
    var ms = bench.info_vec[0].result.mean("ms")
    print(
        "sphere_trace:",
        round(ms, 3),
        "ms per sweep ->",
        round(Float64(BENCH_CARS * BENCH_BEAMS) / ms / 1e6, 2),
        "G rays/s,",
        round(Float64(BENCH_CARS) / ms / 1e3, 1),
        "M lidar-only env-steps/s",
    )


# ANCHOR_END: lidar_benchmark


def main() raises:
    if len(argv()) != 2:
        raise Error("usage: p01 --simple | --lidar | --benchmark")
    var mode = argv()[1]
    if mode == "--benchmark":
        run_benchmark()
        return

    with DeviceContext() as ctx:
        var mem = PuzzleMemory[dtype](ctx)
        var n_cars = NUM_CARS if mode == "--lidar" else 1
        var out = mem.output(n_cars * NUM_BEAMS)

        # Wall distance field: for every drivable pixel, the distance to the
        # nearest wall pixel (centre to centre). warporacer gets this from
        # scipy's distance_transform_edt; brute force is fine for 40 x 16.
        var rows = MAP.split("\n")
        var edt = ctx.enqueue_create_buffer[dtype](H * W)
        edt.enqueue_fill(0)
        with edt.map_to_host() as edt_host:
            for r in range(H):
                for c in range(W):
                    if rows[r].as_bytes()[c] == UInt8(ord("#")):
                        continue
                    var best = H * H + W * W
                    for wr in range(H):
                        for wc in range(W):
                            if rows[wr].as_bytes()[wc] == UInt8(ord("#")):
                                best = min(best, (r - wr) ** 2 + (c - wc) ** 2)
                    edt_host[r * W + c] = sqrt(Scalar[dtype](best))

        # Beam directions in the car's frame: x forward, y left, 270 degree fan.
        var dirs = ctx.enqueue_create_buffer[dtype](NUM_BEAMS * 2)
        dirs.enqueue_fill(0)
        with dirs.map_to_host() as dirs_host:
            for j in range(NUM_BEAMS):
                var theta = -FOV / 2 + FOV * Float64(j) / Float64(NUM_BEAMS - 1)
                dirs_host[j * 2] = Scalar[dtype](cos(theta))
                dirs_host[j * 2 + 1] = Scalar[dtype](sin(theta))

        # (col, row, heading) per car. Car 0 is the one `--simple` uses.
        var cars = ctx.enqueue_create_buffer[dtype](NUM_CARS * 3)
        cars.enqueue_fill(0)
        with cars.map_to_host() as cars_host:
            cars_host[0] = CAR_COL
            cars_host[1] = CAR_ROW
            cars_host[2] = 0
            cars_host[3] = 6.5
            cars_host[4] = 12.5
            cars_host[5] = Scalar[dtype](pi / 2)
            cars_host[6] = 34.5
            cars_host[7] = 3.5
            cars_host[8] = Scalar[dtype](-3 * pi / 4)

        var edt_tensor = TileTensor(edt, edt_layout)
        var dirs_tensor = TileTensor(dirs, dirs_layout)
        if mode == "--simple":
            var out_tensor = TileTensor(out, out_layout)
            ctx.enqueue_function[ray_march[out_tensor.Engine]](
                out_tensor,
                edt_tensor,
                dirs_tensor,
                grid_dim=BLOCKS_PER_GRID,
                block_dim=THREADS_PER_BLOCK,
            )
        elif mode == "--lidar":
            var obs_tensor = TileTensor(out, obs_layout)
            var cars_tensor = TileTensor(cars, cars_layout)
            ctx.enqueue_function[lidar[obs_tensor.Engine]](
                obs_tensor,
                cars_tensor,
                edt_tensor,
                dirs_tensor,
                grid_dim=BLOCKS_PER_GRID_2,
                block_dim=THREADS_PER_BLOCK_2,
            )
        else:
            raise Error("Invalid argument")

        ctx.synchronize()
        var expected = ctx.enqueue_create_host_buffer[dtype](n_cars * NUM_BEAMS)
        expected.enqueue_fill(0)

        # Reference: the same march, one ray at a time on the CPU.
        with edt.map_to_host() as edt_host, dirs.map_to_host() as dirs_host, cars.map_to_host() as cars_host:
            for car in range(n_cars):
                var ch = cos(cars_host[car * 3 + 2])
                var sh = sin(cars_host[car * 3 + 2])
                for beam in range(NUM_BEAMS):
                    var bx = dirs_host[beam * 2]
                    var by = dirs_host[beam * 2 + 1]
                    var d_col = ch * bx - sh * by
                    var d_row = -(sh * bx + ch * by)
                    var col = cars_host[car * 3]
                    var row = cars_host[car * 3 + 1]
                    var dist: Scalar[dtype] = 0
                    while dist < MAX_RANGE:
                        var c = Int(col)
                        var r = Int(row)
                        if c < 0 or c >= W or r < 0 or r >= H:
                            break
                        var step = edt_host[r * W + c]
                        if step == 0:
                            break
                        col += d_col * step
                        row += d_row * step
                        dist += step
                    expected[car * NUM_BEAMS + beam] = min(dist, MAX_RANGE)

        with out.map_to_host() as out_host:
            print("out:", out_host)
            print("expected:", expected)
            for i in range(n_cars * NUM_BEAMS):
                assert_almost_equal(out_host[i], expected[i], rtol=1e-5)
        mem.verify()
        print("Puzzle 01 complete ✅")
