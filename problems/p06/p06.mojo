from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from layout import Coord, RowMajorLayout, TileTensor
from layout.tile_layout import row_major
from std.math import cos, sin
from std.math.constants import pi
from std.python import Python
from std.sys import argv, exit
from std.time import perf_counter_ns

from harness.track import Track, load_tracks, spawn, stack

comptime dtype = DType.float32
comptime Vec2 = SIMD[dtype, 2]

comptime NUM_LIDAR = 108
comptime LIDAR_FOV = 3 * pi / 2  # 270 degrees
comptime LIDAR_RANGE: Scalar[dtype] = 20.0  # m
comptime LIDAR_MOUNT_X: Scalar[dtype] = 0.2733  # m, base_link -> laser offset
comptime MAX_ACTIVE_MAPS = 8  # warporacer trains on a rotating pool of this many maps
comptime TPB = 128  # threads per block, warporacer's block_dim

# The fleet's sizes are only known at run time, so these layouts carry their shape.
comptime Dyn1 = RowMajorLayout[Int]
comptime Dyn2 = RowMajorLayout[Int, Int]
comptime Dyn3 = RowMajorLayout[Int, Int, Int]
comptime dirs_layout = row_major[NUM_LIDAR, 2]()
comptime DirsLayout = type_of(dirs_layout)

comptime Scan = TileTensor[
    mut=True, dtype, Dyn2, MutAnyOrigin
]  # (car, beam) range in m
comptime Cars = TileTensor[mut=False, dtype, Dyn2, ImmutAnyOrigin]  # x, y, psi
comptime MapId = TileTensor[mut=False, DType.int32, Dyn1, ImmutAnyOrigin]
comptime Edt = TileTensor[
    mut=False, dtype, Dyn3, ImmutAnyOrigin
]  # (n_maps, H, W), zero-padded
comptime Dirs = TileTensor[mut=False, dtype, DirsLayout, ImmutAnyOrigin]
comptime Origin = TileTensor[
    mut=False, dtype, Dyn2, ImmutAnyOrigin
]  # world origin per map
comptime Res = TileTensor[
    mut=False, dtype, Dyn1, ImmutAnyOrigin
]  # meters/px per map
comptime MapH = TileTensor[
    mut=False, DType.int32, Dyn1, ImmutAnyOrigin
]  # unpadded rows per map


def raycast(
    edt: Edt, mid: Int, var p: Vec2, d: Vec2, max_px: Scalar[dtype]
) -> Scalar[dtype]:
    """Sphere-march from pixel `p` along unit direction `d` (col, row); returns
    distance in px. Padding rows/cols are zero (= wall), so rays stop at each map's
    true bounds."""
    var dist: Scalar[dtype] = 0
    while dist < max_px:
        var col = Int(p[0])
        var row = Int(p[1])
        if (
            col < 0
            or col >= Int(edt.dim[2]())
            or row < 0
            or row >= Int(edt.dim[1]())
        ):
            break
        var step = edt[mid, row, col]
        if step == 0:
            break
        p += d * step
        dist += step
    return min(dist, max_px)


def raycast_fixed_step(
    edt: Edt, mid: Int, var p: Vec2, d: Vec2, max_px: Scalar[dtype]
) -> Scalar[dtype]:
    """The march you would write without a distance field: advance 1 px at a time and
    test for wall. Same stopping rules as raycast(), one read per pixel of range.
    """
    var dist: Scalar[dtype] = 0
    while dist < max_px:
        var col = Int(p[0])
        var row = Int(p[1])
        if (
            col < 0
            or col >= Int(edt.dim[2]())
            or row < 0
            or row >= Int(edt.dim[1]())
        ):
            break
        if edt[mid, row, col] == 0:
            break
        p += d
        dist += 1
    return min(dist, max_px)


def laser_px(
    x: Scalar[dtype],
    y: Scalar[dtype],
    ch: Scalar[dtype],
    sh: Scalar[dtype],
    org: Vec2,
    r: Scalar[dtype],
    map_h: Int,
) -> Vec2:
    """Pixel position of the laser mount for a car at world (x, y) with heading
    (cos, sin): the Puzzle 5 start point, shared by every kernel below."""
    return Vec2(
        (x + LIDAR_MOUNT_X * ch - org[0]) / r,
        Scalar[dtype](map_h - 1) - (y + LIDAR_MOUNT_X * sh - org[1]) / r,
    )


def beam_px(b: Vec2, ch: Scalar[dtype], sh: Scalar[dtype]) -> Vec2:
    """A car-frame beam rotated into the heading frame, with the row axis flipped.
    """
    return Vec2(ch * b[0] - sh * b[1], -(sh * b[0] + ch * b[1]))


def lidar_kernel(
    scan: Scan,
    cars: Cars,
    map_id: MapId,
    edt: Edt,
    lidar_dirs: Dirs,
    origin: Origin,
    res: Res,
    map_h: MapH,
):
    """One block per car and one thread per beam, beams adjacent: the layout
    warporacer ships."""
    var i = Int(block_idx.y)
    var j = Int(thread_idx.x)
    if j < NUM_LIDAR:
        var mid = Int(map_id[i])
        var r = res[mid]
        var ch = cos(cars[i, 2])
        var sh = sin(cars[i, 2])
        var p = laser_px(
            cars[i, 0],
            cars[i, 1],
            ch,
            sh,
            Vec2(origin[mid, 0], origin[mid, 1]),
            r,
            Int(map_h[mid]),
        )
        var d = beam_px(Vec2(lidar_dirs[j, 0], lidar_dirs[j, 1]), ch, sh)
        scan[i, j] = raycast(edt, mid, p, d, LIDAR_RANGE / r) * r


def lidar_fixed_step_kernel(
    scan: Scan,
    cars: Cars,
    map_id: MapId,
    edt: Edt,
    lidar_dirs: Dirs,
    origin: Origin,
    res: Res,
    map_h: MapH,
):
    """`lidar_kernel` with the 1 px stepper in place of the sphere march: the baseline the
    distance field replaces."""
    var i = Int(block_idx.y)
    var j = Int(thread_idx.x)
    if j < NUM_LIDAR:
        var mid = Int(map_id[i])
        var r = res[mid]
        var ch = cos(cars[i, 2])
        var sh = sin(cars[i, 2])
        var p = laser_px(
            cars[i, 0],
            cars[i, 1],
            ch,
            sh,
            Vec2(origin[mid, 0], origin[mid, 1]),
            r,
            Int(map_h[mid]),
        )
        var d = beam_px(Vec2(lidar_dirs[j, 0], lidar_dirs[j, 1]), ch, sh)
        scan[i, j] = raycast_fixed_step(edt, mid, p, d, LIDAR_RANGE / r) * r


def lidar_transposed_kernel(
    scan: Scan,
    cars: Cars,
    map_id: MapId,
    edt: Edt,
    lidar_dirs: Dirs,
    origin: Origin,
    res: Res,
    map_h: MapH,
):
    """`lidar_kernel` with the grid turned round: a block holds TPB adjacent cars looking
    along one beam (block_idx.y) instead of one car's adjacent beams."""
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    var j = Int(block_idx.y)
    if i < Int(scan.dim[0]()):
        var mid = Int(map_id[i])
        var r = res[mid]
        var ch = cos(cars[i, 2])
        var sh = sin(cars[i, 2])
        var p = laser_px(
            cars[i, 0],
            cars[i, 1],
            ch,
            sh,
            Vec2(origin[mid, 0], origin[mid, 1]),
            r,
            Int(map_h[mid]),
        )
        var d = beam_px(Vec2(lidar_dirs[j, 0], lidar_dirs[j, 1]), ch, sh)
        scan[i, j] = raycast(edt, mid, p, d, LIDAR_RANGE / r) * r


# ANCHOR: lidar_car_kernel
def lidar_car_kernel(
    scan: Scan,
    cars: Cars,
    map_id: MapId,
    edt: Edt,
    lidar_dirs: Dirs,
    origin: Origin,
    res: Res,
    map_h: MapH,
):
    """One thread per car that loops over its beams: 108x fewer threads, each doing
    108x the work. Same rays, same answers; only the layout differs."""
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i < Int(scan.dim[0]()):
        # FILL ME IN (roughly 8 lines)
        pass


# ANCHOR_END: lidar_car_kernel


struct Fleet:
    """The lidar half of warporacer's Env: a padded map stack, per-map origin, resolution
    and height, a map id per car, and cars spawned at random drivable poses."""

    var tracks: List[Track]
    var num_envs: Int
    var n_maps: Int
    var h: Int  # rows of the padded stack, the largest map's
    var w: Int
    var edt: DeviceBuffer[dtype]
    var origin: DeviceBuffer[dtype]
    var res: DeviceBuffer[dtype]
    var map_h: DeviceBuffer[DType.int32]
    var lidar_dirs: DeviceBuffer[dtype]
    var cars: DeviceBuffer[dtype]
    var map_id: DeviceBuffer[DType.int32]
    var scan: DeviceBuffer[dtype]

    def __init__(
        out self,
        ctx: DeviceContext,
        var tracks: List[Track],
        num_envs: Int,
        seed: Int,
    ) raises:
        self.num_envs = num_envs
        self.n_maps = len(tracks)
        self.h = 0
        self.w = 0
        for t in tracks:
            self.h = max(self.h, t.h)
            self.w = max(self.w, t.w)
        self.edt = stack(ctx, tracks)
        self.origin = ctx.enqueue_create_buffer[dtype](self.n_maps * 2)
        self.res = ctx.enqueue_create_buffer[dtype](self.n_maps)
        self.map_h = ctx.enqueue_create_buffer[DType.int32](self.n_maps)
        with self.origin.map_to_host() as o, self.res.map_to_host() as r, self.map_h.map_to_host() as mh:
            for k in range(self.n_maps):
                o[k * 2] = tracks[k].ox
                o[k * 2 + 1] = tracks[k].oy
                r[k] = tracks[k].res
                mh[k] = Int32(tracks[k].h)
        self.lidar_dirs = ctx.enqueue_create_buffer[dtype](NUM_LIDAR * 2)
        with self.lidar_dirs.map_to_host() as d:
            for j in range(NUM_LIDAR):
                var theta = -LIDAR_FOV / 2 + LIDAR_FOV * Float64(j) / Float64(
                    NUM_LIDAR - 1
                )
                d[j * 2] = Scalar[dtype](cos(theta))
                d[j * 2 + 1] = Scalar[dtype](sin(theta))

        var np = Python.import_module("numpy")
        var spawned = spawn(ctx, tracks, num_envs, np.random.default_rng(seed))
        self.cars = spawned.cars
        self.map_id = spawned.map_id
        self.scan = ctx.enqueue_create_buffer[dtype](num_envs * NUM_LIDAR)
        self.scan.enqueue_fill(0)
        self.tracks = tracks^
        ctx.synchronize()


# The four launches `--benchmark` times.
comptime BEAM = 0  # lidar_kernel: one car's beams per block
comptime CAR = 1  # lidar_car_kernel: one thread per car
comptime TRANSPOSED = 2  # lidar_transposed_kernel: adjacent cars per block
comptime FIXED = 3  # lidar_fixed_step_kernel: same grid as BEAM, 1 px steps


def launch[variant: Int](ctx: DeviceContext, mut fleet: Fleet) raises:
    """One sweep of the whole fleet with the given kernel and its grid."""
    var scan = TileTensor(
        fleet.scan, row_major(Coord(fleet.num_envs, NUM_LIDAR))
    )
    var cars = TileTensor(fleet.cars, row_major(Coord(fleet.num_envs, 3)))
    var map_id = TileTensor(fleet.map_id, row_major(Coord(fleet.num_envs)))
    var edt = TileTensor(
        fleet.edt, row_major(Coord(fleet.n_maps, fleet.h, fleet.w))
    )
    var dirs = TileTensor(fleet.lidar_dirs, dirs_layout)
    var origin = TileTensor(fleet.origin, row_major(Coord(fleet.n_maps, 2)))
    var res = TileTensor(fleet.res, row_major(Coord(fleet.n_maps)))
    var map_h = TileTensor(fleet.map_h, row_major(Coord(fleet.n_maps)))
    var blocks = (fleet.num_envs + TPB - 1) // TPB
    comptime if variant == CAR:
        ctx.enqueue_function[lidar_car_kernel](
            scan,
            cars,
            map_id,
            edt,
            dirs,
            origin,
            res,
            map_h,
            grid_dim=blocks,
            block_dim=TPB,
        )
    elif variant == TRANSPOSED:
        ctx.enqueue_function[lidar_transposed_kernel](
            scan,
            cars,
            map_id,
            edt,
            dirs,
            origin,
            res,
            map_h,
            grid_dim=(blocks, NUM_LIDAR),
            block_dim=(TPB, 1),
        )
    elif variant == FIXED:
        ctx.enqueue_function[lidar_fixed_step_kernel](
            scan,
            cars,
            map_id,
            edt,
            dirs,
            origin,
            res,
            map_h,
            grid_dim=(1, fleet.num_envs),
            block_dim=(TPB, 1),
        )
    else:
        ctx.enqueue_function[lidar_kernel](
            scan,
            cars,
            map_id,
            edt,
            dirs,
            origin,
            res,
            map_h,
            grid_dim=(1, fleet.num_envs),
            block_dim=(TPB, 1),
        )


def scan_of[
    variant: Int
](ctx: DeviceContext, mut fleet: Fleet) raises -> HostBuffer[dtype]:
    """The fleet's scan after one sweep with `variant`, on the host."""
    fleet.scan.enqueue_fill(0)
    launch[variant](ctx, fleet)
    var host = ctx.enqueue_create_host_buffer[dtype](fleet.num_envs * NUM_LIDAR)
    fleet.scan.enqueue_copy_to(host)
    ctx.synchronize()
    return host^


def bench[
    variant: Int
](ctx: DeviceContext, mut fleet: Fleet, steps: Int) raises -> Float64:
    """Milliseconds per step over `steps` sweeps, synchronizing only at the ends. The first launch
    also compiles the kernel, hence the warm-up."""
    for _ in range(3):
        launch[variant](ctx, fleet)
    ctx.synchronize()
    var start = perf_counter_ns()
    for _ in range(steps):
        launch[variant](ctx, fleet)
    ctx.synchronize()
    return Float64(perf_counter_ns() - start) / Float64(steps) / 1e6


def pad(s: String, width: Int) -> String:
    """`s` right-padded with spaces to `width`."""
    var out = s
    while out.byte_length() < width:
        out += " "
    return out


def num(x: Float64, width: Int, decimals: Int) -> String:
    """`x` rounded to `decimals`, left-padded with spaces to `width`."""
    var s = String(round(x, decimals))
    while s.byte_length() < width:
        s = " " + s
    return s


def ratio(slow: Float64, fast: Float64) -> String:
    """`slow / fast` as a speed-up, e.g. "5.76x"."""
    return String(round(slow / fast, 2)) + "x"


def main() raises:
    var maps = String("maps")
    var num_envs = 8192
    var steps = 500
    var max_active_maps = MAX_ACTIVE_MAPS
    var seed = 0
    var args = argv()
    var k = 1
    while k < len(args):
        var a = String(args[k])
        if a == "--num-envs":
            num_envs = atol(args[k + 1])
            k += 2
        elif a == "--steps":
            steps = atol(args[k + 1])
            k += 2
        elif a == "--max-active-maps":
            max_active_maps = atol(args[k + 1])
            k += 2
        elif a == "--seed":
            seed = atol(args[k + 1])
            k += 2
        elif a.startswith("-"):
            raise Error(
                "usage: p06 [maps] [--num-envs N] [--steps N]"
                " [--max-active-maps N] [--seed N]"
            )
        else:
            maps = a
            k += 1

    with DeviceContext() as ctx:
        var np = Python.import_module("numpy")
        var rng = np.random.default_rng(seed)
        var tracks = load_tracks(ctx, maps, max_active_maps, rng)
        var fleet = Fleet(ctx, tracks^, num_envs, seed)

        print("Puzzle 06: Lidar Benchmark")
        print("=" * 70)
        print()
        print(
            "Configuration: "
            + String(num_envs)
            + " envs x "
            + String(NUM_LIDAR)
            + " beams, "
            + String(LIDAR_RANGE)
            + " m range, device "
            + ctx.name()
        )
        var names = String()
        for k in range(fleet.n_maps):
            names += (", " if k > 0 else "") + fleet.tracks[k].name
        print("maps:", names)
        print(
            "edt stack: ("
            + String(fleet.n_maps)
            + ", "
            + String(fleet.h)
            + ", "
            + String(fleet.w)
            + ") float32, "
            + String(
                Int(round(Float64(fleet.n_maps * fleet.h * fleet.w) * 4 / 1e6))
            )
            + " MB"
        )
        print("-" * 60)

        var BEAM_NAME = "per (car, beam)"
        var CAR_NAME = "per car"
        var TRANSPOSED_NAME = "per (beam, car)"
        var FIXED_NAME = "per (car, beam), 1 px steps"

        print("Testing Correctness...")
        var reference = scan_of[BEAM](ctx, fleet)
        var n = fleet.num_envs * NUM_LIDAR
        var tol: Scalar[dtype] = 0  # one pixel of the coarsest map
        var ref_max: Scalar[dtype] = 0
        for t in fleet.tracks:
            tol = max(tol, t.res)
        for i in range(n):
            ref_max = max(ref_max, reference[i])
        var ok = ref_max > 0
        var others = [
            scan_of[CAR](ctx, fleet),
            scan_of[TRANSPOSED](ctx, fleet),
            scan_of[FIXED](ctx, fleet),
        ]
        var other_names = [CAR_NAME, TRANSPOSED_NAME, FIXED_NAME]
        for v in range(3):
            var max_diff: Scalar[dtype] = 0
            var within = 0
            for i in range(n):
                var diff = abs(others[v][i] - reference[i])
                max_diff = max(max_diff, diff)
                if diff <= 2 * tol:
                    within += 1
            if v == 2:
                # Grazing rays: a sphere trace and a 1 px stepper can stop on different
                # sides of a wall pixel's corner, so the two only have to agree on
                # nearly every ray.
                var pct = Float64(within) / Float64(n)
                ok = ok and pct >= 0.95
                print(
                    "   "
                    + pad(other_names[v], 28)
                    + " within 2 px of per (car, beam) on "
                    + String(round(pct * 100, 1))
                    + "% of rays"
                )
            else:
                ok = ok and max_diff <= tol
                print(
                    "   "
                    + pad(other_names[v], 28)
                    + " max difference vs per (car, beam): "
                    + String(max_diff)
                    + " m"
                )
        if not ok:
            print("   ❌ Layouts disagree")
            exit(1)
        print("   ✅ All layouts agree")

        print()
        print("Benchmarking (" + String(steps) + " steps)...")
        print()
        print("Performance Results:")
        var ms = [
            bench[BEAM](ctx, fleet, steps),
            bench[CAR](ctx, fleet, steps),
            bench[TRANSPOSED](ctx, fleet, steps),
            bench[FIXED](ctx, fleet, steps),
        ]
        var all_names = [BEAM_NAME, CAR_NAME, TRANSPOSED_NAME, FIXED_NAME]
        for v in range(4):
            var rays = Float64(n) / (ms[v] * 1e-3)
            var envs = Float64(num_envs) / (ms[v] * 1e-3)
            print(
                "   "
                + pad(all_names[v], 28)
                + " "
                + num(ms[v], 8, 3)
                + " ms/step "
                + num(rays / 1e6, 9, 1)
                + " M rays/s "
                + num(envs / 1e6, 7, 2)
                + " M env-steps/s"
            )
        print(
            "   sphere tracing is "
            + ratio(ms[FIXED], ms[BEAM])
            + " faster than 1 px steps"
        )
        if ms[BEAM] < ms[CAR]:
            print(
                "   "
                + BEAM_NAME
                + " is "
                + ratio(ms[CAR], ms[BEAM])
                + " faster than "
                + CAR_NAME
            )
        else:
            print(
                "   "
                + CAR_NAME
                + " is "
                + ratio(ms[BEAM], ms[CAR])
                + " faster than "
                + BEAM_NAME
            )
        print(
            "   "
            + BEAM_NAME
            + " is "
            + ratio(ms[TRANSPOSED], ms[BEAM])
            + " faster than "
            + TRANSPOSED_NAME
        )

        print()
        print("Key Learning Points:")
        print(
            "• The distance field turns a loop over every pixel of range into a"
            " loop over a handful of lookups"
        )
        print(
            "• Same rays, same results, two thread layouts: 108x more threads"
            " or 108x more work per thread"
        )
        print(
            "• A GPU needs far more than 8k threads in flight to hide memory"
            " latency; a CPU core does not"
        )
        print(
            "• Which 32 rays share a warp decides how long it idles for its"
            " slowest lane; only a GPU shows this"
        )
        print(
            "• The timed loop launches the kernel `steps` times and syncs once,"
            " so it measures the kernel, not the host"
        )
        print(
            "• warporacer's whole step (physics + lidar + PPO bookkeeping) runs"
            " ~7M env-steps/s at 8k envs on one GPU"
        )
        print("Puzzle 06 complete ✅")
