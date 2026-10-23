from max.gpu import block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from layout import Coord, RowMajorLayout, TileTensor
from layout.tile_layout import row_major
from std.math import cos, sin
from std.math.constants import pi
from std.python import Python
from std.testing import assert_almost_equal

from harness.canary import PuzzleMemory
from harness.track import Track

comptime dtype = DType.float32
comptime Vec2 = SIMD[dtype, 2]

# ANCHOR: lidar
comptime MAP = "maps/Spielberg_map.yaml"
comptime NUM_CARS = 4
comptime NUM_LIDAR = 108
comptime LIDAR_FOV = 3 * pi / 2  # 270 degrees
comptime LIDAR_RANGE: Scalar[dtype] = 20.0  # m
comptime LIDAR_MOUNT_X: Scalar[dtype] = 0.2733  # m, base_link -> laser offset
comptime TPB = 128  # one block per car, one thread per beam
comptime EdtLayout = RowMajorLayout[Int, Int]  # (H, W), sizes known at run time
comptime cars_layout = row_major[NUM_CARS, 3]()
comptime dirs_layout = row_major[NUM_LIDAR, 2]()
comptime scan_layout = row_major[NUM_CARS, NUM_LIDAR]()
comptime CarsLayout = type_of(cars_layout)
comptime DirsLayout = type_of(dirs_layout)
comptime ScanLayout = type_of(scan_layout)


def raycast(
    edt: TileTensor[mut=False, dtype, EdtLayout, ImmutAnyOrigin],
    var p: Vec2,
    d: Vec2,
    max_px: Scalar[dtype],
) -> Scalar[dtype]:
    """Sphere-march from pixel `p` along unit direction `d` (col, row); returns
    distance in px."""
    var dist: Scalar[dtype] = 0
    while dist < max_px:
        var col = Int(p[0])
        var row = Int(p[1])
        if (
            col < 0
            or col >= Int(edt.dim[1]())
            or row < 0
            or row >= Int(edt.dim[0]())
        ):
            break
        var step = edt[row, col]
        if step == 0:
            break
        p += d * step
        dist += step
    return min(dist, max_px)


def lidar(
    scan: TileTensor[
        mut=True, dtype, ScanLayout, MutAnyOrigin
    ],  # (car, beam) range in m
    cars: TileTensor[mut=False, dtype, CarsLayout, ImmutAnyOrigin],  # x, y, psi
    edt: TileTensor[mut=False, dtype, EdtLayout, ImmutAnyOrigin],
    lidar_dirs: TileTensor[
        mut=False, dtype, DirsLayout, ImmutAnyOrigin
    ],  # unit beam directions in the car frame
    res: Scalar[dtype],
    ox: Scalar[dtype],  # world x, y of the bottom-left pixel
    oy: Scalar[dtype],
):
    """One block per car and one thread per beam: rays from the mount point, rotated
    into the heading frame (the row axis flips y)."""
    var i = block_idx.y
    var j = thread_idx.x
    if j < NUM_LIDAR:
        var ch = cos(cars[i, 2])
        var sh = sin(cars[i, 2])
        # FILL ME IN (roughly 9 lines)


# ANCHOR_END: lidar


def beam_dirs(ctx: DeviceContext) raises -> DeviceBuffer[dtype]:
    """Unit beam directions in the car frame (x forward, y left), NUM_LIDAR of them
    spread evenly over LIDAR_FOV."""
    var dirs = ctx.enqueue_create_buffer[dtype](NUM_LIDAR * 2)
    with dirs.map_to_host() as d:
        for j in range(NUM_LIDAR):
            var theta = -LIDAR_FOV / 2 + LIDAR_FOV * Float64(j) / Float64(
                NUM_LIDAR - 1
            )
            d[j * 2] = Scalar[dtype](cos(theta))
            d[j * 2 + 1] = Scalar[dtype](sin(theta))
    return dirs^


def every_18th(buf: HostBuffer[dtype], row: Int) -> String:
    """Beams 0, 18, 36, ... of one car's scan, as a list."""
    var s = String("[")
    for j in range(0, NUM_LIDAR, 18):
        s += String(buf[row * NUM_LIDAR + j]) + (
            " " if j + 18 < NUM_LIDAR else "]"
        )
    return s


def main() raises:
    with DeviceContext() as ctx:
        var track = Track(ctx, MAP)
        var np = Python.import_module("numpy")
        var rng = np.random.default_rng(0)
        var cars = track.sample_poses(ctx, NUM_CARS, rng)
        var lidar_dirs = beam_dirs(ctx)

        var mem = PuzzleMemory[dtype](ctx)
        var scan = mem.output(NUM_CARS * NUM_LIDAR)
        var scan_tensor = TileTensor(scan, scan_layout)
        var cars_tensor = TileTensor(cars, cars_layout)
        var edt_tensor = TileTensor(
            track.edt, row_major(Coord(track.h, track.w))
        )
        var dirs_tensor = TileTensor(lidar_dirs, dirs_layout)
        ctx.enqueue_function[lidar](
            scan_tensor,
            cars_tensor,
            edt_tensor,
            dirs_tensor,
            track.res,
            track.ox,
            track.oy,
            grid_dim=(1, NUM_CARS),
            block_dim=(TPB, 1),
        )
        ctx.synchronize()

        # The same scan on the CPU, in float32 like the kernel.
        var expected = ctx.enqueue_create_host_buffer[dtype](
            NUM_CARS * NUM_LIDAR
        )
        var max_px = LIDAR_RANGE / track.res
        with cars.map_to_host() as c, lidar_dirs.map_to_host() as dirs, track.edt.map_to_host() as e:
            for i in range(NUM_CARS):
                var ch = cos(c[i * 3 + 2])
                var sh = sin(c[i * 3 + 2])
                var px = (c[i * 3] + LIDAR_MOUNT_X * ch - track.ox) / track.res
                var py = (
                    Scalar[dtype](track.h - 1)
                    - (c[i * 3 + 1] + LIDAR_MOUNT_X * sh - track.oy) / track.res
                )
                for j in range(NUM_LIDAR):
                    var bx = dirs[j * 2]
                    var by = dirs[j * 2 + 1]
                    var dx = ch * bx - sh * by
                    var dy = -(sh * bx + ch * by)
                    var x = px
                    var y = py
                    var dist: Scalar[dtype] = 0
                    while dist < max_px:
                        var col = Int(x)
                        var row = Int(y)
                        if (
                            col < 0
                            or col >= track.w
                            or row < 0
                            or row >= track.h
                        ):
                            break
                        var step = e[row * track.w + col]
                        if step == 0:
                            break
                        x += dx * step
                        y += dy * step
                        dist += step
                    expected[i * NUM_LIDAR + j] = min(dist, max_px) * track.res

        print(
            "map: "
            + track.name
            + " ("
            + String(track.h)
            + " x "
            + String(track.w)
            + " px, "
            + String(track.res)
            + " m/px)"
        )
        with cars.map_to_host() as c:
            print("car 0 pose (x, y, psi):", c[0], c[1], c[2])
        with scan.map_to_host() as out:
            print("out (car 0, every 18th beam):", every_18th(out, 0))
            print("expected:", every_18th(expected, 0))
            var max_diff: Scalar[dtype] = 0
            for k in range(NUM_CARS * NUM_LIDAR):
                max_diff = max(max_diff, abs(out[k] - expected[k]))
            print("max |out - expected|:", max_diff, "m")
            for k in range(NUM_CARS * NUM_LIDAR):
                assert_almost_equal(
                    out[k], expected[k], atol=Float64(track.res)
                )  # within one pixel
        mem.verify()
        print("Puzzle 04 complete ✅")
