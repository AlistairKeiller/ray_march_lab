from max.gpu import block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from layout import Coord, RowMajorLayout, TileTensor
from layout.tile_layout import row_major
from std.math import cos, sin
from std.math.constants import pi
from std.python import Python
from std.testing import assert_almost_equal

from harness.canary import PuzzleMemory
from harness.track import Track, stack, to_device

comptime dtype = DType.float32
comptime Vec2 = SIMD[dtype, 2]

# ANCHOR: lidar_multi
comptime NUM_MAPS = 3  # berlin, skirk and Spielberg, see main()
comptime CARS_PER_MAP = 2
comptime NUM_CARS = CARS_PER_MAP * NUM_MAPS
comptime NUM_LIDAR = 108
comptime LIDAR_FOV = 3 * pi / 2  # 270 degrees
comptime LIDAR_RANGE: Scalar[dtype] = 20.0  # m
comptime LIDAR_MOUNT_X: Scalar[dtype] = 0.2733  # m, base_link -> laser offset
comptime TPB = 128  # one block per car, one thread per beam
comptime StackLayout = RowMajorLayout[
    Int, Int, Int
]  # (n_maps, H, W), sizes known at run time
comptime cars_layout = row_major[NUM_CARS, 3]()
comptime map_id_layout = row_major[NUM_CARS]()
comptime dirs_layout = row_major[NUM_LIDAR, 2]()
comptime origin_layout = row_major[NUM_MAPS, 2]()
comptime per_map_layout = row_major[NUM_MAPS]()
comptime scan_layout = row_major[NUM_CARS, NUM_LIDAR]()
comptime CarsLayout = type_of(cars_layout)
comptime MapIdLayout = type_of(map_id_layout)
comptime DirsLayout = type_of(dirs_layout)
comptime OriginLayout = type_of(origin_layout)
comptime PerMapLayout = type_of(per_map_layout)
comptime ScanLayout = type_of(scan_layout)


def raycast(
    edt: TileTensor[mut=False, dtype, StackLayout, ImmutAnyOrigin],
    mid: Int,
    p: Vec2,
    d: Vec2,
    max_px: Scalar[dtype],
) -> Scalar[dtype]:
    """Sphere-march from pixel `p` along unit direction `d` (col, row); returns
    distance in px. Padding rows/cols are zero (= wall), so rays stop at each map's
    true bounds."""
    var pos = p
    var dist: Scalar[dtype] = 0
    while dist < max_px:
        var col = Int(pos[0])
        var row = Int(pos[1])
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
        pos += d * step
        dist += step
    return min(dist, max_px)


def lidar_multi(
    scan: TileTensor[mut=True, dtype, ScanLayout, MutAnyOrigin],
    cars: TileTensor[mut=False, dtype, CarsLayout, ImmutAnyOrigin],  # x, y, psi
    map_id: TileTensor[
        mut=False, DType.int32, MapIdLayout, ImmutAnyOrigin
    ],  # which map each car drives on
    edt: TileTensor[
        mut=False, dtype, StackLayout, ImmutAnyOrigin
    ],  # (n_maps, H, W), zero-padded to the largest map
    lidar_dirs: TileTensor[mut=False, dtype, DirsLayout, ImmutAnyOrigin],
    origin: TileTensor[
        mut=False, dtype, OriginLayout, ImmutAnyOrigin
    ],  # world origin per map
    res: TileTensor[
        mut=False, dtype, PerMapLayout, ImmutAnyOrigin
    ],  # meters/px per map
    map_h: TileTensor[
        mut=False, DType.int32, PerMapLayout, ImmutAnyOrigin
    ],  # unpadded rows per map
):
    var i = block_idx.y
    var j = thread_idx.x
    if j < NUM_LIDAR:
        # FILL ME IN (roughly 12 lines)
        pass


# ANCHOR_END: lidar_multi


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
        var maps = [
            "maps/berlin.yaml",
            "maps/skirk.yaml",
            "maps/Spielberg_map.yaml",
        ]
        var tracks = List[Track]()
        for k in range(NUM_MAPS):
            tracks.append(Track(ctx, maps[k]))
        var np = Python.import_module("numpy")
        var rng = np.random.default_rng(0)
        var poses = Python.list()
        for k in range(NUM_MAPS):
            poses.append(tracks[k].py.sample_poses(CARS_PER_MAP, rng))
        var cars = to_device[dtype](ctx, np.concatenate(poses))
        var map_id = ctx.enqueue_create_buffer[DType.int32](NUM_CARS)
        with map_id.map_to_host() as m:
            for i in range(NUM_CARS):
                m[i] = Int32(i // CARS_PER_MAP)
        var lidar_dirs = beam_dirs(ctx)

        var edt = stack(ctx, tracks)
        var h = 0
        var w = 0
        for t in tracks:
            h = max(h, t.h)
            w = max(w, t.w)
        var origin = ctx.enqueue_create_buffer[dtype](NUM_MAPS * 2)
        var res = ctx.enqueue_create_buffer[dtype](NUM_MAPS)
        var map_h = ctx.enqueue_create_buffer[DType.int32](NUM_MAPS)
        with origin.map_to_host() as o, res.map_to_host() as r, map_h.map_to_host() as mh:
            for k in range(NUM_MAPS):
                o[k * 2] = tracks[k].ox
                o[k * 2 + 1] = tracks[k].oy
                r[k] = tracks[k].res
                mh[k] = Int32(tracks[k].h)

        var mem = PuzzleMemory[dtype](ctx)
        var scan = mem.output(NUM_CARS * NUM_LIDAR)
        var scan_tensor = TileTensor(scan, scan_layout)
        var cars_tensor = TileTensor(cars, cars_layout)
        var map_id_tensor = TileTensor(map_id, map_id_layout)
        var edt_tensor = TileTensor(edt, row_major(Coord(NUM_MAPS, h, w)))
        var dirs_tensor = TileTensor(lidar_dirs, dirs_layout)
        var origin_tensor = TileTensor(origin, origin_layout)
        var res_tensor = TileTensor(res, per_map_layout)
        var map_h_tensor = TileTensor(map_h, per_map_layout)
        ctx.enqueue_function[lidar_multi](
            scan_tensor,
            cars_tensor,
            map_id_tensor,
            edt_tensor,
            dirs_tensor,
            origin_tensor,
            res_tensor,
            map_h_tensor,
            grid_dim=(1, NUM_CARS),
            block_dim=(TPB, 1),
        )
        ctx.synchronize()

        # The same scans on the CPU, one map at a time, in float32 like the kernel.
        var expected = ctx.enqueue_create_host_buffer[dtype](
            NUM_CARS * NUM_LIDAR
        )
        with cars.map_to_host() as c, lidar_dirs.map_to_host() as dirs:
            for k in range(NUM_MAPS):
                ref t = tracks[k]
                var max_px = LIDAR_RANGE / t.res
                with t.edt.map_to_host() as e:
                    for i in range(k * CARS_PER_MAP, (k + 1) * CARS_PER_MAP):
                        var ch = cos(c[i * 3 + 2])
                        var sh = sin(c[i * 3 + 2])
                        var px = (c[i * 3] + LIDAR_MOUNT_X * ch - t.ox) / t.res
                        var py = (
                            Scalar[dtype](t.h - 1)
                            - (c[i * 3 + 1] + LIDAR_MOUNT_X * sh - t.oy) / t.res
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
                                    or col >= t.w
                                    or row < 0
                                    or row >= t.h
                                ):
                                    break
                                var step = e[row * t.w + col]
                                if step == 0:
                                    break
                                x += dx * step
                                y += dy * step
                                dist += step
                            expected[i * NUM_LIDAR + j] = (
                                min(dist, max_px) * t.res
                            )

        var names = String()
        for k in range(NUM_MAPS):
            ref t = tracks[k]
            names += (
                (", " if k > 0 else "")
                + t.name
                + " ("
                + String(t.h)
                + " x "
                + String(t.w)
                + ")"
            )
        print(
            "edt stack: ("
            + String(NUM_MAPS)
            + ", "
            + String(h)
            + ", "
            + String(w)
            + ") from "
            + names
        )
        var tol: Scalar[dtype] = 0
        for t in tracks:
            tol = max(tol, t.res)
        with scan.map_to_host() as out:
            for k in range(NUM_MAPS):
                var i = k * CARS_PER_MAP
                print(tracks[k].name, "car", i)
                print("  out (every 18th beam):", every_18th(out, i))
                print("  expected:             ", every_18th(expected, i))
            var max_diff: Scalar[dtype] = 0
            for k in range(NUM_CARS * NUM_LIDAR):
                max_diff = max(max_diff, abs(out[k] - expected[k]))
            print("max |out - expected|:", max_diff, "m")
            for k in range(NUM_CARS * NUM_LIDAR):
                assert_almost_equal(out[k], expected[k], atol=Float64(tol))
        mem.verify()
        print("Puzzle 05 complete ✅")
