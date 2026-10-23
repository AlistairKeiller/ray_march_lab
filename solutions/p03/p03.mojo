from max.gpu import thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from layout import TileTensor, TensorLayout
from layout.tile_layout import row_major
from std.math import cos, sin, sqrt
from std.math.constants import pi
from std.testing import assert_almost_equal

from harness.canary import PuzzleMemory

comptime dtype = DType.float32
comptime Vec2 = SIMD[dtype, 2]

# ANCHOR: raycast_solution
comptime SIZE = 16  # a SIZE x SIZE px room with a 1 px wall all round and a 2 x 2 px pillar
comptime NUM_RAYS = 8
comptime MAX_PX: Scalar[
    dtype
] = 8.0  # give up, and report MAX_PX, after marching this far
comptime edt_layout = row_major[SIZE, SIZE]()
comptime rays_layout = row_major[NUM_RAYS, 2]()
comptime out_layout = row_major[NUM_RAYS]()
comptime EdtLayout = type_of(edt_layout)
comptime RaysLayout = type_of(rays_layout)
comptime OutLayout = type_of(out_layout)


def raycast[
    EdtLayoutT: TensorLayout
](
    edt: TileTensor[mut=False, dtype, EdtLayoutT, ImmutAnyOrigin],
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


def march(
    output: TileTensor[mut=True, dtype, OutLayout, MutAnyOrigin],
    edt: TileTensor[mut=False, dtype, EdtLayout, ImmutAnyOrigin],
    starts: TileTensor[
        mut=False, dtype, RaysLayout, ImmutAnyOrigin
    ],  # (col, row) per ray
    dirs: TileTensor[
        mut=False, dtype, RaysLayout, ImmutAnyOrigin
    ],  # unit (col, row) per ray
    max_px: Scalar[dtype],
):
    var i = thread_idx.x
    output[i] = raycast(
        edt,
        Vec2(starts[i, 0], starts[i, 1]),
        Vec2(dirs[i, 0], dirs[i, 1]),
        max_px,
    )


# ANCHOR_END: raycast_solution


def room(ctx: DeviceContext, size: Int) raises -> DeviceBuffer[dtype]:
    """A size x size room with a 1 px wall all round and a 2 x 2 px pillar at rows 4-5,
    cols 10-11: for every pixel, the distance to the nearest wall pixel, centre to centre
    (brute force; warporacer gets this from scipy's distance_transform_edt)."""
    var wall = List[Bool](capacity=size * size)
    for r in range(size):
        for c in range(size):
            var border = r == 0 or r == size - 1 or c == 0 or c == size - 1
            var pillar = 4 <= r < 6 and 10 <= c < 12
            wall.append(border or pillar)
    var edt = ctx.enqueue_create_buffer[dtype](size * size)
    with edt.map_to_host() as host:
        for r in range(size):
            for c in range(size):
                var best = 2 * size * size
                for wr in range(size):
                    for wc in range(size):
                        if wall[wr * size + wc]:
                            best = min(best, (r - wr) ** 2 + (c - wc) ** 2)
                host[r * size + c] = sqrt(Scalar[dtype](best))
    return edt^


def main() raises:
    with DeviceContext() as ctx:
        var mem = PuzzleMemory[dtype](ctx)
        var out = mem.output(NUM_RAYS)
        var edt = room(ctx, SIZE)
        # Every ray starts at the centre of pixel (8, 8), one every 45 degrees.
        var starts = ctx.enqueue_create_buffer[dtype](NUM_RAYS * 2)
        var dirs = ctx.enqueue_create_buffer[dtype](NUM_RAYS * 2)
        with starts.map_to_host() as s, dirs.map_to_host() as d:
            for k in range(NUM_RAYS):
                var angle = Float64(k) * 2 * pi / NUM_RAYS
                s[k * 2] = Scalar[dtype](SIZE // 2) + 0.5
                s[k * 2 + 1] = Scalar[dtype](SIZE // 2) + 0.5
                d[k * 2] = Scalar[dtype](cos(angle))
                d[k * 2 + 1] = Scalar[dtype](sin(angle))

        var out_tensor = TileTensor(out, out_layout)
        var edt_tensor = TileTensor(edt, edt_layout)
        var starts_tensor = TileTensor(starts, rays_layout)
        var dirs_tensor = TileTensor(dirs, rays_layout)
        ctx.enqueue_function[march](
            out_tensor,
            edt_tensor,
            starts_tensor,
            dirs_tensor,
            MAX_PX,
            grid_dim=1,
            block_dim=NUM_RAYS,
        )
        ctx.synchronize()

        # The same march on the CPU, in float32 like the kernel.
        var expected = ctx.enqueue_create_host_buffer[dtype](NUM_RAYS)
        with edt.map_to_host() as e, starts.map_to_host() as s, dirs.map_to_host() as d:
            for k in range(NUM_RAYS):
                var x = s[k * 2]
                var y = s[k * 2 + 1]
                var dist: Scalar[dtype] = 0
                while dist < MAX_PX:
                    var col = Int(x)
                    var row = Int(y)
                    if col < 0 or col >= SIZE or row < 0 or row >= SIZE:
                        break
                    var step = e[row * SIZE + col]
                    if step == 0:
                        break
                    x += d[k * 2] * step
                    y += d[k * 2 + 1] * step
                    dist += step
                expected[k] = min(dist, MAX_PX)

        with out.map_to_host() as out_host:
            print("out:", out_host)
            print("expected:", expected)
            for k in range(NUM_RAYS):
                assert_almost_equal(out_host[k], expected[k], atol=1e-4)
        mem.verify()
        print("Puzzle 03 complete ✅")
