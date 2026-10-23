from max.gpu import thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from layout import TileTensor
from layout.tile_layout import row_major
from std.testing import assert_equal

from harness.canary import PuzzleMemory

comptime dtype = DType.float32
comptime Vec2 = SIMD[dtype, 2]

# ANCHOR: clearance_solution
comptime NUM_CARS = 4
comptime SIZE = 8  # the map is a SIZE x SIZE px room with a 1 px wall all round
comptime RES: Scalar[dtype] = 0.5  # m/px, coarse so the room fits in a print
comptime ORIGIN = Vec2(-2.0, -2.0)  # world x, y of the bottom-left pixel
comptime CAR_W: Scalar[dtype] = 0.42  # m, the F1TENTH's width
comptime CAR_L: Scalar[dtype] = 0.58  # m, and length
comptime HALF_DIAG: Scalar[dtype] = 0.35805028  # m, hypot(CAR_W, CAR_L) / 2
comptime cars_layout = row_major[NUM_CARS, 3]()
comptime edt_layout = row_major[SIZE, SIZE]()
comptime out_layout = row_major[NUM_CARS]()
comptime CarsLayout = type_of(cars_layout)
comptime EdtLayout = type_of(edt_layout)
comptime OutLayout = type_of(out_layout)


def clearance(
    output: TileTensor[mut=True, dtype, OutLayout, MutAnyOrigin],
    cars: TileTensor[mut=False, dtype, CarsLayout, ImmutAnyOrigin],  # x, y, psi
    edt: TileTensor[
        mut=False, dtype, EdtLayout, ImmutAnyOrigin
    ],  # (row, col) px distance to nearest wall
    res: Scalar[dtype],
    ox: Scalar[dtype],  # world x, y of the bottom-left pixel
    oy: Scalar[dtype],
):
    var i = thread_idx.x
    var col = Int((cars[i, 0] - ox) / res).clamp(0, Int(edt.dim[1]()) - 1)
    var row = Int(
        Scalar[dtype](Int(edt.dim[0]()) - 1) - (cars[i, 1] - oy) / res
    ).clamp(0, Int(edt.dim[0]()) - 1)
    output[i] = edt[row, col] * res - HALF_DIAG


# ANCHOR_END: clearance_solution


def room(ctx: DeviceContext, size: Int) raises -> DeviceBuffer[dtype]:
    """Distance-to-wall raster of a square room: free inside, wall on the border. The
    nearest wall pixel is always straight across, so the distance is the smaller of the
    four offsets to the border."""
    var edt = ctx.enqueue_create_buffer[dtype](size * size)
    with edt.map_to_host() as host:
        for r in range(size):
            for c in range(size):
                host[r * size + c] = Scalar[dtype](
                    min(min(r, c), min(size - 1 - r, size - 1 - c))
                )
    return edt^


def main() raises:
    with DeviceContext() as ctx:
        var mem = PuzzleMemory[dtype](ctx)
        var out = mem.output(NUM_CARS)
        var edt = room(ctx, SIZE)
        var cars = ctx.enqueue_create_buffer[dtype](NUM_CARS * 3)
        with cars.map_to_host() as c:
            c[0], c[1], c[2] = 0.0, 0.0, 0.0  # middle of the room
            c[3], c[4], c[5] = (
                1.2,
                -1.4,
                0.0,
            )  # one pixel from the bottom-right corner
            c[6], c[7], c[8] = -0.9, 0.3, 0.0  # two pixels from the left wall
            c[9], c[10], c[11] = 5.0, 9.0, 0.0  # off the map entirely

        var out_tensor = TileTensor(out, out_layout)
        var cars_tensor = TileTensor(cars, cars_layout)
        var edt_tensor = TileTensor(edt, edt_layout)
        ctx.enqueue_function[clearance](
            out_tensor,
            cars_tensor,
            edt_tensor,
            RES,
            ORIGIN[0],
            ORIGIN[1],
            grid_dim=1,
            block_dim=NUM_CARS,
        )
        ctx.synchronize()

        # The same transform on the CPU, in float32 like the kernel.
        var expected = ctx.enqueue_create_host_buffer[dtype](NUM_CARS)
        with cars.map_to_host() as c, edt.map_to_host() as e:
            for i in range(NUM_CARS):
                var col = Int((c[i * 3] - ORIGIN[0]) / RES).clamp(0, SIZE - 1)
                var row = Int(
                    SIZE - 1 - (c[i * 3 + 1] - ORIGIN[1]) / RES
                ).clamp(0, SIZE - 1)
                expected[i] = e[row * SIZE + col] * RES - HALF_DIAG

        print("room (px to wall):")
        with edt.map_to_host() as e:
            for r in range(SIZE):
                var line = String()
                for c in range(SIZE):
                    line += String(Int(e[r * SIZE + c])) + " "
                print(" ", line)
        with out.map_to_host() as out_host:
            print("out:", out_host)
            print("expected:", expected)
            for i in range(NUM_CARS):
                assert_equal(out_host[i], expected[i])
        mem.verify()
        print("Puzzle 02 complete ✅")
