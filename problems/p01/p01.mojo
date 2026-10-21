from max.gpu import thread_idx
from max.gpu.host import DeviceContext
from layout import TileTensor
from layout.tile_layout import row_major
from std.testing import assert_equal

from harness.canary import PuzzleMemory

comptime dtype = DType.float32

# ANCHOR: px_to_m
comptime SIZE = 4
comptime RES: Scalar[dtype] = 0.05  # meters per pixel, the F1TENTH map default
comptime layout = row_major[SIZE]()
comptime Layout = type_of(layout)


def px_to_m(
    output: TileTensor[mut=True, dtype, Layout, MutAnyOrigin],
    edt: TileTensor[mut=False, dtype, Layout, ImmutAnyOrigin],
    res: Scalar[dtype],
):
    var i = thread_idx.x
    # FILL ME IN (roughly 1 line)


# ANCHOR_END: px_to_m


def main() raises:
    with DeviceContext() as ctx:
        var mem = PuzzleMemory[dtype](ctx)
        var out = mem.output(SIZE)
        var edt = ctx.enqueue_create_buffer[dtype](SIZE)
        with edt.map_to_host() as edt_host:
            for i in range(SIZE):
                edt_host[i] = Scalar[dtype](i)

        var out_tensor = TileTensor(out, layout)
        var edt_tensor = TileTensor(edt, layout)
        ctx.enqueue_function[px_to_m](
            out_tensor, edt_tensor, RES, grid_dim=1, block_dim=SIZE
        )
        ctx.synchronize()

        # The same conversion on the CPU, in float32 like the kernel.
        var expected = ctx.enqueue_create_host_buffer[dtype](SIZE)
        with edt.map_to_host() as edt_host:
            for i in range(SIZE):
                expected[i] = edt_host[i] * RES
        with out.map_to_host() as out_host:
            print("out:", out_host)
            print("expected:", expected)
            for i in range(SIZE):
                assert_equal(out_host[i], expected[i])
        mem.verify()
        print("Puzzle 01 complete ✅")
