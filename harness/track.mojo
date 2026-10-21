# ===----------------------------------------------------------------------=== #
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""The puzzles' map loader: `harness/track.py`, driven through Python interop.

`Track` wraps the Python `Track` (map yaml + image -> wall-distance field, via SciPy's
distance transform) and keeps a device copy of its field. Loading maps in Python means
both labs read them with exactly the same code, and the Mojo side never parses a PNG.
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.python import Python, PythonObject
from std.python.numpy import from_numpy_array


def to_device[
    dtype: DType
](ctx: DeviceContext, array: PythonObject) raises -> DeviceBuffer[dtype]:
    """Copies a NumPy array of `dtype` into a new device buffer, in C order."""
    var flat = array.ravel()
    var span = from_numpy_array[dtype](flat)
    var buf = ctx.enqueue_create_buffer[dtype](len(span))
    buf.enqueue_copy_from(span)
    ctx.synchronize()  # `flat` goes away when this returns, so the copy must be done
    return buf^


struct Track(Copyable, Movable):
    """One map: `edt` is its (h, w) float32 wall-distance field in px, on the device;
    `res` is m/px and (ox, oy) the world origin of the bottom-left pixel. Image row 0 is
    the top of the map, so rows run against y."""

    comptime dtype = DType.float32

    var py: PythonObject
    """The `harness.track.Track` this wraps."""
    var name: String
    var h: Int
    var w: Int
    var res: Float32
    var ox: Float32
    var oy: Float32
    var edt: DeviceBuffer[Self.dtype]

    def __init__(out self, ctx: DeviceContext, yaml_path: StringSlice) raises:
        """Loads the map at `yaml_path` (a ROS map yaml next to its image)."""
        var track = Python.import_module("harness.track").Track(
            String(yaml_path)
        )
        self = Self(ctx, py=track)

    def __init__(out self, ctx: DeviceContext, *, py: PythonObject) raises:
        """Wraps an already loaded `harness.track.Track`."""
        self.py = py
        self.name = String(py=py.name)
        self.h = Int(py=py.h)
        self.w = Int(py=py.w)
        self.res = Float32(py=py.res)
        self.ox = Float32(py=py.ox)
        self.oy = Float32(py=py.oy)
        self.edt = to_device[Self.dtype](ctx, py.edt)

    def sample_poses(
        self, ctx: DeviceContext, n: Int, rng: PythonObject
    ) raises -> DeviceBuffer[Self.dtype]:
        """n random (x, y, psi) rows at pixel centres at least 0.5 m from a wall.
        """
        return to_device[Self.dtype](ctx, self.py.sample_poses(n, rng))


def stack(
    ctx: DeviceContext, tracks: List[Track]
) raises -> DeviceBuffer[Track.dtype]:
    """Pads `tracks` into one (n_maps, H, W) device raster, H and W the largest map's.
    Padding is zero, i.e. wall, so a ray that leaves a small map's true bounds stops
    there instead of wandering into the padding."""
    var py_tracks = Python.list()
    for t in tracks:
        py_tracks.append(t.py)
    var padded = Python.import_module("harness.track").stack(py_tracks)
    return to_device[Track.dtype](ctx, padded)


def load_tracks(
    ctx: DeviceContext, maps: StringSlice, k: Int, rng: PythonObject
) raises -> List[Track]:
    """Up to k maps from a yaml or a directory of them, sampled like warporacer does.
    """
    var loaded = Python.import_module("harness.track").load_tracks(
        String(maps), k, rng
    )
    var tracks = List[Track]()
    for t in loaded:
        tracks.append(Track(ctx, py=t))
    return tracks^


@fieldwise_init
struct Spawned(Copyable, Movable):
    """A fleet's spawn: `cars` (n, 3) float32 rows of (x, y, psi) and `map_id` (n,) int32,
    the map each car drives on."""

    var cars: DeviceBuffer[DType.float32]
    var map_id: DeviceBuffer[DType.int32]


def spawn(
    ctx: DeviceContext, tracks: List[Track], num_envs: Int, rng: PythonObject
) raises -> Spawned:
    """A map id per car, uniform over `tracks`, and a random drivable pose on that map, the
    way warporacer's Env spawns its fleet."""
    var py_tracks = Python.list()
    for t in tracks:
        py_tracks.append(t.py)
    var spawned = Python.import_module("harness.track").spawn(
        py_tracks, num_envs, rng
    )
    return Spawned(
        to_device[DType.float32](ctx, spawned[0]),
        to_device[DType.int32](ctx, spawned[1]),
    )
