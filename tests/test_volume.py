import numpy as np
import pytest

from copickmod.volume import PLANE_AXES, ChunkCache, MultiscaleVolume, parse_size


def _expected_plane(data, axis, index):
    row_ax, col_ax = PLANE_AXES[axis]
    sheet = np.take(data, index, axis=axis)
    return sheet.T if row_ax > col_ax else sheet


def _complete_plane(vol, cache, *args, **kwargs):
    vol.plane(cache, *args, **kwargs)
    cache.wait(10)
    data, missing = vol.plane(cache, *args, **kwargs)
    assert missing is None
    return data


def test_metadata(ome_group, pyramid):
    vol = MultiscaleVolume(ome_group, name="t")
    assert vol.nlevels == 3
    assert vol.shape == pyramid[0].shape
    assert vol.voxel_size == 10.0
    np.testing.assert_allclose(vol.factor(2), [4, 4, 4])
    assert vol.levels[0].chunks == (16, 16, 16)


@pytest.mark.parametrize("level", [0, 1, 2])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_full_planes_match_numpy(ome_group, pyramid, level, axis):
    vol = MultiscaleVolume(ome_group)
    cache = ChunkCache()
    data = pyramid[level]
    for index in (0, data.shape[axis] // 2, data.shape[axis] - 1):
        got = _complete_plane(vol, cache, level, axis, index)
        np.testing.assert_array_equal(got, _expected_plane(data, axis, index))


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_roi_planes(ome_group, pyramid, axis):
    vol = MultiscaleVolume(ome_group)
    cache = ChunkCache()
    full = _expected_plane(pyramid[0], axis, 17)
    got = _complete_plane(vol, cache, 0, axis, 17, rows=(5, 33), cols=(-4, 21))
    np.testing.assert_array_equal(got, full[5:33, 0:21])


def test_missing_chunks_are_reported_then_filled(ome_group):
    vol = MultiscaleVolume(ome_group)
    cache = ChunkCache()
    _, missing = vol.plane(cache, 0, 0, 3, request=False)
    assert missing is not None and missing.all()
    assert cache.pending() == 0
    _, missing = vol.plane(cache, 0, 0, 3)
    cache.wait(10)
    _, missing = vol.plane(cache, 0, 0, 3)
    assert missing is None


def test_out_of_range_plane(ome_group):
    vol = MultiscaleVolume(ome_group)
    data, missing = vol.plane(ChunkCache(), 0, 0, 999)
    assert missing.all()


def test_gather_matches_fancy_indexing(ome_group, pyramid):
    vol = MultiscaleVolume(ome_group)
    cache = ChunkCache()
    rng = np.random.default_rng(1)
    n = 5000
    zi = rng.integers(-3, 43, n)
    yi = rng.integers(-3, 73, n)
    xi = rng.integers(-3, 93, n)
    vol.gather(cache, 0, zi, yi, xi)
    cache.wait(10)
    vals, ok = vol.gather(cache, 0, zi, yi, xi)
    inside = (zi >= 0) & (zi < 40) & (yi >= 0) & (yi < 70) & (xi >= 0) & (xi < 90)
    np.testing.assert_array_equal(ok, inside)
    np.testing.assert_allclose(vals[ok], pyramid[0][zi[ok], yi[ok], xi[ok]])


def test_gather_many_chunks_uses_sort_path(ome_group, pyramid):
    vol = MultiscaleVolume(ome_group)
    cache = ChunkCache()
    z, y, x = np.indices((40, 70, 90))
    vol.gather(cache, 0, z, y, x)
    cache.wait(10)
    vals, ok = vol.gather(cache, 0, z, y, x)
    assert ok.all()
    np.testing.assert_allclose(vals.reshape(40, 70, 90), pyramid[0])


def test_sample_trilinear_on_ramp(tmp_path):
    from conftest import write_ome_zarr

    z, y, x = np.indices((20, 20, 20), dtype=np.float32)
    ramp = 3 * x + 5 * y + 7 * z
    vol = MultiscaleVolume(write_ome_zarr(tmp_path / "ramp.zarr", [ramp], chunks=(8, 8, 8)))
    cache = ChunkCache()
    # corner coordinates: the centre of voxel i is at i + 0.5
    coords = np.array([[3.5, 4.25, 10.0], [2.5, 7.75, 11.0], [1.5, 1.5, 12.3]]).T  # (3, N) zyx
    vol.sample(cache, 0, coords)
    cache.wait(10)
    vals, ok = vol.sample(cache, 0, coords)
    assert ok.all()
    idx = coords - 0.5
    expected = 3 * idx[2] + 5 * idx[1] + 7 * idx[0]
    np.testing.assert_allclose(vals, expected, rtol=1e-5)
    near, ok = vol.sample(cache, 0, coords, linear=False)
    np.testing.assert_allclose(near, ramp[tuple(np.floor(coords).astype(int))])


def test_lru_budget(ome_group):
    vol = MultiscaleVolume(ome_group)
    one = 16 * 16 * 16 * 4
    cache = ChunkCache(budget_bytes=3 * one)
    for i in range(5):
        cache.fetch(vol, 0, (0, 0, i))
    assert len(cache) == 3
    assert cache.nbytes <= 3 * one
    assert cache.get(vol, 0, (0, 0, 0)) is None
    assert cache.get(vol, 0, (0, 0, 4)) is not None


def test_request_dedupes(ome_group):
    vol = MultiscaleVolume(ome_group)
    cache = ChunkCache(workers=1)
    import threading

    gate = threading.Event()
    orig = vol.read_chunk

    def slow(level, cidx):
        gate.wait(5)
        return orig(level, cidx)

    vol.read_chunk = slow
    f1 = cache.request(vol, 0, (0, 0, 0))
    f2 = cache.request(vol, 0, (0, 0, 0))
    assert f1 is f2
    f3 = cache.request(vol, 0, (0, 0, 1))  # queued behind f1
    assert cache.cancel_except({ChunkCache.key(vol, 0, (0, 0, 0))}) == 1
    assert f3.cancelled()
    gate.set()
    cache.wait(10)
    assert cache.get(vol, 0, (0, 0, 0)) is not None


def test_failed_chunk_reported(ome_group):
    vol = MultiscaleVolume(ome_group)
    cache = ChunkCache()
    errors = []
    cache.on_error = lambda key, exc: errors.append(key)

    def boom(level, cidx):
        raise OSError("network down")

    vol.read_chunk = boom
    cache.request(vol, 0, (0, 0, 0))
    cache.wait(10)
    assert errors and cache.request(vol, 0, (0, 0, 0)) is None  # not retried in a loop


def test_parse_size():
    assert parse_size("2G") == 2 * 1024**3
    assert parse_size("512m") == 512 * 1024**2
    assert parse_size("1.5GB") == int(1.5 * 1024**3)
    assert parse_size("1000") == 1000
