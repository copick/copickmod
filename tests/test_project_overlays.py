import numpy as np
import pytest
from conftest import VOXEL

from copickmod import overlays, project
from copickmod.project import CmodError, classify_uri
from copickmod.volume import ChunkCache, MultiscaleVolume


@pytest.mark.parametrize(
    "uri,kind",
    [
        ("wbp@10.0", "tomogram"),
        ("wbp", "tomogram"),
        ("wbp-*@*", "tomogram"),
        ("ribosome:*/*", "picks"),
        ("ribosome", "tomogram"),  # bare names are tomogram types; overlays need a colon
        ("ribosome:alice", "picks"),
        ("membrane:alice/1@10.0", "segmentation"),
        ("membrane:*/*?multilabel=true", "segmentation"),
        ("re:ribo.*:.*/.*", "picks"),
    ],
)
def test_classify(uri, kind):
    assert classify_uri(uri) == kind


def test_feature_uri_rejected():
    with pytest.raises(CmodError, match="feature"):
        classify_uri("wbp@10.0:sobel")


def test_open_root_errors(tmp_path):
    with pytest.raises(CmodError, match="export CFG"):
        project.open_root(None)
    with pytest.raises(CmodError, match="not found"):
        project.open_root(str(tmp_path / "nope.json"))


def test_unknown_run_suggests(copick_config):
    _, root = copick_config
    with pytest.raises(CmodError, match="Did you mean: TS_001"):
        project.get_run(root, "TS_01")


def test_resolve_full(copick_config):
    _, root = copick_config
    run = project.get_run(root, "TS_001")
    sel = project.resolve(root, run, ["wbp@10", "ribosome:*/*", "membrane:*/*@10.0", "labels:*/*?multilabel=true"])
    assert sel.tomogram.tomo_type == "wbp"
    assert len(sel.picks) == 1
    assert {s.name for s in sel.segmentations} == {"membrane", "labels"}


def test_resolve_errors(copick_config):
    _, root = copick_config
    run = project.get_run(root, "TS_001")
    with pytest.raises(CmodError, match="has 2 tomograms"):
        project.resolve(root, run, [])
    with pytest.raises(CmodError, match="No tomogram matches"):
        project.resolve(root, run, ["nope@10"])
    with pytest.raises(CmodError, match="No tomogram matches"):
        project.resolve(root, run, ["wbp", "nope@10"])


def test_resolve_several_tomograms(copick_config):
    """Several tomogram URIs, or a glob matching several, open in one ZaP: the first is shown."""
    _, root = copick_config
    run = project.get_run(root, "TS_001")
    sel = project.resolve(root, run, ["*@10"])
    assert [t.tomo_type for t in [sel.tomogram, *sel.more_tomograms]] == ["denoised", "wbp"]
    sel = project.resolve(root, run, ["wbp", "denoised", "wbp"])
    assert sel.tomogram.tomo_type == "wbp" and [t.tomo_type for t in sel.more_tomograms] == ["denoised"]
    with pytest.raises(CmodError, match="No picks, filaments or meshes"):
        project.resolve(root, run, ["wbp", "ghost:*/*"])


def test_available_lists_everything(copick_config):
    _, root = copick_config
    avail = project.available(project.get_run(root, "TS_001"))
    assert sorted(avail["tomogram"]) == ["denoised@10.0", "wbp@10.0"]
    assert avail["picks"] == ["ribosome:alice/1"]
    assert len(avail["segmentation"]) == 2


def _tomo_vol(root):
    run = root.get_run("TS_001")
    tomo = run.get_voxel_spacing(VOXEL).get_tomograms("wbp")[0]
    return MultiscaleVolume(tomo.zarr(), fallback_voxel_size=VOXEL)


def test_picks_to_world_and_spheres(copick_config):
    _, root = copick_config
    run = root.get_run("TS_001")
    ref = _tomo_vol(root)
    layer = overlays.picks_layer(root, run.picks[0], ref, "ribosome:alice/1")
    assert layer.rgba == (0, 255, 0, 255)
    assert layer.radius == pytest.approx(6.0)  # 60 A / 10 A
    # (x, y, z) = (455, 305, 205) A -> voxel centre (45.5, 30.5, 20.5) -> world (z, y, x) = (21, 31, 46)
    np.testing.assert_allclose(layer.points[1], [21.0, 31.0, 46.0])
    rows, cols, radii, on = overlays.sphere_sections(layer.points, layer.radius, 0, 21.0)
    # pick 0 is at z 15.5 (5.5 away, inside the 6-voxel sphere), pick 1 on the plane, pick 2 at z 30.5
    np.testing.assert_allclose(radii, [np.sqrt(36 - 5.5**2), 6.0])
    np.testing.assert_array_equal(on, [False, True])
    rows, cols, radii, on = overlays.sphere_sections(layer.points, layer.radius, 0, 25.0)
    np.testing.assert_allclose(radii, [np.sqrt(36 - 16), np.sqrt(36 - 5.5**2)])
    assert not on.any()
    # XZ plane: rows are z, cols are x
    rows, cols, _, _ = overlays.sphere_sections(layer.points, layer.radius, 1, 31.0)
    assert (rows[0], cols[0]) == (21.0, 46.0)


def test_filaments_layer(copick_config):
    _, root = copick_config
    run = root.get_run("TS_001")
    fils = getattr(run, "filaments", [])
    if not fils:
        pytest.skip("copick without filaments")
    layer = overlays.filaments_layer(root, fils[0], _tomo_vol(root), "actin")
    assert layer.kind == "filaments" and len(layer.polylines) == 1
    segs = overlays.slab_segments(layer.polylines, layer.radius, 0, layer.polylines[0][0, 0])
    assert segs and segs[0].shape[1] == 2


def test_segmentation_planes(copick_config):
    _, root = copick_config
    run = root.get_run("TS_001")
    ref = _tomo_vol(root)
    cache = ChunkCache()
    seg = run.get_segmentations(name="labels", is_multilabel=True)[0]
    layer = overlays.segmentation_layer(root, seg, "labels")
    rows = np.arange(70) + 0.5
    cols = np.arange(90) + 0.5
    overlays.seg_plane_rgba(layer, cache, ref, 0, 15.5, rows, cols, VOXEL)
    cache.wait(10)
    rgba, complete = overlays.seg_plane_rgba(layer, cache, ref, 0, 15.5, rows, cols, VOXEL)
    assert complete
    assert tuple(rgba[25, 40]) == (255, 0, 255, 255)  # label 2 = membrane colour
    assert rgba[5, 5, 3] == 0
    assert rgba[:, :, 3].astype(bool).sum() == 20 * 30
    # Binary segmentation uses the object colour.
    bseg = run.get_segmentations(name="membrane", is_multilabel=False)[0]
    blayer = overlays.segmentation_layer(root, bseg, "membrane")
    rgba, _ = overlays.seg_plane_rgba(blayer, cache, ref, 0, 15.5, rows, cols, VOXEL)
    cache.wait(10)
    rgba, _ = overlays.seg_plane_rgba(blayer, cache, ref, 0, 15.5, rows, cols, VOXEL)
    assert tuple(rgba[25, 40]) == (255, 0, 255, 255)


def test_segmentation_with_different_voxel_size(tmp_path):
    """A 20 A segmentation must line up with a 10 A tomogram through physical coordinates."""
    from conftest import synthetic_volume, write_ome_zarr

    ref = MultiscaleVolume(write_ome_zarr(tmp_path / "t.zarr", [synthetic_volume()], voxel=10.0))
    labels = np.zeros((20, 35, 45), dtype=np.uint8)
    labels[5, 10, 20] = 1
    seg_vol = MultiscaleVolume(write_ome_zarr(tmp_path / "s.zarr", [labels], voxel=20.0))
    layer = overlays.SegLayer(
        name="s",
        volume=seg_vol,
        lut=np.array([[0, 0, 0, 0], [9, 9, 9, 255]], np.uint8),
        rgba=(9, 9, 9, 255),
    )
    cache = ChunkCache()
    rows, cols = np.arange(70) + 0.5, np.arange(90) + 0.5
    overlays.seg_plane_rgba(layer, cache, ref, 0, 10.5, rows, cols, 10.0)
    cache.wait(10)
    rgba, _ = overlays.seg_plane_rgba(layer, cache, ref, 0, 10.5, rows, cols, 10.0)
    on = np.argwhere(rgba[:, :, 3] > 0)
    # seg voxel (z5, y10, x20) at 20 A covers tomogram voxels y 20-21, x 40-41 on z 10-11
    assert sorted(map(tuple, on)) == [(20, 40), (20, 41), (21, 40), (21, 41)]


def test_instance_colors_are_stable():
    layer = overlays.SegLayer(name="i", volume=None, lut=None, rgba=(0, 0, 0, 0))
    a = overlays.colorize(layer, np.array([[0, 7, 7, 9]]))
    assert a[0, 0, 3] == 0 and tuple(a[0, 1]) == tuple(a[0, 2]) and tuple(a[0, 1]) != tuple(a[0, 3])
