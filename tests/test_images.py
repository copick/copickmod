"""Several tomograms and MRC files per window set; several window sets; the Options that choose between them."""

import os
import subprocess
import sys

import numpy as np
import pytest
from conftest import synthetic_volume, write_mrc, write_ome_zarr
from qtpy.QtCore import Qt

from copickmod import project, settings
from copickmod.mrc import MrcVolume, is_mrc_path
from copickmod.overlays import PointLayer
from copickmod.state import Image, Session
from copickmod.volume import ChunkCache, MultiscaleVolume

SHAPE = (24, 40, 48)


@pytest.fixture
def two_images(tmp_path, qapp):
    """A 10 A OME-Zarr tomogram with a pick, and an MRC file of the same field at 5 A."""
    data = synthetic_volume(SHAPE)
    vol = MultiscaleVolume(write_ome_zarr(tmp_path / "v.zarr", [data, data[::2, ::2, ::2]], chunks=(8, 16, 16)))
    fine = np.repeat(np.repeat(np.repeat(data, 2, 0), 2, 1), 2, 2)
    path = write_mrc(tmp_path / "fine.mrc", fine, voxel=5.0)
    layer = PointLayer(name="p", kind="picks", rgba=(0, 255, 0, 255), radius=3.0, points=np.array([[12.5, 20.5, 24.5]]))
    s = Session(vol, ChunkCache(), point_layers=[layer], image=Image(vol, "wbp@10.0", run=_Run("TS_001")))
    yield s, path, fine
    s.close()


class _Run:
    def __init__(self, name):
        self.name = name


def test_mrc_volume(tmp_path):
    data = np.random.default_rng(0).normal(size=(6, 600, 300)).astype(np.float32)
    path = write_mrc(tmp_path / "a.rec", data, voxel=7.0, origin=(70.0, 0.0, 0.0))
    assert is_mrc_path(path) and not is_mrc_path(str(tmp_path / "missing.mrc"))
    vol = MrcVolume(path)
    assert vol.shape == data.shape and vol.voxel_size == pytest.approx(7.0)
    assert vol.levels[0].translation[2] == pytest.approx(70.0)  # MRC2014: voxel i at origin + i * pixel
    assert vol.nlevels == 2  # one subsampled level: 600 / 2 >= 256, 600 / 4 < 256
    assert vol.header_range == pytest.approx((data.min(), data.max()))  # 3dmod's slider range for files
    cache = ChunkCache()
    plane, missing = vol.plane(cache, 0, 0, 3)
    assert missing is not None  # nothing cached yet: chunks are fetched in the background
    cache.wait(10)
    plane, missing = vol.plane(cache, 0, 0, 3)
    assert missing is None and np.array_equal(plane, data[3])
    coarse = vol.read_chunk(1, (0, 0, 0))
    assert np.array_equal(coarse, data[::2, ::2, ::2][:, :256, :256])  # every 2nd voxel
    thin = MrcVolume(write_mrc(tmp_path / "img.mrc", data[:1]))  # a single image keeps its one section
    assert thin.levels[1].shape == (1, 300, 150)
    thin.close()
    cache.shutdown()
    vol.close()


def test_switching_keeps_place_overlays_and_contrast(two_images, qtbot):
    s, path, fine = two_images
    for z in range(SHAPE[0]):
        s.volume.plane(s.cache, 0, 0, z)
    s.cache.wait(10)
    s.set_point(z=12, y=20, x=24)
    assert s.initial_contrast()
    black0 = s.contrast.black
    seen = []
    s.image_changed.connect(lambda old: seen.append(old))
    assert s.open_files([path]) == 1
    assert s.current == 1 and len(s.images) == 2 and seen and seen[0].shape == SHAPE
    assert s.volume.voxel_size == pytest.approx(5.0)
    assert s.point == [24, 40, 48]  # the same place in angstroms
    pick = s.point_layers[0]
    assert pick.points[0] == pytest.approx([24.5, 40.5, 48.5]) and pick.radius == pytest.approx(6.0)
    assert s.image.run.name == "TS_001"  # a file opened over a run shows the run's overlays
    assert not s.has_contrast and s.auto_pending  # its own contrast, from its own data
    qtbot.waitUntil(
        lambda: (s.volume.plane(s.cache, 0, 0, 24), s.cache.wait(1), s.initial_contrast())[-1],
        timeout=20000,
    )
    assert s.step_image(1) and s.current == 0  # wraps around
    assert s.point == [12, 20, 24] and s.contrast.black == black0
    assert s.point_layers[0].points[0] == pytest.approx([12.5, 20.5, 24.5])
    assert s.open_files([path]) == 1 and len(s.images) == 2  # already open: shown, not added again
    assert s.open_files([str(path) + ".nope"]) == 0


def test_other_run_has_its_own_overlays(two_images):
    s, path, _ = two_images
    vol = s.volume
    s.add_image(Image(vol, "wbp@10.0", run=_Run("TS_002")))
    assert s.point_layers == []
    s.show_image(0)
    assert [layer.name for layer in s.point_layers] == ["p"]


def test_zap_switches_with_arrows(two_images, qtbot):
    from copickmod.zap import ZapWindow

    s, path, _ = two_images
    win = ZapWindow(s)
    qtbot.addWidget(win)
    win.resize(300, 300)
    win.show()
    qtbot.waitExposed(win)
    assert not win.image_bar.isVisibleTo(win)  # one tomogram: no chooser row
    zoom = win.canvas.zoom
    s.open_files([path])
    assert win.image_bar.isVisibleTo(win) and win.image_combo.count() == 2 and win.image_combo.currentIndex() == 1
    assert win.canvas.zoom == pytest.approx(zoom * 0.5)  # 5 A voxels: same field of view
    assert win.z_slider.maximum() == 2 * SHAPE[0] and "[2/2]" in win.windowTitle()
    qtbot.keyClick(win.canvas, Qt.Key_Right)
    assert s.current == 0 and win.canvas.zoom == pytest.approx(zoom)
    qtbot.keyClick(win.canvas, Qt.Key_2)  # 3dmod's next time
    assert s.current == 1
    settings.put("keys/arrows_switch", False)
    x = s.point[2]
    qtbot.keyClick(win.canvas, Qt.Key_Right)  # now the 3dmod meaning: move the point
    assert s.current == 1 and s.point[2] == x + 1
    win.close()


def test_viewer_window_sets(copick_config, qtbot):
    from copickmod.app import Viewer, ViewOptions

    _, root = copick_config
    run1, run2 = project.get_run(root, "TS_001"), project.get_run(root, "TS_002")
    viewer = Viewer(root, ViewOptions())
    first = viewer.open(project.resolve(root, run1, ["wbp", "ribosome:*/*"]))
    qtbot.addWidget(first)
    first.move(120, 80)
    zap = next(w for w in first.open_windows() if type(w).__name__ == "ZapWindow")
    zap.resize(300, 250)
    zap.move(300, 200)
    denoised = next(t for _, t in project.run_tomograms(run1) if t.tomo_type == "denoised")
    # Another tomogram of the run: into its ZaP.
    assert viewer.open_tomogram(run1, denoised) is first
    assert len(first.session.images) == 2 and first.session.image.tomogram.tomo_type == "denoised"
    assert len(first.session.point_layers) == 1  # the run's picks stay
    # Opening it again only switches to it.
    wbp = first.session.images[0].tomogram
    assert viewer.open_tomogram(run1, wbp) is first and first.session.current == 0
    # Another run: a new Info and ZaP window, a little down and right of the current ones.
    tomo2 = project.choose_tomogram(run2)
    second = viewer.open_tomogram(run2, tomo2)
    qtbot.addWidget(second)
    assert second is not first and viewer.infos == [first, second] and viewer.active is second
    from qtpy.QtCore import QPoint

    from copickmod.app import CASCADE

    step = QPoint(CASCADE, CASCADE)  # no window manager offscreen: the default title bar height
    assert second.pos() == first.pos() + step  # cascaded, so the windows underneath stay visible
    zap2 = next(w for w in second.open_windows() if type(w).__name__ == "ZapWindow")
    assert zap2.pos() == zap.pos() + step and zap2.size() == zap.size()
    assert second.session.point_layers == []
    # Options: other runs into the current ZaP.
    settings.put("open/other_run", settings.SAME_ZAP)
    viewer.set_active(first)
    third_run = [t for _, t in project.run_tomograms(run2) if t is not tomo2]
    if third_run:
        assert viewer.open_tomogram(run2, third_run[0]) is second  # run 2 is open: its own ZaP
    # Closing a set: the others stay, and the last one used becomes current.
    viewer.set_active(first)
    first.close()
    assert viewer.infos == [second] and viewer.active is second
    second.close()
    assert viewer.infos == [] and viewer.active is None


def test_option_new_windows_for_same_run(copick_config, qtbot):
    from copickmod.app import Viewer, ViewOptions

    _, root = copick_config
    run1, run2 = project.get_run(root, "TS_001"), project.get_run(root, "TS_002")
    settings.put("open/same_run", settings.NEW_WINDOWS)
    settings.put("open/other_run", settings.SAME_ZAP)
    viewer = Viewer(root, ViewOptions())
    first = viewer.open(project.resolve(root, run1, ["wbp", "ribosome:*/*"]))
    qtbot.addWidget(first)
    denoised = next(t for _, t in project.run_tomograms(run1) if t.tomo_type == "denoised")
    second = viewer.open_tomogram(run1, denoised)
    qtbot.addWidget(second)
    assert second is not first
    assert len(second.session.point_layers) == 1  # shown overlays carry over to the run's new windows
    # Another run goes into the current ZaP, with that run's (no) overlays.
    assert viewer.open_tomogram(run2, project.choose_tomogram(run2)) is second
    assert len(second.session.images) == 2 and second.session.point_layers == []
    second.session.show_image(0)
    assert len(second.session.point_layers) == 1
    for info in list(viewer.infos):
        info.close()


def test_window_positions_remembered(copick_config, qtbot):
    from copickmod.app import Viewer, ViewOptions

    _, root = copick_config
    run = project.get_run(root, "TS_001")
    viewer = Viewer(root, ViewOptions())
    info = viewer.open(project.resolve(root, run, ["wbp"]))
    info.move(150, 90)
    zap = next(w for w in info.open_windows() if type(w).__name__ == "ZapWindow")
    zap.move(400, 260)
    info.close()
    assert (settings.window_pos("info").x(), settings.window_pos("zap").x()) == (150, 400)
    viewer2 = Viewer(root, ViewOptions())
    info2 = viewer2.open(project.resolve(root, run, ["wbp"]))
    qtbot.addWidget(info2)
    zap2 = next(w for w in info2.open_windows() if type(w).__name__ == "ZapWindow")
    assert info2.pos().x() == 150 and zap2.pos().x() == 400
    info2.close()


def test_float_setting_is_remembered(two_images, qtbot):
    from copickmod.info import InfoWindow

    s, _, _ = two_images
    info = InfoWindow(s)
    qtbot.addWidget(info)
    assert not s.float_on and not info.float_box.isChecked()  # 3dmod: off at first
    info.float_box.setChecked(True)
    assert s.float_on and settings.get("contrast/float")
    info.close()


def test_cli_opens_mrc_without_project(tmp_path):
    data = synthetic_volume((8, 32, 40))
    path = write_mrc(tmp_path / "t.mrc", data)
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", XDG_CONFIG_HOME=str(tmp_path))
    env.pop("CFG", None)
    env.pop("COPICK_CONFIG", None)
    out = tmp_path / "snap.png"
    proc = subprocess.run(
        [sys.executable, "-m", "copickmod", "--snapshot", str(out), path],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr
    assert (tmp_path / "snap_zap.png").exists() and (tmp_path / "snap_info.png").exists()
    bad = subprocess.run(
        [sys.executable, "-m", "copickmod", "-W", str(tmp_path / "missing.mrc")],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert bad.returncode == 1 and "No such file" in bad.stderr
