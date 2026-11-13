import os

import numpy as np
import pytest
from conftest import synthetic_volume, write_ome_zarr
from qtpy.QtCore import Qt

from copickmod import project, snapshot
from copickmod.overlays import PointLayer
from copickmod.state import Session
from copickmod.volume import ChunkCache, MultiscaleVolume

SHAPE = (48, 80, 96)


@pytest.fixture
def session(tmp_path, qapp):
    data = synthetic_volume(SHAPE)
    vol = MultiscaleVolume(write_ome_zarr(tmp_path / "v.zarr", [data, data[::2, ::2, ::2]], chunks=(16, 16, 16)))
    picks = PointLayer("ribo", "picks", (0, 255, 0, 255), 3.0, points=np.array([[5.5, 10.5, 10.5], [30.5, 40.5, 50.5]]))
    fil = PointLayer(
        "fil", "filaments", (255, 0, 0, 255), 2.0,
        points=np.array([[20.5, 5.5, 5.5], [22.5, 6.5, 7.5]]), polylines=[np.array([[20.5, 5.5, 5.5], [22.5, 6.5, 7.5]])],
    )  # fmt: skip
    s = Session(vol, ChunkCache(), point_layers=[picks, fil])
    yield s
    s.close()


@pytest.fixture
def info(session, qtbot):
    from copickmod.info import InfoWindow

    win = InfoWindow(session, header="test")
    qtbot.addWidget(win)
    win.show()
    return win


def test_object_navigation_keys(info, session, qtbot):
    zap = info.open_window("zap")
    c = zap.canvas
    assert info.obj_spin.value() == 1
    qtbot.keyClick(c, Qt.Key_BracketRight)  # next point -> first pick
    assert session.point == [5, 10, 10]
    qtbot.keyClick(c, Qt.Key_BracketRight)
    assert session.point == [30, 40, 50]
    qtbot.keyClick(c, Qt.Key_BraceLeft)  # first point
    assert session.point == [5, 10, 10]
    qtbot.keyClick(c, Qt.Key_P)  # next object: the filaments
    assert info.obj_spin.value() == 2 and info.cont_max.text() == "/ 1"
    qtbot.keyClick(c, Qt.Key_BraceRight)  # last point of the filament
    assert session.point == [22, 6, 7]
    qtbot.keyClick(c, Qt.Key_O)
    assert info.obj_spin.value() == 1
    # Shift+PageUp / PageDown: sections holding points of the current object (picks at z 5 and 30)
    session.set_point(z=10)
    qtbot.keyClick(c, Qt.Key_PageUp, Qt.ShiftModifier)
    assert session.point[0] == 30
    qtbot.keyClick(c, Qt.Key_PageDown, Qt.ShiftModifier)
    assert session.point[0] == 5


def test_display_toggles_and_contrast_keys(info, session, qtbot):
    zap = info.open_window("zap")
    c = zap.canvas
    qtbot.keyClick(c, Qt.Key_T)
    assert not session.show_model
    qtbot.keyClick(c, Qt.Key_T, Qt.ShiftModifier)
    assert not session.show_markers
    qtbot.keyClick(c, Qt.Key_T, Qt.ControlModifier)  # current object off
    assert not session.point_layers[0].visible
    qtbot.keyClick(c, Qt.Key_F12)
    assert session.contrast.false_color
    session.contrast.black, session.contrast.white = 1.0, 2.0
    qtbot.keyClick(c, Qt.Key_F10)  # ramp 2 starts from the current setting
    session.contrast.black, session.contrast.white = 5.0, 6.0
    qtbot.keyClick(c, Qt.Key_F9)  # back to ramp 1
    assert (session.contrast.black, session.contrast.white) == (1.0, 2.0)
    qtbot.keyClick(c, Qt.Key_F10)
    assert (session.contrast.black, session.contrast.white) == (5.0, 6.0)


def test_find_max_nearby(session):
    session.set_point(z=10, y=40, x=50)
    y, x, v = session.find_max_nearby(radius=5)
    # the synthetic data grow with x + 100 y: the maximum is at the far edge of the disc in y
    assert (y, x) == (45, 50) and v == pytest.approx(session.volume.read_chunk(0, (0, 2, 3))[10, 13, 2], abs=1)


def test_false_color_table():
    from copickmod.contrast import FALSE_COLOR

    assert FALSE_COLOR.shape == (256, 3)
    assert tuple(FALSE_COLOR[0]) == (100, 75, 200) and tuple(FALSE_COLOR[-1]) == (255, 0, 192)


def test_snapshots(info, session, qtbot, tmp_path):
    from qtpy.QtGui import QImage

    zap = info.open_window("zap")
    qtbot.waitExposed(zap)
    path = snapshot.quick_snapshot(zap, "PNG", directory=str(tmp_path))
    assert os.path.basename(path) == "zap000.png" and os.path.exists(path)
    assert os.path.basename(snapshot.quick_snapshot(zap, "TIFF", directory=str(tmp_path))) == "zap000.tif"
    img, apx = snapshot.compose(zap, scale=2, scale_bar=True, note="hello")
    assert img.width() == 2 * zap.canvas.width() and img.height() > 2 * zap.canvas.height()  # caption strip
    assert apx == pytest.approx(session.volume.voxel_size / (2 * zap.canvas.zoom))
    # Rubber band: only the band area.
    zap.band_button.setChecked(True)
    zap.canvas.band = (10.0, 10.0, 30.0, 50.0)
    img, _ = snapshot.render_window(zap, scale=1, band=True)
    assert abs(img.width() - 40 * zap.canvas.zoom) <= 2 and abs(img.height() - 20 * zap.canvas.zoom) <= 2
    # The tool.
    dialog = info.open_window("snapshot")
    assert dialog.window_combo.count() == 2 and dialog.band.isEnabled()
    dialog.directory = str(tmp_path)
    dialog._suggest_name()
    saved = dialog.save()
    assert saved and QImage(saved).width() > 0
    xyz = info.open_window("xyz")
    qtbot.waitExposed(xyz)
    img, apx = snapshot.render_window(xyz, scale=2)
    assert img.width() == 2 * xyz.canvas.width() and apx > 0


def test_scale_bar_lengths():
    assert snapshot.nice_scale_length(10.0, 500) == 100  # 500 px * 0.2 * 1 nm/px
    assert snapshot.nice_scale_length(10.0, 1300) == 200
    assert snapshot.scale_label(2000) == "2 µm"


def test_choose_tomogram(copick_config):
    _, root = copick_config
    run = root.get_run("TS_001")
    assert project.choose_tomogram(run).tomo_type == "denoised"  # denoised before wbp
    assert project.choose_tomogram(run, "wbp", 10.0).tomo_type == "wbp"  # the current type if the run has it
    assert project.object_kind(run.picks[0]) == "picks"
    assert project.object_kind(run.segmentations[0]) == "segmentation"


def test_browser_and_switching(copick_config, qtbot):
    from copickmod.app import Viewer, ViewOptions

    _, root = copick_config
    run = project.get_run(root, "TS_001")
    sel = project.resolve(root, run, ["wbp", "ribosome:*/*"])
    viewer = Viewer(root, ViewOptions())
    info = viewer.open(sel)
    qtbot.addWidget(info)
    browser = viewer.show_browser()
    qtbot.addWidget(browser)
    assert browser.runs.tree.topLevelItemCount() == 2
    picks = browser.tables["picks"]
    assert picks.tree.topLevelItem(0).checkState(0) == Qt.Checked  # loaded from the command line
    qtbot.waitUntil(lambda: picks.tree.topLevelItem(0).text(3) == "3", timeout=60000)  # count loaded in the background
    # Show a segmentation from the table: it becomes an overlay and an Info object.
    segs = browser.tables["segmentation"]
    segs.tree.topLevelItem(0).setCheckState(0, Qt.Checked)
    assert len(viewer.session.seg_layers) == 1 and info.obj_max.text() == "/ 2"
    # Search.
    segs.search.setText("multi")
    assert [segs.tree.topLevelItem(i).isHidden() for i in range(2)].count(False) == 1
    # Another tomogram of the same run keeps the overlays; another run clears them.
    denoised = next(t for _, t in project.run_tomograms(run) if t.tomo_type == "denoised")
    viewer.open_tomogram(run, denoised)
    assert viewer.sel.tomogram.tomo_type == "denoised" and len(viewer.session.layers()) == 2
    assert [type(w).__name__ for w in viewer.info.open_windows()] == ["ZapWindow"]
    browser.open_run(root.get_run("TS_002"))
    assert viewer.sel.run.name == "TS_002" and viewer.session.layers() == []
    assert browser.objects_label.text().endswith("TS_002</b>")
    qtbot.addWidget(viewer.info)


def test_thumbnail(copick_config):
    from copickmod.browser import make_thumbnail

    _, root = copick_config
    img, caption = make_thumbnail(root.get_run("TS_001"), size=64)
    assert img.dtype == np.uint8 and max(img.shape) <= 64 and img.std() > 0
    assert caption.startswith("denoised")
