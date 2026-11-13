import numpy as np
import pytest
from conftest import synthetic_volume, write_ome_zarr
from qtpy.QtCore import QPoint, Qt

from copickmod.canvas import ZOOMS, zoom_step
from copickmod.overlays import PointLayer
from copickmod.state import Session
from copickmod.volume import ChunkCache, MultiscaleVolume

SHAPE = (48, 80, 96)


@pytest.fixture
def session(tmp_path, qapp):
    data = synthetic_volume(SHAPE)
    vol = MultiscaleVolume(
        write_ome_zarr(tmp_path / "v.zarr", [data, data[::2, ::2, ::2], data[::4, ::4, ::4]], chunks=(16, 16, 16)),
    )
    layer = PointLayer(name="p", kind="picks", rgba=(0, 255, 0, 255), radius=3.0, points=np.array([[24.5, 40.5, 48.5]]))
    s = Session(vol, ChunkCache(), point_layers=[layer])
    yield s
    s.close()


def drag_to(widget, pos):
    """A mouse move with the left button held (QTest.mouseMove sends no button state)."""
    from qtpy.QtCore import QEvent, QPointF
    from qtpy.QtGui import QMouseEvent
    from qtpy.QtWidgets import QApplication

    ev = QMouseEvent(QEvent.MouseMove, QPointF(pos), Qt.NoButton, Qt.LeftButton, Qt.NoModifier)
    QApplication.sendEvent(widget, ev)


def settle(qtbot, session, widget, timeout=15.0):
    """Repaint until the widget has all its data, contrast is final and nothing is being fetched."""
    from qtpy.QtWidgets import QApplication

    win = widget.window()

    def done():
        widget.repaint()
        QApplication.processEvents()
        session.cache.wait(2)
        QApplication.processEvents()
        from copickmod.app import window_complete

        return (
            session.cache.pending() == 0 and session.has_contrast and not session.auto_pending and window_complete(win)
        )

    qtbot.waitUntil(done, timeout=int(timeout * 1000))


def test_zoom_table():
    assert zoom_step(1.0, True) == 1.5 and zoom_step(1.0, False) == 0.75
    assert zoom_step(0.8, False) == 0.75 and zoom_step(0.8, True) == 1
    assert zoom_step(ZOOMS[-1], True) == ZOOMS[-1]


def test_level_choice(session):
    assert session.choose_level(1.0) == 0
    assert session.choose_level(0.75) == 1  # 2x binned shown at 1.5x
    assert session.choose_level(0.3333) == 2
    session.min_level = 1
    assert session.choose_level(2.0) == 1


@pytest.fixture
def zap(session, qtbot):
    from copickmod.zap import ZapWindow

    win = ZapWindow(session)
    qtbot.addWidget(win)
    win.resize(400, 400)
    win.show()
    qtbot.waitExposed(win)
    settle(qtbot, session, win.canvas)
    return win


def test_zap_renders_and_sets_contrast(zap, session, qtbot):
    assert session.has_contrast and not session.auto_pending
    assert zap.canvas.view.complete
    img = zap.canvas.grab().toImage()
    center = img.pixelColor(img.width() // 2 + 20, img.height() // 2 + 20)
    assert center.red() == center.green() == center.blue()  # grey data, not background


def test_zap_section_keys(zap, session, qtbot):
    c = zap.canvas
    z0 = session.point[0]
    qtbot.keyClick(c, Qt.Key_PageUp)
    assert session.point[0] == z0 + 1
    qtbot.keyClick(c, Qt.Key_PageDown)
    qtbot.keyClick(c, Qt.Key_PageDown)
    assert session.point[0] == z0 - 1
    qtbot.keyClick(c, Qt.Key_Home)
    assert session.point[0] == SHAPE[0] - 1
    qtbot.keyClick(c, Qt.Key_End)
    assert session.point[0] == 0
    qtbot.keyClick(c, Qt.Key_Home, Qt.ShiftModifier)
    assert session.point[0] == SHAPE[0] // 2
    qtbot.keyClick(c, Qt.Key_Asterisk, Qt.KeypadModifier)
    assert session.point[0] == SHAPE[0] // 2 + 10
    assert zap.z_edit.text() == str(session.point[0] + 1)  # toolbar follows
    qtbot.keyClick(c, Qt.Key_Right)
    qtbot.keyClick(c, Qt.Key_Up)
    assert session.point[2] == SHAPE[2] // 2 + 1 and session.point[1] == SHAPE[1] // 2 + 1


def test_zap_zoom_keys(zap, qtbot):
    c = zap.canvas
    z = c.zoom
    qtbot.keyClick(c, Qt.Key_Equal)
    assert c.zoom == zoom_step(z, True)
    qtbot.keyClick(c, Qt.Key_Minus)
    qtbot.keyClick(c, Qt.Key_Minus)
    assert c.zoom == zoom_step(z, False)
    assert zap.zoom_edit.text() == f"{c.zoom:.4g}"


def test_contrast_keys(zap, session, qtbot):
    c = zap.canvas
    b, w = session.contrast.black_slider, session.contrast.white_slider
    qtbot.keyClick(c, Qt.Key_F2)
    assert session.contrast.black_slider == b + 3
    qtbot.keyClick(c, Qt.Key_F3)
    assert session.contrast.white_slider == w - 3
    qtbot.keyClick(c, Qt.Key_F11)
    assert session.contrast.invert
    qtbot.keyClick(c, Qt.Key_A, Qt.ShiftModifier)
    assert (session.contrast.black_slider, session.contrast.white_slider) != (b + 3, w - 3)


def test_click_sets_point(zap, session, qtbot):
    c = zap.canvas
    z = session.point[0]
    q = c.view.to_screen(c.rect_f(), c.zoom, c.center, 30.5, 20.5)  # world (y, x) -> screen
    pos = QPoint(int(q.x()), int(q.y()))  # the pixel containing that point
    qtbot.mouseClick(c, Qt.LeftButton, pos=pos)
    assert session.point == [z, 30, 20]


def test_fitting_image_stays_centered(zap, qtbot):
    """3dmod: an image smaller than the window cannot be dragged around."""
    c = zap.canvas
    before = list(c.center)
    qtbot.mousePress(c, Qt.LeftButton, pos=QPoint(100, 100))
    drag_to(c, QPoint(130, 100))
    drag_to(c, QPoint(160, 120))
    qtbot.mouseRelease(c, Qt.LeftButton, pos=QPoint(160, 120))
    assert c.center == pytest.approx(before)
    assert c.center == pytest.approx([SHAPE[1] / 2, SHAPE[2] / 2], abs=1)


def test_drag_pans_like_3dmod(zap, session, qtbot):
    c = zap.canvas
    c.set_zoom(8.0)  # 768 x 640 image in a ~400 pixel window
    z = session.point
    cy, cx = c.center
    qtbot.mousePress(c, Qt.LeftButton, pos=QPoint(100, 100))
    drag_to(c, QPoint(110, 100))
    drag_to(c, QPoint(120, 95))
    qtbot.mouseRelease(c, Qt.LeftButton, pos=QPoint(120, 95))
    # At zoom >= 1, 3dmod moves the image one image pixel per screen pixel of mouse motion.
    assert c.center[1] == pytest.approx(cx - 20)
    assert c.center[0] == pytest.approx(cy - 5)
    assert session.point == z  # a drag does not move the point
    # Dragging far stops with at most 1/6 of the window showing past the image edge.
    qtbot.mousePress(c, Qt.LeftButton, pos=QPoint(50, 50))
    drag_to(c, QPoint(60, 50))
    for x in range(100, 2000, 100):
        drag_to(c, QPoint(x, 50))
    qtbot.mouseRelease(c, Qt.LeftButton, pos=QPoint(1900, 50))
    left_edge = c.view.to_screen(c.rect_f(), c.zoom, c.center, 0, 0).x()
    assert 0 < left_edge <= c.width() / 6 + 1


def test_wheel_does_nothing(zap, session, qtbot):
    from qtpy.QtCore import QPointF
    from qtpy.QtGui import QWheelEvent
    from qtpy.QtWidgets import QApplication

    c = zap.canvas
    zoom, point = c.zoom, list(session.point)
    ev = QWheelEvent(
        QPointF(50, 50),
        QPointF(50, 50),
        QPoint(0, 0),
        QPoint(0, 120),
        Qt.NoButton,
        Qt.NoModifier,
        Qt.NoScrollPhase,
        False,
    )
    QApplication.sendEvent(c, ev)
    assert c.zoom == zoom and session.point == point


def test_keypad_moves_image_and_no_auto_follow(zap, session, qtbot):
    c = zap.canvas
    c.set_zoom(8.0)
    cy, cx = c.center
    qtbot.keyClick(c, Qt.Key_Left, Qt.KeypadModifier)  # the image moves left, so the view moves right
    assert c.center[1] == pytest.approx(cx + 5)
    before = list(c.center)
    session.set_point(y=2, x=2)  # far off screen: in movie mode the ZaP does not follow
    assert c.center == pytest.approx(before)
    session.center_on_point.emit()  # going to a model point does bring it into view
    q = c.view.to_screen(c.rect_f(), c.zoom, c.center, 2.5, 2.5)
    assert c.rect_f().contains(q)
    qtbot.keyClick(c, Qt.Key_K)  # keep centered
    session.set_point(y=40, x=50)
    assert c.center == pytest.approx([40.5, 50.5], abs=1)


def test_movie(zap, session, qtbot):
    z = session.point[0]
    qtbot.mouseClick(zap.canvas, Qt.MiddleButton, pos=QPoint(50, 50))
    qtbot.waitUntil(lambda: session.point[0] != z, timeout=20000)
    qtbot.mouseClick(zap.canvas, Qt.MiddleButton, pos=QPoint(50, 50))
    assert not session.movie_timer.isActive()


def test_movie_bounces_at_the_ends(session):
    """3dmod's default movie goes back and forth: the end section shows for one more frame, then it turns."""
    n = session.volume.shape[0]
    session.set_point(z=n - 2)
    session.toggle_movie(1, 0)
    seen = []
    for _ in range(4):
        session._movie_step()
        seen.append(session.point[0])
    assert seen == [n - 1, n - 1, n - 2, n - 3]
    session.toggle_movie(1, 0)  # either direction key stops a running movie
    assert not session.movie_timer.isActive()


def test_movie_rate_follows_3dmod(session):
    from copickmod.state import movie_interval

    assert session.movie_timer.interval() == 50  # 3dmod default: 20 frames/s
    for _ in range(3):
        session.movie_speed(faster=True)
    assert session.movie_timer.interval() == 25  # three steps double the rate
    for _ in range(100):
        session.movie_speed(faster=True)
    assert 1000 / movie_interval(session.movie_rate) <= 200  # 3dmod MAXRATE
    for _ in range(200):
        session.movie_speed(faster=False)
    assert 1000 / movie_interval(session.movie_rate) >= 0.2  # MINRATE
    assert session.movie_timer.interval() > 4000


def test_movie_reads_ahead(zap, session, qtbot):
    settle(qtbot, session, zap.canvas)
    session.set_point(z=20)  # chunks are 16 deep: the next layer starts at z = 32
    settle(qtbot, session, zap.canvas)
    level = zap.canvas.view.level
    nxt = int(session.volume.level_index(level, session.volume, 32.5, 0)) // session.volume.levels[level].chunks[0]
    keys = [k for k in session.cache._lru if k[1] == level and k[2] == nxt]
    assert not keys
    session.toggle_movie(1, 0)
    zap.canvas.repaint()
    session.stop_movie()
    session.cache.wait(5)
    assert [k for k in session.cache._lru if k[1] == level and k[2] == nxt]


def test_float_applied_before_repaint(session):
    session.cache.wait(1)
    for z in range(SHAPE[0]):
        for level in range(session.volume.nlevels):
            session.volume.plane(session.cache, level, 0, z)
    session.cache.wait(10)
    assert session.initial_contrast()
    session.float_on = True
    session.apply_float()  # sets the Float reference section
    seen = []
    session.point_changed.connect(lambda: seen.append(session.contrast.black))
    before = session.contrast.black
    session.set_point(z=session.point[0] + 10)
    assert seen and seen[0] != before  # contrast already floated when views are told to repaint


def test_info_window(session, qtbot):
    from copickmod.info import InfoWindow

    info = InfoWindow(session, header="Image size 96 x 80, 48 sections.")
    qtbot.addWidget(info)
    info.show()
    zap = info.open_window("zap")
    settle(qtbot, session, zap.canvas)
    session.set_point(z=5, y=6, x=7)
    assert [info.point_spins[a].value() for a in (2, 1, 0)] == [8, 7, 6]
    assert "Image size" in info.output.toPlainText()
    info.black_slider.setValue(info.white_slider.value() + 10)  # pushing black past white drags white along
    assert session.contrast.white_slider == session.contrast.black_slider + 1
    assert info.black_pos.text() == str(session.contrast.black_slider)
    # Object / Point go to a pick.
    assert info.obj_spin.value() == 1 and info.point_max.text() == "/ 1"
    info.point_spin.setValue(1)
    assert session.point == [24, 40, 48]
    # The Object List turns overlays on and off.
    info.show_object_list()
    info.object_list.list.item(0).setCheckState(Qt.Unchecked)
    assert not session.point_layers[0].visible
    info.object_list.close()
    assert info.open_window("xyz") is info.open_window("xyz")  # only one XYZ window
    session.message.emit("hello")
    assert "hello" in info.output.toPlainText()


def test_float_keeps_appearance(tmp_path, qapp, qtbot):
    """3dmod Float: changing section rescales black/white so mean and SD keep their grey levels."""
    from copickmod.contrast import stats

    rng = np.random.default_rng(3)
    data = rng.normal(0, 1, (8, 64, 64)).astype(np.float32)
    data *= (1 + 0.1 * np.arange(8, dtype=np.float32))[:, None, None]  # SD grows with Z
    data += 0.5 * np.arange(8, dtype=np.float32)[:, None, None]  # and so does the mean
    vol = MultiscaleVolume(write_ome_zarr(tmp_path / "f.zarr", [data], chunks=(8, 64, 64)))
    s = Session(vol, ChunkCache())
    s.float_on = True  # off at first, as in 3dmod
    s.cache.fetch(vol, 0, (0, 0, 0))
    s.set_point(z=1)
    assert s.initial_contrast()
    s.apply_float()  # sets the reference section
    b0, w0 = s.contrast.black, s.contrast.white
    st1 = stats(data[1])
    s.set_point(z=6)
    s.apply_float()
    st6 = stats(data[6])
    ratio = st6.sd / st1.sd
    black = st6.mean - (st1.mean - b0) * ratio
    white = black + (w0 - b0) * ratio
    assert s._float_bw == pytest.approx((black, white), rel=1e-4)  # the reference is kept unrounded
    step = (s.contrast.hi - s.contrast.lo) / 255  # the display uses whole slider steps, like 3dmod
    lo, hi = s.contrast.lo, s.contrast.hi  # values beyond the slider range are clamped, as in 3dmod
    assert s.contrast.black == pytest.approx(min(max(black, lo), hi), abs=step / 2 + 1e-6)
    assert s.contrast.white == pytest.approx(min(max(white, lo), hi), abs=step / 2 + 1e-6)
    s.set_float(False)
    b6 = s.contrast.black
    s.set_point(z=2)
    s.apply_float()
    assert s.contrast.black == b6
    s.cache.shutdown()


@pytest.fixture
def xyz(session, qtbot):
    from copickmod.xyz import XyzWindow

    win = XyzWindow(session)
    qtbot.addWidget(win)
    win.fit_to_image()
    win.show()
    qtbot.waitExposed(win)
    settle(qtbot, session, win.canvas)
    return win


def _gl_to_qt(c, L, x, y):
    """The Qt pixel containing a GL position."""
    return QPoint(int(np.floor(x)), int(np.floor(L.H - y)))


def test_xyz_layout_like_3dmod(xyz):
    from copickmod.xyz import ALL_BORDER, BSIZE, GSIZE

    c = xyz.canvas
    L = c.layout_geometry()
    nz, ny, nx = SHAPE
    assert L.xd1 + L.xd2 == L.W - ALL_BORDER and L.yd1 + L.yd2 == L.H - ALL_BORDER
    assert L.xd2 == L.yd2  # square Z panels
    assert BSIZE + GSIZE + L.xd1 == L.X2
    # The Z share follows nz / (n + nz) in the limiting direction (the other is widened to keep Z square).
    shares = (L.xd2 / (L.xd1 + L.xd2), L.yd2 / (L.yd1 + L.yd2))
    targets = (nz / (nx + nz), nz / (ny + nz))
    assert min(abs(a - b) for a, b in zip(shares, targets)) < 0.03
    assert all(v.complete for v in c.views.values())
    # Side panels show image data, not background.
    img = c.grab().toImage()
    col = img.pixelColor(L.panel(2).center().toPoint() + QPoint(7, 9))  # off the current-point cross
    assert col.red() == col.green() == col.blue()


def test_xyz_clicks_and_gutters(xyz, session, qtbot):
    from copickmod.xyz import BOX_AXIS, X_GADGET_BOX, Y_SLICE_BOX

    c = xyz.canvas
    L = c.layout_geometry()
    z = c.zoom
    y0 = session.point[1]
    # Click in the XZ panel at x = 10, z = 40.
    pos = _gl_to_qt(c, L, L.bx + 10.5 * z, L.by2 + 40.5 * z)
    assert c.getxyz(pos.x(), pos.y())[0] == Y_SLICE_BOX
    qtbot.mouseClick(c, Qt.LeftButton, pos=pos)
    assert session.point == [40, y0, 10]
    # Click in the top gutter sets X only.
    pos = _gl_to_qt(c, L, L.bx + 30.5 * z, L.Y2 - GS // 2)
    assert c.getxyz(pos.x(), pos.y())[0] == X_GADGET_BOX
    qtbot.mouseClick(c, Qt.LeftButton, pos=pos)
    assert session.point == [40, y0, 30]
    # PageUp over the YZ panel moves X; the arrows there follow the panel's axes (left/right = Z).
    hover = L.panel(2).center()
    c._mouse = hover
    qtbot.keyClick(c, Qt.Key_PageUp)
    assert session.point[2] == 31
    qtbot.keyClick(c, Qt.Key_Right)
    assert session.point[0] == 41
    assert BOX_AXIS[Y_SLICE_BOX] == 1


GS = 16


def test_xyz_drag_z_gadget_and_fraction(xyz, session, qtbot):
    c = xyz.canvas
    L = c.layout_geometry()
    z = c.zoom
    # Drag the Z cross in the top-right square.
    start = _gl_to_qt(c, L, L.X2 + 2, L.Y2 + 2)
    qtbot.mousePress(c, Qt.LeftButton, pos=start)
    target_z = 30
    end = _gl_to_qt(c, L, L.bx2 + target_z * z, L.by2 + target_z * z)
    drag_to(c, end)
    qtbot.mouseRelease(c, Qt.LeftButton, pos=end)
    assert abs(session.point[0] - target_z) <= 1
    # Drag the blue box: the Z panels get bigger.
    frac = c.xz_fraction
    box = _gl_to_qt(c, L, L.X2 - GS // 2 - 1, L.Y2 - GS // 2 - 1)
    qtbot.mousePress(c, Qt.LeftButton, pos=box)
    drag_to(c, QPoint(box.x() - 20, box.y() + 20))
    qtbot.mouseRelease(c, Qt.LeftButton, pos=QPoint(box.x() - 20, box.y() + 20))
    assert c.xz_fraction > frac


def test_xyz_panning_is_shared(xyz, session, qtbot):
    c = xyz.canvas
    c.set_zoom(8.0)
    L = c.layout_geometry()
    bx, by, bx2 = L.bx, L.by, L.bx2
    start = L.panel(0).center().toPoint()
    qtbot.mousePress(c, Qt.LeftButton, pos=start)
    drag_to(c, QPoint(start.x() + 10, start.y()))
    drag_to(c, QPoint(start.x() + 20, start.y() + 10))
    qtbot.mouseRelease(c, Qt.LeftButton, pos=QPoint(start.x() + 20, start.y() + 10))
    L2 = c.layout_geometry()
    assert L2.bx > bx  # XY moved right (so does XZ, which shares X)
    assert L2.by < by  # and down (so does YZ, which shares Y)
    assert L2.bx2 == bx2  # Z untouched
    c.center_on_point()  # K
    L3 = c.layout_geometry()
    cz, cy, cx = session.point
    # Centred to within 3dmod's integer rounding (two image pixels).
    assert abs((L3.bx + (cx + 0.5) * c.zoom) - (L3.X1 + L3.xd1 / 2)) <= 2 * c.zoom + 1


def test_slicer(session, qtbot):
    from copickmod.slicer import SlicerWindow

    win = SlicerWindow(session)
    qtbot.addWidget(win)
    win.show()
    qtbot.waitExposed(win)
    c = win.canvas
    settle(qtbot, session, c)
    assert c.complete
    qtbot.keyClick(c, Qt.Key_8, Qt.KeypadModifier)
    assert c.angles[0] == pytest.approx(0.5)
    assert win.angle_sliders[0].value() == 5
    win.angle_sliders[2].setValue(300)  # Z slider -> last adjusted angle is Z
    assert c.angles[2] == pytest.approx(30.0) and c.last_angle == 2
    qtbot.keyClick(c, Qt.Key_3, Qt.KeypadModifier)
    assert c.angles[2] == pytest.approx(45.0)
    qtbot.keyClick(c, Qt.Key_0, Qt.KeypadModifier)
    assert c.angles[2] == 0.0
    qtbot.keyClick(c, Qt.Key_Plus)
    assert c.thickness == 2 and win.thick_spin.value() == 2
    c.angles = [0.0, 0.0, 0.0]
    z = session.point[0]
    qtbot.keyClick(c, Qt.Key_PageUp)
    assert session.point[0] == z + 1
    settle(qtbot, session, c)
    assert c.complete


def test_slicer_panning_and_following(session, qtbot):
    from copickmod.slicer import SlicerWindow

    win = SlicerWindow(session)
    qtbot.addWidget(win)
    win.show()
    qtbot.waitExposed(win)
    c = win.canvas
    center = c.center.copy()
    # Arrow keys shift the image a pixel in their direction: the centre moves the other way.
    qtbot.keyClick(c, Qt.Key_Left)
    assert c.center[2] == pytest.approx(center[2] + 1)
    # Dragging moves the centre of rotation (follows the mouse below zoom 4).
    c.set_zoom(2.0)
    qtbot.mousePress(c, Qt.LeftButton, pos=QPoint(100, 100))
    drag_to(c, QPoint(110, 100))
    drag_to(c, QPoint(130, 100))
    qtbot.mouseRelease(c, Qt.LeftButton, pos=QPoint(130, 100))
    assert c.center[2] == pytest.approx(center[2] + 1 - 15)
    # New mode: follows a Z change and single X/Y steps made elsewhere, not a jump to a new point.
    before = c.center.copy()
    session.move(0, 3)
    assert c.center[0] == pytest.approx(before[0] + 3)
    session.move(2, 1)
    assert c.center[2] == pytest.approx(before[2] + 1)
    after = c.center.copy()
    session.set_point(y=5, x=5)
    assert c.center == pytest.approx(after)
    # Classic mode (K) keeps the slice centred on the current point.
    qtbot.keyClick(c, Qt.Key_K)
    assert c.center == pytest.approx(np.asarray(session.point) + 0.5)


def test_snapshot_cli(copick_config, tmp_path):
    import os
    import subprocess
    import sys

    cfg, _ = copick_config
    out = tmp_path / "snap.png"
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", CFG=cfg)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "copickmod",
            "--snapshot",
            str(out),
            "-xyz",
            "-S",
            "-Z",
            "TS_001",
            "wbp@10",
            "ribosome:*/*",
            "labels:*/*?multilabel=true",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr
    from qtpy.QtGui import QImage

    for kind in ("info", "zap", "xyz", "slicer"):
        path = tmp_path / f"snap_{kind}.png"
        assert path.exists(), proc.stdout + proc.stderr
        img = QImage(str(path))
        colors = {img.pixel(x, y) for x in range(0, img.width(), 7) for y in range(0, img.height(), 7)}
        assert len(colors) > 20, kind  # not blank


def test_closed_windows_do_not_receive_signals(session, qtbot):
    from copickmod.info import InfoWindow

    info = InfoWindow(session, header="t")
    qtbot.addWidget(info)
    info.show()
    for kind in ("zap", "xyz", "slicer"):
        win = info.open_window(kind)
        qtbot.waitExposed(win)
        win.close()
    qtbot.wait(50)  # let deleteLater run
    assert info.open_windows() == []
    session.set_point(z=3)
    session.step_contrast(3, 0)
    session.set_opacity(0.2)
    session.data_arrived.emit()
    qtbot.wait(50)


def test_reopen_windows_after_closing(session, qtbot):
    """Regression: closing a window, then opening another from the Image menu, touched the deleted window."""
    from copickmod.info import InfoWindow

    info = InfoWindow(session, header="t")
    qtbot.addWidget(info)
    info.show()
    zap = info.open_window("zap")
    qtbot.waitExposed(zap)
    zap.close()
    qtbot.wait(50)  # the ZaP is deleted (WA_DeleteOnClose)
    image_menu = next(a.menu() for a in info.menuBar().actions() if a.text() == "&Image")
    for action in image_menu.actions():  # ZaP, XYZ, Slicer, Model View, as a user would click them
        action.trigger()
    kinds = sorted(type(w).__name__ for w in info.open_windows())
    assert kinds == ["ModelViewWindow", "SlicerWindow", "XyzWindow", "ZapWindow"]
    for w in info.open_windows():
        w.close()
    qtbot.wait(50)
    info.open_window("xyz")  # again after everything was closed
    assert [type(w).__name__ for w in info.open_windows()] == ["XyzWindow"]


def test_zap_toolbar_like_3dmod(zap, session, qtbot):
    from qtpy.QtWidgets import QToolBar

    assert isinstance(zap.toolbar, QToolBar)
    assert zap.size_label.text().strip() == f"{zap.canvas.width()}x{zap.canvas.height()}"  # window size, as 3dmod
    # Lock: the ZaP keeps its section; PageUp in it moves only its own section.
    z = session.point[0]
    zap.lock_button.setChecked(True)
    session.set_point(z=z + 5)
    assert zap.canvas.shown_point()[0] == z and zap.z_edit.text() == str(z + 1)
    qtbot.keyClick(zap.canvas, Qt.Key_PageUp)
    assert zap.canvas.locked_z == z + 1 and session.point[0] == z + 5
    zap.lock_button.setChecked(False)
    assert zap.canvas.shown_point()[0] == z + 5


def test_zap_rubber_band_and_arrow(zap, session, qtbot):
    c = zap.canvas
    zap.band_button.setChecked(True)
    qtbot.mousePress(c, Qt.LeftButton, pos=QPoint(100, 100))
    drag_to(c, QPoint(120, 110))
    drag_to(c, QPoint(160, 140))
    qtbot.mouseRelease(c, Qt.LeftButton, pos=QPoint(160, 140))
    band = c.band_screen_rect()
    assert band is not None and 55 <= band.width() <= 65 and 35 <= band.height() <= 45
    assert zap.size_label.text().strip() == f"{int(band.width())}x{int(band.height())}"
    zoom = c.zoom
    qtbot.keyClick(c, Qt.Key_R, Qt.ShiftModifier | Qt.ControlModifier)  # zoom the band to fill the window
    assert c.zoom > zoom and c.band is None and not zap.band_button.isChecked()
    zap.arrow_button.setChecked(True)
    qtbot.mousePress(c, Qt.LeftButton, pos=QPoint(50, 50))
    drag_to(c, QPoint(60, 60))
    drag_to(c, QPoint(90, 80))
    qtbot.mouseRelease(c, Qt.LeftButton, pos=QPoint(90, 80))
    assert c.arrow is not None
    zap.arrow_button.setChecked(False)
    assert c.arrow is None


def test_xyz_lock_and_sum(xyz, session, qtbot):
    c = xyz.canvas
    xyz.lock_button.setChecked(True)
    before = list(c.point)
    session.set_point(z=1, y=2, x=3)
    assert c.point == before  # a locked XYZ ignores the global point
    c._mouse = c.layout_geometry().panel(0).center()
    qtbot.keyClick(c, Qt.Key_PageUp)
    assert c.point[0] == before[0] + 1 and session.point == [1, 2, 3]
    xyz.lock_button.setChecked(False)
    assert c.point == [1, 2, 3]
    xyz.sum_spin.setValue(5)
    assert all(v.thickness == 5 for v in c.views.values())
    settle(qtbot, session, c)
    assert all(v.complete for v in c.views.values())


def test_info_icon_row(session, qtbot):
    from copickmod.info import InfoWindow

    info = InfoWindow(session, header="x")
    qtbot.addWidget(info)
    info.show()
    zap = info.open_window("zap")
    info.show_box.setChecked(False)  # "Show" off: choosing a point does not move the current point
    before = list(session.point)
    info.point_spin.setValue(1)
    assert session.point == before
    info.show_box.setChecked(True)
    info.point_spin.setValue(0)
    info.point_spin.setValue(1)
    assert session.point == [24, 40, 48]
    info.pin_button.setChecked(True)
    assert info.windowFlags() & Qt.WindowStaysOnTopHint
    info.raise_all()
    assert zap.isVisible()
