import os
import shutil
import subprocess
import sys

import numpy as np
import pytest
from conftest import VOXEL, synthetic_volume, write_ome_zarr
from qtpy.QtCore import Qt

from copickmod import overlays, project
from copickmod.modelview import STANDARD_VIEWS, Camera, pick_nearest, rot_x, rot_y
from copickmod.state import Session
from copickmod.volume import ChunkCache, MultiscaleVolume


def test_camera_projection_and_rotation():
    cam = Camera((40, 70, 90))  # z, y, x
    np.testing.assert_allclose(cam.center, [45, 35, 20])
    s = cam.to_screen(np.array([[45.0, 35.0, 20.0]]), 400, 300)
    np.testing.assert_allclose(s[0, :2], [200, 150])  # the centre is in the middle of the window
    right = cam.to_screen(np.array([[55.0, 35.0, 20.0]]), 400, 300)
    up = cam.to_screen(np.array([[45.0, 45.0, 20.0]]), 400, 300)
    assert right[0, 0] > 200 and up[0, 1] < 150  # x right, y up, like the ZaP
    for _ in range(50):
        cam.rotate_view(rot_x(7) @ rot_y(-3))
    np.testing.assert_allclose(cam.R @ cam.R.T, np.eye(3), atol=1e-9)


def test_standard_views():
    for name, R in STANDARD_VIEWS.items():
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12, err_msg=name)
        assert np.linalg.det(R) == pytest.approx(1.0), name
    np.testing.assert_allclose(STANDARD_VIEWS["top"], np.eye(3))
    for side in ("front", "back", "left", "right"):  # side views have Z up
        np.testing.assert_allclose(STANDARD_VIEWS[side] @ [0, 0, 1], [0, 1, 0], atol=1e-12, err_msg=side)
    np.testing.assert_allclose(STANDARD_VIEWS["front"] @ [0, -1, 0], [0, 0, 1], atol=1e-12)  # seen from -Y


def test_pick_nearest_prefers_front():
    screen = np.array([[100, 100, -5.0], [102, 101, 3.0], [300, 300, 9.0]])
    assert pick_nearest(screen, 101, 100) == 1  # both close: the nearer one in depth wins
    assert pick_nearest(screen, 200, 200) is None


def test_isosurface_and_cap(tmp_path):
    lab = np.zeros((20, 30, 40), np.uint8)
    lab[5:15, 10:20, 10:30] = 2
    lab[2:4, 2:4, 2:4] = 1
    ref = MultiscaleVolume(write_ome_zarr(tmp_path / "r.zarr", [np.zeros((20, 30, 40), np.float32)]))
    seg = MultiscaleVolume(write_ome_zarr(tmp_path / "s.zarr", [lab]))
    lut = np.zeros((4, 4), np.uint8)
    lut[1] = (0, 255, 0, 255)
    lut[2] = (255, 0, 255, 255)
    meshes = overlays.seg_isosurfaces(overlays.SegLayer("s", seg, lut, (0, 0, 0, 255)), ref)
    assert [m.rgba for m in meshes] == [(0, 255, 0, 255), (255, 0, 255, 255)]
    box = meshes[1]
    np.testing.assert_allclose(box.vertices.min(0), [10, 10, 5], atol=0.6)  # world xyz corners of the label
    np.testing.assert_allclose(box.vertices.max(0), [30, 20, 15], atol=0.6)
    v, n, f = box.vertices, box.normals, box.faces
    assert n[np.argmax(v[:, 0])][0] > 0.5  # outward normals
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    assert np.mean(np.sum(np.cross(b - a, c - a) * n[f[:, 0]], axis=1) > 0) > 0.99  # counter-clockwise outside
    capped = overlays.seg_isosurfaces(overlays.SegLayer("s", seg, lut, (0, 0, 0, 255)), ref, max_faces=200)
    assert len(capped[1].faces) < len(box.faces)


def test_sphere_batch():
    v, n, f = overlays.sphere_batch(np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]), 2.0, 1)
    assert v.shape == (84, 3) and f.max() == 83
    np.testing.assert_allclose(v[:42].mean(0), [3, 2, 1], atol=1e-5)  # xyz centre of the first pick
    np.testing.assert_allclose(np.linalg.norm(v[:42] - [3, 2, 1], axis=1), 2.0, rtol=1e-5)


def test_meshes_are_resolved_and_loaded(copick_config):
    _, root = copick_config
    run = project.get_run(root, "TS_001")
    sel = project.resolve(root, run, ["wbp", "membrane:*/*"])
    assert len(sel.meshes) == 1 and not sel.picks
    assert project.available(run)["mesh"] == ["membrane:alice/1"]
    ref = MultiscaleVolume(sel.tomogram.zarr(), fallback_voxel_size=VOXEL)
    layer = overlays.mesh_layer(root, sel.meshes[0], ref, "membrane:alice/1")
    assert layer.rgba == (255, 0, 255, 255)
    # A 100 A sphere at (450, 350, 200) A is a 10 voxel sphere around voxel centre (45.5, 35.5, 20.5).
    np.testing.assert_allclose(layer.vertices.mean(0), [45.5, 35.5, 20.5], atol=0.2)
    np.testing.assert_allclose(np.linalg.norm(layer.vertices - [45.5, 35.5, 20.5], axis=1), 10.0, rtol=0.02)


@pytest.fixture
def session(tmp_path, qapp):
    data = synthetic_volume((48, 80, 96))
    vol = MultiscaleVolume(write_ome_zarr(tmp_path / "v.zarr", [data, data[::2, ::2, ::2]], chunks=(16, 16, 16)))
    layer = overlays.PointLayer(
        "p",
        "picks",
        (0, 255, 0, 255),
        3.0,
        points=np.array([[24.5, 40.5, 48.5], [10.5, 10.5, 10.5]]),
    )
    s = Session(vol, ChunkCache(), point_layers=[layer])
    yield s
    s.close()


def test_model_view_keys(session, qtbot):
    from copickmod.modelview import ModelViewWindow

    win = ModelViewWindow(session)
    qtbot.addWidget(win)
    win.resize(400, 400)
    win.show()
    c = win.canvas
    qtbot.keyClick(c, Qt.Key_F)
    np.testing.assert_allclose(c.camera.R, STANDARD_VIEWS["front"])
    qtbot.keyClick(c, Qt.Key_T)
    np.testing.assert_allclose(c.camera.R, np.eye(3))
    qtbot.keyClick(c, Qt.Key_8, Qt.KeypadModifier)
    assert not np.allclose(c.camera.R, np.eye(3))
    z = c.camera.zoom
    qtbot.keyClick(c, Qt.Key_Equal)
    assert c.camera.zoom == pytest.approx(z * 1.05)
    qtbot.keyClick(c, Qt.Key_Underscore)
    assert c.camera.zoom == pytest.approx(z * 1.05 / 2)
    assert c.show_planes[0] and not c.show_planes[2]
    qtbot.keyClick(c, Qt.Key_Z)
    qtbot.keyClick(c, Qt.Key_X, Qt.ShiftModifier)
    assert not c.show_planes[0] and c.show_planes[2]
    qtbot.keyClick(c, Qt.Key_U)
    assert c.spin_timer.isActive()
    qtbot.keyClick(c, Qt.Key_U)
    assert not c.spin_timer.isActive()
    # Right click on a pick makes it the current point.
    qtbot.keyClick(c, Qt.Key_T)
    sx, sy, _ = c.camera.to_screen(np.array([[10.5, 10.5, 10.5]]), c.width(), c.height())[0]
    assert c.select_point(sx, sy)
    assert session.point == [10, 10, 10]


@pytest.mark.skipif(shutil.which("xvfb-run") is None, reason="needs xvfb-run for real OpenGL")
def test_model_view_renders_with_opengl(copick_config, tmp_path):
    from qtpy.QtGui import QImage

    cfg, _ = copick_config
    out = tmp_path / "m.png"
    env = {k: v for k, v in os.environ.items() if k != "QT_QPA_PLATFORM"}
    env["CFG"] = cfg
    proc = subprocess.run(
        ["xvfb-run", "-a", "-s", "-screen 0 1280x1024x24", sys.executable, "-m", "copickmod", "--snapshot", str(out),
         "-V", "TS_001", "wbp@10", "ribosome:*/*", "labels:*/*?multilabel=true", "membrane:*/*"],
        env=env, capture_output=True, text=True, timeout=300,
    )  # fmt: skip
    assert proc.returncode == 0, proc.stderr[-2000:]
    img = QImage(str(tmp_path / "m_modelview.png"))
    assert not img.isNull()
    colors = {img.pixel(x, y) & 0xFFFFFF for x in range(0, img.width(), 3) for y in range(0, img.height(), 3)}
    greys = sum(1 for c in colors if (c >> 16) & 255 == (c >> 8) & 255 == c & 255)
    assert greys > 30  # the image plane
    magenta = sum(1 for c in colors if (c >> 16) & 255 > 100 and (c >> 8) & 255 < 60 and c & 255 > 100)
    assert magenta > 3  # the label-2 surface and the membrane mesh are magenta
