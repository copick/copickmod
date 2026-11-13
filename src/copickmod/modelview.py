"""The Model View window: a 3D view of picks, filaments, meshes, segmentation surfaces and image planes.

Modelled on 3dmod's Model View (imodv): fixed-function OpenGL with an orthographic camera, black background and a
light at the viewer; first-button drag moves the model, second-button drag rotates it, Shift + third button rotates
around the viewing axis, the wheel zooms, a third-button click selects the nearest point; the keypad rotates,
T/B/F/K/L/R show the standard views and Z, Shift+X, Shift+Y toggle the image planes through the current point.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, Tuple

import numpy as np
from qtpy.QtCore import QSize, Qt, QTimer, Signal
from qtpy.QtGui import QSurfaceFormat
from qtpy.QtWidgets import QAction, QMainWindow, QOpenGLWidget

from copickmod import overlays
from copickmod.contrast import gray_to_rgb
from copickmod.snapshot import quick_snapshot, snapshot_format
from copickmod.state import Session
from copickmod.style import CURPOINT, keep_alive

log = logging.getLogger(__name__)

PLANE_MAX_PIXELS = 1024  # image planes are drawn from the finest level at most this many pixels across
MAX_SPHERE_VERTICES = 3_000_000  # lower the sphere quality for very large pick sets
ZOOM_SMALL, ZOOM_BIG = 1.05, 2.0  # 3dmod: -/= and _/+


# ---------------------------------------------------------------------------------------------- view math
def rot_x(deg: float) -> np.ndarray:
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(deg: float) -> np.ndarray:
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(deg: float) -> np.ndarray:
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


#: Standard views (model xyz -> view xyz; view x right, y up, z toward the viewer), as 3dmod's T/B/F/K/L/R keys.
STANDARD_VIEWS = {
    "top": np.eye(3),  # looking down Z, like the ZaP
    "bottom": rot_x(180),
    "front": rot_x(-90),  # from -Y, Z up
    "back": np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=float),  # from +Y, Z up
    "left": np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]], dtype=float),  # from -X, Z up
    "right": np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float),  # from +X, Z up
}


class Camera:
    """Orthographic view: ``view = R @ (p - center) + trans``; the window shows ``radius / zoom`` vertically."""

    def __init__(self, shape_zyx):
        nz, ny, nx = shape_zyx
        self.center = np.array([nx, ny, nz], dtype=float) / 2.0
        self.radius = 0.5 * float(np.linalg.norm([nx, ny, nz]))
        self.R = np.eye(3)
        self.trans = np.zeros(3)
        self.zoom = 1.0

    def half_extents(self, width: int, height: int) -> Tuple[float, float]:
        half_h = self.radius / self.zoom
        return half_h * width / max(height, 1), half_h

    def pixel_size(self, height: int) -> float:
        return 2.0 * self.radius / self.zoom / max(height, 1)

    def to_view(self, pts_xyz: np.ndarray) -> np.ndarray:
        return (np.asarray(pts_xyz, dtype=float) - self.center) @ self.R.T + self.trans

    def to_screen(self, pts_xyz: np.ndarray, width: int, height: int) -> np.ndarray:
        """Screen (x, y) in Qt pixels and view depth (larger is nearer), shape (N, 3)."""
        v = self.to_view(pts_xyz)
        hw, hh = self.half_extents(width, height)
        sx = (v[:, 0] / hw * 0.5 + 0.5) * width
        sy = (0.5 - v[:, 1] / hh * 0.5) * height
        return np.stack([sx, sy, v[:, 2]], axis=1)

    def rotate_view(self, R_inc: np.ndarray) -> None:
        """Rotate the model about axes fixed to the screen."""
        self.R = R_inc @ self.R
        # keep it orthonormal
        u, _, vt = np.linalg.svd(self.R)
        self.R = u @ vt

    def modelview_matrix(self) -> np.ndarray:
        """Column-major 4x4 for glLoadMatrix."""
        M = np.eye(4)
        M[:3, :3] = self.R
        M[:3, 3] = self.trans - self.R @ self.center
        return M.T.astype(np.float64)


def pick_nearest(screen: np.ndarray, x: float, y: float, max_dist: float = 10.0) -> Optional[int]:
    """Index of the frontmost point within ``max_dist`` pixels of (x, y), as 3dmod's Model View point selection."""
    if len(screen) == 0:
        return None
    d2 = (screen[:, 0] - x) ** 2 + (screen[:, 1] - y) ** 2
    near = np.flatnonzero(d2 <= max_dist**2)
    if not len(near):
        return None
    return int(near[np.argmax(screen[near, 2])])


# ---------------------------------------------------------------------------------------------- the GL widget
class ModelViewGL(QOpenGLWidget):
    iso_ready = Signal(object, object)  # (segmentation layer, (generation, list of MeshLayer or exception))
    status_changed = Signal(str)

    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        fmt = QSurfaceFormat()
        fmt.setDepthBufferSize(24)
        self.setFormat(fmt)
        self.session = session
        self.camera = Camera(session.volume.shape)
        self.show_planes = {0: True, 1: False, 2: False}  # XY (Z key), XZ (Shift+Y), YZ (Shift+X)
        self.show_point = True
        self.show_bbox = True
        self.sphere_quality = 2
        self.delta_rot = 2.0  # degrees per keypad step
        self._spin = (1, 1.0)  # axis and direction of continuous rotation (Keypad Enter / 5 / U)
        self.gl_error: Optional[str] = None
        self._sphere_cache: Dict[int, tuple] = {}
        self._gpu: Dict[tuple, tuple] = {}  # uploaded triangle buffers
        self._textures: Dict[int, tuple] = {}  # axis -> (texture id, key)
        self._iso: Dict[int, object] = {}  # id(seg layer) -> list[MeshLayer] | "pending" | Exception
        self._iso_gen = 0  # surfaces made for an image shown before are dropped
        self._iso_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cmod-iso")
        self._mouse = None
        self._press = None
        self._moved = False
        self.spin_timer = QTimer(self)
        self.spin_timer.setInterval(40)
        self.spin_timer.timeout.connect(self._spin_step)
        self.iso_ready.connect(self._iso_done)
        session.layers_changed.connect(self._start_isosurfaces)
        session.image_changed.connect(self._image_changed)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(200, 200)
        for sig in (
            session.point_changed,
            session.contrast_changed,
            session.overlays_changed,
            session.data_arrived,
        ):
            sig.connect(self.update)
        self._start_isosurfaces()

    def sizeHint(self) -> QSize:
        return QSize(640, 640)

    # ------------------------------------------------------------------ isosurfaces (background)
    def _start_isosurfaces(self) -> None:
        for layer in self.session.seg_layers:
            if id(layer) in self._iso:
                continue
            self._iso[id(layer)] = "pending"
            fut = self._iso_pool.submit(overlays.seg_isosurfaces, layer, self.session.volume)
            gen = self._iso_gen
            fut.add_done_callback(
                lambda f, lay=layer, g=gen: self.iso_ready.emit(lay, (g, f.exception() or f.result())),
            )

    def _iso_done(self, layer, packed) -> None:
        gen, result = packed
        if gen != self._iso_gen:
            return  # made in the coordinates of an image that is no longer shown
        self._iso[id(layer)] = result
        if isinstance(result, Exception):
            self.session.message.emit(f"Model View: could not make a surface for {layer.name}: {result}")
        self.update()

    def _release(self, arrays) -> None:
        entry = self._gpu.pop(tuple(id(a) for a in arrays), None)
        if entry is not None:
            for buf in entry[:3]:
                buf.delete()

    def _image_changed(self, old) -> None:
        """Another tomogram is shown: same orientation and zoom; surfaces and buffers are made again."""
        camera = Camera(self.session.volume.shape)
        camera.R, camera.zoom = self.camera.R, self.camera.zoom
        camera.trans = self.camera.trans * (old.voxel_size / self.session.volume.voxel_size)
        self.camera = camera
        self._iso_gen += 1
        self._iso.clear()
        self._sphere_cache.clear()
        if self.gl_error is None and (self._textures or self._gpu) and self.isValid():
            from OpenGL import GL

            self.makeCurrent()
            for entry in self._gpu.values():
                for buf in entry[:3]:
                    buf.delete()
            GL.glDeleteTextures([t for t, _ in self._textures.values()])
            self.doneCurrent()
        self._gpu.clear()
        self._textures.clear()
        self._start_isosurfaces()
        self.update()

    def pending_isosurfaces(self) -> int:
        return sum(1 for v in self._iso.values() if isinstance(v, str))

    # ------------------------------------------------------------------ GL
    def initializeGL(self) -> None:
        try:
            from OpenGL import GL

            GL.glEnable(GL.GL_DEPTH_TEST)
            GL.glShadeModel(GL.GL_SMOOTH)
            GL.glEnable(GL.GL_NORMALIZE)
            GL.glLightModeli(GL.GL_LIGHT_MODEL_TWO_SIDE, GL.GL_TRUE)
            GL.glLightfv(GL.GL_LIGHT0, GL.GL_AMBIENT, (0.25, 0.25, 0.25, 1.0))
            GL.glLightfv(GL.GL_LIGHT0, GL.GL_DIFFUSE, (0.8, 0.8, 0.8, 1.0))
            GL.glLightfv(GL.GL_LIGHT0, GL.GL_SPECULAR, (0.3, 0.3, 0.3, 1.0))
            GL.glMaterialfv(GL.GL_FRONT_AND_BACK, GL.GL_SPECULAR, (0.3, 0.3, 0.3, 1.0))
            GL.glMaterialf(GL.GL_FRONT_AND_BACK, GL.GL_SHININESS, 20.0)
            GL.glColorMaterial(GL.GL_FRONT_AND_BACK, GL.GL_AMBIENT_AND_DIFFUSE)
        except Exception as exc:  # noqa: BLE001 - report instead of crashing without OpenGL
            self.gl_error = f"OpenGL is not available: {exc}"
            log.warning(self.gl_error)
            self.session.message.emit(self.gl_error)

    def paintGL(self) -> None:
        if self.gl_error is None:
            try:
                self._draw_gl()
            except Exception as exc:  # noqa: BLE001
                self.gl_error = f"OpenGL drawing failed: {exc}"
                log.exception("Model View drawing failed")
                self.session.message.emit(self.gl_error)
        self.status_changed.emit(self.status_text())

    def status_text(self) -> str:
        if self.gl_error:
            return self.gl_error
        if self.pending_isosurfaces():
            return "Computing segmentation surfaces..."
        return ""

    def _draw_gl(self, size: Optional[Tuple[int, int]] = None) -> None:
        from OpenGL import GL

        dpr = self.devicePixelRatioF()
        w, h = size if size is not None else (int(self.width() * dpr), int(self.height() * dpr))
        cam = self.camera
        # Fixed-function drawing: make sure no shader program or generic attribute array is left active.
        GL.glUseProgram(0)
        for i in range(8):
            GL.glDisableVertexAttribArray(i)
        GL.glViewport(0, 0, w, h)
        GL.glClearColor(0.0, 0.0, 0.0, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDisable(GL.GL_CULL_FACE)
        GL.glDisable(GL.GL_BLEND)
        GL.glDepthMask(GL.GL_TRUE)

        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        hw, hh = cam.half_extents(w, h)
        depth = 4.0 * cam.radius + float(np.abs(cam.trans[2]))
        GL.glOrtho(-hw, hw, -hh, hh, -depth, depth)
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_POSITION, (0.0, 0.0, 1.0, 0.0))  # directional, from the viewer
        GL.glLoadMatrixd(cam.modelview_matrix())

        opaque, transparent = self._surfaces() if self.session.show_model else ([], [])
        GL.glEnable(GL.GL_LIGHTING)
        GL.glEnable(GL.GL_LIGHT0)
        GL.glEnable(GL.GL_COLOR_MATERIAL)
        for rgba, verts, normals, faces in opaque:
            self._draw_triangles(rgba, verts, normals, faces)
        GL.glDisable(GL.GL_LIGHTING)

        if self.session.show_model:
            self._draw_filaments()
        self._draw_planes()
        if self.show_bbox:
            self._draw_bbox()
        if self.show_point and self.session.show_markers:
            self._draw_current_point()

        if transparent:
            GL.glEnable(GL.GL_BLEND)
            # Blend colour only: QOpenGLWidget composites its framebuffer as premultiplied alpha, so the
            # destination alpha must stay 1 or Qt shows garbage colours.
            GL.glBlendFuncSeparate(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA, GL.GL_ZERO, GL.GL_ONE)
            GL.glDepthMask(GL.GL_FALSE)
            GL.glEnable(GL.GL_LIGHTING)
            for rgba, verts, normals, faces in transparent:
                self._draw_triangles(rgba, verts, normals, faces)
            GL.glDisable(GL.GL_LIGHTING)
            GL.glDepthMask(GL.GL_TRUE)
            GL.glDisable(GL.GL_BLEND)
        GL.glDisable(GL.GL_COLOR_MATERIAL)

    def _surfaces(self):
        """Lit triangle sets to draw: (rgba, vertices, normals, faces), split by transparency."""
        s = self.session
        items = []
        for layer in s.seg_layers:
            if not layer.visible:
                continue
            result = self._iso.get(id(layer))
            if isinstance(result, list):
                items += [(m.rgba, m.vertices, m.normals, m.faces) for m in result]
        for layer in s.mesh_layers:
            if layer.visible and len(layer.faces):
                items.append((layer.rgba, layer.vertices, layer.normals, layer.faces))
        for layer in s.point_layers:
            if layer.visible and layer.kind == "picks" and len(layer.points):
                items.append((layer.rgba, *self._spheres(layer)))
        opaque = [it for it in items if it[0][3] >= 255]
        transparent = [it for it in items if it[0][3] < 255]
        return opaque, transparent

    def _spheres(self, layer):
        quality = self.sphere_quality
        tv, _ = overlays.sphere_template(quality)
        while quality > 0 and len(layer.points) * len(tv) > MAX_SPHERE_VERTICES:
            quality -= 1
            tv, _ = overlays.sphere_template(quality)
        key = id(layer)
        cached = self._sphere_cache.get(key)
        if cached is None or cached[0] != quality:
            if cached is not None:
                self._release(cached[1:])
            cached = (quality, *overlays.sphere_batch(layer.points, layer.radius, quality))
            self._sphere_cache[key] = cached
        return cached[1:]

    def _buffers(self, verts, normals, faces):
        """GPU buffers for a triangle set, uploaded once (llvmpipe redraws a million triangles far faster)."""
        from OpenGL import GL
        from OpenGL.arrays import vbo

        key = (id(verts), id(normals), id(faces))
        entry = self._gpu.get(key)
        if entry is None:
            entry = (
                vbo.VBO(np.ascontiguousarray(verts, dtype=np.float32)),
                vbo.VBO(np.ascontiguousarray(normals, dtype=np.float32)),
                vbo.VBO(np.ascontiguousarray(faces, dtype=np.uint32), target=GL.GL_ELEMENT_ARRAY_BUFFER),
                int(faces.size),
                (verts, normals, faces),  # keep the arrays (and so their ids) alive
            )
            self._gpu[key] = entry
        return entry

    def _draw_triangles(self, rgba, verts, normals, faces) -> None:
        from OpenGL import GL

        vv, vn, vi, count, _ = self._buffers(verts, normals, faces)
        GL.glColor4ub(*[int(c) for c in rgba[:3]], int(rgba[3]))
        GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
        GL.glEnableClientState(GL.GL_NORMAL_ARRAY)
        vv.bind()
        GL.glVertexPointer(3, GL.GL_FLOAT, 0, vv)
        vn.bind()
        GL.glNormalPointer(GL.GL_FLOAT, 0, vn)
        vi.bind()
        GL.glDrawElements(GL.GL_TRIANGLES, count, GL.GL_UNSIGNED_INT, None)
        vi.unbind()
        vn.unbind()
        vv.unbind()
        GL.glDisableClientState(GL.GL_NORMAL_ARRAY)
        GL.glDisableClientState(GL.GL_VERTEX_ARRAY)

    def _draw_filaments(self) -> None:
        from OpenGL import GL

        GL.glLineWidth(2.0)
        GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
        for layer in self.session.point_layers:
            if not layer.visible or layer.kind != "filaments":
                continue
            GL.glColor4ub(*[int(c) for c in layer.rgba[:3]], 255)
            for line in layer.polylines:
                xyz = np.ascontiguousarray(line[:, ::-1], dtype=np.float32)
                GL.glVertexPointer(3, GL.GL_FLOAT, 0, xyz)
                GL.glDrawArrays(GL.GL_LINE_STRIP, 0, len(xyz))
        GL.glDisableClientState(GL.GL_VERTEX_ARRAY)
        GL.glLineWidth(1.0)

    def _draw_bbox(self) -> None:
        from OpenGL import GL

        nz, ny, nx = self.session.volume.shape
        c = np.array(
            [[x, y, z] for z in (0, nz) for y in (0, ny) for x in (0, nx)],
            dtype=np.float32,
        )
        edges = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
        GL.glColor3ub(110, 110, 110)
        GL.glBegin(GL.GL_LINES)
        for a, b in edges:
            GL.glVertex3fv(c[a])
            GL.glVertex3fv(c[b])
        GL.glEnd()

    def _draw_current_point(self) -> None:
        from OpenGL import GL

        z, y, x = (v + 0.5 for v in self.session.point)
        arm = 0.03 * self.camera.radius
        GL.glColor3ub(CURPOINT.red(), CURPOINT.green(), CURPOINT.blue())
        GL.glBegin(GL.GL_LINES)
        for d in ((arm, 0, 0), (0, arm, 0), (0, 0, arm)):
            GL.glVertex3f(x - d[0], y - d[1], z - d[2])
            GL.glVertex3f(x + d[0], y + d[1], z + d[2])
        GL.glEnd()

    # ------------------------------------------------------------------ image planes (3dmod "Image View")
    def plane_level(self, axis: int) -> int:
        vol = self.session.volume
        row_ax, col_ax = {0: (1, 2), 1: (0, 2), 2: (1, 0)}[axis]
        for level in range(vol.nlevels):
            shape = vol.levels[level].shape
            if max(shape[row_ax], shape[col_ax]) <= PLANE_MAX_PIXELS:
                return level
        return vol.nlevels - 1

    def _plane_image(self, axis: int):
        """(uint8 image, level, complete) of the plane through the current point at a moderate resolution."""
        s = self.session
        vol = s.volume
        level = self.plane_level(axis)
        index = int(vol.level_index(level, vol, s.point[axis] + 0.5, axis))
        index = min(max(index, 0), vol.levels[level].shape[axis] - 1)
        data, missing = vol.plane(s.cache, level, axis, index)
        img = s.contrast.map(data)
        if missing is not None:
            img[missing] = 0
        return img, level, index, missing is None

    def _draw_planes(self) -> None:
        from OpenGL import GL

        s = self.session
        if not s.has_contrast or not any(self.show_planes.values()):
            return
        vol = s.volume
        z, y, x = (v + 0.5 for v in s.point)
        GL.glEnable(GL.GL_TEXTURE_2D)
        GL.glTexEnvi(GL.GL_TEXTURE_ENV, GL.GL_TEXTURE_ENV_MODE, GL.GL_REPLACE)
        for axis, on in self.show_planes.items():
            if not on:
                continue
            img, level, index, complete = self._plane_image(axis)
            if axis == 0:
                s.report_level(id(self), level)
            c = s.contrast
            key = (level, index, c.black, c.white, c.invert, c.false_color, complete)
            tex, old_key = self._textures.get(axis, (None, None))
            if tex is None:
                tex = int(GL.glGenTextures(1))
            if key != old_key:
                GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
                GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
                h, w = img.shape
                if c.false_color:
                    rgb = np.ascontiguousarray(gray_to_rgb(img, True))
                    GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB, w, h, 0, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, rgb)
                else:
                    GL.glTexImage2D(
                        GL.GL_TEXTURE_2D, 0, GL.GL_LUMINANCE, w, h, 0, GL.GL_LUMINANCE, GL.GL_UNSIGNED_BYTE,
                        np.ascontiguousarray(img),
                    )  # fmt: skip
                self._textures[axis] = (tex, key)
            GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
            # World extent of the level (it can overhang the level-0 size by a voxel for odd sizes).
            ex = float(vol.world_coord(level, vol, vol.levels[level].shape[2], 2))
            ey = float(vol.world_coord(level, vol, vol.levels[level].shape[1], 1))
            ez = float(vol.world_coord(level, vol, vol.levels[level].shape[0], 0))
            if axis == 0:  # rows y, cols x, at z
                quad = [(0, 0, 0, 0, z), (1, 0, ex, 0, z), (1, 1, ex, ey, z), (0, 1, 0, ey, z)]
            elif axis == 1:  # rows z, cols x, at y
                quad = [(0, 0, 0, y, 0), (1, 0, ex, y, 0), (1, 1, ex, y, ez), (0, 1, 0, y, ez)]
            else:  # rows y, cols z, at x
                quad = [(0, 0, x, 0, 0), (1, 0, x, 0, ez), (1, 1, x, ey, ez), (0, 1, x, ey, 0)]
            GL.glColor3f(1, 1, 1)
            GL.glBegin(GL.GL_QUADS)
            for s_, t_, qx, qy, qz in quad:
                GL.glTexCoord2f(s_, t_)
                GL.glVertex3f(qx, qy, qz)
            GL.glEnd()
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glDisable(GL.GL_TEXTURE_2D)

    # ------------------------------------------------------------------ interaction
    def _rotate(self, R_inc: np.ndarray) -> None:
        self.camera.rotate_view(R_inc)
        self.update()

    def rotate_axis(self, axis: int, degrees: float) -> None:
        """Rotate about a screen axis (0 = X, 1 = Y, 2 = Z); also sets the axis of continuous rotation."""
        self._spin = (axis, 1.0 if degrees >= 0 else -1.0)
        self._rotate({0: rot_x, 1: rot_y, 2: rot_z}[axis](degrees))

    def set_view(self, name: str) -> None:
        self.camera.R = STANDARD_VIEWS[name].copy()
        self.update()

    def zoom_by(self, factor: float) -> None:
        self.camera.zoom = float(np.clip(self.camera.zoom * factor, 0.01, 200.0))
        self.update()

    def toggle_spin(self) -> None:
        if self.spin_timer.isActive():
            self.spin_timer.stop()
        else:
            self.spin_timer.start()

    def _spin_step(self) -> None:
        axis, sign = self._spin
        self._rotate({0: rot_x, 1: rot_y, 2: rot_z}[axis](self.delta_rot * sign))

    def mousePressEvent(self, event) -> None:
        self._press = (event.button(), event.pos(), event.modifiers())
        self._mouse = event.pos()
        self._moved = False

    def mouseMoveEvent(self, event) -> None:
        if self._press is None:
            return
        button, start, mods = self._press
        pos = event.pos()
        dx, dy = pos.x() - self._mouse.x(), pos.y() - self._mouse.y()
        if (pos - start).manhattanLength() > 3:
            self._moved = True
        cam = self.camera
        if button == Qt.LeftButton:  # move the model
            px = cam.pixel_size(self.height())
            cam.trans += np.array([dx * px, -dy * px, 0.0])
            self.update()
        elif button == Qt.MiddleButton:  # rotate about the axis perpendicular to the motion
            scale = 180.0 / (np.pi * 0.4 * min(self.width(), self.height()))
            self._rotate(rot_x(scale * dy) @ rot_y(scale * dx))
        elif button == Qt.RightButton and mods & Qt.ShiftModifier:  # rotate about the viewing axis
            cx, cy = self.width() / 2.0, self.height() / 2.0
            a0 = np.degrees(np.arctan2(-(self._mouse.y() - cy), self._mouse.x() - cx))
            a1 = np.degrees(np.arctan2(-(pos.y() - cy), pos.x() - cx))
            self._rotate(rot_z(a1 - a0))
        self._mouse = pos

    def mouseReleaseEvent(self, event) -> None:
        press = self._press
        if press is not None and press[0] == Qt.RightButton and not self._moved and not press[2] & Qt.ShiftModifier:
            self.select_point(event.pos().x() + 0.5, event.pos().y() + 0.5)
        self._press = None

    def wheelEvent(self, event) -> None:
        # 3dmod: each notch zooms by 1.05 (scrolling up zooms out)
        self.zoom_by(ZOOM_SMALL ** (-event.angleDelta().y() / 120.0))

    def select_point(self, x: float, y: float) -> bool:
        """Third-button click: make the nearest, frontmost pick or filament point the current point."""
        candidates = []
        for layer in self.session.point_layers:
            if layer.visible and len(layer.points):
                candidates.append(layer.points)
        if not candidates:
            return False
        pts = np.concatenate(candidates)  # zyx world
        screen = self.camera.to_screen(pts[:, ::-1], self.width(), self.height())
        i = pick_nearest(screen, x, y)
        if i is None:
            return False
        z, yy, xx = (int(v) for v in np.floor(pts[i]))
        self.session.set_point(z, yy, xx)
        self.session.center_on_point.emit()
        return True

    def keyPressEvent(self, event) -> None:
        from copickmod import keys

        key, mods = event.key(), event.modifiers()
        shift = bool(mods & Qt.ShiftModifier)
        ctrl = bool(mods & Qt.ControlModifier)
        keypad = bool(mods & Qt.KeypadModifier)
        cam = self.camera
        d = self.delta_rot
        rot_keys = {  # keypad and the substitute cluster around U
            Qt.Key_2: (0, d), Qt.Key_8: (0, -d), Qt.Key_4: (1, -d), Qt.Key_6: (1, d), Qt.Key_3: (2, -d), Qt.Key_9: (2, d),
            Qt.Key_Down: (0, d), Qt.Key_Up: (0, -d), Qt.Key_Left: (1, -d), Qt.Key_Right: (1, d),
            Qt.Key_PageDown: (2, -d), Qt.Key_PageUp: (2, d),
        }  # fmt: skip
        if keypad and key in rot_keys:
            self.rotate_axis(*rot_keys[key])
        elif (keypad and key in (Qt.Key_Enter, Qt.Key_5, Qt.Key_Clear)) or (key == Qt.Key_U and not ctrl):
            self.toggle_spin()
        elif key in (Qt.Key_7, Qt.Key_J) and not keypad and not shift:
            self.rotate_axis(0, d if key == Qt.Key_7 else -d)
        elif key in (Qt.Key_Y, Qt.Key_I) and not shift and not ctrl:
            self.rotate_axis(1, d if key == Qt.Key_Y else -d)
        elif key in (Qt.Key_6, Qt.Key_H) and not keypad and not shift:
            self.rotate_axis(2, d if key == Qt.Key_6 else -d)
        elif key in (Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down):  # translate the model
            px = cam.pixel_size(self.height()) * 10
            delta = {Qt.Key_Left: (-px, 0), Qt.Key_Right: (px, 0), Qt.Key_Up: (0, px), Qt.Key_Down: (0, -px)}[key]
            cam.trans[:2] += delta
            self.update()
        elif key in (Qt.Key_PageUp, Qt.Key_PageDown):
            cam.trans[2] += cam.pixel_size(self.height()) * 10 * (1 if key == Qt.Key_PageUp else -1)
            self.update()
        elif key == Qt.Key_Minus:
            self.zoom_by(1.0 / ZOOM_SMALL)
        elif key == Qt.Key_Underscore:
            self.zoom_by(1.0 / ZOOM_BIG)
        elif key == Qt.Key_Equal or (key == Qt.Key_Plus and keypad):
            self.zoom_by(ZOOM_SMALL)
        elif key == Qt.Key_Plus:
            self.zoom_by(ZOOM_BIG)
        elif key == Qt.Key_T and not shift:
            self.set_view("top")
        elif key == Qt.Key_B and not shift:
            self.set_view("bottom")
        elif key == Qt.Key_F and not shift and not ctrl:
            self.set_view("front")
        elif key == Qt.Key_K and not shift and not ctrl:
            self.set_view("back")
        elif key == Qt.Key_L and shift:
            self.session.open("objects")
        elif key == Qt.Key_L and not ctrl:
            self.set_view("left")
        elif key == Qt.Key_R and not shift and not ctrl:
            self.set_view("right")
        elif key == Qt.Key_Z and not shift and not ctrl:
            self.window().toggle_plane(0)
        elif key == Qt.Key_Y and shift:
            self.window().toggle_plane(1)
        elif key == Qt.Key_X and shift:
            self.window().toggle_plane(2)
        elif key == Qt.Key_P and not shift and not ctrl:
            self.show_point = not self.show_point
            self.update()
        elif key == Qt.Key_G:
            self.sphere_quality = int(np.clip(self.sphere_quality + (-1 if shift else 1), 0, 3))
            self.session.message.emit(f"Model View sphere quality {self.sphere_quality}")
            self.update()
        elif key in (Qt.Key_Comma, Qt.Key_Period):
            self.delta_rot = float(np.clip(self.delta_rot * (1.5 if key == Qt.Key_Period else 1 / 1.5), 0.1, 45))
            self.session.message.emit(f"Model View rotation increment {self.delta_rot:.1f} degrees")
        elif snapshot_format(event):
            quick_snapshot(self.window(), snapshot_format(event))
        elif key in (Qt.Key_Escape, Qt.Key_Q):
            self.window().close()
        elif not keys.handle_global(self.session, event, axis=0):
            super().keyPressEvent(event)

    def render_offscreen(self, scale: int = 1, cursor: bool = True, overlays: bool = True):
        """Render the view into an image ``scale`` times the window size (an offscreen framebuffer)."""
        from OpenGL import GL
        from qtpy.QtGui import QOpenGLFramebufferObject, QOpenGLFramebufferObjectFormat

        s = self.session
        width, height = self.width() * scale, self.height() * scale
        saved = (s.show_model, s.show_markers)
        s.show_model, s.show_markers = saved[0] and overlays, saved[1] and cursor
        self.makeCurrent()
        try:
            fmt = QOpenGLFramebufferObjectFormat()
            fmt.setAttachment(QOpenGLFramebufferObject.CombinedDepthStencil)
            fbo = QOpenGLFramebufferObject(width, height, fmt)
            fbo.bind()
            self._draw_gl(size=(width, height))
            img = fbo.toImage()
            fbo.release()
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.defaultFramebufferObject())
        finally:
            s.show_model, s.show_markers = saved
            self.doneCurrent()
        return img

    def cleanup(self) -> None:
        self.spin_timer.stop()
        self._iso_pool.shutdown(wait=False, cancel_futures=True)
        if self.gl_error is None and (self._textures or self._gpu) and self.isValid():
            try:
                from OpenGL import GL

                self.makeCurrent()
                GL.glDeleteTextures([t for t, _ in self._textures.values()])
                for entry in self._gpu.values():
                    for buf in entry[:3]:
                        buf.delete()
                self.doneCurrent()
            except Exception:  # noqa: BLE001 - best effort at shutdown
                pass
        self._textures.clear()
        self._gpu.clear()


class ModelViewWindow(QMainWindow):
    """3dmod's Model View window: the GL view plus File / Edit / View / Help menus."""

    def __init__(self, session: Session):
        super().__init__(None)
        self.session = session
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowTitle(f"cmod Model View: {session.title}")
        keep_alive(self)
        for sig in (session.image_changed, session.images_changed):
            sig.connect(lambda *_: self.setWindowTitle(f"cmod Model View: {self.session.title}"))
        self.canvas = ModelViewGL(session, self)
        self.setCentralWidget(self.canvas)
        self.canvas.status_changed.connect(self._show_status)
        self._plane_actions: Dict[int, QAction] = {}
        mb = self.menuBar()
        file_menu = mb.addMenu("&File")
        self._action(file_menu, "Snapshot (PNG)\tShift+S", self.snapshot)
        self._action(file_menu, "Snapshot Tool...\tShift+H", lambda: session.open("snapshot"))
        self._action(file_menu, "Close\tEsc", self.close)
        edit = mb.addMenu("&Edit")
        self._action(edit, "Objects...\tShift+L", lambda: session.open("objects"))
        view = mb.addMenu("&View")
        for axis, label in ((0, "Z Image Plane\tZ"), (1, "Y Image Plane\tShift+Y"), (2, "X Image Plane\tShift+X")):
            a = self._action(view, label, lambda _=False, ax=axis: self.toggle_plane(ax), checkable=True)
            a.setChecked(self.canvas.show_planes[axis])
            self._plane_actions[axis] = a
        self._point_action = self._action(view, "Current Point\tP", self._toggle_point, checkable=True)
        self._point_action.setChecked(True)
        self._bbox_action = self._action(view, "Bounding Box", self._toggle_bbox, checkable=True)
        self._bbox_action.setChecked(True)
        view.addSeparator()
        for name, key in (("Top", "T"), ("Bottom", "B"), ("Front", "F"), ("Back", "K"), ("Left", "L"), ("Right", "R")):
            self._action(view, f"{name}\t{key}", lambda _=False, n=name.lower(): self.canvas.set_view(n))
        help_menu = mb.addMenu("&Help")
        self._action(help_menu, "Hot keys", lambda: session.open("help"))
        self.canvas.setFocus()

    def _action(self, menu, text, slot, checkable: bool = False) -> QAction:
        a = QAction(text, self)
        a.setCheckable(checkable)
        a.triggered.connect(lambda _=False: slot())
        menu.addAction(a)
        return a

    def _show_status(self, text: str) -> None:
        if text:
            self.statusBar().showMessage(text)
        elif self.statusBar().currentMessage():
            self.statusBar().clearMessage()

    def toggle_plane(self, axis: int) -> None:
        on = not self.canvas.show_planes[axis]
        self.canvas.show_planes[axis] = on
        self._plane_actions[axis].setChecked(on)
        self.canvas.update()

    def _toggle_point(self) -> None:
        self.canvas.show_point = self._point_action.isChecked()
        self.canvas.update()

    def _toggle_bbox(self) -> None:
        self.canvas.show_bbox = self._bbox_action.isChecked()
        self.canvas.update()

    def fit_to_image(self) -> None:
        self.resize(self.sizeHint())

    def sizeHint(self) -> QSize:
        return QSize(700, 700)

    def render_image(self, scale: int = 1, overlays: bool = True, cursor: bool = True, band: bool = True):
        """The 3D view at ``scale`` times the window size, and its angstroms per pixel."""
        c = self.canvas
        img = c.render_offscreen(scale, cursor=cursor, overlays=overlays)
        return img, c.camera.pixel_size(c.height() * scale) * self.session.volume.voxel_size

    def snapshot(self) -> str:
        return quick_snapshot(self, "PNG")

    def keyPressEvent(self, event) -> None:
        self.canvas.keyPressEvent(event)

    def closeEvent(self, event) -> None:
        self.canvas.cleanup()
        super().closeEvent(event)
