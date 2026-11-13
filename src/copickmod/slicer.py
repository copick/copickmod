"""The Slicer window: an oblique slice through the current point (3dmod slicer.cpp)."""

from __future__ import annotations

import numpy as np
from qtpy.QtCore import QPointF, QRectF, QSize, Qt
from qtpy.QtGui import QColor, QPainter, QPainterPath, QPen
from qtpy.QtWidgets import QGridLayout, QHBoxLayout, QLabel, QMainWindow, QSlider, QSpinBox, QWidget

from copickmod import keys, overlays, slicer_geom
from copickmod.canvas import (
    CURSOR_COLOR,
    ClickDragMixin,
    PlaneView,
    draw_cross,
    fit_zoom,
    pixel_center,
    qcolor,
    rgba_image,
    zoom_step,
)
from copickmod.contrast import gray_to_rgb
from copickmod.snapshot import quick_snapshot, snapshot_format
from copickmod.state import Session
from copickmod.style import (
    BACKGROUND,
    center_icon,
    checker_icon,
    keep_alive,
    lock_icon,
    make_toolbar,
    tb_push,
    tb_toggle,
    tb_zoom_tools,
    zoom_text,
)

MAX_SAMPLES = 1_500_000
#: Keypad angle steps (3dmod): key -> degrees added to the last adjusted angle.
ANGLE_KEYS = {
    Qt.Key_4: -0.1, Qt.Key_Left: -0.1, Qt.Key_6: 0.1, Qt.Key_Right: 0.1,
    Qt.Key_2: -0.5, Qt.Key_Down: -0.5, Qt.Key_8: 0.5, Qt.Key_Up: 0.5,
    Qt.Key_1: -15.0, Qt.Key_End: -15.0, Qt.Key_3: 15.0, Qt.Key_PageDown: 15.0,
}  # fmt: skip


def wrap_angle(a: float) -> float:
    a = (a + 180.0) % 360.0 - 180.0
    return 180.0 if a == -180.0 else round(a, 3)


class SlicerCanvas(ClickDragMixin, QWidget):
    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.session = session
        self.view_id = PlaneView(session, 0).view_id  # just for the wanted-chunk bookkeeping
        self.angles = [0.0, 0.0, 0.0]  # x, y, z in degrees
        self.last_angle = 0
        _, ny, nx = session.volume.shape
        self.zoom = fit_zoom(nx, ny, 640, 640)
        self.hq = False  # 3dmod's Slicer starts with fast (nearest neighbour) rendering
        self.thickness = 1
        self.center = np.asarray(session.point, dtype=float) + 0.5  # centre of rotation, world zyx
        self.pan = [0.0, 0.0]  # always 0: like 3dmod, panning moves the centre of rotation itself
        self.classic = False  # 3dmod classic mode (K): the slice stays centred on the current point
        self.locked = False  # lock: ignore changes of the current point made elsewhere
        self._last_point = list(session.point)
        self._own_move = False
        self._mouse = QPointF(0, 0)
        self.complete = False
        self._overlays = True  # cleared while rendering snapshots without overlays
        self._cursor = True
        self._render_geom = None
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        self.setMinimumSize(100, 100)
        session.point_changed.connect(self._point_changed)
        session.image_changed.connect(self._image_changed)
        for sig in (session.contrast_changed, session.overlays_changed, session.data_arrived):
            sig.connect(self.update)

    def sizeHint(self) -> QSize:
        return QSize(640, 640)

    def _image_changed(self, old) -> None:
        """Another tomogram is shown: the same centre (in angstroms), field of view and angles."""
        s = self.session
        self.center = np.array([s.map_world(old, a, self.center[a]) for a in range(3)], dtype=float)
        ratio = s.volume.voxel_size / old.voxel_size
        if abs(ratio - 1.0) > 1e-6:
            self.zoom = float(min(max(self.zoom * ratio, 0.005), 50.0))
        self._last_point = list(s.point)  # the point moved with the image, not by the user
        self.update()

    # ------------------------------------------------------------------ geometry
    def _point_changed(self) -> None:
        """3dmod's new-mode rules: follow Z changes made elsewhere, and single steps in X or Y (arrow keys), but not a
        click on a new point. Classic mode keeps the slice centred on the current point."""
        new = list(self.session.point)
        if not self._own_move and not self.locked:
            if self.classic:
                self.center = np.asarray(new, dtype=float) + 0.5
            else:
                dz, dy, dx = (a - b for a, b in zip(new, self._last_point))
                if dz and not dy and not dx:
                    self.center[0] = min(max(self.center[0] + dz, 0.0), self.session.volume.shape[0] - 0.5)
                elif dy and not dz and not dx and abs(dy) == 1:
                    self.center[1] += dy
                elif dx and not dz and not dy and abs(dx) == 1:
                    self.center[2] += dx
        self._last_point = new
        self.update()
        self.window().sync_controls()

    def set_classic(self, on: bool) -> None:
        self.classic = bool(on)
        if on:
            self.center = np.asarray(self.session.point, dtype=float) + 0.5
        self.update()

    def translate_rotated(self, du: float, dv: float, dn: float = 0.0) -> bool:
        """SlicerFuncs::translateByRotatedVec: move the centre within or normal to the slice, staying in the volume."""
        eu, ev, en = slicer_geom.slice_axes(self.angles)
        new = self.center + du * eu + dv * ev + dn * en
        nz, ny, nx = self.session.volume.shape
        if 0 <= new[2] < nx and 0 <= new[1] < ny and 0 <= new[0] < nz - 0.5:
            self.center = new
            self.update()
            return True
        return False

    def screen_to_world(self, pos) -> np.ndarray:
        eu, ev, _ = slicer_geom.slice_axes(self.angles)
        W, H, Z = self._geom()
        u = self.pan[0] + (pos.x() - W / 2.0) / Z
        v = self.pan[1] - (pos.y() - H / 2.0) / Z
        return self.center + u * eu + v * ev

    def world_to_screen(self, u: np.ndarray, v: np.ndarray):
        W, H, Z = self._geom()
        return W / 2.0 + (u - self.pan[0]) * Z, H / 2.0 - (v - self.pan[1]) * Z

    def _geom(self):
        """(width, height, zoom) being drawn: the widget's, or a snapshot's while one is rendered."""
        return self._render_geom if self._render_geom is not None else (self.width(), self.height(), self.zoom)

    def normal_position(self) -> float:
        """Position of the centre along the slice normal, relative to the volume centre (the view-axis slider)."""
        _, _, en = slicer_geom.slice_axes(self.angles)
        return float((self.center - np.asarray(self.session.volume.shape) / 2.0) @ en)

    def move_along_normal(self, delta: float) -> None:
        """PageUp/PageDown: move the centre along the normal, and the current Z with it (keeps a ZaP in sync)."""
        _, _, en = slicer_geom.slice_axes(self.angles)
        shape = np.asarray(self.session.volume.shape, dtype=float)
        self.center = np.clip(self.center + delta * en, 0.0, shape - 1e-3)
        self._own_move = True
        try:
            z, y, x = (int(v) for v in np.floor(self.center))
            self.session.set_point(z, y, x)
        finally:
            self._own_move = False
        self.update()
        self.window().sync_controls()

    # ------------------------------------------------------------------ painting
    def paintEvent(self, event) -> None:
        p = QPainter(self)
        self.paint_scene(p, self.width(), self.height(), self.zoom)
        if not self.complete:
            p.setPen(Qt.yellow)
            p.drawText(6, self.height() - 6, "loading...")
        p.end()

    def paint_scene(self, p: QPainter, W: int, H: int, zoom: float, overlays: bool = True, cursor: bool = True) -> bool:
        """Draw the slice into a W x H area at ``zoom`` (the widget, or a high-resolution snapshot)."""
        self._render_geom = (W, H, zoom)
        self._overlays, self._cursor = overlays, cursor
        try:
            self._paint(p, W, H, zoom)
        finally:
            self._render_geom = None
            self._overlays = self._cursor = True
        return self.complete

    def _paint(self, p: QPainter, W: int, H: int, zoom: float) -> None:
        p.fillRect(QRectF(0, 0, W, H), BACKGROUND)
        s = self.session
        vol = s.volume
        level = s.choose_level(zoom)
        s.report_level(self.view_id, level)
        step = float(vol.factor(level)[2])  # one sample per voxel of the chosen level
        w = max(1, int(np.ceil(W / (zoom * step))))
        h = max(1, int(np.ceil(H / (zoom * step))))
        if w * h > MAX_SAMPLES:
            k = float(np.sqrt(w * h / MAX_SAMPLES))
            step *= k
            w, h = max(1, int(w / k)), max(1, int(h / k))
        wanted: set = set()
        shape = np.asarray(vol.shape, dtype=float)
        offsets = np.arange(self.thickness) - (self.thickness - 1) / 2.0
        acc = np.zeros((h, w), dtype=np.float32)
        ok_all = np.ones((h, w), dtype=bool)
        mid = slicer_geom.plane_coords(self.center, self.angles, w, h, step, self.pan)
        for off in offsets:
            coords = mid if off == 0 else slicer_geom.plane_coords(self.center, self.angles, w, h, step, self.pan, off)
            lc = np.stack([vol.level_index(level, vol, coords[a], a) for a in range(3)])
            vals, ok = vol.sample(s.cache, level, lc, linear=self.hq, wanted=wanted)
            acc += vals
            ok_all &= ok
        acc /= len(offsets)
        inside = np.all((mid >= 0) & (mid < shape[:, None, None]), axis=0)
        missing = inside & ~ok_all
        self.complete = not missing.any()
        if missing.any():
            # Fill holes from coarser levels (nearest neighbour) while the chosen level loads.
            acc[missing] = np.nan
            for coarse in range(vol.nlevels - 1, level, -1):
                pts = mid[:, missing]
                lc = np.stack([vol.level_index(coarse, vol, pts[a], a) for a in range(3)])
                cvals, cok = vol.sample(s.cache, coarse, lc, linear=False, wanted=wanted)
                sub = acc[missing]
                sub[cok] = cvals[cok]
                acc[missing] = sub
        s.set_wanted(self.view_id, wanted)

        gray = s.contrast.map(acc)
        rgba = np.empty((h, w, 4), dtype=np.uint8)
        rgba[..., :3] = gray_to_rgb(gray, s.contrast.false_color)
        rgba[..., 3] = np.where(inside & s.has_contrast, 255, 0)
        span_w, span_h = w * step * zoom, h * step * zoom
        target = QRectF(W / 2.0 - span_w / 2.0, H / 2.0 - span_h / 2.0, span_w, span_h)
        p.setRenderHint(QPainter.SmoothPixmapTransform, self.hq)
        p.drawImage(target, rgba_image(rgba[::-1]))

        visible_segs = [layer for layer in s.seg_layers if layer.visible and self._overlays and s.show_model]
        if visible_segs:
            p.setRenderHint(QPainter.SmoothPixmapTransform, False)
            p.setOpacity(s.opacity)
            for layer in visible_segs:
                srgba = overlays.seg_points_rgba(layer, s.cache, vol, mid, float(vol.levels[0].scale[2] * step))
                p.drawImage(target, rgba_image(srgba[::-1]))
            p.setOpacity(1.0)
        if self._overlays and s.show_model:
            self._draw_points(p)
        if self._cursor and s.show_markers:
            self._draw_cursor(p)

    def _draw_points(self, p: QPainter) -> None:
        s = self.session
        p.setRenderHint(QPainter.Antialiasing, True)
        for layer in s.point_layers:
            if not layer.visible or len(layer.points) == 0:
                continue
            color = qcolor(layer.rgba, 255)
            if layer.kind == "filaments":
                p.setPen(QPen(color, 2.0))
                for line in layer.polylines:
                    u, v, d = slicer_geom.project(line, self.center, self.angles)
                    near = np.abs(d) < layer.radius
                    if not near.any():
                        continue
                    xs, ys = self.world_to_screen(u, v)
                    path = QPainterPath()
                    started = False
                    for i in range(len(line)):
                        if not near[i]:
                            started = False
                            continue
                        q = QPointF(xs[i], ys[i])
                        if started:
                            path.lineTo(q)
                        else:
                            path.moveTo(q)
                            started = True
                    p.drawPath(path)
                continue
            u, v, d = slicer_geom.project(layer.points, self.center, self.angles)
            sel = np.abs(d) < layer.radius
            xs, ys = self.world_to_screen(u[sel], v[sel])
            radii = np.sqrt(np.maximum(layer.radius**2 - d[sel] ** 2, 0.0)) * self._geom()[2]
            for x, y, r, dd in zip(xs, ys, radii, d[sel]):
                p.setPen(QPen(color, 2.0 if abs(dd) <= 0.5 else 1.0))
                p.drawEllipse(QPointF(x, y), r, r)
        p.setRenderHint(QPainter.Antialiasing, False)

    def _draw_cursor(self, p: QPainter) -> None:
        s = self.session
        u, v, d = slicer_geom.project(np.asarray(s.point, dtype=float)[None] + 0.5, self.center, self.angles)
        x, y = self.world_to_screen(u, v)
        draw_cross(p, QPointF(float(x[0]), float(y[0])), CURSOR_COLOR if abs(d[0]) < 1 else QColor(255, 150, 150, 120))

    # ------------------------------------------------------------------ interaction
    def set_angle(self, index: int, value: float) -> None:
        self.angles[index] = wrap_angle(value)
        self.last_angle = index
        self.update()
        self.window().sync_controls()

    def set_zoom(self, zoom: float) -> None:
        self.zoom = float(min(max(zoom, 0.005), 50.0))
        self.update()
        self.window().sync_controls()

    def set_thickness(self, n: int) -> None:
        self.thickness = int(min(max(n, 1), 200))
        self.update()
        self.window().sync_controls()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._press(event)

    def mouseMoveEvent(self, event) -> None:
        self._mouse = QPointF(event.pos())
        delta = self._move(event)
        if delta is not None:
            fac = 1.0 / self.zoom if self.zoom < 4.0 else 0.25  # 3dmod slicer pan rate
            if self.translate_rotated(-delta[0] * fac, delta[1] * fac):
                self.window().sync_controls()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self._release_is_click(event):
            world = self.screen_to_world(pixel_center(event.pos()))
            self._own_move = True  # 3dmod "new" mode: clicking does not recentre the slice
            try:
                z, y, x = (int(v) for v in np.floor(world))
                self.session.set_point(z, y, x)
            finally:
                self._own_move = False

    def wheelEvent(self, event) -> None:
        event.ignore()  # 3dmod does nothing with the wheel in movie mode

    def keyPressEvent(self, event) -> None:
        key, mods = event.key(), event.modifiers()
        keypad = bool(mods & Qt.KeypadModifier)
        if keypad and key in ANGLE_KEYS:
            self.set_angle(self.last_angle, self.angles[self.last_angle] + ANGLE_KEYS[key])
        elif keypad and key in (Qt.Key_0, Qt.Key_Insert):
            self.set_angle(self.last_angle, 0.0)
        elif key == Qt.Key_Minus:
            self.set_zoom(zoom_step(self.zoom, False))
        elif key == Qt.Key_Equal or (key == Qt.Key_Plus and keypad):
            self.set_zoom(zoom_step(self.zoom, True))
        elif key == Qt.Key_Underscore:
            self.set_thickness(self.thickness - 1)
        elif key == Qt.Key_Plus:
            self.set_thickness(self.thickness + 1)
        elif key in (Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down):
            # 3dmod new mode: the image shifts one pixel in the arrow's direction (0.2 with Ctrl).
            step = 0.2 if mods & Qt.ControlModifier else 1.0
            du = {Qt.Key_Left: step, Qt.Key_Right: -step}.get(key, 0.0)
            dv = {Qt.Key_Up: -step, Qt.Key_Down: step}.get(key, 0.0)
            self.translate_rotated(du, dv)
        elif key == Qt.Key_K and not mods & Qt.ShiftModifier:
            self.window().center_button.toggle()
        elif key in (Qt.Key_PageUp, Qt.Key_PageDown):
            self.move_along_normal(1.0 if key == Qt.Key_PageUp else -1.0)
        elif snapshot_format(event):
            quick_snapshot(self.window(), snapshot_format(event))
        elif key == Qt.Key_Escape:
            self.window().close()
        elif not keys.handle_global(self.session, event, axis=0):
            super().keyPressEvent(event)


class SlicerWindow(QMainWindow):
    def __init__(self, session: Session, angles=None):
        super().__init__(None)
        self.session = session
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowTitle(f"cmod Slicer: {session.title}")
        keep_alive(self)
        self.canvas = SlicerCanvas(session, self)
        self.setCentralWidget(self.canvas)
        if angles is not None:
            self.canvas.angles = [wrap_angle(a) for a in angles]

        tb = make_toolbar(self, "Slicer Toolbar")
        self.toolbar = tb
        self.zoom_edit = tb_zoom_tools(
            tb,
            lambda upward: self.canvas.set_zoom(zoom_step(self.canvas.zoom, upward)),
            self._zoom_typed,
        )
        self.hq_button = tb_toggle(
            checker_icon(False),
            checker_icon(True),
            "Toggle between fast rendering and slower but higher quality image",
        )
        self.hq_button.toggled.connect(self._hq_toggled)
        self.lock_button = tb_toggle(
            lock_icon(False),
            lock_icon(True),
            "Lock window at current position, ignoring changes of the current point elsewhere",
        )
        self.lock_button.toggled.connect(self._lock_toggled)
        self.center_button = tb_toggle(
            center_icon(False),
            center_icon(True),
            "Keep image centered on current point (classic mode, K)",
        )
        self.center_button.toggled.connect(self.canvas.set_classic)
        for b in (self.hq_button, self.lock_button, self.center_button):
            tb.addWidget(b)
        tb.addWidget(tb_push("Help", "Open help window", lambda: session.open("help")))

        # Slice control tool bar: angle sliders, view axis position and image thickness.
        self.addToolBarBreak()
        control = make_toolbar(self, "Slice Control")
        panel = QWidget()
        grid = QGridLayout(panel)
        grid.setContentsMargins(2, 0, 2, 2)
        grid.setHorizontalSpacing(4)
        grid.setVerticalSpacing(0)
        self.angle_sliders = []
        self.angle_labels = []
        for i, name in enumerate("XYZ"):
            slider = QSlider(Qt.Horizontal)
            slider.setRange(-1800, 1800)
            slider.setFocusPolicy(Qt.NoFocus)
            slider.setToolTip(f"Rotation about the {name} axis (applied to the volume in the order Z, Y, X)")
            slider.valueChanged.connect(lambda v, i=i: self.canvas.set_angle(i, v / 10.0))
            label = QLabel()
            label.setMinimumWidth(label.fontMetrics().horizontalAdvance("-180.0") + 4)
            label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            grid.addWidget(QLabel(name), i, 0)
            grid.addWidget(slider, i, 1)
            grid.addWidget(label, i, 2)
            self.angle_sliders.append(slider)
            self.angle_labels.append(label)
        self.axis_slider = QSlider(Qt.Horizontal)
        half = int(np.ceil(np.linalg.norm(session.volume.shape) / 2.0))
        self.axis_slider.setRange(-half, half)
        self.axis_slider.setFocusPolicy(Qt.NoFocus)
        self.axis_slider.setToolTip("View axis position: move along the slice normal")
        self.axis_slider.valueChanged.connect(self._axis_moved)
        self.axis_label = QLabel()
        self.axis_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        grid.addWidget(QLabel("View axis"), 3, 0)
        grid.addWidget(self.axis_slider, 3, 1)
        grid.addWidget(self.axis_label, 3, 2)
        self.thick_spin = QSpinBox()
        self.thick_spin.setRange(1, 200)
        self.thick_spin.setKeyboardTracking(False)
        self.thick_spin.setFocusPolicy(Qt.ClickFocus)
        self.thick_spin.setToolTip("Set thickness of image slice to display (_ / +)")
        self.thick_spin.valueChanged.connect(self.canvas.set_thickness)
        thick = QHBoxLayout()
        thick.addWidget(QLabel("Img"))
        thick.addWidget(self.thick_spin)
        thick.addStretch(1)
        grid.addLayout(thick, 4, 0, 1, 3)
        control.addWidget(panel)

        session.point_changed.connect(self.sync_controls)
        session.image_changed.connect(self._image_changed)
        session.images_changed.connect(self._retitle)
        self.sync_controls()
        self.canvas.setFocus()

    def _retitle(self) -> None:
        self.setWindowTitle(f"cmod Slicer: {self.session.title}")

    def _image_changed(self, old) -> None:
        half = int(np.ceil(np.linalg.norm(self.session.volume.shape) / 2.0))
        self.axis_slider.blockSignals(True)
        self.axis_slider.setRange(-half, half)
        self.axis_slider.blockSignals(False)
        self._retitle()
        self.sync_controls()
        if hasattr(self, "zoom_changed"):
            self.zoom_changed()

    def _lock_toggled(self, on: bool) -> None:
        self.canvas.locked = on

    def render_image(self, scale: int = 1, overlays: bool = True, cursor: bool = True, band: bool = True):
        """The slice at ``scale`` times the window size (finer data), and its angstroms per pixel."""
        from copickmod.snapshot import render_until_complete

        c, s = self.canvas, self.session
        width, height, zoom = c.width() * scale, c.height() * scale, c.zoom * scale
        live_id = c.view_id
        c.view_id = PlaneView(s, 0).view_id  # separate chunk bookkeeping for the snapshot
        try:
            img = render_until_complete(
                lambda p: c.paint_scene(p, width, height, zoom, overlays, cursor), width, height, s,
            )  # fmt: skip
        finally:
            s.drop_view(c.view_id)
            c.view_id = live_id
        return img, s.volume.voxel_size / zoom

    def sync_controls(self) -> None:
        c = self.canvas
        for i, (slider, label) in enumerate(zip(self.angle_sliders, self.angle_labels)):
            slider.blockSignals(True)
            slider.setValue(int(round(c.angles[i] * 10)))
            slider.blockSignals(False)
            label.setText(f"{c.angles[i]:.1f}")
        pos = c.normal_position()
        self.axis_slider.blockSignals(True)
        self.axis_slider.setValue(int(round(pos)))
        self.axis_slider.blockSignals(False)
        self.axis_label.setText(f"{pos:.0f}")
        self.zoom_edit.setText(zoom_text(c.zoom))
        self.thick_spin.blockSignals(True)
        self.thick_spin.setValue(c.thickness)
        self.thick_spin.blockSignals(False)

    def _axis_moved(self, value: int) -> None:
        self.canvas.move_along_normal(value - self.canvas.normal_position())

    def _zoom_typed(self, zoom: float) -> None:
        self.canvas.set_zoom(zoom)
        self.canvas.setFocus()

    def _hq_toggled(self, on: bool) -> None:
        self.canvas.hq = on
        self.canvas.update()

    def keyPressEvent(self, event) -> None:
        self.canvas.keyPressEvent(event)

    def closeEvent(self, event) -> None:
        self.session.drop_view(self.canvas.view_id)
        super().closeEvent(event)
