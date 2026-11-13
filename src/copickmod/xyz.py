"""The XYZ window: XY, XZ and YZ planes through the current point, behaving like 3dmod's xyz.cpp.

The window is split into fixed panels: XY at the bottom left, XZ above it, YZ to its right and a Z "gadget" square at
the top right, separated by 16-pixel gutters. Panel sizes follow the fraction of Z in each direction (adjustable by
dragging the blue box where the gutters cross). Images pan inside their panels with three shared translations (X, Y
and Z), so dragging one panel moves the matching axis of the others. The gutters hold the X (green) and Y (red)
position markers, which can be clicked or dragged, and the Z gadget shows the Z position as a cross.

Geometry is computed in 3dmod's OpenGL convention (origin at the bottom left) and converted to Qt when drawing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from qtpy.QtCore import QPointF, QRectF, QSize, Qt
from qtpy.QtGui import QPainter, QPen
from qtpy.QtWidgets import QLabel, QMainWindow, QSlider, QSpinBox, QWidget

from copickmod import keys
from copickmod.canvas import CLICK_MS, DRAG_PX, PlaneView, fit_zoom, image_offset, trans_factor, zoom_step
from copickmod.snapshot import quick_snapshot, snapshot_format
from copickmod.state import Session
from copickmod.style import (
    BACKGROUND,
    BGNPOINT,
    ENDPOINT,
    FOREGROUND,
    FRACTION_BOX,
    center_icon,
    checker_icon,
    keep_alive,
    lock_icon,
    make_toolbar,
    section_slider_width,
    tb_button,
    tb_push,
    tb_toggle,
    tb_zoom_tools,
    zoom_text,
    zscale_icon,
)
from copickmod.zap import available_size

BSIZE = 8  # border
GSIZE = 16  # gutter
ALL_BORDER = 2 * BSIZE + GSIZE
GRAB_LENGTH = 7
GRAB_WIDTH = 3

# Mouse regions (xyz.cpp). The "X slice" is the YZ panel (X fixed), the "Y slice" the XZ panel, "Z slice" the XY.
NOT_IN_BOX, X_SLICE_BOX, Y_SLICE_BOX, Z_SLICE_BOX, Z_GADGET_BOX, X_GADGET_BOX, Y_GADGET_BOX, FRACTION_BOX_ID = range(8)
#: Plane axis (0 = XY, 1 = XZ, 2 = YZ) of each slice box.
BOX_AXIS = {Z_SLICE_BOX: 0, Y_SLICE_BOX: 1, X_SLICE_BOX: 2}


def allocate_dim(winsize: int, z_fraction: float) -> Tuple[int, int]:
    """Split a window dimension into the main panel and the Z panel (XyzWindow::allocateDim)."""
    dim2 = int(np.floor(z_fraction * (winsize - ALL_BORDER) + 0.5))
    dim1 = winsize - ALL_BORDER - dim2
    if dim1 < 2:
        dim2 = max(2, dim2 - (2 - dim1))
        dim1 = 2
    elif dim2 < 2:
        dim1 = max(2, dim1 - (2 - dim2))
        dim2 = 2
    return dim1, dim2


@dataclass
class Layout:
    """Panel geometry in 3dmod GL coordinates (y up) plus the image positions."""

    W: int
    H: int
    xd1: int
    xd2: int
    yd1: int
    yd2: int
    X1: int
    X2: int
    Y1: int
    Y2: int
    bx: float = 0.0  # GL x of world x = 0 in XY/XZ
    bx2: float = 0.0  # GL x of world z = 0 in YZ
    by: float = 0.0  # GL y of world y = 0 in XY/YZ
    by2: float = 0.0  # GL y of world z = 0 in XZ

    def qrect(self, x: float, y: float, w: float, h: float) -> QRectF:
        """GL rectangle (lower-left corner) to a Qt rectangle."""
        return QRectF(x, self.H - y - h, w, h)

    def qpoint(self, x: float, y: float) -> QPointF:
        return QPointF(x, self.H - y)

    def panel(self, axis: int) -> QRectF:
        if axis == 0:
            return self.qrect(self.X1, self.Y1, self.xd1, self.yd1)
        if axis == 1:
            return self.qrect(self.X1, self.Y2, self.xd1, self.yd2)
        return self.qrect(self.X2, self.Y1, self.xd2, self.yd1)


def in_box(llx, lly, urx, ury, x, y) -> bool:
    return llx <= x <= urx and lly <= y <= ury


class XyzCanvas(QWidget):
    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.session = session
        self.views = {axis: PlaneView(session, axis) for axis in (0, 1, 2)}
        self.locked_point = None  # lock: keep showing this location regardless of the global point
        for view in self.views.values():
            view.point_source = lambda: self.point
        nz, ny, nx = session.volume.shape
        aw, ah = available_size()
        self.zoom = fit_zoom(nx + nz, ny + nz, aw * 0.9 - ALL_BORDER, ah * 0.8 - ALL_BORDER)
        self.hq = True
        self.xtrans1 = self.ytrans1 = self.xtrans2 = self.ytrans2 = 0
        self.reset_fractions()
        self._pressed = False
        self._press_t = 0.0
        self._first = (0, 0)
        self._last = (0, 0)
        self._which = NOT_IN_BOX
        self._mouse = QPointF(0, 0)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        self.setMinimumSize(ALL_BORDER + 20, ALL_BORDER + 20)
        session.point_changed.connect(self.update)
        session.image_changed.connect(self._image_changed)
        for sig in (session.contrast_changed, session.overlays_changed, session.data_arrived):
            sig.connect(self.update)

    def _image_changed(self, old) -> None:
        """Another tomogram is shown: keep the field of view in angstroms; a lock moves to the same place."""
        s = self.session
        ratio = s.volume.voxel_size / old.voxel_size
        if abs(ratio - 1.0) > 1e-6:
            self.zoom = float(min(max(self.zoom * ratio, 0.005), 50.0))
            self.xtrans1, self.xtrans2, self.ytrans1, self.ytrans2 = (
                int(round(t / ratio)) for t in (self.xtrans1, self.xtrans2, self.ytrans1, self.ytrans2)
            )
        if self.locked_point is not None:
            self.locked_point = [
                int(np.clip(np.floor(s.map_world(old, a, v + 0.5)), 0, s.volume.shape[a] - 1))
                for a, v in enumerate(self.locked_point)
            ]
        self.update()

    def reset_fractions(self) -> None:
        nz, ny, nx = self.session.volume.shape
        self.xz_fraction = nz / (nx + nz)
        self.yz_fraction = nz / (ny + nz)

    def fit_size(self) -> QSize:
        nz, ny, nx = self.session.volume.shape
        return QSize(int(self.zoom * (nx + nz)) + ALL_BORDER, int(self.zoom * (ny + nz)) + ALL_BORDER)

    def sizeHint(self) -> QSize:
        aw, ah = available_size()
        s = self.fit_size()
        return QSize(min(s.width(), int(aw * 0.95)), min(s.height(), int(ah * 0.85)))

    # ------------------------------------------------------------------ geometry
    def layout_geometry(self, W: Optional[int] = None, H: Optional[int] = None, zoom: Optional[float] = None) -> Layout:
        """Panel sizes (GetCIImages) and image offsets (DrawImage) for the window size and pan.

        With an explicit size and zoom (for high-resolution snapshots) the pan is not modified.
        """
        nz, ny, nx = self.session.volume.shape
        commit = W is None
        W = self.width() if W is None else W
        H = self.height() if H is None else H
        z = self.zoom if zoom is None else zoom
        trans = [self.xtrans1, self.xtrans2, self.ytrans1, self.ytrans2]
        xd1, xd2 = allocate_dim(W, self.xz_fraction)
        yd1, yd2 = allocate_dim(H, self.yz_fraction)
        if xd2 > yd2:  # the Z panels are square
            xd1 += xd2 - yd2
            xd2 = yd2
        elif yd2 > xd2:
            yd1 += yd2 - xd2
            yd2 = xd2
        L = Layout(W, H, xd1, xd2, yd1, yd2, BSIZE, BSIZE + GSIZE + xd1, BSIZE, BSIZE + GSIZE + yd1)
        woff, doff, trans[0] = image_offset(xd1, nx, z, trans[0])
        L.bx = L.X1 + woff - doff * z
        woff, doff, trans[1] = image_offset(xd2, nz, z, trans[1])
        L.bx2 = L.X2 + woff - doff * z
        woff, doff, trans[2] = image_offset(yd1, ny, z, trans[2])
        L.by = L.Y1 + woff - doff * z
        woff, doff, trans[3] = image_offset(yd2, nz, z, trans[3])
        L.by2 = L.Y2 + woff - doff * z
        if commit:
            self.xtrans1, self.xtrans2, self.ytrans1, self.ytrans2 = trans
        return L

    def panel_center(self, L: Layout, axis: int, zoom: Optional[float] = None) -> Tuple[float, float]:
        """World (row, col) at the centre of a panel, for PlaneView."""
        z = self.zoom if zoom is None else zoom
        if axis == 0:
            return (L.Y1 + L.yd1 / 2 - L.by) / z, (L.X1 + L.xd1 / 2 - L.bx) / z
        if axis == 1:
            return (L.Y2 + L.yd2 / 2 - L.by2) / z, (L.X1 + L.xd1 / 2 - L.bx) / z
        return (L.Y1 + L.yd1 / 2 - L.by) / z, (L.X2 + L.xd2 / 2 - L.bx2) / z

    # ------------------------------------------------------------------ painting
    def paintEvent(self, event) -> None:
        p = QPainter(self)
        L = self.layout_geometry()
        complete = self.paint_scene(p, L, self.zoom, self.views)
        self.session.report_level(self.views[0].view_id, self.views[0].level)
        if not complete:
            p.setPen(Qt.yellow)
            p.drawText(L.X2 + 4, self.height() - L.Y2 - L.yd2 + 14, "loading...")
        p.end()

    def paint_scene(
        self,
        p: QPainter,
        L: Layout,
        zoom: float,
        views,
        overlays: bool = True,
        cursor: bool = True,
    ) -> bool:
        """Draw the panels and markers for a layout; also used for high-resolution snapshots."""
        p.fillRect(QRectF(0, 0, L.W, L.H), BACKGROUND)
        complete = True
        for axis, view in views.items():
            center = self.panel_center(L, axis, zoom)
            complete &= view.render(p, L.panel(axis), zoom, center, self.hq, cursor=cursor, overlays=overlays)
        self._draw_current_lines(p, L, zoom)
        return complete

    def _draw_current_lines(self, p: QPainter, L: Layout, zoom: Optional[float] = None) -> None:
        """XyzWindow::DrawCurrentLines: panel boxes, gutter markers, Z gadget cross and the fraction box."""
        cz, cy, cx = self.point
        z = self.zoom if zoom is None else zoom
        hlen = GRAB_LENGTH // 2
        hwidth = GRAB_WIDTH // 2
        xline_y = L.Y2 - GSIZE // 2 - 1
        yline_x = L.X2 - GSIZE // 2 - 1

        def line(color, x0, y0, x1, y1):
            p.setPen(QPen(color, 1))
            p.drawLine(L.qpoint(x0, y0), L.qpoint(x1, y1))

        def box(color, x, y, w, h):
            p.setPen(QPen(color, 1))
            p.setBrush(Qt.NoBrush)
            p.drawRect(L.qrect(x, y, w, h))

        def fill(color, x, y, w, h):
            p.fillRect(L.qrect(x, y, w, h), color)

        # Z: cross in the gadget square, and the box around the XY panel.
        cenx = int(L.bx2 + z * (cz + 0.5))
        ceny = int(L.by2 + z * (cz + 0.5))
        cenxlim = min(max(cenx, L.X2 - 1), L.X2 + L.xd2 + 1)
        cenylim = min(max(ceny, L.Y2), L.Y2 + L.yd2)
        if ceny == cenylim:
            line(FOREGROUND, L.X2, ceny, L.X2 + L.xd2, ceny)
        if cenx == cenxlim:
            line(FOREGROUND, cenx, L.Y2, cenx, L.Y2 + L.yd2)
        if cenx == cenxlim and ceny == cenylim:
            fill(FOREGROUND, cenx - hlen, ceny - hlen, GRAB_LENGTH, GRAB_LENGTH)
        box(FOREGROUND, L.X1 - 1, L.Y1 - 1, L.xd1 + 1, L.yd1 + 1)

        # X: green line in the top gutter from the X position to the right, and the YZ box.
        cenx = int(L.bx + z * (cx + 0.5))
        cenxlim = min(max(cenx, L.X1 - 1), L.X1 + L.xd1)
        line(BGNPOINT, cenxlim, xline_y, L.X2 + L.xd2, xline_y)
        if cenx == cenxlim:
            fill(BGNPOINT, cenx - hwidth, xline_y - hlen, GRAB_WIDTH, GRAB_LENGTH)
        box(BGNPOINT, L.X2 - 1, L.Y1 - 1, L.xd2 + 1, L.yd1 + 1)

        # Y: red line in the right gutter from the top down to the Y position, and the XZ box.
        ceny = int(L.by + z * (cy + 0.5))
        cenylim = min(max(ceny, L.Y1 - 1), L.Y1 + L.yd1)
        line(ENDPOINT, yline_x, L.Y2 + L.yd2, yline_x, cenylim)
        if ceny == cenylim:
            fill(ENDPOINT, yline_x - hlen, ceny - hwidth, GRAB_LENGTH, GRAB_WIDTH)
        box(ENDPOINT, L.X1 - 1, L.Y2 - 1, L.xd1 + 1, L.yd2 + 1)

        # Grab box for re-apportioning the window between the panels.
        fill(FRACTION_BOX, yline_x - hlen - 2, xline_y - hlen - 2, GRAB_LENGTH + 4, GRAB_LENGTH + 4)

    # ------------------------------------------------------------------ hit testing (XyzWindow::Getxyz)
    def getxyz(self, qx: float, qy: float):
        """Region under a mouse position, with the coordinates it selects: ``(box, x, y, z)`` (floats)."""
        L = self.layout_geometry()
        x, y = qx + 0.5, L.H - qy - 0.5  # centre of the pixel under the mouse, in GL coordinates
        cz, cy, cx = self.point
        mx, my, mz = cx + 0.5, cy + 0.5, cz + 0.5
        scale = 1.0 / self.zoom
        if in_box(L.X1, L.Y1, L.X1 + L.xd1, L.Y1 + L.yd1, x, y):
            return Z_SLICE_BOX, (x - L.bx) * scale, (y - L.by) * scale, mz
        if in_box(L.X1, L.Y2, L.X1 + L.xd1, L.Y2 + L.yd2, x, y):
            return Y_SLICE_BOX, (x - L.bx) * scale, my, (y - L.by2) * scale
        if in_box(L.X2, L.Y1, L.X2 + L.xd2, L.Y1 + L.yd1, x, y):
            return X_SLICE_BOX, mx, (y - L.by) * scale, (x - L.bx2) * scale
        if in_box(L.X2 - 1, L.Y2 - 1, L.X2 + L.xd2 + 1, L.Y2 + L.yd2 + 1, x, y):
            return Z_GADGET_BOX, mx, my, 0.5 * ((y - L.by2) + (x - L.bx2)) * scale
        if in_box(L.X1 - 1, L.Y1 + L.yd1, L.X1 + L.xd1 + 1, L.Y2, x, y):
            return X_GADGET_BOX, (x - L.bx) * scale, my, mz
        if in_box(L.X1 + L.xd1, L.Y1 - 1, L.X2, L.Y1 + L.yd1 + 1, x, y):
            return Y_GADGET_BOX, mx, (y - L.by) * scale, mz
        if in_box(L.X1 + L.xd1, L.Y1 + L.yd1, L.X2, L.Y2, x, y):
            return FRACTION_BOX_ID, mx, my, mz
        return NOT_IN_BOX, mx, my, mz

    def panel_at(self, pos) -> Optional[int]:
        """Plane axis of the panel under a position, if any."""
        return BOX_AXIS.get(self.getxyz(pos.x(), pos.y())[0])

    @property
    def point(self):
        return self.locked_point if self.locked_point is not None else self.session.point

    def set_location(self, x: float, y: float, z: float) -> None:
        zyx = [int(np.floor(z)), int(np.floor(y)), int(np.floor(x))]
        if self.locked_point is None:
            self.session.set_point(*zyx)
            return
        shape = self.session.volume.shape
        self.locked_point = [int(np.clip(v, 0, n - 1)) for v, n in zip(zyx, shape)]
        self.update()
        self.window()._sync()

    def move(self, axis: int, step: int) -> None:
        zyx = list(self.point)
        zyx[axis] += step
        self.set_location(zyx[2] + 0.5, zyx[1] + 0.5, zyx[0] + 0.5)

    def set_sum(self, n: int) -> None:
        for view in self.views.values():
            view.thickness = max(1, int(n))
        self.update()

    # ------------------------------------------------------------------ mouse
    def mousePressEvent(self, event) -> None:
        pos = event.pos()
        if event.button() == Qt.LeftButton:
            self._press_t = time.monotonic()
            self._which = self.getxyz(pos.x(), pos.y())[0]
            self._pressed = True
        elif event.button() in (Qt.MiddleButton, Qt.RightButton):
            axis = self.panel_at(pos)
            self.session.toggle_movie(1 if event.button() == Qt.MiddleButton else -1, 0 if axis is None else axis)
        self._first = self._last = (pos.x(), pos.y())

    def mouseMoveEvent(self, event) -> None:
        pos = event.pos()
        self._mouse = QPointF(pos)
        if self._pressed:
            dx, dy = self._first[0] - pos.x(), self._first[1] - pos.y()
            held = (time.monotonic() - self._press_t) * 1000 > CLICK_MS
            if self._which == NOT_IN_BOX or self._which > Z_SLICE_BOX or held or dx * dx + dy * dy > DRAG_PX**2:
                self.b1_drag(pos.x(), pos.y())
        self._last = (pos.x(), pos.y())

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self._pressed:
            self._pressed = False
            if (time.monotonic() - self._press_t) * 1000 <= CLICK_MS:
                self.b1_press(event.pos().x(), event.pos().y())

    def wheelEvent(self, event) -> None:
        event.ignore()  # 3dmod does nothing with the wheel in movie mode

    def b1_press(self, qx: int, qy: int) -> None:
        box, mx, my, mz = self.getxyz(qx, qy)
        if box in (NOT_IN_BOX, FRACTION_BOX_ID):
            return
        self.set_location(mx, my, mz)

    def b1_drag(self, x: int, y: int) -> None:
        """XyzWindow::B1Drag: pan the panels, drag a position marker, or re-apportion the panels."""
        L = self.layout_geometry()
        H = L.H
        box = self._which
        if box == Y_GADGET_BOX:  # do not drag out of a gutter
            yinv = H - y
            if yinv < L.Y1 + 1:
                y = H - (L.Y1 + 1)
            elif yinv > L.Y1 + L.yd1:
                y = H - (L.Y1 + L.yd1)
        elif box == X_GADGET_BOX:
            x = min(max(x, L.X1), L.X1 + L.xd1 - 1)
        elif box == Z_GADGET_BOX:
            x = min(max(x, L.X2), L.X2 + L.xd2 - 1)
            yinv = H - y
            if yinv < L.Y2 + 1:
                y = H - (L.Y2 + 1)
            elif yinv > L.Y2 + L.yd2:
                y = H - (L.Y2 + L.yd2)

        scale = 1.0 / self.zoom
        fac = trans_factor(self.zoom)
        lmx, lmy = self._last
        delx = int(np.floor(fac * (x - lmx) + 0.5))
        dely = int(np.floor(fac * (y - lmy) + 0.5))
        cz, cy, cx = self.point
        if box == NOT_IN_BOX:
            return
        if box == X_SLICE_BOX:
            self.xtrans2 += delx
            self.ytrans1 -= dely
            self.ytrans2 += delx
        elif box == Y_SLICE_BOX:
            self.xtrans1 += delx
            self.ytrans2 -= dely
            self.xtrans2 -= dely
        elif box == Z_SLICE_BOX:
            self.xtrans1 += delx
            self.ytrans1 -= dely
        elif box == Z_GADGET_BOX:
            self.set_location(cx, cy, 0.5 * ((H - y - 0.5 - L.by2) + (x + 0.5 - L.bx2)) * scale)
            return
        elif box == X_GADGET_BOX:
            self.set_location((x + 0.5 - L.bx) * scale, cy, cz)
            return
        elif box == Y_GADGET_BOX:
            self.set_location(cx, (H - y - 0.5 - L.by) * scale, cz)
            return
        elif box == FRACTION_BOX_ID:
            self._drag_fraction(L, x, y, lmx, lmy)
        self.update()

    def _drag_fraction(self, L: Layout, x: int, y: int, lmx: int, lmy: int) -> None:
        nz, ny, nx = self.session.volume.shape
        delz = -0.5 * (x - lmx + lmy - y)
        xfrac = L.xd2 / (L.xd1 + L.xd2)
        yfrac = L.yd2 / (L.yd1 + L.yd2)
        if yfrac / self.yz_fraction > xfrac / self.xz_fraction:
            delfrac = ((L.yd2 + delz) / (L.yd1 + L.yd2)) / yfrac
        else:
            delfrac = ((L.xd2 + delz) / (L.xd1 + L.xd2)) / xfrac
        self.xz_fraction *= delfrac
        self.yz_fraction *= delfrac
        total = nx + nz
        self.xz_fraction = max(2.0 / total, min((total - 2.0) / total, self.xz_fraction))
        total = ny + nz
        self.yz_fraction = max(2.0 / total, min((total - 2.0) / total, self.yz_fraction))

    # ------------------------------------------------------------------ zoom / centre
    def set_zoom(self, zoom: float) -> None:
        self.zoom = float(min(max(zoom, 0.005), 50.0))
        self.update()
        win = self.window()
        if hasattr(win, "zoom_changed"):
            win.zoom_changed()

    def center_on_point(self) -> None:
        """XyzWindow::centerClicked (K): centre every panel on the current point."""
        nz, ny, nx = self.session.volume.shape
        cz, cy, cx = self.point
        self.xtrans1 = int(nx / 2.0 - cx)
        self.ytrans1 = int(ny / 2.0 - cy)
        self.xtrans2 = int(nz / 2.0 - cz)
        self.ytrans2 = self.xtrans2
        self.update()

    # ------------------------------------------------------------------ keys
    def keyPressEvent(self, event) -> None:
        key, mods = event.key(), event.modifiers()
        shift = bool(mods & Qt.ShiftModifier)
        ctrl = bool(mods & Qt.ControlModifier)
        keypad = bool(mods & Qt.KeypadModifier)
        s = self.session
        box = self.getxyz(self._mouse.x(), self._mouse.y())[0]
        if key == Qt.Key_Minus:
            self.set_zoom(zoom_step(self.zoom, False))
        elif key in (Qt.Key_Equal, Qt.Key_Plus):
            self.set_zoom(zoom_step(self.zoom, True))
        elif key == Qt.Key_R and shift:
            self.window().fit_to_image()
        elif key == Qt.Key_R and not ctrl:
            self.window().hq_button.toggle()
            s.message.emit(f"High-resolution mode {'ON' if self.hq else 'OFF'}")
        elif key == Qt.Key_K:
            self.center_on_point()
        elif snapshot_format(event):
            quick_snapshot(self.window(), snapshot_format(event))
        elif key in (Qt.Key_PageUp, Qt.Key_PageDown, Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down) and not keypad:
            # Over the side panels, PageUp/Down step through that panel's planes and the arrows follow its axes.
            use = key
            if box == X_SLICE_BOX:  # YZ: horizontal is Z
                use = {
                    Qt.Key_PageUp: Qt.Key_Right,
                    Qt.Key_PageDown: Qt.Key_Left,
                    Qt.Key_Left: Qt.Key_PageDown,
                    Qt.Key_Right: Qt.Key_PageUp,
                }.get(key, key)
            elif box == Y_SLICE_BOX:  # XZ: vertical is Z
                use = {
                    Qt.Key_PageUp: Qt.Key_Up,
                    Qt.Key_PageDown: Qt.Key_Down,
                    Qt.Key_Down: Qt.Key_PageDown,
                    Qt.Key_Up: Qt.Key_PageUp,
                }.get(key, key)
            axis, step = {
                Qt.Key_PageUp: (0, 1),
                Qt.Key_PageDown: (0, -1),
                Qt.Key_Right: (2, 1),
                Qt.Key_Left: (2, -1),
                Qt.Key_Up: (1, 1),
                Qt.Key_Down: (1, -1),
            }[use]
            self.move(axis, step)
        elif key == Qt.Key_Escape:
            self.window().close()
        elif not keys.handle_global(s, event, axis=BOX_AXIS.get(box, 0)):
            super().keyPressEvent(event)


class XyzWindow(QMainWindow):
    def __init__(self, session: Session):
        super().__init__(None)
        self.session = session
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowTitle(f"cmod XYZ Window: {session.title}")
        keep_alive(self)
        self.canvas = XyzCanvas(session, self)
        self.setCentralWidget(self.canvas)

        tb = make_toolbar(self, "XYZ Toolbar")
        self.toolbar = tb
        self.zoom_edit = tb_zoom_tools(
            tb,
            lambda upward: self.canvas.set_zoom(zoom_step(self.canvas.zoom, upward)),
            self._zoom_typed,
        )
        self.hq_button = tb_toggle(
            checker_icon(False),
            checker_icon(True),
            "Toggle between regular and high-resolution (interpolated) image (R)",
            checked=True,
        )
        self.hq_button.toggled.connect(self._hq_toggled)
        self.lock_button = tb_toggle(
            lock_icon(False),
            lock_icon(True),
            "Keep display at set location regardless of global location",
        )
        self.lock_button.toggled.connect(self._lock_toggled)
        zscale = tb_toggle(zscale_icon(False), zscale_icon(True), "Apply model Z scaling to display")
        zscale.setEnabled(False)
        zscale.setToolTip(zscale.toolTip() + " (copick volumes are isotropic)")
        for b in (self.hq_button, self.lock_button, zscale):
            tb.addWidget(b)
        tb.addWidget(
            tb_button(
                center_icon(True),
                "Center windows on current image or model point (K)",
                self.canvas.center_on_point,
            ),
        )
        tb.addWidget(QLabel("Sum"))
        self.sum_spin = QSpinBox()
        self.sum_spin.setRange(1, 999)
        self.sum_spin.setFocusPolicy(Qt.ClickFocus)
        self.sum_spin.setKeyboardTracking(False)
        self.sum_spin.setToolTip("Number of slices to sum in each plane")
        self.sum_spin.valueChanged.connect(self._sum_changed)
        tb.addWidget(self.sum_spin)
        self.sliders: Dict[int, QSlider] = {}
        for axis, name in ((2, "X"), (1, "Y"), (0, "Z")):
            label = QLabel(f" {name}")
            label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            tb.addWidget(label)
            slider = QSlider(Qt.Horizontal)
            n = session.volume.shape[axis]
            slider.setRange(1, n)
            slider.setFocusPolicy(Qt.NoFocus)
            slider.setFixedWidth(section_slider_width(n, slider))
            slider.setToolTip(f"Set {name} coordinate")
            slider.valueChanged.connect(lambda v, a=axis: self._slider_moved(a, v - 1))
            self.sliders[axis] = slider
            tb.addWidget(slider)
        tb.addWidget(tb_push("Help", "Open help window", lambda: session.open("help")))

        session.point_changed.connect(self._sync)
        session.image_changed.connect(self._image_changed)
        session.images_changed.connect(self._retitle)
        self._sync()
        self.zoom_changed()
        self.canvas.setFocus()

    def _retitle(self) -> None:
        self.setWindowTitle(f"cmod XYZ Window: {self.session.title}")

    def _image_changed(self, old) -> None:
        for axis, slider in self.sliders.items():
            n = self.session.volume.shape[axis]
            slider.blockSignals(True)
            slider.setRange(1, n)
            slider.blockSignals(False)
            slider.setFixedWidth(section_slider_width(n, slider))
        self._retitle()
        self._sync()
        self.zoom_changed()

    def _sync(self) -> None:
        for axis, slider in self.sliders.items():
            slider.blockSignals(True)
            slider.setValue(self.canvas.point[axis] + 1)
            slider.blockSignals(False)

    def _slider_moved(self, axis: int, value: int) -> None:
        zyx = list(self.canvas.point)
        zyx[axis] = value
        self.canvas.set_location(zyx[2] + 0.5, zyx[1] + 0.5, zyx[0] + 0.5)

    def zoom_changed(self) -> None:
        self.zoom_edit.setText(zoom_text(self.canvas.zoom))

    def _zoom_typed(self, zoom: float) -> None:
        self.canvas.set_zoom(zoom)
        self.canvas.setFocus()

    def _hq_toggled(self, on: bool) -> None:
        self.canvas.hq = on
        self.canvas.update()

    def _lock_toggled(self, on: bool) -> None:
        self.canvas.locked_point = list(self.session.point) if on else None
        self._sync()
        self.canvas.update()

    def _sum_changed(self, n: int) -> None:
        self.canvas.set_sum(n)
        self.canvas.setFocus()

    def fit_to_image(self) -> None:
        """Shift+R (XyzWindow::resizeToFit): reset the panel fractions and fit the window to the image."""
        self.canvas.reset_fractions()
        hint = self.canvas.sizeHint()
        self.resize(hint.width(), hint.height() + self.toolbar.sizeHint().height())
        self.canvas.update()

    def render_image(self, scale: int = 1, overlays: bool = True, cursor: bool = True, band: bool = True):
        """The XYZ window at ``scale`` times its size (finer data), and its angstroms per pixel."""
        from copickmod.snapshot import render_until_complete

        c, s = self.canvas, self.session
        width, height, zoom = c.width() * scale, c.height() * scale, c.zoom * scale
        views = {axis: PlaneView(s, axis) for axis in (0, 1, 2)}
        for axis, view in views.items():
            view.point_source = lambda: c.point
            view.thickness = c.views[axis].thickness
        layout = c.layout_geometry(width, height, zoom)
        try:
            img = render_until_complete(
                lambda p: c.paint_scene(p, layout, zoom, views, overlays, cursor), width, height, s,
            )  # fmt: skip
        finally:
            for view in views.values():
                s.drop_view(view.view_id)
        return img, s.volume.voxel_size / zoom

    def keyPressEvent(self, event) -> None:
        self.canvas.keyPressEvent(event)

    def closeEvent(self, event) -> None:
        for view in self.canvas.views.values():
            self.session.drop_view(view.view_id)
        super().closeEvent(event)
