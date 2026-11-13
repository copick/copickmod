"""Rendering shared by the ZaP and XYZ windows: an axis-aligned plane with overlays, 3dmod zoom steps."""

from __future__ import annotations

import itertools
import time
from typing import Optional, Tuple

import numpy as np
from qtpy.QtCore import QPointF, QRectF
from qtpy.QtGui import QColor, QImage, QPainter, QPainterPath, QPen

from copickmod import overlays
from copickmod.contrast import gray_to_rgb
from copickmod.state import Session
from copickmod.style import BACKGROUND, CURPOINT, draw_plus  # noqa: F401 - BACKGROUND re-exported
from copickmod.volume import PLANE_AXES

#: 3dmod's zoom table (preferences.cpp).
ZOOMS = [0.01, 0.02, 0.04, 0.07, 0.1, 0.1667, 0.25, 0.3333, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20]
CURSOR_COLOR = CURPOINT
CLICK_MS = 250  # 3dmod: a press released sooner is a click
DRAG_PX = 6  # 3dmod: or the mouse moved further than this

_view_ids = itertools.count(1)


def zoom_step(zoom: float, up: bool) -> float:
    """Next zoom in 3dmod's table above or below ``zoom``."""
    if up:
        return next((z for z in ZOOMS if z > zoom * 1.001), ZOOMS[-1])
    return next((z for z in reversed(ZOOMS) if z < zoom * 0.999), ZOOMS[0])


def fit_zoom(width: float, height: float, avail_w: float, avail_h: float) -> float:
    """3dmod start-up zoom: 1.0, stepped down the table until the image fits."""
    zoom = 1.0
    while zoom > ZOOMS[0] and (width * zoom > avail_w or height * zoom > avail_h):
        zoom = zoom_step(zoom, up=False)
    return zoom


def gray_image(data: np.ndarray) -> QImage:
    """uint8 2D array (row 0 at the bottom) -> QImage (row 0 at the top)."""
    buf = np.ascontiguousarray(data[::-1])
    h, w = buf.shape
    return QImage(buf.data, w, h, w, QImage.Format_Grayscale8).copy()


def rgb_image(data: np.ndarray) -> QImage:
    """(H, W, 3) uint8 (row 0 at the bottom) -> QImage."""
    buf = np.ascontiguousarray(data[::-1])
    h, w = buf.shape[:2]
    return QImage(buf.data, w, h, 3 * w, QImage.Format_RGB888).copy()


def rgba_image(data: np.ndarray) -> QImage:
    buf = np.ascontiguousarray(data[::-1])
    h, w = buf.shape[:2]
    return QImage(buf.data, w, h, 4 * w, QImage.Format_RGBA8888).copy()


def qcolor(rgba, alpha: Optional[int] = None) -> QColor:
    r, g, b = rgba[:3]
    a = rgba[3] if len(rgba) > 3 else 255
    return QColor(int(r), int(g), int(b), int(a if alpha is None else alpha))


def draw_cross(p: QPainter, pos: QPointF, color: QColor = CURSOR_COLOR) -> None:
    """3dmod's current image point: a small plus."""
    draw_plus(p, pos.x(), pos.y(), color)


def image_offset(winsize: int, imsize: int, zoom: float, offset: int) -> Tuple[int, int, int]:
    """Port of 3dmod's ``b3dSetImageOffset`` for one axis: where the image sits in a window.

    ``offset`` is the pan, in image pixels, of the image centre from the window centre. An image that fits is always
    centred (panning has no effect); a larger one can be panned until at most 1/6 of the window shows past its edge.

    Returns ``(woff, doff, offset)``: the window position of the first drawn image pixel, the first drawn image pixel,
    and the offset (clamped when it was out of bounds, as 3dmod does to prevent image creep).
    """
    if int(imsize * zoom) <= winsize:
        return int((winsize - int(imsize * zoom)) / 2), 0, offset
    woff = 0
    drawsize = int(winsize / zoom)
    doff = int((imsize // 2) - (winsize / zoom / 2)) - offset
    if doff < 0:
        maxwoff = winsize // 6
        woff = int(-doff * zoom)
        if woff > maxwoff:
            woff = maxwoff
            offset = int(imsize * 0.5 - ((winsize * 0.5 - woff) / zoom))
        doff = 0
    elif doff + drawsize > imsize - 1:
        minds = int((winsize * 0.8333333) / zoom)
        drawsize = imsize - doff
        if drawsize < minds:
            drawsize = minds
            doff = imsize - drawsize
            offset = max(offset, int(imsize * 0.5 - doff - (winsize * 0.5) / zoom - 1.0))
    return woff, doff, offset


def axis_center(winsize: int, imsize: int, zoom: float, offset: int) -> Tuple[float, int]:
    """World coordinate shown at the middle of a window axis, and the (possibly clamped) pan offset."""
    woff, doff, offset = image_offset(winsize, imsize, zoom, offset)
    return doff + (winsize / 2.0 - woff) / zoom, offset


def trans_factor(zoom: float) -> float:
    """3dmod drag rate: the image follows the mouse below zoom 1, else moves 1 image pixel per screen pixel."""
    return 1.0 / zoom if zoom < 1.0 else 1.0


class PlaneView:
    """Draws one axis-aligned plane of the session volume into a rectangle.

    The mapping is: screen x grows with the plane's column axis, screen y *decreases* with the row axis (3dmod draws
    Y, and Z in the XZ panel, pointing up). ``center`` is the world (row, col) coordinate shown at the rectangle centre.
    """

    def __init__(self, session: Session, axis: int):
        self.session = session
        self.axis = axis
        self.view_id = next(_view_ids)
        self.level = 0
        self.complete = False  # until the first paint has found all its data
        self.point_source = None  # optional callable giving the point to show (a locked window's own point)
        self.thickness = 1  # number of level-0 sections summed (3dmod XYZ "Sum")

    def point(self):
        return self.point_source() if self.point_source is not None else self.session.point

    @property
    def row_ax(self) -> int:
        return PLANE_AXES[self.axis][0]

    @property
    def col_ax(self) -> int:
        return PLANE_AXES[self.axis][1]

    # ------------------------------------------------------------------ coordinates
    @staticmethod
    def to_screen(rect: QRectF, zoom: float, center, row: float, col: float) -> QPointF:
        c = rect.center()
        return QPointF(c.x() + (col - center[1]) * zoom, c.y() - (row - center[0]) * zoom)

    @staticmethod
    def to_world(rect: QRectF, zoom: float, center, pos: QPointF) -> Tuple[float, float]:
        c = rect.center()
        return center[0] - (pos.y() - c.y()) / zoom, center[1] + (pos.x() - c.x()) / zoom

    # ------------------------------------------------------------------ rendering
    def render(
        self,
        p: QPainter,
        rect: QRectF,
        zoom: float,
        center: Tuple[float, float],
        hq: bool,
        clip: Optional[QRectF] = None,
        cursor: bool = True,
        overlays: bool = True,
    ) -> bool:
        """Draw the image, segmentations, points and the current-point cross. Returns True if complete."""
        s = self.session
        vis = rect if clip is None else rect.intersected(clip)
        wanted: set = set()
        self.complete = True
        p.save()
        p.setClipRect(vis)
        if vis.width() > 0 and vis.height() > 0:
            show_model = overlays and s.show_model
            self._draw_image(p, rect, vis, zoom, center, hq, wanted, show_model)
            if show_model:
                self._draw_overlays(p, rect, vis, zoom, center, wanted)
            if cursor and s.show_markers:
                pt = self.point()
                draw_cross(p, self.to_screen(rect, zoom, center, pt[self.row_ax] + 0.5, pt[self.col_ax] + 0.5))
        p.restore()
        s.set_wanted(self.view_id, wanted)
        return self.complete

    def _visible_world(self, rect, vis, zoom, center):
        top_row, left_col = self.to_world(rect, zoom, center, vis.topLeft())
        bottom_row, right_col = self.to_world(rect, zoom, center, vis.bottomRight())
        return (bottom_row, top_row), (left_col, right_col)

    def _draw_image(self, p, rect, vis, zoom, center, hq, wanted, segmentations: bool = True) -> None:
        s = self.session
        vol = s.volume
        level = s.choose_level(zoom)
        self.level = level
        lv = vol.levels[level]
        ra, ca = self.row_ax, self.col_ax
        (row_lo, row_hi), (col_lo, col_hi) = self._visible_world(rect, vis, zoom, center)
        r0 = max(0, int(np.floor(vol.level_index(level, vol, row_lo, ra))))
        r1 = min(lv.shape[ra], int(np.ceil(vol.level_index(level, vol, row_hi, ra))))
        c0 = max(0, int(np.floor(vol.level_index(level, vol, col_lo, ca))))
        c1 = min(lv.shape[ca], int(np.ceil(vol.level_index(level, vol, col_hi, ca))))
        if r1 <= r0 or c1 <= c0:
            return
        plane_world = self.point()[self.axis] + 0.5
        fixed = int(np.clip(np.floor(vol.level_index(level, vol, plane_world, self.axis)), 0, lv.shape[self.axis] - 1))
        data, missing = vol.plane(s.cache, level, self.axis, fixed, (r0, r1), (c0, c1), request=False)
        if missing is None and self.thickness > 1:
            data = self._summed(level, fixed, (r0, r1), (c0, c1), data, wanted)
        if missing is not None:
            self.complete = False
            data = data.astype(np.float32)
            data[missing] = np.nan
            row_w = vol.world_coord(level, vol, np.arange(r0, r1) + 0.5, ra)
            col_w = vol.world_coord(level, vol, np.arange(c0, c1) + 0.5, ca)
            # Coarsest first, so its chunks are fetched first and finer levels overwrite it.
            for coarse in range(vol.nlevels - 1, level, -1):
                clv = vol.levels[coarse]
                cf = int(
                    np.clip(
                        np.floor(vol.level_index(coarse, vol, plane_world, self.axis)),
                        0,
                        clv.shape[self.axis] - 1,
                    ),
                )
                rr = np.clip(np.floor(vol.level_index(coarse, vol, row_w, ra)).astype(int), 0, clv.shape[ra] - 1)
                cc = np.clip(np.floor(vol.level_index(coarse, vol, col_w, ca)).astype(int), 0, clv.shape[ca] - 1)
                cdata, cmiss = vol.plane(
                    s.cache,
                    coarse,
                    self.axis,
                    cf,
                    (rr.min(), rr.max() + 1),
                    (cc.min(), cc.max() + 1),
                    wanted=wanted,
                )
                sub = cdata[np.ix_(rr - rr.min(), cc - cc.min())]
                fill = missing if cmiss is None else missing & ~cmiss[np.ix_(rr - rr.min(), cc - cc.min())]
                data[fill] = sub[fill]
            vol.plane(s.cache, level, self.axis, fixed, (r0, r1), (c0, c1), request=True, wanted=wanted)
        ahead = s.movie_ahead(self.axis)
        if ahead:
            # 3dmod movies run from memory; here the next layer of chunks is read while the movie is in this one.
            depth = lv.chunks[self.axis]
            layer = fixed // depth + ahead
            if not 0 <= layer < -(-lv.shape[self.axis] // depth):
                layer = fixed // depth - ahead  # the movie turns around at the end: the layer it reaches next
            nxt = min(max(layer, 0), (lv.shape[self.axis] - 1) // depth) * depth
            vol.request_plane(s.cache, level, self.axis, nxt, (r0, r1), (c0, c1), wanted=wanted)
        if not s.has_contrast:
            return  # nothing loaded yet: keep the background until the first section arrives
        gray = s.contrast.map(data)
        img = rgb_image(gray_to_rgb(gray, True)) if s.contrast.false_color else gray_image(gray)
        top_left = self.to_screen(
            rect,
            zoom,
            center,
            vol.world_coord(level, vol, r1, ra),
            vol.world_coord(level, vol, c0, ca),
        )
        bottom_right = self.to_screen(
            rect,
            zoom,
            center,
            vol.world_coord(level, vol, r0, ra),
            vol.world_coord(level, vol, c1, ca),
        )
        p.setRenderHint(QPainter.SmoothPixmapTransform, hq)
        p.drawImage(QRectF(top_left, bottom_right), img)

        if segmentations and any(layer.visible for layer in s.seg_layers):
            row_w = vol.world_coord(level, vol, np.arange(r0, r1) + 0.5, ra)
            col_w = vol.world_coord(level, vol, np.arange(c0, c1) + 0.5, ca)
            p.setRenderHint(QPainter.SmoothPixmapTransform, False)
            p.setOpacity(s.opacity)
            for layer in s.seg_layers:
                if not layer.visible:
                    continue
                rgba, complete = overlays.seg_plane_rgba(
                    layer,
                    s.cache,
                    vol,
                    self.axis,
                    plane_world,
                    row_w,
                    col_w,
                    float(lv.scale[2]),
                )
                self.complete &= complete
                if rgba is not None:
                    p.drawImage(QRectF(top_left, bottom_right), rgba_image(rgba))
            p.setOpacity(1.0)

    def _summed(self, level, fixed, rows, cols, data, wanted):
        """Average of ``thickness`` sections around ``fixed`` (3dmod XYZ "Sum"); the single plane until all load."""
        s = self.session
        vol = s.volume
        n = max(1, int(round(self.thickness / float(vol.factor(level)[self.axis]))))
        if n == 1:
            return data
        size = vol.levels[level].shape[self.axis]
        indices = [i for i in range(fixed - (n - 1) // 2, fixed + n // 2 + 1) if 0 <= i < size]
        total = np.zeros(data.shape, dtype=np.float32)
        for i in indices:
            plane, missing = vol.plane(s.cache, level, self.axis, i, rows, cols, wanted=wanted)
            if missing is not None:
                self.complete = False
                return data
            total += plane
        return total / len(indices)

    def _draw_overlays(self, p, rect, vis, zoom, center, wanted) -> None:
        s = self.session
        plane_world = self.point()[self.axis] + 0.5
        p.setRenderHint(QPainter.Antialiasing, True)
        for layer in s.point_layers:
            if not layer.visible:
                continue
            color = qcolor(layer.rgba, 255)
            if layer.kind == "filaments":
                p.setPen(QPen(color, 2.0))
                for seg in overlays.slab_segments(layer.polylines, layer.radius, self.axis, plane_world):
                    path = QPainterPath()
                    for i, (r, c) in enumerate(seg):
                        q = self.to_screen(rect, zoom, center, r, c)
                        path.moveTo(q) if i == 0 else path.lineTo(q)
                    if len(seg) == 1:
                        q = self.to_screen(rect, zoom, center, seg[0][0], seg[0][1])
                        p.drawEllipse(q, 2.0, 2.0)
                    p.drawPath(path)
                continue
            rows, cols, radii, on = overlays.sphere_sections(layer.points, layer.radius, self.axis, plane_world)
            for r, c, rad, is_on in zip(rows, cols, radii, on):
                q = self.to_screen(rect, zoom, center, r, c)
                p.setPen(QPen(color, 2.0 if is_on else 1.0))
                p.drawEllipse(q, rad * zoom, rad * zoom)
                if is_on:
                    p.drawPoint(q)
        p.setRenderHint(QPainter.Antialiasing, False)


def pixel_center(pos) -> QPointF:
    """Mouse positions are integer pixel indices; the pixel's centre is half a pixel further."""
    return QPointF(pos.x() + 0.5, pos.y() + 0.5)


class ClickDragMixin:
    """3dmod left-button semantics: a press released within 250 ms is a click (acted on at release); the image is
    dragged once the button has been held for 250 ms or the mouse moved more than 6 pixels."""

    def _press(self, event) -> None:
        self._press_pos = event.pos()
        self._press_t = time.monotonic()
        self._last_pos = event.pos()
        self._dragging = False

    def _move(self, event) -> Optional[Tuple[float, float]]:
        """Return the mouse movement since the last event once a drag is under way."""
        if getattr(self, "_press_pos", None) is None:  # set only while the left button is down
            return None
        delta = event.pos() - self._last_pos
        self._last_pos = event.pos()
        if not self._dragging:
            moved = event.pos() - self._press_pos
            held = (time.monotonic() - self._press_t) * 1000 > CLICK_MS
            if held or moved.x() ** 2 + moved.y() ** 2 > DRAG_PX**2:
                self._dragging = True
            else:
                return None
        return float(delta.x()), float(delta.y())

    def _release_is_click(self, event) -> bool:
        if getattr(self, "_press_pos", None) is None:
            return False
        click = not self._dragging and (time.monotonic() - self._press_t) * 1000 <= CLICK_MS
        self._press_pos = None
        self._dragging = False
        return click
