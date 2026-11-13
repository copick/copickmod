"""Run gallery: central XY slices of every run as cards (copick-web's gallery, drawn in 3dmod's style).

Thumbnails are rendered in the background (visible cards first, four at a time) and cached in memory and on disk
(``$XDG_CACHE_HOME/cmod/thumbnails``), so a project opens with its previews straight away the next time.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import numpy as np
from qtpy.QtCore import QObject, QRect, QSize, Qt, Signal
from qtpy.QtGui import QColor, QFont, QFontMetrics, QImage, QPainter, QPen
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QSlider,
    QStyle,
    QStyledItemDelegate,
    QVBoxLayout,
    QWidget,
)

from copickmod.project import choose_tomogram
from copickmod.style import BACKGROUND, keep_alive

log = logging.getLogger(__name__)

STORED = 256  # thumbnails are rendered and cached at this size
PAD = 6


def make_thumbnail(run, size: int = STORED):
    """Central XY plane of the run's default tomogram, block-averaged to ``size`` with a 2-98% stretch.

    Returns ``(uint8 image, caption)`` or None when the run has no tomogram.
    """
    from copickmod.volume import MultiscaleVolume

    tomo = choose_tomogram(run)
    if tomo is None:
        return None
    vol = MultiscaleVolume(tomo.zarr(), fallback_voxel_size=float(tomo.voxel_spacing.voxel_size))
    best = None
    for level, lv in enumerate(vol.levels):
        edge = max(lv.shape[1:])
        if edge > 2048:
            continue
        if best is None or abs(edge - size) < abs(max(vol.levels[best].shape[1:]) - size):
            best = level
    best = vol.nlevels - 1 if best is None else best
    plane = vol.read_plane(best, vol.levels[best].shape[0] // 2).astype(np.float32)
    f = max(1, int(np.ceil(max(plane.shape) / size)))
    if f > 1:
        h, w = (plane.shape[0] // f) * f, (plane.shape[1] // f) * f
        plane = plane[:h, :w].reshape(h // f, f, w // f, f).mean(axis=(1, 3))
    finite = plane[np.isfinite(plane)]
    lo, hi = np.percentile(finite, [2, 98]) if finite.size else (0.0, 1.0)
    img = np.clip((plane - lo) / max(hi - lo, 1e-12) * 255, 0, 255).astype(np.uint8)
    return img[::-1].copy(), f"{tomo.tomo_type} · {float(tomo.voxel_spacing.voxel_size):.2f} Å"


def cache_root() -> str:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "cmod", "thumbnails")


def project_key(root) -> str:
    """Identifies a project for the disk cache (name, overlay location and portal datasets)."""
    cfg = getattr(root, "config", None)
    parts = [str(getattr(cfg, a, "")) for a in ("name", "overlay_root", "static_root", "dataset_ids")]
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


class ThumbnailCache(QObject):
    """Thumbnails of runs, shared by every gallery of a project: memory, then disk, then rendered in the background.

    Entries are ``("ok", QImage, caption)``, ``("none", None, "No tomogram")`` or ``("error", None, message)``.
    """

    ready = Signal(str)  # run name
    _done = Signal(str, object)

    def __init__(self, root, workers: int = 4):
        super().__init__()
        self.directory = os.path.join(cache_root(), project_key(root))
        self.entries: Dict[str, tuple] = {}
        self._queue: List = []
        self._inflight: set = set()
        self._workers = workers
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="cmod-thumb")
        self._done.connect(self._finish)
        self._closed = False

    def get(self, name: str) -> Optional[tuple]:
        return self.entries.get(name)

    def pending(self, name: str) -> bool:
        return name in self._inflight or any(r.name == name for r in self._queue)

    def request(self, runs) -> None:
        """Queue runs (in priority order: visible cards first); already cached or running ones are skipped."""
        names = {r.name for r in runs}
        queued = [r for r in self._queue if r.name not in names]
        self._queue = [r for r in runs if r.name not in self.entries and r.name not in self._inflight] + queued
        self._pump()

    def _pump(self) -> None:
        while not self._closed and self._queue and len(self._inflight) < self._workers:
            run = self._queue.pop(0)
            if run.name in self.entries or run.name in self._inflight:
                continue
            self._inflight.add(run.name)
            fut = self._pool.submit(self._load, run)
            fut.add_done_callback(lambda f, n=run.name: self._done.emit(n, f.exception() or f.result()))

    def _path(self, name: str) -> str:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
        return os.path.join(self.directory, f"{safe}.png")

    def _load(self, run) -> tuple:
        path = self._path(run.name)
        if os.path.exists(path):
            img = QImage(path)
            if not img.isNull():
                return "ok", img, img.text("caption")
        result = make_thumbnail(run)
        if result is None:
            return "none", None, "No tomogram"
        data, caption = result
        h, w = data.shape
        img = QImage(np.ascontiguousarray(data).data, w, h, w, QImage.Format_Grayscale8).copy()
        img.setText("caption", caption)
        try:
            os.makedirs(self.directory, exist_ok=True)
            img.save(path, "PNG")
        except OSError as exc:  # a read-only home is fine: just no disk cache
            log.debug("Could not cache thumbnail %s: %s", path, exc)
        return "ok", img, caption

    def _finish(self, name: str, result) -> None:
        self._inflight.discard(name)
        if isinstance(result, Exception):
            log.debug("Thumbnail for %s failed: %s", name, result)
            result = ("error", None, "Preview unavailable")
        self.entries[name] = result
        self.ready.emit(name)
        self._pump()

    def clear(self, disk: bool = True) -> None:
        """Forget all thumbnails (Reload project), also on disk."""
        self.entries.clear()
        if disk and os.path.isdir(self.directory):
            for f in os.listdir(self.directory):
                if f.endswith(".png"):
                    with contextlib.suppress(OSError):
                        os.remove(os.path.join(self.directory, f))

    def shutdown(self) -> None:
        self._closed = True
        self._queue.clear()
        self._pool.shutdown(wait=False, cancel_futures=True)


class CardDelegate(QStyledItemDelegate):
    """Draws a run as a card: thumbnail on 3dmod's slate background, bold name, grey caption."""

    def __init__(self, gallery: "Gallery"):
        super().__init__(gallery)
        self.gallery = gallery

    def _fonts(self, option):
        bold = QFont(option.font)
        bold.setBold(True)
        small = QFont(option.font)
        small.setPointSizeF(max(6.0, option.font.pointSizeF() - 1))
        return bold, small

    def sizeHint(self, option, index) -> QSize:
        bold, small = self._fonts(option)
        t = self.gallery.thumb_size
        text_h = QFontMetrics(bold).height() + QFontMetrics(small).height() + 6
        return QSize(t + 2 * PAD + 2, t + 3 * PAD + text_h)

    def paint(self, p: QPainter, option, index) -> None:
        run = index.data(Qt.UserRole)
        if run is None:
            return
        pal = option.palette
        rect = option.rect.adjusted(2, 2, -2, -2)
        hovered = bool(option.state & QStyle.State_MouseOver)
        current = run.name == self.gallery.current_run
        p.save()
        p.setRenderHint(QPainter.Antialiasing, False)
        p.fillRect(rect, pal.base() if not hovered else pal.alternateBase())
        border = pal.highlight().color() if current else (pal.dark().color() if hovered else pal.mid().color())
        p.setPen(QPen(border, 2 if current else 1))
        p.drawRect(rect.adjusted(0, 0, -1, -1))

        t = self.gallery.thumb_size
        well = QRect(rect.x() + PAD, rect.y() + PAD, rect.width() - 2 * PAD, t)
        p.fillRect(well, BACKGROUND)
        entry = self.gallery.cache.get(run.name)
        caption = ""
        if entry is not None and entry[0] == "ok":
            img = entry[1].scaled(well.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            p.drawImage(
                well.x() + (well.width() - img.width()) // 2,
                well.y() + (well.height() - img.height()) // 2,
                img,
            )
            caption = entry[2]
        else:
            p.setPen(QColor(200, 200, 220))
            p.drawText(well, Qt.AlignCenter, entry[2] if entry is not None else "Loading...")
            if entry is not None:
                caption = entry[2]

        bold, small = self._fonts(option)
        y = well.bottom() + PAD
        p.setFont(bold)
        p.setPen(pal.text().color())
        fm = QFontMetrics(bold)
        p.drawText(QRect(well.x(), y, well.width(), fm.height()), Qt.AlignLeft | Qt.AlignVCenter,
                   fm.elidedText(run.name, Qt.ElideRight, well.width()))  # fmt: skip
        y += fm.height() + 2
        p.setFont(small)
        p.setPen(pal.color(pal.Disabled, pal.Text) if caption else pal.text().color())
        fs = QFontMetrics(small)
        p.drawText(QRect(well.x(), y, well.width(), fs.height()), Qt.AlignLeft | Qt.AlignVCenter,
                   fs.elidedText(caption if entry is not None and entry[0] == "ok" else "", Qt.ElideMiddle, well.width()))  # fmt: skip
        p.restore()


class Gallery(QWidget):
    """Cards of all runs; click one to open it."""

    def __init__(self, browser, cache: ThumbnailCache, in_window: bool = False):
        super().__init__()
        self.browser = browser
        self.cache = cache
        self.current_run: Optional[str] = None
        self.thumb_size = 200 if in_window else 132
        self._started = False
        self._items: Dict[str, QListWidgetItem] = {}

        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(2)
        row = QHBoxLayout()
        if in_window:
            self.search = QLineEdit()
            self.search.setPlaceholderText("Search runs...")
            self.search.setClearButtonEnabled(True)
            self.search.textChanged.connect(self.filter)
            row.addWidget(self.search, 1)
        else:
            row.addWidget(QLabel("Central XY slices; click a run to open it."), 1)
        row.addWidget(QLabel("Size"))
        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setRange(80, 320)
        self.size_slider.setValue(self.thumb_size)
        self.size_slider.setFixedWidth(100)
        self.size_slider.setToolTip("Thumbnail size")
        self.size_slider.valueChanged.connect(self.set_size)
        row.addWidget(self.size_slider)
        if not in_window:
            pop = QPushButton("Window")
            pop.setToolTip("Show the gallery in its own, larger window")
            pop.clicked.connect(browser.show_gallery_window)
            row.addWidget(pop)
        lay.addLayout(row)

        self.list = QListWidget()
        self.list.setViewMode(QListView.IconMode)
        self.list.setResizeMode(QListView.Adjust)
        self.list.setMovement(QListView.Static)
        self.list.setUniformItemSizes(True)
        self.list.setSpacing(4)
        self.list.setSelectionMode(QListWidget.NoSelection)
        self.list.setMouseTracking(True)
        self.list.setVerticalScrollMode(QListView.ScrollPerPixel)
        self.list.verticalScrollBar().setSingleStep(24)
        self.list.setItemDelegate(CardDelegate(self))
        self.list.itemClicked.connect(lambda item: browser.open_run(item.data(Qt.UserRole)))
        self.list.verticalScrollBar().valueChanged.connect(lambda _v: self.request_visible())
        lay.addWidget(self.list, 1)
        self.empty = QLabel()
        self.empty.setAlignment(Qt.AlignCenter)
        self.empty.hide()
        lay.addWidget(self.empty)
        cache.ready.connect(self._ready)

    # ------------------------------------------------------------------ content
    def populate(self, runs) -> None:
        self.list.clear()
        self._items.clear()
        for run in runs:
            item = QListWidgetItem()
            item.setData(Qt.UserRole, run)
            item.setToolTip(f"{run.name}\nClick to open")
            self.list.addItem(item)
            self._items[run.name] = item
        self._relayout()
        self.request_visible()

    def set_current(self, run_name: Optional[str]) -> None:
        self.current_run = run_name
        self.list.viewport().update()
        item = self._items.get(run_name or "")
        if item is not None and self.isVisible():
            self.list.scrollToItem(item, QListView.EnsureVisible)

    def set_size(self, size: int) -> None:
        self.thumb_size = int(size)
        self._relayout()
        self.request_visible()

    def _relayout(self) -> None:
        delegate = self.list.itemDelegate()
        hint = delegate.sizeHint(self.list.viewOptions(), self.list.model().index(0, 0))
        self.list.setGridSize(hint + QSize(4, 4))
        self.list.doItemsLayout()

    def filter(self, query: str) -> None:
        from copickmod.browser import matches

        shown = 0
        for item in self._items.values():
            hide = not matches(query, [item.data(Qt.UserRole).name])
            item.setHidden(hide)
            shown += not hide
        self.empty.setText(f"No runs match “{query.strip()}”")
        self.empty.setVisible(bool(self._items) and not shown)
        self.request_visible()

    # ------------------------------------------------------------------ loading
    def start(self) -> None:
        self._started = True
        self.request_visible()

    def request_visible(self) -> None:
        """Render the cards on screen first, then the rest."""
        if not self._started or not self._items:
            return
        view = self.list.viewport().rect()
        visible, rest = [], []
        for item in self._items.values():
            if item.isHidden():
                continue
            run = item.data(Qt.UserRole)
            (visible if self.list.visualItemRect(item).intersects(view) else rest).append(run)
        self.cache.request(visible + rest)

    def _ready(self, name: str) -> None:
        item = self._items.get(name)
        if item is not None:
            entry = self.cache.get(name)
            if entry is not None:
                item.setToolTip(f"{name}\n{entry[2]}\nClick to open")
            self.list.update(self.list.indexFromItem(item))

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.request_visible()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.request_visible()


class GalleryWindow(QMainWindow):
    """The gallery in its own, larger window."""

    def __init__(self, browser, cache: ThumbnailCache, title: str):
        super().__init__(None)
        self.setWindowTitle(f"cmod Run Gallery: {title}")
        keep_alive(self)
        self.gallery = Gallery(browser, cache, in_window=True)
        self.setCentralWidget(self.gallery)
        self.resize(1000, 760)
