"""Snapshots: 3dmod's Ctrl+S / Shift+S / Ctrl+Shift+S, and a "Grab with Note"-style Snapshot tool (Shift+H).

Any window can be saved as it looks, or re-rendered at a higher resolution (finer pyramid data, waiting for it to
load), optionally limited to the ZaP rubber band, with a scale bar and a note printed under the image. Images are
saved as PNG, TIFF or JPEG with 3dmod-style names (zap000.png, xyz000.png, slicer000.png, modv0000.png) or copied to
the clipboard.
"""

from __future__ import annotations

import math
import os
import time
from typing import Callable, List, Optional, Tuple

from qtpy.QtCore import QRectF, Qt
from qtpy.QtGui import QColor, QFont, QImage, QPainter, QPen
from qtpy.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

FORMATS = {"PNG": "png", "TIFF": "tif", "JPEG": "jpg"}
#: Snapshot file prefixes, as 3dmod names them.
PREFIXES = {
    "ZapWindow": ("zap", 3),
    "XyzWindow": ("xyz", 3),
    "SlicerWindow": ("slicer", 3),
    "ModelViewWindow": ("modv", 4),
}


# ---------------------------------------------------------------------------------------------- rendering
def render_until_complete(paint: Callable[[QPainter], bool], width: int, height: int, session, timeout: float = 120.0):
    """Render with ``paint`` into a new image, repeating while it reports missing data (up to ``timeout`` s)."""
    deadline = time.monotonic() + timeout
    while True:
        img = QImage(int(width), int(height), QImage.Format_RGB32)
        img.fill(Qt.black)
        p = QPainter(img)
        complete = paint(p)
        p.end()
        if complete or time.monotonic() > deadline:
            return img
        session.cache.wait(1.0)  # fetch threads fill the cache; no event processing, so live views stay quiet


def window_name(win) -> str:
    kind = type(win).__name__
    return {
        "ZapWindow": "ZaP",
        "XyzWindow": "XYZ",
        "SlicerWindow": "Slicer",
        "ModelViewWindow": "Model View",
        "InfoWindow": "Info",
    }.get(kind, kind)


def render_window(win, scale: int = 1, overlays: bool = True, cursor: bool = True, band: bool = True):
    """Image of a cmod window and its angstroms per pixel (None if not an image view)."""
    render = getattr(win, "render_image", None)
    if render is None:
        return win.grab().toImage(), None
    return render(scale=scale, overlays=overlays, cursor=cursor, band=band)


# ---------------------------------------------------------------------------------------------- decorations
def nice_scale_length(angstrom_per_pixel: float, width_px: int, fraction: float = 0.2) -> float:
    """A round scale-bar length in nm (1, 2 or 5 times a power of ten) near ``fraction`` of the image width."""
    target = width_px * fraction * angstrom_per_pixel / 10.0
    if target <= 0:
        return 0.0
    exp = math.floor(math.log10(target))
    candidates = [m * 10.0**e for e in (exp - 1, exp, exp + 1) for m in (1, 2, 5)]
    return min(candidates, key=lambda c: abs(math.log(c / target)))


def scale_label(nm: float) -> str:
    if nm >= 1000:
        return f"{nm / 1000:g} µm"
    return f"{nm:g} nm"


def draw_scale_bar(img: QImage, angstrom_per_pixel: float, length_nm: float = 0.0, ui_scale: float = 1.0) -> float:
    """3dmod-style scale bar in the lower right corner; returns the length used (nm)."""
    if not angstrom_per_pixel:
        return 0.0
    length_nm = length_nm or nice_scale_length(angstrom_per_pixel, img.width())
    length_px = length_nm * 10.0 / angstrom_per_pixel
    margin = 16 * ui_scale
    thick = max(3.0, 4 * ui_scale)
    x1 = img.width() - margin
    x0 = x1 - length_px
    y0 = img.height() - margin - thick
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing, True)
    p.setPen(QPen(Qt.black, 1))
    p.setBrush(Qt.white)
    p.drawRect(QRectF(x0, y0, length_px, thick))
    font = QFont(p.font())
    font.setPixelSize(int(round(13 * ui_scale)))
    font.setBold(True)
    p.setFont(font)
    text_rect = QRectF(x0 - 40 * ui_scale, y0 - 20 * ui_scale, length_px + 80 * ui_scale, 18 * ui_scale)
    p.setPen(QColor(0, 0, 0, 200))
    p.drawText(text_rect.translated(1, 1), Qt.AlignHCenter | Qt.AlignBottom, scale_label(length_nm))
    p.setPen(Qt.white)
    p.drawText(text_rect, Qt.AlignHCenter | Qt.AlignBottom, scale_label(length_nm))
    p.end()
    return length_nm


def add_caption(img: QImage, text: str, ui_scale: float = 1.0) -> QImage:
    """A white strip with the note under the image (3dmod "Grab with Note")."""
    text = text.strip()
    if not text:
        return img
    font = QFont(QApplication.font())
    font.setPixelSize(int(round(13 * ui_scale)))
    probe = QImage(1, 1, QImage.Format_RGB32)
    p = QPainter(probe)
    p.setFont(font)
    pad = int(8 * ui_scale)
    bounds = p.boundingRect(QRectF(pad, 0, img.width() - 2 * pad, 10_000), Qt.TextWordWrap, text)
    p.end()
    out = QImage(img.width(), img.height() + int(bounds.height()) + 2 * pad, QImage.Format_RGB32)
    out.fill(Qt.white)
    p = QPainter(out)
    p.drawImage(0, 0, img)
    p.setFont(font)
    p.setPen(Qt.black)
    p.drawText(QRectF(pad, img.height() + pad, img.width() - 2 * pad, bounds.height()), Qt.TextWordWrap, text)
    p.end()
    return out


def next_name(directory: str, prefix: str, digits: int, ext: str) -> str:
    """First unused name like zap000.png in ``directory``."""
    i = 0
    while True:
        path = os.path.join(directory, f"{prefix}{i:0{digits}d}.{ext}")
        if not os.path.exists(path):
            return path
        i += 1


def compose(win, scale: int = 1, overlays: bool = True, cursor: bool = True, band: bool = True,
            scale_bar: bool = False, bar_nm: float = 0.0, note: str = "") -> Tuple[QImage, Optional[float]]:  # fmt: skip
    img, apx = render_window(win, scale, overlays, cursor, band)
    if scale_bar and apx:
        draw_scale_bar(img, apx, bar_nm, ui_scale=scale)
    if note:
        img = add_caption(img, note, ui_scale=scale)
    return img, apx


def quick_snapshot(win, fmt: str = "PNG", directory: str = "") -> str:
    """Ctrl+S (TIFF), Shift+S (PNG), Ctrl+Shift+S (JPEG): the window as shown, rubber band area only if one is drawn."""
    session = win.session
    prefix, digits = PREFIXES.get(type(win).__name__, ("snap", 3))
    path = next_name(directory or os.getcwd(), prefix, digits, FORMATS[fmt])
    img, _ = compose(win, cursor=session.show_markers)
    if img.save(path, fmt):
        session.message.emit(f"Wrote {fmt} snapshot to {os.path.abspath(path)}")
    else:
        session.message.emit(f"Could not write {path}")
    return path


def snapshot_format(event) -> Optional[str]:
    """Snapshot format for a key event: Ctrl+S TIFF, Shift+S PNG, Ctrl+Shift+S JPEG; None if not a snapshot key."""
    if event.key() != Qt.Key_S:
        return None
    mods = event.modifiers()
    ctrl, shift = bool(mods & Qt.ControlModifier), bool(mods & Qt.ShiftModifier)
    if ctrl and shift:
        return "JPEG"
    if ctrl:
        return "TIFF"
    if shift:
        return "PNG"
    return None


# ---------------------------------------------------------------------------------------------- the tool
class SnapshotDialog(QDialog):
    """cmod Snapshot tool, modelled on 3dmod's Grab with Note."""

    def __init__(self, info):
        super().__init__(info)
        self.info = info
        self.session = info.session
        self.setWindowTitle("cmod Snapshot")
        self._windows: List[QWidget] = []
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)
        grid = QGridLayout()
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(3)

        self.window_combo = QComboBox()
        self.window_combo.currentIndexChanged.connect(self._window_changed)
        grid.addWidget(QLabel("Window"), 0, 0)
        grid.addWidget(self.window_combo, 0, 1, 1, 3)

        self.whole = QRadioButton("Whole window")
        self.band = QRadioButton("Rubber band")
        self.whole.setChecked(True)
        area = QButtonGroup(self)
        area.addButton(self.whole)
        area.addButton(self.band)
        self.band.setToolTip("Only the area inside the ZaP rubber band (Shift+B)")
        grid.addWidget(QLabel("Area"), 1, 0)
        grid.addWidget(self.whole, 1, 1)
        grid.addWidget(self.band, 1, 2)

        self.scale = QSpinBox()
        self.scale.setRange(1, 6)
        self.scale.setSuffix(" x")
        self.scale.setToolTip("Render at this multiple of the window size, with finer image data")
        self.scale.valueChanged.connect(self._update_size)
        self.size_label = QLabel()
        grid.addWidget(QLabel("Resolution"), 2, 0)
        grid.addWidget(self.scale, 2, 1)
        grid.addWidget(self.size_label, 2, 2, 1, 2)

        self.overlays = QCheckBox("Overlays")
        self.overlays.setChecked(True)
        self.marker = QCheckBox("Current point")
        self.marker.setToolTip("Include the current point marker")
        grid.addWidget(self.overlays, 3, 1)
        grid.addWidget(self.marker, 3, 2)

        self.scale_bar = QCheckBox("Scale bar")
        self.scale_bar.setChecked(True)
        self.bar_nm = QSpinBox()
        self.bar_nm.setRange(0, 100_000)
        self.bar_nm.setSpecialValueText("auto")
        self.bar_nm.setSuffix(" nm")
        self.bar_nm.setToolTip("Scale bar length (auto: a round length near a fifth of the image width)")
        grid.addWidget(self.scale_bar, 4, 1)
        grid.addWidget(self.bar_nm, 4, 2)
        lay.addLayout(grid)

        lay.addWidget(QLabel("Note"))
        self.note = QPlainTextEdit()
        self.note.setFixedHeight(self.note.fontMetrics().lineSpacing() * 3 + 10)
        self.note.setPlaceholderText("Text to print under the image")
        lay.addWidget(self.note)

        row = QHBoxLayout()
        row.addWidget(QLabel("Format"))
        self.format = QComboBox()
        self.format.addItems(list(FORMATS))
        self.format.currentIndexChanged.connect(lambda _i: self._suggest_name())
        row.addWidget(self.format)
        row.addWidget(QLabel("File"))
        self.file = QLineEdit()
        row.addWidget(self.file, 1)
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse)
        row.addWidget(browse)
        lay.addLayout(row)

        buttons = QHBoxLayout()
        copy = QPushButton("Copy")
        copy.setToolTip("Copy the snapshot to the clipboard")
        copy.clicked.connect(self.copy)
        save = QPushButton("Save")
        save.setDefault(True)
        save.clicked.connect(self.save)
        done = QPushButton("Done")
        done.clicked.connect(self.close)
        for b in (copy, save, done):
            buttons.addWidget(b)
        lay.addLayout(buttons)
        self.directory = os.getcwd()

    # ------------------------------------------------------------------ state
    def refresh(self) -> None:
        """List the open windows (image windows first, the Info window last)."""
        current = self._current()
        self._windows = [*self.info.open_windows(), self.info]
        self.window_combo.blockSignals(True)
        self.window_combo.clear()
        counts = {}
        for w in self._windows:
            name = window_name(w)
            counts[name] = counts.get(name, 0) + 1
            self.window_combo.addItem(name if counts[name] == 1 else f"{name} {counts[name]}")
        if current in self._windows:
            self.window_combo.setCurrentIndex(self._windows.index(current))
        self.window_combo.blockSignals(False)
        self._window_changed()

    def _current(self):
        i = self.window_combo.currentIndex()
        return self._windows[i] if 0 <= i < len(self._windows) else None

    def _window_changed(self, *_args) -> None:
        win = self._current()
        image_view = win is not None and hasattr(win, "render_image")
        has_band = win is not None and getattr(getattr(win, "canvas", None), "band", None) is not None
        self.band.setEnabled(has_band)
        if not has_band:
            self.whole.setChecked(True)
        for w in (self.scale, self.overlays, self.marker, self.scale_bar, self.bar_nm):
            w.setEnabled(image_view)
        self._update_size()
        self._suggest_name()

    def _update_size(self, *_args) -> None:
        win = self._current()
        if win is None:
            self.size_label.setText("")
            return
        canvas = getattr(win, "canvas", win)
        k = self.scale.value() if hasattr(win, "render_image") else 1
        self.size_label.setText(f"{canvas.width() * k} x {canvas.height() * k} pixels")

    def _suggest_name(self) -> None:
        win = self._current()
        prefix, digits = PREFIXES.get(type(win).__name__, ("info", 3))
        self.file.setText(
            os.path.basename(next_name(self.directory, prefix, digits, FORMATS[self.format.currentText()])),
        )

    def _browse(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save snapshot", os.path.join(self.directory, self.file.text()))
        if path:
            self.directory = os.path.dirname(path)
            self.file.setText(os.path.basename(path))

    # ------------------------------------------------------------------ actions
    def render(self) -> Optional[QImage]:
        win = self._current()
        if win is None:
            return None
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            img, _ = compose(
                win,
                scale=self.scale.value() if hasattr(win, "render_image") else 1,
                overlays=self.overlays.isChecked(),
                cursor=self.marker.isChecked(),
                band=self.band.isChecked(),
                scale_bar=self.scale_bar.isChecked(),
                bar_nm=float(self.bar_nm.value()),
                note=self.note.toPlainText(),
            )
        finally:
            QApplication.restoreOverrideCursor()
        return img

    def copy(self) -> None:
        img = self.render()
        if img is not None:
            QApplication.clipboard().setImage(img)
            self.session.message.emit(f"Copied a {img.width()} x {img.height()} snapshot to the clipboard")

    def save(self) -> Optional[str]:
        img = self.render()
        if img is None:
            return None
        name = self.file.text().strip() or "snapshot"
        path = name if os.path.isabs(name) else os.path.join(self.directory, name)
        fmt = self.format.currentText()
        if not os.path.splitext(path)[1]:
            path += "." + FORMATS[fmt]
        if img.save(path, fmt):
            self.session.message.emit(f"Wrote {img.width()} x {img.height()} {fmt} snapshot to {os.path.abspath(path)}")
            self._suggest_name()
            return path
        self.session.message.emit(f"Could not write {path}")
        return None
