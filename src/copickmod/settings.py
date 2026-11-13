"""User settings, kept with QSettings in ``~/.config/cmod/cmod.conf``, and the Options dialog that edits them."""

from __future__ import annotations

from typing import Any, Optional

from qtpy.QtCore import QPoint, QSettings, QSize
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QVBoxLayout,
)

SAME_ZAP = "same"
NEW_WINDOWS = "new"

DEFAULTS = {
    # Opening a tomogram from the run browser (or Gallery):
    "open/same_run": SAME_ZAP,  # of a run that is open already: add it to that run's ZaP, or open new windows
    "open/other_run": NEW_WINDOWS,  # of another run: open a new ZaP and Info window, or add it to the current ZaP
    "keys/arrows_switch": True,  # Left/Right arrows switch between the tomograms of a ZaP (1/2 always do)
    "windows/remember": True,  # reopen the Info and ZaP windows where they were; new runs' windows go on top
    "contrast/float": False,  # 3dmod's Float and Subarea check boxes: off at first, then remembered (as 3dmod)
    "contrast/subarea": False,
}

#: False while cmod renders snapshots (``--snapshot``): those windows must not overwrite the saved positions.
save_geometry = True


def _store() -> QSettings:
    return QSettings(QSettings.IniFormat, QSettings.UserScope, "cmod", "cmod")


def get(key: str) -> Any:
    default = DEFAULTS[key]
    value = _store().value(key, default)
    if isinstance(default, bool):
        return value in (True, "true", "True", "1", 1)
    return type(default)(value) if value is not None else default


def put(key: str, value: Any) -> None:
    store = _store()
    store.setValue(key, value)
    store.sync()


def save_window(name: str, widget) -> None:
    """Remember a top-level window's position and size."""
    if not save_geometry or not get("windows/remember") or not widget.isVisible():
        return
    store = _store()
    store.setValue(f"geometry/{name}_pos", widget.pos())
    store.setValue(f"geometry/{name}_size", widget.size())
    store.sync()


def window_pos(name: str) -> Optional[QPoint]:
    """Where a window was last (None if unknown, off screen, or remembering is off)."""
    if not get("windows/remember"):
        return None
    pos = _store().value(f"geometry/{name}_pos")
    if not isinstance(pos, QPoint):
        return None
    from qtpy.QtGui import QGuiApplication

    if QGuiApplication.screenAt(pos + QPoint(40, 20)) is None:  # that screen is gone (e.g. another VNC desktop)
        return None
    return pos


def window_size(name: str) -> Optional[QSize]:
    size = _store().value(f"geometry/{name}_size")
    return size if isinstance(size, QSize) and size.isValid() else None


class SettingsDialog(QDialog):
    """cmod's Options (3dmod keeps its settings under Edit > Options too)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("cmod Options")
        layout = QVBoxLayout(self)

        opening = QGroupBox("Opening tomograms from the run browser")
        form = QFormLayout(opening)
        self.same_run = QComboBox()
        self.same_run.addItem("Add it to the run's ZaP (switch with Left/Right)", SAME_ZAP)
        self.same_run.addItem("Open a new ZaP and Info window", NEW_WINDOWS)
        self.other_run = QComboBox()
        self.other_run.addItem("Open a new ZaP and Info window", NEW_WINDOWS)
        self.other_run.addItem("Add it to the current ZaP (switch with Left/Right)", SAME_ZAP)
        form.addRow("Another tomogram of an open run:", self.same_run)
        form.addRow("A tomogram of another run:", self.other_run)
        layout.addWidget(opening)

        self.arrows = QCheckBox("Left/Right arrow keys switch between a ZaP's tomograms (1 and 2 always do)")
        self.remember = QCheckBox("Remember where the Info and ZaP windows are, and open new runs' windows there")
        layout.addWidget(self.arrows)
        layout.addWidget(self.remember)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.RestoreDefaults)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.RestoreDefaults).clicked.connect(lambda: self.load(defaults=True))
        layout.addWidget(buttons)
        self.load()

    def load(self, defaults: bool = False) -> None:
        value = DEFAULTS.get if defaults else get
        for combo, key in ((self.same_run, "open/same_run"), (self.other_run, "open/other_run")):
            combo.setCurrentIndex(max(0, combo.findData(value(key))))
        self.arrows.setChecked(bool(value("keys/arrows_switch")))
        self.remember.setChecked(bool(value("windows/remember")))

    def accept(self) -> None:
        put("open/same_run", self.same_run.currentData())
        put("open/other_run", self.other_run.currentData())
        put("keys/arrows_switch", self.arrows.isChecked())
        put("windows/remember", self.remember.isChecked())
        super().accept()
