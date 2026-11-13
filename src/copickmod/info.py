"""The Info window, cmod's main window, laid out like 3dmod's (form_info.cpp), and the Object List."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QColor, QIcon, QPalette, QPixmap
from qtpy.QtWidgets import (
    QAction,
    QCheckBox,
    QDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from copickmod import __version__, keys, settings
from copickmod.state import Session
from copickmod.style import keep_alive, pin_icon, raise_icon, tb_button, tb_toggle, undo_icon


def swatch(rgba) -> QIcon:
    pix = QPixmap(14, 14)
    pix.fill(QColor(int(rgba[0]), int(rgba[1]), int(rgba[2])))
    return QIcon(pix)


def layer_kind(layer) -> str:
    return "segmentation" if hasattr(layer, "volume") else layer.kind


def _spin(maximum: int, minimum: int = 1) -> QSpinBox:
    spin = QSpinBox()
    spin.setRange(minimum, max(minimum, maximum))
    spin.setKeyboardTracking(False)
    spin.setFixedWidth(spin.fontMetrics().horizontalAdvance("0000") + 26)
    return spin


def _panel() -> QFrame:
    """3dmod's thin plain frame around the Object/Contour/Point and X/Y/Z boxes."""
    frame = QFrame()
    frame.setFrameStyle(QFrame.Panel | QFrame.Plain)
    frame.setLineWidth(1)
    return frame


def _fill(widget, color: Optional[QColor]) -> None:
    """Paint a widget's background (labels) or field (spin boxes) with the object colour, as 3dmod does."""
    pal = widget.palette()
    role = QPalette.Base if isinstance(widget, QSpinBox) else QPalette.Window
    pal.setColor(role, color if color is not None else widget.style().standardPalette().color(role))
    widget.setAutoFillBackground(color is not None or isinstance(widget, QSpinBox))
    widget.setPalette(pal)


class ObjectList(QDialog):
    """3dmod's Object List: turn overlays on and off (plus the segmentation opacity)."""

    def __init__(self, session: Session, layers, on_select, parent=None):
        super().__init__(parent)
        self.session = session
        self.setWindowTitle("cmod Object List")
        lay = QVBoxLayout(self)
        self.list = QListWidget()
        for i, layer in enumerate(layers):
            count = f" ({len(layer)})" if hasattr(layer, "__len__") else ""
            item = QListWidgetItem(swatch(layer.rgba), f"{i + 1}: {layer.name}  [{layer_kind(layer)}]{count}")
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if layer.visible else Qt.Unchecked)
            item.setData(Qt.UserRole, layer)
            self.list.addItem(item)
        self.list.itemChanged.connect(
            lambda item: session.set_layer_visible(item.data(Qt.UserRole), item.checkState() == Qt.Checked),
        )
        self.list.itemDoubleClicked.connect(lambda item: on_select(self.list.row(item)))
        session.overlays_changed.connect(self.sync)
        self.list.setMinimumWidth(min(700, self.list.sizeHintForColumn(0) + 40))
        lay.addWidget(self.list)
        row = QHBoxLayout()
        for text, state in (("All On", Qt.Checked), ("All Off", Qt.Unchecked)):
            b = QPushButton(text)
            b.clicked.connect(lambda _=False, st=state: self._set_all(st))
            row.addWidget(b)
        lay.addLayout(row)
        orow = QHBoxLayout()
        orow.addWidget(QLabel("Segmentation opacity"))
        self.opacity = QSlider(Qt.Horizontal)
        self.opacity.setRange(0, 100)
        self.opacity.setValue(int(session.opacity * 100))
        self.opacity.valueChanged.connect(lambda v: session.set_opacity(v / 100.0))
        orow.addWidget(self.opacity, 1)
        lay.addLayout(orow)
        done = QPushButton("Done")
        done.clicked.connect(self.close)
        lay.addWidget(done)

    def sync(self) -> None:
        for i in range(self.list.count()):
            item = self.list.item(i)
            want = Qt.Checked if item.data(Qt.UserRole).visible else Qt.Unchecked
            if item.checkState() != want:
                self.list.blockSignals(True)
                item.setCheckState(want)
                self.list.blockSignals(False)

    def _set_all(self, state) -> None:
        for i in range(self.list.count()):
            self.list.item(i).setCheckState(state)


class InfoWindow(QMainWindow):
    closed = Signal(bool)  # emitted when closed; True if another tomogram is replacing this one

    def __init__(self, session: Session, header: str = ""):
        super().__init__(None)
        self.session = session
        self.windows: List[QWidget] = []
        self.slicer_angles: Optional[list] = None
        self.layers = session.layers()  # the "objects"
        self.viewer = None  # set by the application controller (run browser, switching tomograms)
        self.replacing = False  # closed because another tomogram is being opened: do not quit
        self.object_list: Optional[ObjectList] = None
        self.snapshot_dialog = None
        self.setWindowTitle(f"cmod: {session.title}")
        keep_alive(self)
        # Float and Subarea start as they were last left (3dmod keeps them in its settings too).
        session.float_on = settings.get("contrast/float")
        session.subarea = settings.get("contrast/subarea")
        self.setAcceptDrops(True)  # drop MRC files to open them in this window's ZaP
        session.opener = self.open_window
        session.navigator = self.navigate
        self._build_menus()

        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(4, 2, 4, 2)
        outer.setSpacing(4)

        # Icon row: keep on top, raise windows, undo, redo, Show.
        icons = QHBoxLayout()
        icons.setSpacing(4)
        self.pin_button = tb_toggle(pin_icon(False), pin_icon(True), "Keep this window on top of all other windows")
        self.pin_button.toggled.connect(self._keep_on_top)
        raise_button = tb_button(raise_icon(), "Raise all cmod windows above other windows", self.raise_all)
        undo = tb_button(undo_icon(False), "Undo changes to model (cmod is view-only)")
        redo = tb_button(undo_icon(True), "Redo changes that were undone (cmod is view-only)")
        undo.setEnabled(False)
        redo.setEnabled(False)
        self.show_box = QCheckBox("Show")
        self.show_box.setChecked(True)
        self.show_box.setFocusPolicy(Qt.NoFocus)
        self.show_box.setToolTip("Show current point when changing object or contour with spin boxes below")
        for w in (self.pin_button, raise_button, undo, redo, self.show_box):
            icons.addWidget(w)
        icons.addStretch(1)
        outer.addLayout(icons)

        # Object / Contour / Point (the overlays) and X / Y / Z, in thin plain frames.
        top = QHBoxLayout()
        top.setSpacing(4)
        ocp = _panel()
        g = QGridLayout(ocp)
        g.setContentsMargins(3, 3, 3, 3)
        g.setHorizontalSpacing(3)
        g.setVerticalSpacing(1)
        self.obj_label = QLabel(" Object ")
        self.obj_spin = _spin(len(self.layers), 0 if not self.layers else 1)
        self.cont_spin = _spin(1, 0)
        self.point_spin = _spin(1, 0)
        for spin in (self.cont_spin, self.point_spin):
            spin.setSpecialValueText("--x--")
        if not self.layers:
            self.obj_spin.setSpecialValueText("--x--")
        self.obj_spin.setToolTip("Set current overlay object")
        self.cont_spin.setToolTip("Set current contour (filament)")
        self.point_spin.setToolTip("Go to a point of the current object")
        self.obj_max = QLabel(f"/ {len(self.layers)}")
        self.cont_max = QLabel("/ 0")
        self.point_max = QLabel("/ 0")
        for row, (label, spin, mx) in enumerate(
            (
                (self.obj_label, self.obj_spin, self.obj_max),
                (QLabel("Contour"), self.cont_spin, self.cont_max),
                (QLabel("Point"), self.point_spin, self.point_max),
            ),
        ):
            label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            mx.setMinimumWidth(mx.fontMetrics().horizontalAdvance("/ 0000"))
            g.addWidget(label, row, 0)
            g.addWidget(spin, row, 1)
            g.addWidget(mx, row, 2)
        self.obj_spin.valueChanged.connect(self._object_changed)
        self.cont_spin.valueChanged.connect(self._contour_changed)
        self.point_spin.valueChanged.connect(self._point_selected)
        top.addWidget(ocp)

        xyz = _panel()
        g = QGridLayout(xyz)
        g.setContentsMargins(3, 3, 3, 3)
        g.setHorizontalSpacing(3)
        g.setVerticalSpacing(1)
        self.point_spins: Dict[int, QSpinBox] = {}
        self.point_sizes: Dict[int, QLabel] = {}
        for row, (axis, name) in enumerate(((2, "X"), (1, "Y"), (0, "Z"))):
            n = session.volume.shape[axis]
            spin = _spin(n)
            spin.setToolTip(f"Set current {name} coordinate")
            spin.valueChanged.connect(lambda v, a=axis: session.set_axis(a, v - 1))
            self.point_spins[axis] = spin
            self.point_sizes[axis] = QLabel(f"/ {n}")
            g.addWidget(QLabel(name), row, 0)
            g.addWidget(spin, row, 1)
            g.addWidget(self.point_sizes[axis], row, 2)
        top.addWidget(xyz)
        top.addStretch(1)
        outer.addLayout(top)

        # Black / White, Float / Subarea / Auto, and the Mode box.
        mid = QHBoxLayout()
        mid.setSpacing(6)
        left = QVBoxLayout()
        left.setSpacing(2)
        grid = QGridLayout()
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(1)
        self.black_slider, self.black_pos = self._slider_row(grid, 0, "Black")
        self.white_slider, self.white_pos = self._slider_row(grid, 1, "White")
        self.black_slider.valueChanged.connect(self._black_moved)
        self.white_slider.valueChanged.connect(self._white_moved)
        left.addLayout(grid)
        row = QHBoxLayout()
        row.setSpacing(6)
        self.float_box = QCheckBox("Float")
        self.float_box.setChecked(session.float_on)
        self.float_box.setToolTip("Adjust contrast to keep the same mean and SD when changing sections")
        self.float_box.toggled.connect(lambda on: (session.set_float(on), settings.put("contrast/float", on)))
        self.subarea_box = QCheckBox("Subarea")
        self.subarea_box.setChecked(session.subarea)
        self.subarea_box.setToolTip("Base Float and Auto on the area visible in the ZaP window")
        self.subarea_box.toggled.connect(lambda on: (session.set_subarea(on), settings.put("contrast/subarea", on)))
        auto = QPushButton("Auto")
        auto.setToolTip("Set contrast automatically from the current section (Shift+A)")
        auto.setFixedWidth(auto.fontMetrics().horizontalAdvance("Auto") + 14)
        auto.clicked.connect(session.autocontrast)
        for w in (self.float_box, self.subarea_box, auto):
            w.setFocusPolicy(Qt.NoFocus)
        row.addWidget(self.float_box)
        row.addStretch(1)
        row.addWidget(self.subarea_box)
        row.addStretch(1)
        row.addWidget(auto)
        left.addLayout(row)
        mid.addLayout(left, 1)
        mode = QGroupBox("Mode")
        ml = QVBoxLayout(mode)
        ml.setContentsMargins(6, 2, 6, 2)
        ml.setSpacing(2)
        movie = QRadioButton("Movie")
        movie.setChecked(True)
        model = QRadioButton("Model")
        model.setEnabled(False)
        model.setToolTip("cmod is view-only")
        for w in (movie, model):
            w.setFocusPolicy(Qt.NoFocus)
            ml.addWidget(w)
        mid.addWidget(mode)
        outer.addLayout(mid)

        self.invert_box = QCheckBox()  # holds the F11 state; 3dmod shows inversion only in the colour ramp
        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setMaximumBlockCount(2000)
        self.output.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.output.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.output.setFixedHeight(self.output.fontMetrics().lineSpacing() * 3 + 12)
        self.output.setFocusPolicy(Qt.ClickFocus)
        outer.addWidget(self.output)
        self.setCentralWidget(central)

        session.layers_changed.connect(self._layers_changed)
        session.image_changed.connect(self._image_changed)
        session.images_changed.connect(self._retitle)
        session.point_changed.connect(self._sync_point)
        session.contrast_changed.connect(self._sync_contrast)
        session.message.connect(self.log)
        self._sync_point()
        self._sync_contrast()
        self._object_changed(self.obj_spin.value())
        for line in header.splitlines():
            self.log(line)
        self.output.moveCursor(self.output.textCursor().Start)
        self.resize(max(320, self.minimumSizeHint().width()), self.minimumSizeHint().height())

    def _retitle(self) -> None:
        self.setWindowTitle(f"cmod: {self.session.title}")

    def _image_changed(self, old) -> None:
        for axis, spin in self.point_spins.items():
            n = self.session.volume.shape[axis]
            spin.blockSignals(True)
            spin.setRange(1, n)
            spin.blockSignals(False)
            self.point_sizes[axis].setText(f"/ {n}")
        self._retitle()
        self._sync_point()

    def open_files_dialog(self) -> None:
        """File > Open Image File: MRC files to show in this window's ZaP (switch with Left/Right or 1/2)."""
        from qtpy.QtWidgets import QFileDialog

        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Open image files",
            "",
            "MRC images (*.mrc *.rec *.st *.ali *.preali *.map *.mrcs);;All files (*)",
        )
        if paths:
            self.session.open_files(paths)

    def dragEnterEvent(self, event) -> None:
        from copickmod.mrc import dropped_paths

        if dropped_paths(event):
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        from copickmod.mrc import dropped_paths

        paths = dropped_paths(event)
        if paths:
            event.acceptProposedAction()
            self.session.open_files(paths)

    def show_options(self) -> None:
        settings.SettingsDialog(self).exec_()

    # ------------------------------------------------------------------ construction helpers
    def _build_menus(self) -> None:
        mb = self.menuBar()
        file_menu = mb.addMenu("&File")
        open_files = QAction("&Open Image File...\tCtrl+O", self)
        open_files.triggered.connect(self.open_files_dialog)
        file_menu.addAction(open_files)
        browser = QAction("&Run Browser...\tCtrl+B", self)
        browser.triggered.connect(lambda: self.open_window("browser"))
        file_menu.addAction(browser)
        snap = QAction("&Snapshot Tool...\tShift+H", self)
        snap.triggered.connect(lambda: self.open_window("snapshot"))
        file_menu.addAction(snap)
        file_menu.addSeparator()
        quit_action = QAction("&Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)
        edit = mb.addMenu("&Edit")
        objects = QAction("&Object List...", self)
        objects.triggered.connect(self.show_object_list)
        edit.addAction(objects)
        options = QAction("O&ptions...", self)
        options.triggered.connect(self.show_options)
        edit.addAction(options)
        image = mb.addMenu("&Image")
        for label, kind, shortcut in (
            ("&ZaP", "zap", "Z"),
            ("&XYZ", "xyz", "Ctrl+X"),
            ("&Slicer", "slicer", "\\"),
            ("&Model View", "model", "V"),
        ):
            action = QAction(f"{label}\t{shortcut}", self)
            action.triggered.connect(lambda _=False, k=kind: self.open_window(k))
            image.addAction(action)
        image.addSeparator()
        for label, delta, shortcut in (("&Next Tomogram", 1, "2"), ("&Previous Tomogram", -1, "1")):
            action = QAction(f"{label}\t{shortcut}", self)
            action.triggered.connect(lambda _=False, d=delta: self.session.step_image(d))
            image.addAction(action)
        help_menu = mb.addMenu("&Help")
        hk = QAction("&Hot keys", self)
        hk.triggered.connect(lambda: self.open_window("help"))
        help_menu.addAction(hk)

    def _slider_row(self, grid: QGridLayout, row: int, name: str):
        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, 255)
        slider.setFocusPolicy(Qt.NoFocus)
        slider.setMinimumWidth(100)
        pos = QLabel()
        pos.setMinimumWidth(pos.fontMetrics().horizontalAdvance("0000"))
        pos.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        grid.addWidget(QLabel(name), row, 0)
        grid.addWidget(slider, row, 1)
        grid.addWidget(pos, row, 2)
        return slider, pos

    # ------------------------------------------------------------------ objects (overlays)
    def show_object_list(self) -> None:
        if not self.layers:
            self.log("No overlays were loaded (add picks or segmentation URIs after the tomogram URI).")
            return
        if self.object_list is None:
            self.object_list = ObjectList(self.session, self.layers, lambda i: self.obj_spin.setValue(i + 1), self)
        self.object_list.show()
        self.object_list.raise_()

    def _layers_changed(self) -> None:
        """Overlays were added (run browser): renumber the objects."""
        self.layers = self.session.layers()
        n = len(self.layers)
        self.obj_spin.blockSignals(True)
        self.obj_spin.setSpecialValueText("" if n else "--x--")
        self.obj_spin.setRange(1 if n else 0, max(n, 1 if n else 0))
        self.obj_spin.blockSignals(False)
        self.obj_max.setText(f"/ {n}")
        self._object_changed(self.obj_spin.value())
        if self.object_list is not None:
            was_visible = self.object_list.isVisible()
            self.object_list.close()
            self.object_list.deleteLater()
            self.object_list = None
            if was_visible:
                self.show_object_list()

    def _current_layer(self):
        i = self.obj_spin.value() - 1
        return self.layers[i] if 0 <= i < len(self.layers) else None

    def _object_changed(self, _value: int) -> None:
        layer = self._current_layer()
        color = None if layer is None else QColor(*[int(c) for c in layer.rgba[:3]])
        for w in (self.obj_label, self.obj_spin, self.obj_max):  # 3dmod tints the whole Object row
            _fill(w, color)
        if layer is None:
            for spin, mx in ((self.cont_spin, self.cont_max), (self.point_spin, self.point_max)):
                spin.setRange(0, 0)
                mx.setText("/ 0")
            return
        kind = layer_kind(layer)
        ncont = 0 if kind in ("segmentation", "mesh") else (1 if kind == "picks" else len(layer.polylines))
        self.cont_spin.blockSignals(True)
        self.cont_spin.setRange(0, ncont)
        self.cont_spin.setValue(1 if ncont else 0)
        self.cont_spin.blockSignals(False)
        self.cont_max.setText(f"/ {ncont}")
        self._contour_changed(self.cont_spin.value())

    def _contour_points(self) -> np.ndarray:
        layer = self._current_layer()
        if layer is None or layer_kind(layer) in ("segmentation", "mesh"):
            return np.zeros((0, 3))
        if layer.kind == "picks":
            return layer.points
        c = self.cont_spin.value() - 1
        return layer.polylines[c] if 0 <= c < len(layer.polylines) else np.zeros((0, 3))

    def _contour_changed(self, _value: int) -> None:
        n = len(self._contour_points())
        self.point_spin.blockSignals(True)
        self.point_spin.setRange(0, n)
        self.point_spin.setValue(0)
        self.point_spin.blockSignals(False)
        self.point_max.setText(f"/ {n}")

    def navigate(self, action: str) -> None:
        """3dmod's object / contour / point hot keys (O, P, C, Shift+C, [, ], {, }, E, Ctrl+T, Shift+PageUp/Down)."""
        if not self.layers:
            self.log("No overlay objects are loaded")
            return
        o, c, pt = self.obj_spin, self.cont_spin, self.point_spin
        if action in ("next_object", "prev_object"):
            o.setValue(o.value() + (1 if action == "next_object" else -1))
        elif action in ("next_contour", "prev_contour"):
            if c.maximum() > 0:
                c.setValue(max(1, c.value() + (1 if action == "next_contour" else -1)))
        elif action in ("next_point", "prev_point"):
            if pt.maximum() > 0:
                pt.setValue(min(pt.maximum(), max(1, pt.value() + (1 if action == "next_point" else -1))))
        elif action == "first_point" and pt.maximum() > 0:
            pt.setValue(1)
        elif action == "last_point" and pt.maximum() > 0:
            pt.setValue(pt.maximum())
        elif action == "unselect_point":
            pt.setValue(0)
        elif action == "toggle_object":
            layer = self._current_layer()
            self.session.set_layer_visible(layer, not layer.visible)
            self.log(f"Object {o.value()} ({layer.name}) display {'ON' if layer.visible else 'OFF'}")
        elif action in ("next_section", "prev_section"):
            layer = self._current_layer()
            if not self.session.next_section_with_points(layer, 1 if action == "next_section" else -1):
                self.log(f"No {'later' if action == 'next_section' else 'earlier'} section with points of {layer.name}")

    def _point_selected(self, value: int) -> None:
        """Go to a pick or filament point, and bring it into view in the ZaP windows."""
        pts = self._contour_points()
        if not 1 <= value <= len(pts) or not self.show_box.isChecked():
            return
        z, y, x = (int(v) for v in np.floor(pts[value - 1]))
        self.session.set_point(z, y, x)
        self.session.center_on_point.emit()

    # ------------------------------------------------------------------ syncing
    def _sync_point(self) -> None:
        for axis, spin in self.point_spins.items():
            spin.blockSignals(True)
            spin.setValue(self.session.point[axis] + 1)
            spin.blockSignals(False)

    def _sync_contrast(self) -> None:
        c = self.session.contrast
        for slider, pos, value in (
            (self.black_slider, self.black_pos, c.black),
            (self.white_slider, self.white_pos, c.white),
        ):
            p = c.to_slider(value)
            slider.blockSignals(True)
            slider.setValue(max(0, min(255, p)))
            slider.blockSignals(False)
            pos.setText(str(max(0, min(255, p))))
            slider.setToolTip(f"Data value {value:.5g}")
        self.invert_box.setChecked(c.invert)

    def _black_moved(self, b: int) -> None:
        w = self.white_slider.value()
        self.session.set_sliders(b, max(w, b + 1))  # pushing black past white drags white along

    def _white_moved(self, w: int) -> None:
        b = self.black_slider.value()
        self.session.set_sliders(min(b, w - 1), w)

    def _keep_on_top(self, on: bool) -> None:
        self.setWindowFlag(Qt.WindowStaysOnTopHint, on)
        self.show()

    def raise_all(self) -> None:
        for w in self.open_windows():
            w.raise_()
        self.raise_()
        self.activateWindow()

    def log(self, text: str) -> None:
        self.output.appendPlainText(text)

    # ------------------------------------------------------------------ windows
    def open_window(self, kind: str) -> Optional[QWidget]:
        from copickmod.slicer import SlicerWindow
        from copickmod.xyz import XyzWindow
        from copickmod.zap import ZapWindow

        if kind == "quit":
            self.close()
            return None
        if kind == "objects":
            self.show_object_list()
            return self.object_list
        if kind == "info":
            self.raise_()
            self.activateWindow()
            return self
        if kind == "raise":
            self.raise_all()
            return self
        if kind == "open_files":
            self.open_files_dialog()
            return self
        if kind == "options":
            self.show_options()
            return self
        if kind == "browser":
            if self.viewer is None:
                self.log("The run browser needs a project (start cmod from the command line)")
                return None
            return self.viewer.show_browser()
        if kind == "snapshot":
            from copickmod.snapshot import SnapshotDialog

            if self.snapshot_dialog is None:
                self.snapshot_dialog = SnapshotDialog(self)
            self.snapshot_dialog.refresh()
            self.snapshot_dialog.show()
            self.snapshot_dialog.raise_()
            return self.snapshot_dialog
        if kind == "help":
            box = QMessageBox(self)
            box.setWindowTitle("cmod hot keys")
            box.setText("<pre>" + keys.HELP_TEXT.replace("&", "&amp;").replace("<", "&lt;") + "</pre>")
            box.setModal(False)
            box.show()
            return box
        single = {"xyz": "XyzWindow", "slicer": "SlicerWindow", "model": "ModelViewWindow"}
        if kind in single:  # one of each, like 3dmod
            for w in self.open_windows():  # only live ones: closed windows are deleted (WA_DeleteOnClose)
                if type(w).__name__ == single[kind]:
                    w.raise_()
                    w.activateWindow()
                    return w
        if kind == "zap":
            win = ZapWindow(self.session)
        elif kind == "xyz":
            win = XyzWindow(self.session)
        elif kind == "slicer":
            win = SlicerWindow(self.session, angles=self.slicer_angles)
        elif kind == "model":
            from copickmod.modelview import ModelViewWindow

            win = ModelViewWindow(self.session)
        else:
            raise ValueError(kind)
        self.windows = [*self.open_windows(), win]
        if hasattr(win, "fit_to_image"):
            win.fit_to_image()
        else:
            win.adjustSize()
        win.show()
        win.raise_()
        return win

    def open_windows(self) -> List[QWidget]:
        """The image windows that are shown. Windows being set up (not shown yet) are kept, deleted ones dropped."""
        self.windows = [w for w in self.windows if _alive(w)]
        return [w for w in self.windows if w.isVisible()]

    def keyPressEvent(self, event) -> None:
        if not keys.handle_global(self.session, event, axis=0):
            super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        settings.save_window("info", self)
        for w in self.open_windows():
            w.close()
        for dialog in (self.object_list, self.snapshot_dialog):
            if dialog is not None:
                dialog.close()
        self.session.close()
        super().closeEvent(event)
        self.closed.emit(self.replacing)  # the application quits when the Info window is closed (3dmod)


def startup_text(sel, session) -> str:
    """The lines 3dmod prints in the Info window when it starts."""
    from importlib.metadata import version

    from copickmod.project import uri_of

    vol = session.volume
    z, y, x = vol.shape
    try:
        copick_version = version("copick")
    except Exception:  # noqa: BLE001
        copick_version = "?"
    img = session.image
    where = f"{uri_of(img.tomogram)} in run {img.run_name}" if img.tomogram is not None else img.path or img.name
    lines = [
        f"Image size {x} x {y}, {z} sections.",
        f"cmod {__version__} (copick {copick_version})",
        f"{where}, {vol.voxel_size:g} A/pixel",
    ]
    if len(session.images) > 1:
        lines.append(f"{len(session.images)} tomograms in this window: Left/Right (or 1/2) switch between them.")
    n = len(session.seg_layers) + len(session.point_layers) + len(session.mesh_layers)
    if n:
        lines.append(f"{n} overlay object{'s' if n > 1 else ''}: Edit > Object List shows or hides them.")
    return "\n".join(lines)


def _alive(widget) -> bool:
    try:
        widget.objectName()  # raises if the C++ object was deleted (WA_DeleteOnClose)
        return True
    except RuntimeError:
        return False
