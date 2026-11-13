"""Run browser and object table (not in 3dmod; modelled on copick-web, styled like 3dmod's dialogs).

* Runs: a tree of run > voxel spacing > tomogram with a search box; clicking a tomogram opens it (overlays carry
  over within a run), double-clicking a run opens its preferred tomogram, the context menu shows Run Info.
* Gallery: central-slice thumbnails of every run, rendered in the background; clicking one opens the run.
* Objects: picks, segmentations, filaments and meshes of the open run with visibility check boxes, rows tinted with
  the object colour, a search box and a "Shown only" filter per tab; counts are loaded in the background and
  filaments expand into a list that jumps to each filament.
"""

from __future__ import annotations

import contextlib
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import numpy as np
from qtpy.QtCore import QObject, QSize, Qt, QUrl, Signal
from qtpy.QtGui import QBrush, QColor, QConicalGradient, QDesktopServices, QFont, QIcon, QPainter, QPixmap
from qtpy.QtWidgets import (
    QAction,
    QApplication,
    QCheckBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStyle,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from copickmod.gallery import Gallery, GalleryWindow, ThumbnailCache, make_thumbnail  # noqa: F401 - re-exported
from copickmod.project import (
    FILAMENTS,
    MESH,
    PICKS,
    SEGMENTATION,
    Selection,
    choose_tomogram,
    run_tomograms,
    uri_of,
)
from copickmod.settings import SettingsDialog
from copickmod.style import keep_alive

log = logging.getLogger(__name__)

PORTAL = "https://cryoetdataportal.czscience.com"
THUMB = 160
TYPE_LABEL = {"binary": "binary", "multilabel": "multi", "instance": "inst", "panoptic": "pan"}


def matches(query: str, fields) -> bool:
    """copick-web search: case-insensitive substring of any field; an empty query matches everything."""
    q = query.strip().lower()
    return not q or any(q in str(f).lower() for f in fields if f is not None)


def object_rgba(root, name: str):
    obj = root.get_object(name)
    if obj is None or obj.color is None:
        return (100, 100, 100, 255)
    c = [int(v) for v in obj.color]
    return tuple(c + [255] * (4 - len(c)))


def swatch(rgba, rainbow: bool = False) -> QIcon:
    pix = QPixmap(12, 12)
    pix.fill(Qt.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing)
    if rainbow:
        g = QConicalGradient(6, 6, 0)
        for i, c in enumerate((Qt.red, Qt.yellow, Qt.green, Qt.cyan, Qt.blue, Qt.magenta, Qt.red)):
            g.setColorAt(i / 6, QColor(c))
        p.setBrush(QBrush(g))
    else:
        p.setBrush(QColor(*[int(c) for c in rgba[:3]]))
    p.setPen(QColor(60, 60, 60))
    p.drawEllipse(1, 1, 10, 10)
    p.end()
    return QIcon(pix)


def tint(rgba) -> QBrush:
    """Row background: the object colour at ~12% (copick-web)."""
    return QBrush(QColor(int(rgba[0]), int(rgba[1]), int(rgba[2]), 32))


def seg_type(seg) -> str:
    t = getattr(seg, "segmentation_type", None)
    if t:
        return t
    return "multilabel" if seg.is_multilabel else "binary"


class _Bridge(QObject):
    """Delivers results from worker threads to the GUI thread."""

    done = Signal(object, object)


def search_bar(what: str, on_change) -> tuple:
    edit = QLineEdit()
    edit.setPlaceholderText(f"Search {what}...")
    edit.setClearButtonEnabled(True)
    edit.textChanged.connect(lambda _t: on_change())

    def key(event, _orig=edit.keyPressEvent):
        if event.key() == Qt.Key_Escape:
            edit.clear()
        else:
            _orig(event)

    edit.keyPressEvent = key
    counter = QLabel()
    counter.setMinimumWidth(counter.fontMetrics().horizontalAdvance("0000/0000") + 4)
    counter.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
    return edit, counter


# ---------------------------------------------------------------------------------------------- runs
class RunTree(QWidget):
    def __init__(self, browser: "BrowserWindow"):
        super().__init__()
        self.browser = browser
        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(2)
        row = QHBoxLayout()
        self.search, self.counter = search_bar("runs", self.filter)
        row.addWidget(self.search, 1)
        row.addWidget(self.counter)
        lay.addLayout(row)
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setUniformRowHeights(True)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._menu)
        self.tree.itemExpanded.connect(self._expanded)
        self.tree.itemClicked.connect(self._clicked)
        self.tree.itemDoubleClicked.connect(self._double_clicked)
        lay.addWidget(self.tree, 1)
        self.empty = QLabel()
        self.empty.setAlignment(Qt.AlignCenter)
        self.empty.hide()
        lay.addWidget(self.empty)
        style = self.style()
        self.icons = {
            "run": style.standardIcon(QStyle.SP_DirIcon),
            "run_open": style.standardIcon(QStyle.SP_DirOpenIcon),
            "tomo": style.standardIcon(QStyle.SP_FileIcon),
        }

    def populate(self, runs) -> None:
        self.tree.clear()
        for run in runs:
            item = QTreeWidgetItem([run.name])
            item.setIcon(0, self.icons["run"])
            item.setData(0, Qt.UserRole, ("run", run))
            item.addChild(QTreeWidgetItem(["Loading..."]))
            self.tree.addTopLevelItem(item)
        self.filter()

    def filter(self) -> None:
        q = self.search.text()
        shown = 0
        n = self.tree.topLevelItemCount()
        for i in range(n):
            item = self.tree.topLevelItem(i)
            hide = not matches(q, [item.text(0)])
            item.setHidden(hide)
            shown += not hide
        self.counter.setText(f"{shown}/{n}" if q.strip() else str(n))
        if n and not shown:
            self.empty.setText(f"No runs match “{q.strip()}”")
        self.empty.setVisible(bool(n and not shown))

    def _load(self, item: QTreeWidgetItem) -> None:
        kind, run = item.data(0, Qt.UserRole)
        if item.childCount() == 1 and item.child(0).data(0, Qt.UserRole) is None:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                item.takeChildren()
                spacings: Dict[float, QTreeWidgetItem] = {}
                for vs, tomo in run_tomograms(run):
                    parent = spacings.get(vs)
                    if parent is None:
                        parent = QTreeWidgetItem([f"{vs:.2f} Å"])
                        parent.setData(0, Qt.UserRole, ("vs", run))
                        item.addChild(parent)
                        spacings[vs] = parent
                    child = QTreeWidgetItem([tomo.tomo_type])
                    child.setIcon(0, self.icons["tomo"])
                    child.setData(0, Qt.UserRole, ("tomo", run, tomo))
                    parent.addChild(child)
                if not spacings:
                    item.addChild(QTreeWidgetItem(["No voxel spacings"]))
            finally:
                QApplication.restoreOverrideCursor()

    def _expanded(self, item: QTreeWidgetItem) -> None:
        data = item.data(0, Qt.UserRole)
        if data and data[0] == "run":
            self._load(item)
            item.setIcon(0, self.icons["run_open"])

    def _clicked(self, item: QTreeWidgetItem, _col: int) -> None:
        data = item.data(0, Qt.UserRole)
        if data and data[0] == "tomo":
            self.browser.viewer.open_tomogram(data[1], data[2])

    def _double_clicked(self, item: QTreeWidgetItem, _col: int) -> None:
        data = item.data(0, Qt.UserRole)
        if data and data[0] == "run":
            self.browser.open_run(data[1])

    def _menu(self, pos) -> None:
        item = self.tree.itemAt(pos)
        data = item.data(0, Qt.UserRole) if item is not None else None
        if not data:
            return
        run = data[1]
        menu = QMenu(self)
        menu.addAction("Open", lambda: self.browser.open_run(run) if data[0] != "tomo" else None)
        menu.addAction("Run Info...", lambda: self.browser.run_info(run))
        menu.exec_(self.tree.viewport().mapToGlobal(pos))

    def reveal(self, run_name: str, voxel: float, tomo_type: str) -> None:
        """Expand to the open tomogram and show it bold and selected."""
        bold = QFont(self.tree.font())
        bold.setBold(True)
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            is_run = item.text(0) == run_name
            item.setFont(0, bold if is_run else self.tree.font())
            if not is_run:
                continue
            self._load(item)
            item.setExpanded(True)
            for j in range(item.childCount()):
                vs_item = item.child(j)
                for k in range(vs_item.childCount()):
                    t = vs_item.child(k)
                    data = t.data(0, Qt.UserRole)
                    hit = bool(data) and t.text(0) == tomo_type and vs_item.text(0) == f"{voxel:.2f} Å"
                    t.setFont(0, bold if hit else self.tree.font())
                    if hit:
                        vs_item.setExpanded(True)
                        self.tree.setCurrentItem(t)
                        self.tree.scrollToItem(t)


# ---------------------------------------------------------------------------------------------- objects
class EntityTable(QWidget):
    """One tab of the object table."""

    COLUMNS = {
        PICKS: ["Object", "User", "Session", "Count"],
        FILAMENTS: ["Object", "User", "Session", "Count"],
        SEGMENTATION: ["Name", "Type", "User", "Session", "Voxel"],
        MESH: ["Object", "User", "Session", ""],
    }

    def __init__(self, browser: "BrowserWindow", kind: str, what: str):
        super().__init__()
        self.browser = browser
        self.kind = kind
        self.what = what
        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(2)
        row = QHBoxLayout()
        self.search, self.counter = search_bar(what, self.filter)
        self.shown_only = QCheckBox("Shown only")
        self.shown_only.setToolTip(f"List only the {what} shown in the viewer")
        self.shown_only.toggled.connect(lambda _on: self.filter())
        row.addWidget(self.search, 1)
        row.addWidget(self.counter)
        row.addWidget(self.shown_only)
        lay.addLayout(row)
        self.tree = QTreeWidget()
        self.tree.setColumnCount(len(self.COLUMNS[kind]))
        self.tree.setHeaderLabels(self.COLUMNS[kind])
        self.tree.setRootIsDecorated(kind == FILAMENTS)
        self.tree.setUniformRowHeights(True)
        self.tree.itemChanged.connect(self._item_changed)
        self.tree.itemDoubleClicked.connect(self._double_clicked)
        self.tree.itemClicked.connect(self._clicked)
        self.tree.itemExpanded.connect(self._expanded)
        lay.addWidget(self.tree, 1)
        self.empty = QLabel()
        self.empty.setAlignment(Qt.AlignCenter)
        self.empty.hide()
        lay.addWidget(self.empty)
        self._bridge = _Bridge()
        self._bridge.done.connect(self._count_done)
        self._items: Dict[str, QTreeWidgetItem] = {}

    # ------------------------------------------------------------------ content
    def populate(self, objects: List, pool: ThreadPoolExecutor) -> None:
        root = self.browser.viewer.root
        self.tree.blockSignals(True)
        self.tree.clear()
        self._items.clear()
        for obj in objects:
            name = uri_of(obj)
            if self.kind == SEGMENTATION:
                stype = seg_type(obj)
                rgba = object_rgba(root, obj.name)
                cols = [
                    obj.name,
                    TYPE_LABEL.get(stype, stype),
                    obj.user_id,
                    obj.session_id,
                    f"{float(obj.voxel_size):.2f} Å",
                ]
                icon = swatch(rgba, rainbow=stype != "binary")
            else:
                rgba = object_rgba(root, obj.pickable_object_name)
                cols = [obj.pickable_object_name, obj.user_id, obj.session_id, ""]
                if self.kind == MESH:
                    cols[3] = "Model View"
                icon = swatch(rgba)
            item = QTreeWidgetItem(cols)
            item.setIcon(0, icon)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(0, Qt.Unchecked)
            item.setData(0, Qt.UserRole, obj)
            item.setToolTip(0, name)
            if self.kind in (PICKS, FILAMENTS):
                item.setTextAlignment(3, Qt.AlignRight | Qt.AlignVCenter)
                item.setText(3, "...")
                fut = pool.submit(_count, obj, self.kind)
                fut.add_done_callback(lambda f, n=name: self._bridge.done.emit(n, f.exception() or f.result()))
            if self.kind == FILAMENTS:
                item.addChild(QTreeWidgetItem(["Loading..."]))
            for c in range(len(cols)):
                item.setBackground(c, tint(rgba))
            self.tree.addTopLevelItem(item)
            self._items[name] = item
        for c in range(self.tree.columnCount()):
            self.tree.resizeColumnToContents(c)
        self.tree.blockSignals(False)
        self.sync()

    def _count_done(self, name: str, result) -> None:
        item = self._items.get(name)
        if item is not None:
            item.setText(3, "?" if isinstance(result, Exception) else str(result))

    def sync(self) -> None:
        """Check boxes follow what the viewer shows."""
        session = self.browser.session
        self.tree.blockSignals(True)
        for name, item in self._items.items():
            layer = session.find_layer(name) if session is not None else None
            on = layer is not None and layer.visible
            item.setCheckState(0, Qt.Checked if on else Qt.Unchecked)
            font = QFont(item.font(0))
            font.setBold(on)
            item.setFont(0, font)
        self.tree.blockSignals(False)
        self.filter()

    def shown_count(self) -> int:
        return sum(item.checkState(0) == Qt.Checked for item in self._items.values())

    def filter(self) -> None:
        q = self.search.text()
        only = self.shown_only.isChecked()
        shown = 0
        for item in self._items.values():
            fields = [item.text(c) for c in range(self.tree.columnCount())]
            if self.kind == SEGMENTATION:
                fields.append(item.text(1))  # short type labels: binary, multi, inst, pan
            ok = matches(q, fields) and (not only or item.checkState(0) == Qt.Checked)
            item.setHidden(not ok)
            shown += ok
        total = len(self._items)
        self.counter.setText(f"{shown}/{total}" if (q.strip() or only) else str(total))
        if not total:
            self.empty.setText(f"No {self.what} found")
        elif not shown:
            if only and not q.strip():
                self.empty.setText(f"None of the {self.what} are shown")
            elif only:
                self.empty.setText(f"No shown {self.what} match “{q.strip()}”")
            else:
                self.empty.setText(f"No {self.what} match “{q.strip()}”")
        self.empty.setVisible(not shown)
        self.browser.update_tab_titles()

    # ------------------------------------------------------------------ interaction
    def _item_changed(self, item: QTreeWidgetItem, col: int) -> None:
        obj = item.data(0, Qt.UserRole)
        if col == 0 and item.parent() is None and obj is not None:
            self.browser.viewer.show_overlay(obj, item.checkState(0) == Qt.Checked)

    def _double_clicked(self, item: QTreeWidgetItem, _col: int) -> None:
        if item.parent() is None:  # double-click toggles visibility, like copick-web
            item.setCheckState(0, Qt.Unchecked if item.checkState(0) == Qt.Checked else Qt.Checked)

    def _expanded(self, item: QTreeWidgetItem) -> None:
        """Filaments: list each filament with its length (loading the set the first time)."""
        if self.kind != FILAMENTS or item.parent() is not None:
            return
        if item.childCount() != 1 or item.child(0).data(0, Qt.UserRole) is not None:
            return
        obj = item.data(0, Qt.UserRole)
        item.takeChildren()
        try:
            lines = obj.numpy()
            ids = obj.instance_ids()
        except Exception as exc:  # noqa: BLE001
            item.addChild(QTreeWidgetItem([f"Could not load: {exc}"]))
            return
        for i, (fid, line) in enumerate(zip(ids, lines)):
            length_nm = float(np.linalg.norm(np.diff(line, axis=0), axis=1).sum()) / 10 if len(line) > 1 else 0.0
            child = QTreeWidgetItem([f"#{int(fid)}", f"{length_nm:.0f} nm"])
            child.setData(0, Qt.UserRole, ("filament", obj, i))
            child.setToolTip(0, "Click to go to the middle of this filament")
            item.addChild(child)

    def _clicked(self, item: QTreeWidgetItem, _col: int) -> None:
        data = item.data(0, Qt.UserRole)
        if not isinstance(data, tuple) or data[0] != "filament":
            return
        _, obj, index = data
        viewer = self.browser.viewer
        viewer.show_overlay(obj, True)
        layer = viewer.session.find_layer(uri_of(obj))
        if layer is not None and index < len(layer.polylines):
            line = layer.polylines[index]
            z, y, x = (int(v) for v in np.floor(line[len(line) // 2]))
            viewer.session.set_point(z, y, x)
            viewer.session.center_on_point.emit()


def _count(obj, kind: str) -> int:
    if kind == FILAMENTS:
        return len(obj.numpy())
    return len(obj.points)


# ---------------------------------------------------------------------------------------------- run info
class RunInfoDialog(QDialog):
    def __init__(self, browser: "BrowserWindow", run):
        super().__init__(browser)
        self.browser = browser
        root = browser.viewer.root
        cdp = "CDP" in type(root).__name__
        self.setWindowTitle(f"Run {run.name}")
        lay = QVBoxLayout(self)
        lay.setSpacing(6)
        head = QLabel(f"<b>Run {run.name}</b> &nbsp; ({'CryoET Data Portal' if cdp else 'File system'})")
        lay.addWidget(head)

        run_id = getattr(run, "portal_run_id", None)
        if run_id is not None:
            box = QGroupBox("CryoET Data Portal")
            row = QHBoxLayout(box)
            dataset = getattr(run, "portal_dataset_id", None)
            row.addWidget(self._link(f"Run {run_id}", f"{PORTAL}/runs/{run_id}"))
            if dataset is not None:
                row.addWidget(self._link(f"Dataset {dataset}", f"{PORTAL}/datasets/{dataset}"))
            row.addStretch(1)
            lay.addWidget(box)

        cfg = getattr(root, "config", None)
        box = QGroupBox("Location")
        g = QGridLayout(box)
        for r, (label, attr) in enumerate((("Static", "static_root"), ("Overlay", "overlay_root"))):
            path = getattr(cfg, attr, None) if cfg is not None else None
            g.addWidget(QLabel(label), r, 0)
            value = QLabel(str(path) if path else "-")
            value.setTextInteractionFlags(Qt.TextSelectableByMouse)
            value.setFont(QFont("Monospace"))
            g.addWidget(value, r, 1)
        lay.addWidget(box)

        counts = []
        for label, attr in (
            ("picks", "picks"),
            ("segmentations", "segmentations"),
            ("meshes", "meshes"),
            ("filaments", "filaments"),
        ):
            items = getattr(run, attr, None)
            if items is not None:
                counts.append(f"{len(items)} {label}")
        lay.addWidget(QLabel("Contents: " + ", ".join(counts)))

        inner = QWidget()
        il = QVBoxLayout(inner)
        il.setContentsMargins(0, 0, 0, 0)
        for vs in run.voxel_spacings:
            box = QGroupBox(f"Voxel spacing {float(vs.voxel_size):.2f} Å")
            bl = QVBoxLayout(box)
            if not vs.tomograms:
                bl.addWidget(QLabel("No tomograms"))
            for tomo in vs.tomograms:
                bl.addWidget(self._tomogram_card(run, vs, tomo))
            il.addWidget(box)
        il.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(inner)
        scroll.setFrameShape(QFrame.NoFrame)
        lay.addWidget(scroll, 1)
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        lay.addWidget(close, 0, Qt.AlignRight)
        self.resize(560, 480)

    @staticmethod
    def _link(text: str, url: str) -> QLabel:
        label = QLabel(f'<a href="{url}">{text}</a>')
        label.setOpenExternalLinks(False)
        label.linkActivated.connect(lambda u: QDesktopServices.openUrl(QUrl(u)))
        return label

    def _tomogram_card(self, run, vs, tomo) -> QWidget:
        w = QWidget()
        row = QHBoxLayout(w)
        row.setContentsMargins(2, 2, 2, 2)
        facts = [f"<b>{tomo.tomo_type}</b>"]
        try:
            import zarr

            group = zarr.open_group(tomo.zarr(), mode="r")
            datasets = group.attrs.get("multiscales", [{}])[0].get("datasets", [{"path": "0"}])
            arr = group[datasets[0]["path"]]
            z, y, x = arr.shape[-3:]
            v = float(vs.voxel_size) / 10
            facts.append(
                f"{x} × {y} × {z} voxels ({x * v:.0f} × {y * v:.0f} × {z * v:.0f} nm) · "
                f"{arr.dtype}, {len(datasets)} pyramid level{'s' if len(datasets) != 1 else ''}",
            )
        except Exception as exc:  # noqa: BLE001
            facts.append(f"(could not read: {exc})")
        text = QLabel("<br>".join(facts))
        text.setWordWrap(True)
        row.addWidget(text, 1)
        open_button = QPushButton("Open")
        open_button.clicked.connect(lambda: (self.browser.viewer.open_tomogram(run, tomo), self.close()))
        row.addWidget(open_button, 0, Qt.AlignTop)
        return w


# ---------------------------------------------------------------------------------------------- window
class BrowserWindow(QMainWindow):
    def __init__(self, viewer):
        super().__init__(None)
        self.viewer = viewer
        self.session = None
        self.sel = None
        self.setAttribute(Qt.WA_DeleteOnClose, False)
        root = viewer.root
        project = getattr(getattr(root, "config", None), "name", None) or "copick project"
        self.setWindowTitle(f"cmod Run Browser: {project}")
        keep_alive(self)
        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="cmod-browse")

        mb = self.menuBar()
        file_menu = mb.addMenu("&File")
        self._action(file_menu, "Reload project", self.reload)
        self._action(file_menu, "Open Image File...", self.open_files_dialog)
        self._action(file_menu, "Options...", lambda: SettingsDialog(self).exec_())
        self._action(file_menu, "Close", self.close)
        view = mb.addMenu("&View")
        self._action(view, "Run Info...", lambda: self.run_info(self.sel.run) if self.sel else None)
        self._action(view, "Gallery", lambda: self.run_tabs.setCurrentIndex(1))
        self._action(view, "Gallery Window", self.show_gallery_window)
        help_menu = mb.addMenu("&Help")
        self._action(help_menu, "Hot keys", lambda: viewer.info.open_window("help") if viewer.info else None)

        self.runs = RunTree(self)
        self.thumbnails = ThumbnailCache(root)
        self.gallery = Gallery(self, self.thumbnails)
        self.gallery_window: Optional[GalleryWindow] = None
        self.runs.search.textChanged.connect(self.gallery.filter)
        self.run_tabs = QTabWidget()
        self.run_tabs.addTab(self.runs, "Runs")
        self.run_tabs.addTab(self.gallery, "Gallery")
        self.run_tabs.currentChanged.connect(lambda i: self.gallery.start() if i == 1 else None)

        lower = QWidget()
        ll = QVBoxLayout(lower)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(2)
        self.objects_label = QLabel()
        ll.addWidget(self.objects_label)
        self.tables = {
            PICKS: EntityTable(self, PICKS, "picks"),
            SEGMENTATION: EntityTable(self, SEGMENTATION, "segmentations"),
            FILAMENTS: EntityTable(self, FILAMENTS, "filaments"),
            MESH: EntityTable(self, MESH, "meshes"),
        }
        self.entity_tabs = QTabWidget()
        for table in self.tables.values():
            self.entity_tabs.addTab(table, "")
        ll.addWidget(self.entity_tabs, 1)

        split = QSplitter(Qt.Vertical)
        split.addWidget(self.run_tabs)
        split.addWidget(lower)
        split.setSizes([300, 300])
        self.setCentralWidget(split)
        self.current_label = QLabel()
        self.statusBar().addWidget(self.current_label, 1)
        info_button = QPushButton("Run Info")
        info_button.clicked.connect(lambda: self.run_info(self.sel.run) if self.sel else None)
        self.statusBar().addPermanentWidget(info_button)
        self.reload(reread=False)
        self.update_tab_titles()

    def _action(self, menu, text, slot) -> QAction:
        a = QAction(text, self)
        a.triggered.connect(lambda _=False: slot())
        menu.addAction(a)
        return a

    def sizeHint(self) -> QSize:
        return QSize(520, 760)

    # ------------------------------------------------------------------ data
    def reload(self, reread: bool = True) -> None:
        root = self.viewer.root
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            if reread and hasattr(root, "refresh"):
                try:
                    root.refresh()
                except Exception as exc:  # noqa: BLE001
                    log.warning("Could not refresh the project: %s", exc)
            runs = list(root.runs)
        finally:
            QApplication.restoreOverrideCursor()
        if reread:
            self.thumbnails.clear()  # tomograms may have changed: render the previews again
        self._runs = runs
        self.runs.populate(runs)
        for gallery in self._galleries():
            gallery.populate(runs)
        if self.run_tabs.currentIndex() == 1:
            self.gallery.start()
        if self.session is not None:
            self.attach(self.session, force=True)

    def attach(self, session, force: bool = False) -> None:
        """Follow a window set (and the tomogram it shows): highlight it and list its run's objects."""
        if session is not self.session:
            old = self.session
            if old is not None:
                for sig, slot in ((old.overlays_changed, self._sync_tables), (old.image_changed, self._image_changed)):
                    with contextlib.suppress(RuntimeError, TypeError):  # closed, or never connected
                        sig.disconnect(slot)
            if session is not None:
                session.overlays_changed.connect(self._sync_tables)
                session.image_changed.connect(self._image_changed)
        elif not force and self.sel is not None and session is not None and self._shows(session):
            return
        self.session = session
        img = session.image if session is not None else None
        run, tomo = (img.run, img.tomogram) if img is not None else (None, None)
        old_run = self.sel.run if self.sel is not None else None
        self.sel = Selection(run=run, tomogram=tomo) if run is not None else None
        if tomo is not None:
            voxel = float(tomo.voxel_spacing.voxel_size)
            self.runs.reveal(run.name, voxel, tomo.tomo_type)
            self.current_label.setText(f"{run.name} · {voxel:.2f} Å · {tomo.tomo_type}")
        else:
            self.current_label.setText(img.title if img is not None else "")
        for gallery in self._galleries():
            gallery.set_current(run.name if run is not None else None)
        if run is None:
            self.objects_label.setText("No copick run is shown")
            for table in self.tables.values():
                table.populate([], self._pool)
            return
        self.objects_label.setText(f"Objects in run <b>{run.name}</b>")
        if not force and old_run is not None and getattr(old_run, "name", None) == run.name:
            self._sync_tables()  # same run (another tomogram of it): the lists stay
            return
        lists = {
            PICKS: list(run.picks),
            SEGMENTATION: list(run.segmentations),
            FILAMENTS: list(getattr(run, "filaments", []) or []),
            MESH: list(getattr(run, "meshes", []) or []),
        }
        for kind, table in self.tables.items():
            table.populate(lists[kind], self._pool)

    def _shows(self, session) -> bool:
        img = session.image
        return self.sel is not None and self.sel.run is img.run and self.sel.tomogram is img.tomogram

    def _image_changed(self, _old=None) -> None:
        if self.session is not None:
            self.attach(self.session)

    def _sync_tables(self) -> None:
        for table in self.tables.values():
            table.sync()

    def update_tab_titles(self) -> None:
        if not hasattr(self, "entity_tabs"):
            return
        names = {PICKS: "Picks", SEGMENTATION: "Segmentations", FILAMENTS: "Filaments", MESH: "Meshes"}
        for i, (kind, table) in enumerate(self.tables.items()):
            shown = table.shown_count()
            self.entity_tabs.setTabText(i, f"{names[kind]} ({shown})" if shown else names[kind])
            self.entity_tabs.setTabToolTip(i, f"{shown} shown in the viewer" if shown else "")

    # ------------------------------------------------------------------ actions
    def open_run(self, run) -> None:
        """Open a run's preferred tomogram: the current type and voxel size if it has them (copick-web)."""
        prefer_type = prefer_vs = None
        if self.sel is not None and self.sel.tomogram is not None:
            prefer_type = self.sel.tomogram.tomo_type
            prefer_vs = float(self.sel.tomogram.voxel_spacing.voxel_size)
        tomo = choose_tomogram(run, prefer_type, prefer_vs)
        if tomo is None:
            self.statusBar().showMessage(f"Run {run.name} has no tomograms", 5000)
            return
        self.viewer.open_tomogram(run, tomo)

    def _galleries(self) -> list:
        galleries = [self.gallery]
        if self.gallery_window is not None:
            galleries.append(self.gallery_window.gallery)
        return galleries

    def show_gallery_window(self) -> None:
        """The gallery in its own, larger window (it shares the thumbnails with the tab)."""
        if self.gallery_window is None:
            project = getattr(getattr(self.viewer.root, "config", None), "name", None) or "copick project"
            self.gallery_window = GalleryWindow(self, self.thumbnails, project)
            g = self.gallery_window.gallery
            g.populate(getattr(self, "_runs", []))
            g.set_current(self.sel.run.name if self.sel is not None else None)
            g.start()
        self.gallery_window.show()
        self.gallery_window.raise_()
        self.gallery_window.activateWindow()

    def open_files_dialog(self) -> None:
        """MRC files into the current window set's ZaP (or new windows if none is open)."""
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Open image files",
            "",
            "MRC images (*.mrc *.rec *.st *.ali *.preali *.map *.mrcs);;All files (*)",
        )
        if paths:
            self.viewer.open_files(paths)

    def run_info(self, run) -> None:
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            dialog = RunInfoDialog(self, run)
        finally:
            QApplication.restoreOverrideCursor()
        dialog.show()

    def keyPressEvent(self, event) -> None:
        from copickmod import keys

        if self.viewer.session is None or not keys.handle_global(self.viewer.session, event, axis=0):
            super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        super().closeEvent(event)
        if not self.viewer.infos:  # no tomogram open: closing the browser quits
            if self.gallery_window is not None:
                self.gallery_window.close()
            self.thumbnails.shutdown()
            self._pool.shutdown(wait=False, cancel_futures=True)
            self.viewer.quit()
