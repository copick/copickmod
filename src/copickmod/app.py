"""Building a viewing session from resolved copick objects and running the Qt application."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence

from copickmod import overlays
from copickmod.project import Selection, uri_of
from copickmod.volume import ChunkCache, MultiscaleVolume

log = logging.getLogger(__name__)


@dataclass
class ViewOptions:
    open_xyz: bool = False
    open_slicer: bool = False
    open_zap: bool = False
    open_model: bool = False
    cache_bytes: int = 2 * 1024**3
    binning: int = 1
    zoom: Optional[float] = None
    center_xyz: Optional[Sequence[int]] = None  # 1-based, like the 3dmod display
    angles_xyz: Optional[Sequence[float]] = None
    snapshot: Optional[str] = None
    snapshot_timeout: float = 300.0
    browser: bool = False


def build_layers(root, sel: Selection, vol: MultiscaleVolume):
    point_layers: List[overlays.PointLayer] = []
    seg_layers: List[overlays.SegLayer] = []
    mesh_layers: List[overlays.MeshLayer] = []
    for picks in sel.picks:
        try:
            point_layers.append(overlays.picks_layer(root, picks, vol, uri_of(picks)))
        except Exception as exc:  # noqa: BLE001 - one bad overlay must not stop the viewer
            log.warning("Could not load picks %s: %s", uri_of(picks), exc)
    for fil in sel.filaments:
        try:
            point_layers.append(overlays.filaments_layer(root, fil, vol, uri_of(fil)))
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not load filaments %s: %s", uri_of(fil), exc)
    for mesh in sel.meshes:
        try:
            mesh_layers.append(overlays.mesh_layer(root, mesh, vol, uri_of(mesh)))
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not load mesh %s: %s", uri_of(mesh), exc)
    for seg in sel.segmentations:
        try:
            seg_layers.append(overlays.segmentation_layer(root, seg, uri_of(seg)))
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not open segmentation %s: %s", uri_of(seg), exc)
    return point_layers, seg_layers, mesh_layers


def tomogram_image(tomo, run, opts: ViewOptions):
    """An :class:`~copickmod.state.Image` for a copick tomogram (its OME-Zarr opened lazily)."""
    from copickmod.state import Image, level_for_binning

    name = uri_of(tomo)
    t0 = time.monotonic()
    vol = MultiscaleVolume(tomo.zarr(), name=name, fallback_voxel_size=float(tomo.voxel_spacing.voxel_size))
    log.debug("Opened %s (%d levels) in %.2fs", name, vol.nlevels, time.monotonic() - t0)
    return Image(vol, name, run=run, tomogram=tomo, min_level=level_for_binning(vol, opts.binning))


def file_image(path: str, opts: ViewOptions):
    """An :class:`~copickmod.state.Image` for an MRC file (memory-mapped)."""
    from copickmod.mrc import MrcVolume
    from copickmod.state import Image, level_for_binning

    vol = MrcVolume(path)
    return Image(vol, vol.name, path=vol.path, min_level=level_for_binning(vol, opts.binning))


def build_session(root, sel: Selection, opts: ViewOptions):
    """A session showing the selection's tomogram first, with its other tomograms and files to switch to."""
    from copickmod.state import Session

    images = [tomogram_image(t, sel.run, opts) for t in [sel.tomogram, *sel.more_tomograms] if t is not None]
    images += [file_image(path, opts) for path in sel.files]
    if not images:
        raise ValueError("nothing to show: no tomogram and no file")
    first = images[0]
    cache = ChunkCache(opts.cache_bytes)
    t0 = time.monotonic()
    point_layers, seg_layers, mesh_layers = (
        build_layers(root, sel, first.volume) if sel.run is not None else ([], [], [])
    )
    log.debug(
        "Loaded %d overlays in %.2fs",
        len(point_layers) + len(seg_layers) + len(mesh_layers),
        time.monotonic() - t0,
    )
    session = Session(
        first.volume,
        cache,
        None,  # contrast is set from the first sections that load
        point_layers,
        seg_layers,
        min_level=first.min_level,
        mesh_layers=mesh_layers,
        image=first,
    )
    session.binning = opts.binning
    for img in images[1:]:
        session.add_image(img, show=False)
    if opts.center_xyz is not None:
        x, y, z = (int(v) - 1 for v in opts.center_xyz)
        session.set_point(z, y, x)
    return session


WINDOW_KINDS = {"ZapWindow": "zap", "XyzWindow": "xyz", "SlicerWindow": "slicer", "ModelViewWindow": "model"}


def _first_zap(info):
    return next((w for w in info.open_windows() if type(w).__name__ == "ZapWindow"), None)


CASCADE = 28  # pixels new windows are moved down and right when the title bar height is unknown


def cascade_pos(like, size):
    """Where a new window goes: one title bar down and right of ``like``, or on top of it if there is no room."""
    from qtpy.QtCore import QPoint
    from qtpy.QtGui import QGuiApplication

    title = like.geometry().top() - like.frameGeometry().top()
    step = title if 16 <= title <= 60 else CASCADE
    pos = like.pos() + QPoint(step, step)
    screen = QGuiApplication.screenAt(like.pos() + QPoint(40, 20)) or QGuiApplication.primaryScreen()
    if screen is not None:
        avail = screen.availableGeometry()
        frame = like.frameGeometry().size() - like.size()  # window manager decorations
        if pos.x() + size.width() + frame.width() > avail.right() + 1 or (
            pos.y() + size.height() + frame.height() > avail.bottom() + 1
        ):
            return like.pos()
    return pos


class Viewer:
    """The application: one copick project, any number of window sets and the run browser.

    A window set is an Info window with its ZaP (and other image windows) and one :class:`Session`; the ZaP can
    hold several tomograms and MRC files and switches between them with Left/Right (or 1/2). Opening a tomogram
    from the browser follows the Options (``settings.py``): by default another tomogram of an open run goes into
    that run's ZaP, and a tomogram of another run gets a new ZaP and Info window, cascaded from the current ones.
    """

    def __init__(self, root, opts: ViewOptions):
        self.root = root
        self.opts = opts
        self.infos: list = []  # Info windows of the open window sets, oldest first
        self.active = None  # the Info window of the set used last (the browser shows its run)
        self.browser = None
        self.quit_on_close = False  # set when this controller runs the application's event loop
        from qtpy.QtWidgets import QApplication

        app = QApplication.instance()
        if app is not None:
            app.focusWindowChanged.connect(self._focus_changed)

    # ------------------------------------------------------------------ the current window set
    @property
    def info(self):
        return self.active

    @property
    def session(self):
        return self.active.session if self.active is not None else None

    @property
    def sel(self) -> Optional[Selection]:
        """The run and tomogram shown in the current window set (what the browser highlights)."""
        s = self.session
        if s is None:
            return None
        return Selection(run=s.image.run, tomogram=s.image.tomogram)

    def set_active(self, info) -> None:
        if info is self.active:
            return
        self.active = info
        if self.browser is not None:
            self.browser.attach(info.session if info is not None else None)

    def _focus_changed(self, window) -> None:
        if window is None:
            return
        for info in self.infos:
            try:
                widgets = [info, *info.open_windows()]
            except RuntimeError:  # being deleted
                continue
            if any(w.windowHandle() is window for w in widgets):
                self.set_active(info)
                return

    # ------------------------------------------------------------------ opening
    def initial_kinds(self) -> List[str]:
        opts = self.opts
        kinds = [
            k for k, on in (("xyz", opts.open_xyz), ("slicer", opts.open_slicer), ("model", opts.open_model)) if on
        ]
        if opts.open_zap or not kinds:
            kinds.insert(0, "zap")
        return kinds

    def open_kinds(self) -> List[str]:
        if self.info is None:
            return []
        return [WINDOW_KINDS[type(w).__name__] for w in self.info.open_windows() if type(w).__name__ in WINDOW_KINDS]

    def open(self, sel: Selection, kinds: Optional[List[str]] = None):
        """Open a new window set (Info window plus ZaP or the given windows) for a selection.

        The first set uses the command-line options and the remembered window positions; later ones go a little
        down and right of the current set's Info and ZaP windows, at the same zoom.
        """
        from qtpy.QtCore import QTimer

        from copickmod import settings
        from copickmod.info import InfoWindow, startup_text

        first = not self.infos
        like = self.active
        like_zap = _first_zap(like) if like is not None else None
        kinds = kinds or (self.initial_kinds() if first else ["zap"])
        opts = self.opts if first else ViewOptions(cache_bytes=self.opts.cache_bytes, binning=self.opts.binning)
        session = build_session(self.root, sel, opts)
        info = InfoWindow(session, header=startup_text(sel, session))
        info.viewer = self
        info.closed.connect(lambda _replacing, i=info: self._info_closed(i))
        info.slicer_angles = list(self.opts.angles_xyz) if first and self.opts.angles_xyz is not None else None
        remember = settings.get("windows/remember")
        if like is not None and remember:
            info.move(cascade_pos(like, info.size()))
        elif first:
            pos = settings.window_pos("info")
            if pos is not None:
                info.move(pos)
        self.infos.append(info)
        info.show()
        for kind in kinds:
            win = info.open_window(kind)
            if kind != "zap" or win is None:
                if first and self.opts.zoom and hasattr(win, "canvas") and hasattr(win.canvas, "set_zoom"):
                    win.canvas.set_zoom(self.opts.zoom)
                continue
            if like_zap is not None and remember:
                ratio = session.volume.voxel_size / like_zap.session.volume.voxel_size
                win.canvas.set_zoom(like_zap.canvas.zoom * ratio)
                win.resize(like_zap.size())
                win.move(cascade_pos(like_zap, like_zap.size()))
            elif first:
                if self.opts.zoom:
                    win.canvas.set_zoom(self.opts.zoom)
                    win.fit_to_image()
                pos = settings.window_pos("zap")
                if pos is not None:
                    win.move(pos)
        self.set_active(info)
        QTimer.singleShot(0, session.initial_contrast)
        return info

    def _target_for(self, run):
        """The window set a tomogram of ``run`` should go into, following the Options; None for a new set."""
        from copickmod import settings

        name = str(getattr(run, "name", ""))
        with_run = [i for i in self.infos if any(img.run_name == name for img in i.session.images)]
        if with_run:
            if settings.get("open/same_run") == settings.SAME_ZAP:
                return self.active if self.active in with_run else with_run[-1]
            return None
        if settings.get("open/other_run") == settings.SAME_ZAP:
            return self.active
        return None

    def _bring_up(self, info) -> None:
        zap = _first_zap(info)
        for w in (info, zap):
            if w is not None:
                w.show()
                w.raise_()
        if zap is not None:
            zap.activateWindow()
        self.set_active(info)

    def open_tomogram(self, run, tomogram):
        """Open a tomogram from the browser: switch to it if it is open, else as the Options say.

        A new window set for a run that is open already gets the overlays shown there, as in copick-web.
        """
        from copickmod.project import object_kind

        for info in self.infos:
            index = info.session.find_image(tomogram=tomogram)
            if index is not None:
                info.session.show_image(index)
                self._bring_up(info)
                return info
        target = self._target_for(run)
        if target is not None:
            try:
                image = tomogram_image(tomogram, run, ViewOptions(binning=self.opts.binning))
            except Exception as exc:  # noqa: BLE001 - report in the Info window
                target.session.message.emit(f"Could not open {uri_of(tomogram)}: {exc}")
                return target
            target.session.add_image(image)
            self._bring_up(target)
            return target
        sel = Selection(run=run, tomogram=tomogram)
        s = self.session
        if s is not None and s.image.run_name == str(getattr(run, "name", "")):
            for layer in s.layers():
                if layer.visible and layer.source is not None:
                    kind = object_kind(layer.source)
                    {
                        "picks": sel.picks,
                        "filaments": sel.filaments,
                        "segmentation": sel.segmentations,
                        "mesh": sel.meshes,
                    }[kind].append(layer.source)
        return self.open(sel)

    def open_files(self, paths: Sequence[str], info=None):
        """Open MRC files into a window set's ZaP (the current one by default), or a new set if none is open."""
        target = info or self.active
        if target is None:
            return self.open(Selection(run=None, tomogram=None, files=list(paths)))
        target.session.open_files(paths)
        self._bring_up(target)
        return target

    def _info_closed(self, info) -> None:
        """Closing an Info window closes its set; closing the last one quits, as in 3dmod (unless the browser
        is open)."""
        if info in self.infos:
            self.infos.remove(info)
        if info is self.active:
            self.active = None
            self.set_active(self.infos[-1] if self.infos else None)
        if self.infos:
            return
        if self.browser is not None and self.browser.isVisible():
            return
        self.quit()

    def quit(self) -> None:
        # Only quit when this controller runs the event loop: QApplication.quit() outside exec() makes every later
        # event loop return at once (which would break an embedding application or a test session).
        if self.quit_on_close:
            from qtpy.QtWidgets import QApplication

            app = QApplication.instance()
            if app is not None:
                app.quit()

    def show_overlay(self, obj, on: bool) -> None:
        """Show or hide a copick object over the current tomogram, loading it the first time."""
        from copickmod.project import object_kind, uri_of

        s = self.session
        if s is None:
            return
        name = uri_of(obj)
        layer = s.find_layer(name)
        if layer is None:
            if not on:
                return
            try:
                layer = overlays.make_layer(self.root, obj, object_kind(obj), s.volume, name)
            except Exception as exc:  # noqa: BLE001 - report and carry on
                s.message.emit(f"Could not load {name}: {exc}")
                return
            s.add_layer(layer)
        elif layer.visible != on:
            s.set_layer_visible(layer, on)

    def show_browser(self):
        from copickmod.browser import BrowserWindow

        if self.root is None:
            if self.session is not None:
                self.session.message.emit("The run browser needs a copick project (-c or $CFG)")
            return None
        if self.browser is None:
            self.browser = BrowserWindow(self)
            if self.session is not None:
                self.browser.attach(self.session)
            self.browser.resize(self.browser.sizeHint())
        self.browser.show()
        self.browser.raise_()
        self.browser.activateWindow()
        return self.browser


def run_viewer(root, sel: Optional[Selection], opts: ViewOptions) -> int:
    from qtpy.QtWidgets import QApplication

    from copickmod.style import apply_app_style

    app = QApplication.instance() or QApplication(["cmod"])
    app.setApplicationName("cmod")
    apply_app_style(app)
    from copickmod import settings

    settings.save_geometry = not opts.snapshot  # snapshot windows must not move the user's saved positions
    viewer = Viewer(root, opts)
    viewer.quit_on_close = True
    if sel is not None:
        viewer.open(sel)
    if root is not None and (sel is None or opts.browser):
        viewer.show_browser()
    if opts.snapshot and viewer.info is not None:
        _schedule_snapshot(app, viewer.info, viewer.session, opts.snapshot, opts.snapshot_timeout)
    app._cmod_viewer = viewer  # keep the controller alive for the lifetime of the application
    return app.exec_()


def window_complete(win) -> bool:
    canvas = getattr(win, "canvas", None)
    if canvas is None:
        return True
    if hasattr(canvas, "pending_isosurfaces"):
        return canvas.pending_isosurfaces() == 0
    if hasattr(canvas, "views"):
        return all(v.complete for v in canvas.views.values())
    if hasattr(canvas, "view"):
        return canvas.view.complete
    return getattr(canvas, "complete", True)


def _schedule_snapshot(app, info, session, path: str, timeout: float) -> None:
    """Wait until every window has its data, save ``<stem>_<window>.png`` files and quit."""
    from qtpy.QtCore import QTimer

    t0 = time.monotonic()
    stem, ext = os.path.splitext(path)
    ext = ext or ".png"

    def check() -> None:
        wins = info.open_windows()
        for w in wins:
            w.repaint()
        busy = session.cache.pending() > 0 or session.auto_pending or not all(window_complete(w) for w in wins)
        log.debug(
            "snapshot check at %.1fs: pending %d, auto %s, complete %s",
            time.monotonic() - t0,
            session.cache.pending(),
            session.auto_pending,
            [window_complete(w) for w in wins],
        )
        if busy and time.monotonic() - t0 < timeout:
            QTimer.singleShot(250, check)
            return
        if busy:
            print(f"cmod: snapshot timed out after {timeout:.0f}s; saving what is loaded", flush=True)
        counts = {}
        for w in [info, *wins]:
            kind = type(w).__name__.replace("Window", "").lower()
            counts[kind] = counts.get(kind, 0) + 1
            suffix = kind if counts[kind] == 1 else f"{kind}{counts[kind]}"
            out = f"{stem}_{suffix}{ext}"
            w.grab().save(out)
            print(f"cmod: saved {out}", flush=True)
        info.close()

    QTimer.singleShot(300, check)
