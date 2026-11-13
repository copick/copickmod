"""State shared by all cmod windows (the equivalent of 3dmod's ViewInfo)."""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
from qtpy.QtCore import QObject, Qt, QTimer, Signal

from copickmod.contrast import Contrast
from copickmod.contrast import stats as contrast_stats
from copickmod.overlays import PointLayer, SegLayer, reframe
from copickmod.volume import ChunkCache, MultiscaleVolume

log = logging.getLogger(__name__)

AXIS_NAMES = ("Z", "Y", "X")

# 3dmod movie rates (moviecon.cpp): BASEINT, RATEFAC, MINRATE, MAXRATE.
MOVIE_BASE_INTERVAL = 50.0  # ms, rate 0
MOVIE_RATE_FACTOR = 1.25992  # cube root of 2
MOVIE_MIN_RATE = 0.2  # frames/s
MOVIE_MAX_RATE = 200.0


def level_for_binning(vol: MultiscaleVolume, binning: int) -> int:
    """The finest pyramid level shown with ``-B binning`` (3dmod's binning on loading)."""
    level = 0
    for i in range(vol.nlevels):
        if vol.factor(i)[2] <= binning + 1e-6:
            level = i
    return level


def movie_interval(rate: int) -> float:
    """Milliseconds between movie frames at 3dmod rate step ``rate`` (higher is slower)."""
    return MOVIE_BASE_INTERVAL * MOVIE_RATE_FACTOR**rate


@dataclass
class Image:
    """One tomogram or MRC file a :class:`Session` can show. The ZaP switches between them like 3dmod's times."""

    volume: MultiscaleVolume
    name: str
    run: object = None  # copick run (files opened into a run's windows get that run, so its overlays show)
    tomogram: object = None  # copick tomogram; None for files
    path: Optional[str] = None  # MRC file
    min_level: int = 0
    saved: Optional[dict] = None  # contrast state while another image is shown

    @property
    def run_name(self) -> str:
        return str(getattr(self.run, "name", "") or "")

    @property
    def title(self) -> str:
        return f"{self.run_name}  {self.name}" if self.run_name else self.name

    def same_as(self, tomogram=None, path=None) -> bool:
        if path is not None:
            return self.path is not None and os.path.abspath(path) == os.path.abspath(self.path)
        if tomogram is None or self.tomogram is None:
            return False
        from copickmod.project import uri_of

        return self.tomogram is tomogram or (
            uri_of(self.tomogram) == uri_of(tomogram)
            and self.run_name == str(getattr(tomogram.voxel_spacing.run, "name", ""))
        )


class Session(QObject):
    """Current point, contrast, overlays and data access for the tomograms shown by one set of windows.

    A session holds one or more :class:`Image` objects and shows one at a time (``volume`` is the shown one's).
    The current point is an integer voxel position at level 0 of that volume, stored zyx (``point[0]`` is Z).
    """

    point_changed = Signal()
    contrast_changed = Signal()
    overlays_changed = Signal()
    data_arrived = Signal()
    message = Signal(str)
    center_on_point = Signal()  # bring the current point into view (e.g. after going to a pick)
    layers_changed = Signal()  # overlays were added
    image_changed = Signal(object)  # another image is shown; the argument is the volume shown before
    images_changed = Signal()  # an image was added
    _chunk_in = Signal()  # emitted from fetch threads; delivered queued on the GUI thread
    _error_in = Signal(str)

    def __init__(
        self,
        volume: MultiscaleVolume,
        cache: ChunkCache,
        contrast: Optional[Contrast] = None,
        point_layers: Sequence[PointLayer] = (),
        seg_layers: Sequence[SegLayer] = (),
        title: str = "",
        mesh_layers: Sequence = (),
        min_level: int = 0,
        image: Optional[Image] = None,
    ):
        super().__init__()
        self.images: List[Image] = [image or Image(volume, title or volume.name, min_level=min_level)]
        self.current = 0
        self.binning = 1  # -B: the finest level images opened later may show
        self._run_layers: Dict[str, tuple] = {}  # overlays of the runs not shown now: (points, segs, meshes, frame)
        self.volume = volume
        self.cache = cache
        self.contrast = contrast if contrast is not None else Contrast(0.0, 1.0, 0.0, 1.0)
        self.point_layers: List[PointLayer] = list(point_layers)
        self.seg_layers: List[SegLayer] = list(seg_layers)
        self.mesh_layers: list = list(mesh_layers)  # copick meshes, shown in the Model View only
        self.min_level = max(0, min(min_level, volume.nlevels - 1))
        self.opacity = 0.5
        self.point = [s // 2 for s in volume.shape]
        # Like 3dmod's autocontrast at start-up, redo the contrast from the displayed section once it has loaded
        # (the initial values come from the coarsest level, whose binned noise is weaker than level 0's).
        self.auto_pending = True
        self.has_contrast = contrast is not None
        self._view_levels: Dict[int, tuple] = {}  # view id -> (is_zap, level)
        # 3dmod "Float" (on by default): keep the mean and SD of the display constant when changing sections, and
        # "Subarea": base Float and Auto on the area visible in the ZaP.
        self.float_on = False  # as 3dmod; the Info window starts it from the remembered setting
        self.subarea = False
        self.subarea_roi: Optional[tuple] = None  # (level, (r0, r1), (c0, c1)) visible in the ZaP
        self._float_ref: Optional[tuple] = None  # (level, z) whose appearance is kept
        self._float_bw: Optional[tuple] = None  # unclamped black/white tracked like 3dmod's sRefBlack/White
        self._stats_memo: Dict[tuple, object] = {}  # (level, z, rows, cols) -> Stats of complete sections
        self._float_timer = QTimer(self)
        self._float_timer.setSingleShot(True)
        self._float_timer.setInterval(0)
        self._float_timer.timeout.connect(self.apply_float)
        self.opener: Optional[Callable[[str], None]] = None  # set by the Info window
        self.navigator: Optional[Callable[[str], None]] = None  # object/contour/point stepping (Info window)
        self.show_model = True  # T: draw overlays at all
        self.show_markers = True  # Shift+T: draw the current point marker
        self.ramp_slots: List[Optional[tuple]] = [None, None, None, None]  # F9/F10: four stored contrast settings
        self.ramp_index = 0
        self._wanted: Dict[int, set] = {}

        # Coalesce chunk arrivals into at most one repaint per 40 ms.
        self._repaint_timer = QTimer(self)
        self._repaint_timer.setSingleShot(True)
        self._repaint_timer.setInterval(40)
        self._repaint_timer.timeout.connect(self._data_arrived)
        self._chunk_in.connect(self._on_chunk)
        self._error_in.connect(self.message.emit)
        self._errors_reported = 0
        cache.on_chunk = lambda key: self._chunk_in.emit()
        cache.on_error = self._report_error

        # Movie through Z (3dmod '#'/'$', middle/right click in ZaP), at 3dmod's rates (moviecon.cpp).
        self.movie_dir = 0
        self.movie_axis = 0
        self.movie_rate = 0  # 3dmod movierate: interval = 50 ms * 1.26 ** rate, so 0 is 20 frames/s
        self.movie_timer = QTimer(self)
        self.movie_timer.setTimerType(Qt.PreciseTimer)
        self.movie_timer.setInterval(round(movie_interval(0)))
        self.movie_timer.timeout.connect(self._movie_step)

    # ------------------------------------------------------------------ data
    def _data_arrived(self) -> None:
        if self.auto_pending:
            self.initial_contrast()
        elif self.float_on and self._float_key() != (self._float_ref or (None, None))[:2]:
            self.apply_float()
        self.data_arrived.emit()

    def initial_contrast(self) -> bool:
        """Set contrast and slider range from the current section at the display level, once it is loaded.

        Until then, the first complete section of a coarser level gives provisional values, so an image can be
        shown as soon as any data has arrived. Returns True once the final contrast is set.
        """
        if not self.auto_pending:
            return True
        for level in range(self.display_level, self.volume.nlevels):
            if level > self.display_level and self.has_contrast:
                break
            z = int(self.volume.level_index(level, self.volume, self.point[0] + 0.5, 0))
            plane, missing = self.volume.plane(self.cache, level, 0, z, request=level == self.display_level)
            if missing is not None:
                continue
            invert = self.contrast.invert
            self.contrast = Contrast.from_stats(contrast_stats(plane), getattr(self.volume, "header_range", None))
            self.contrast.invert = invert
            self.has_contrast = True
            self.auto_pending = level != self.display_level
            self._reset_float()
            self.contrast_changed.emit()
            break
        return not self.auto_pending

    def _on_chunk(self) -> None:
        if not self._repaint_timer.isActive():
            self._repaint_timer.start()

    def _report_error(self, key, exc) -> None:
        self._errors_reported += 1
        if self._errors_reported <= 5:
            self._error_in.emit(f"Error reading data (level {key[1]}, chunk {key[2:]}): {exc}")

    def set_wanted(self, view_id: int, keys: set) -> None:
        """Record the chunks a view is waiting for and cancel queued fetches nobody wants any more."""
        self._wanted[view_id] = keys
        keep = set().union(*self._wanted.values()) if self._wanted else set()
        self.cache.cancel_except(keep)

    def drop_view(self, view_id: int) -> None:
        self._wanted.pop(view_id, None)
        self._view_levels.pop(view_id, None)

    def report_level(self, view_id: int, level: int, zap: bool = False) -> None:
        """Windows report the pyramid level they display; contrast statistics follow :attr:`display_level`."""
        before = self.display_level
        self._view_levels[view_id] = (zap, level)
        if self.display_level != before:
            self.schedule_float()

    def set_subarea_roi(self, level: int, rows: tuple, cols: tuple) -> None:
        self.subarea_roi = (level, tuple(rows), tuple(cols))

    @property
    def display_level(self) -> int:
        """The level contrast is computed on: the ZaP's if one is open, else the finest level shown anywhere.

        Binned levels have less noise, so contrast must come from the level the user is looking at.
        """
        zap_levels = [lv for is_zap, lv in self._view_levels.values() if is_zap]
        levels = zap_levels or [lv for _, lv in self._view_levels.values()]
        return min(min(levels) if levels else self.min_level, self.volume.nlevels - 1)

    def choose_level(self, zoom: float) -> int:
        """3dmod pyramid rule: the coarsest level needing at most 1.5x upsampling at this zoom."""
        best = self.min_level
        for level in range(self.min_level, self.volume.nlevels):
            if self.volume.factor(level)[2] * zoom <= 1.5 + 1e-6:
                best = level
        return best

    def value_at(self, point: Optional[Sequence[int]] = None) -> float:
        """Level-0 voxel value (reads the chunk synchronously if needed)."""
        z, y, x = self.point if point is None else point
        lv = self.volume.levels[0]
        cidx = (z // lv.chunks[0], y // lv.chunks[1], x // lv.chunks[2])
        block = self.cache.fetch(self.volume, 0, cidx)
        return float(block[z - cidx[0] * lv.chunks[0], y - cidx[1] * lv.chunks[1], x - cidx[2] * lv.chunks[2]])

    # ------------------------------------------------------------------ current point
    def set_point(self, z: Optional[int] = None, y: Optional[int] = None, x: Optional[int] = None) -> None:
        new = list(self.point)
        for i, v in enumerate((z, y, x)):
            if v is not None:
                new[i] = int(min(max(int(v), 0), self.volume.shape[i] - 1))
        if new != self.point:
            z_changed = new[0] != self.point[0]
            self.point = new
            if z_changed and self.float_on:
                # Before point_changed, so the new section is painted once, with its floated contrast.
                self.apply_float()
            self.point_changed.emit()

    def move(self, axis: int, delta: int) -> None:
        values: List[Optional[int]] = [None, None, None]
        values[axis] = self.point[axis] + delta
        self.set_point(*values)

    def set_axis(self, axis: int, value: int) -> None:
        values: List[Optional[int]] = [None, None, None]
        values[axis] = value
        self.set_point(*values)

    # ------------------------------------------------------------------ contrast
    def set_sliders(self, black: float, white: float) -> None:
        self.auto_pending = False
        self.contrast.set_sliders(black, white)
        self._reset_float()
        self.contrast_changed.emit()

    def step_contrast(self, dblack: float, dwhite: float) -> None:
        self.auto_pending = False
        self.contrast.step(dblack, dwhite)
        self._reset_float()
        self.contrast_changed.emit()

    # Float: 3dmod's imod_info_bwfloat, in float space.
    def _float_key(self) -> tuple:
        level = self.display_level
        return level, int(self.volume.level_index(level, self.volume, self.point[0] + 0.5, 0))

    def _region(self, level: int):
        """Rows/cols to sample at ``level``: the ZaP's visible area with Subarea on, else the whole section."""
        if self.subarea and self.subarea_roi is not None and self.subarea_roi[0] == level:
            return self.subarea_roi[1], self.subarea_roi[2]
        return None, None

    def section_stats(self, level: int, z: int):
        """Mean/SD of what is loaded of a section (or the subarea); None if too little of it is loaded."""
        rows, cols = self._region(level)
        memo = (self.volume.uid, level, z, rows, cols)
        if memo in self._stats_memo:  # complete sections never change: Float during movies reuses them
            return self._stats_memo[memo]
        data, missing = self.volume.plane(self.cache, level, 0, z, rows, cols, request=False)
        if missing is None:
            result = contrast_stats(data, border=0.0 if rows is not None else 0.05)
            if len(self._stats_memo) >= 4096:
                self._stats_memo.clear()
            self._stats_memo[memo] = result
            return result
        if missing.mean() > 0.75:
            return None
        loaded = np.asarray(data, dtype=np.float32)[~missing]
        return contrast_stats(loaded, border=0.0)

    def _reset_float(self) -> None:
        """The current section and black/white become the reference that Float keeps (3dmod does this whenever
        the contrast is changed by hand)."""
        self._float_ref = None
        self._float_bw = None
        self.schedule_float()

    def schedule_float(self) -> None:
        if self.float_on and not self._float_timer.isActive():
            self._float_timer.start()

    def apply_float(self) -> None:
        if not self.float_on or not self.has_contrast or self.auto_pending:
            return
        key = self._float_key()
        if self._float_ref is None:
            if self.section_stats(*key) is not None:
                self._float_ref = key
                self._float_bw = (self.contrast.black, self.contrast.white)
            return
        if key == self._float_ref:
            return
        cur = self.section_stats(*key)
        ref = self.section_stats(*self._float_ref)
        if cur is None:
            return  # retried when more data arrive
        if ref is None:  # reference no longer cached: start over from here
            self._float_ref, self._float_bw = key, (self.contrast.black, self.contrast.white)
            return
        ratio = cur.sd / ref.sd
        ref_black, ref_white = self._float_bw
        black = cur.mean - (ref.mean - ref_black) * ratio
        white = black + (ref_white - ref_black) * ratio
        self.contrast.set_floated(black, white)
        self._float_ref, self._float_bw = key, (black, white)  # unrounded, like 3dmod's sRefBlack/sRefWhite
        self.contrast_changed.emit()

    def set_float(self, on: bool) -> None:
        self.float_on = bool(on)
        self._reset_float()

    def set_subarea(self, on: bool) -> None:
        self.subarea = bool(on)
        self._reset_float()

    def toggle_invert(self) -> None:
        self.contrast.invert = not self.contrast.invert
        self.contrast_changed.emit()

    def toggle_false_color(self) -> None:
        self.contrast.false_color = not self.contrast.false_color
        self.message.emit(f"False color {'ON' if self.contrast.false_color else 'OFF'}")
        self.contrast_changed.emit()

    def select_ramp(self, index: int) -> None:
        """F9 / F10: switch between four independent black/white settings, like 3dmod's colour ramps."""
        index %= len(self.ramp_slots)
        self.ramp_slots[self.ramp_index] = (self.contrast.black, self.contrast.white)
        self.ramp_index = index
        if self.ramp_slots[index] is not None:
            self.contrast.black, self.contrast.white = self.ramp_slots[index]
        self._reset_float()
        self.message.emit(f"Contrast ramp {index + 1}")
        self.contrast_changed.emit()

    def toggle_model(self) -> None:
        self.show_model = not self.show_model
        self.message.emit(f"Model display {'ON' if self.show_model else 'OFF'}")
        self.overlays_changed.emit()

    def toggle_markers(self) -> None:
        self.show_markers = not self.show_markers
        self.overlays_changed.emit()

    def navigate(self, action: str) -> None:
        if self.navigator is not None:
            self.navigator(action)

    def object_sections(self, layer) -> np.ndarray:
        """Z sections (level 0) holding points of a picks or filaments layer."""
        pts = getattr(layer, "points", None)
        if pts is None or not len(pts):
            return np.zeros(0, dtype=int)
        return np.unique(np.floor(pts[:, 0]).astype(int))

    def next_section_with_points(self, layer, direction: int) -> bool:
        """Shift+PageUp/PageDown: the next section (in ``direction``) holding points of ``layer``."""
        secs = self.object_sections(layer)
        z = self.point[0]
        cand = secs[secs > z] if direction > 0 else secs[secs < z]
        if not len(cand):
            return False
        self.set_axis(0, int(cand.min() if direction > 0 else cand.max()))
        return True

    def find_max_nearby(self, radius: int = 10):
        """Shift+F: the brightest level-0 pixel within ``radius`` pixels of the current point on this section."""
        z, y, x = self.point
        lv = self.volume.levels[0]
        rows = (max(0, y - radius), min(lv.shape[1], y + radius + 1))
        cols = (max(0, x - radius), min(lv.shape[2], x + radius + 1))
        for cy in range(rows[0] // lv.chunks[1], (rows[1] - 1) // lv.chunks[1] + 1):
            for cx in range(cols[0] // lv.chunks[2], (cols[1] - 1) // lv.chunks[2] + 1):
                self.cache.fetch(self.volume, 0, (z // lv.chunks[0], cy, cx))
        data, _ = self.volume.plane(self.cache, 0, 0, z, rows, cols, request=False)
        yy, xx = np.ogrid[rows[0] : rows[1], cols[0] : cols[1]]
        values = np.where((yy - y) ** 2 + (xx - x) ** 2 <= radius**2, data.astype(np.float64), -np.inf)
        i, j = np.unravel_index(int(np.argmax(values)), values.shape)
        return int(rows[0] + i), int(cols[0] + j), float(values[i, j])

    def autocontrast(self) -> None:
        """Shift+A: 3dmod autocontrast on the current Z section (or the Subarea) at the displayed level."""
        self.auto_pending = False
        for level in range(self.display_level, self.volume.nlevels):
            z = int(self.volume.level_index(level, self.volume, self.point[0] + 0.5, 0))
            s = self.section_stats(level, z)
            if s is not None:
                from copickmod.contrast import auto_black_white

                self.contrast.set_auto(*auto_black_white(s))
                self._reset_float()
                self.contrast_changed.emit()
                self.message.emit(
                    f"Autocontrast: black {self.contrast.black_slider}, white {self.contrast.white_slider}",
                )
                return
        self.message.emit("Autocontrast: section not loaded yet")

    # ------------------------------------------------------------------ overlays
    def set_layer_visible(self, layer, visible: bool) -> None:
        layer.visible = visible
        self.overlays_changed.emit()

    def layers(self) -> list:
        """All overlays, as the Info window numbers them: segmentations, picks and filaments, meshes."""
        return [*self.seg_layers, *self.point_layers, *self.mesh_layers]

    def find_layer(self, name: str):
        return next((layer for layer in self.layers() if layer.name == name), None)

    def add_layer(self, layer) -> None:
        if hasattr(layer, "volume"):
            self.seg_layers.append(layer)
        elif getattr(layer, "kind", "") == "mesh":
            self.mesh_layers.append(layer)
        else:
            self.point_layers.append(layer)
        self.layers_changed.emit()
        self.overlays_changed.emit()

    def set_opacity(self, opacity: float) -> None:
        self.opacity = float(np.clip(opacity, 0.0, 1.0))
        self.overlays_changed.emit()

    # ------------------------------------------------------------------ movie
    def toggle_movie(self, direction: int, axis: int = 0) -> None:
        """Start a movie along ``axis``, or stop the one running there in either direction (3dmod setmovievar)."""
        if self.movie_timer.isActive() and self.movie_axis == axis:
            self.stop_movie()
            return
        self.movie_dir = direction
        self.movie_axis = axis
        self.movie_timer.start()

    def stop_movie(self) -> None:
        self.movie_timer.stop()
        self.movie_dir = 0

    def movie_speed(self, faster: bool) -> None:
        """',' / '.': one step of 3dmod's rate table (a factor 2 every three steps), within 0.2-200 frames/s."""
        rate = self.movie_rate + (-1 if faster else 1)
        if MOVIE_MIN_RATE <= 1000.0 / movie_interval(rate) <= MOVIE_MAX_RATE:
            self.movie_rate = rate
        self.movie_timer.setInterval(max(1, round(movie_interval(self.movie_rate))))
        self.message.emit(f"Movie rate: {1000.0 / movie_interval(self.movie_rate):.1f} frames/s")

    def movie_ahead(self, axis: int) -> int:
        """Direction the movie runs along ``axis`` (0 if none), so views can read the coming sections ahead."""
        return self.movie_dir if self.movie_timer.isActive() and self.movie_axis == axis else 0

    def _movie_step(self) -> None:
        n = self.volume.shape[self.movie_axis]
        nxt = self.point[self.movie_axis] + self.movie_dir
        if not 0 <= nxt < n:
            # 3dmod's default (looponeway off, workprocs.cpp movie_inc): stay on the end section for this frame
            # and come back the other way.
            self.movie_dir = -self.movie_dir
            return
        self.set_axis(self.movie_axis, nxt)

    # ------------------------------------------------------------------ images
    @property
    def image(self) -> Image:
        return self.images[self.current]

    @property
    def title(self) -> str:
        n = len(self.images)
        return self.image.title + (f"  [{self.current + 1}/{n}]" if n > 1 else "")

    def map_world(self, old: MultiscaleVolume, axis: int, value):
        """A world coordinate along ``axis`` of volume ``old`` in the shown volume's world (through angstroms)."""
        o, n = old.levels[0], self.volume.levels[0]
        phys = (np.asarray(value, dtype=float) - 0.5) * o.scale[axis] + o.translation[axis]
        return (phys - n.translation[axis]) / n.scale[axis] + 0.5

    def find_image(self, tomogram=None, path=None) -> Optional[int]:
        return next((i for i, img in enumerate(self.images) if img.same_as(tomogram, path)), None)

    def add_image(self, image: Image, show: bool = True) -> int:
        """Add a tomogram or file (shown at once unless ``show`` is False); returns its index."""
        if image.run is None and image.tomogram is None:
            image.run = self.image.run  # a file opened over a run shows that run's overlays
        self.images.append(image)
        self.images_changed.emit()
        if show:
            self.show_image(len(self.images) - 1)
        return len(self.images) - 1

    def open_files(self, paths: Sequence[str]) -> int:
        """Open MRC files into these windows, showing the last one; returns how many could be opened."""
        from copickmod.mrc import MrcVolume

        opened = 0
        for path in paths:
            index = self.find_image(path=path)
            if index is None:
                try:
                    vol = MrcVolume(path)
                except Exception as exc:  # noqa: BLE001 - report and carry on with the others
                    self.message.emit(f"Could not open {path}: {exc}")
                    continue
                index = self.add_image(
                    Image(vol, vol.name, path=vol.path, min_level=level_for_binning(vol, self.binning)),
                    show=False,
                )
            opened += 1
            last = index
        if opened:
            self.show_image(last)
        return opened

    def step_image(self, delta: int) -> bool:
        """Show the next (+1) or previous (-1) image, wrapping around; False if there is only one."""
        if len(self.images) < 2:
            return False
        self.show_image((self.current + delta) % len(self.images))
        return True

    def _contrast_state(self) -> dict:
        return {
            "contrast": self.contrast,
            "has_contrast": self.has_contrast,
            "auto_pending": self.auto_pending,
            "float_ref": self._float_ref,
            "float_bw": self._float_bw,
            "ramp_slots": self.ramp_slots,
            "ramp_index": self.ramp_index,
        }

    def show_image(self, index: int) -> None:
        """Show another of the images at the same place (through angstroms), with its own contrast.

        Overlays belong to a run: they are moved to the new image's voxels, or swapped for the other run's.
        """
        if index == self.current or not 0 <= index < len(self.images):
            return
        old, new = self.image, self.images[index]
        old_vol, new_vol = self.volume, new.volume
        old.saved = self._contrast_state()
        o, n = old_vol.levels[0], new_vol.levels[0]
        point = []
        for axis in range(3):
            phys = self.point[axis] * o.scale[axis] + o.translation[axis]
            value = int(math.floor((phys - n.translation[axis]) / n.scale[axis] + 0.5))
            point.append(min(max(value, 0), new_vol.shape[axis] - 1))

        layers = (self.point_layers, self.seg_layers, self.mesh_layers, old_vol)
        if new.run_name != old.run_name:
            self._run_layers[old.run_name] = layers
            layers = self._run_layers.pop(new.run_name, ([], [], [], new_vol))
        points, segs, meshes, frame = layers
        for layer in (*points, *meshes):
            reframe(layer, frame, new_vol)
        self.point_layers, self.seg_layers, self.mesh_layers = list(points), list(segs), list(meshes)

        invert, false_color = self.contrast.invert, self.contrast.false_color  # display modes stay (F11, F12)
        state = new.saved or {
            "contrast": Contrast(0.0, 1.0, 0.0, 1.0),
            "has_contrast": False,
            "auto_pending": True,
            "float_ref": None,
            "float_bw": None,
            "ramp_slots": [None, None, None, None],
            "ramp_index": 0,
        }
        self.contrast = state["contrast"]
        self.contrast.invert, self.contrast.false_color = invert, false_color
        self.has_contrast, self.auto_pending = state["has_contrast"], state["auto_pending"]
        self._float_ref, self._float_bw = state["float_ref"], state["float_bw"]
        self.ramp_slots, self.ramp_index = state["ramp_slots"], state["ramp_index"]

        self.current = index
        self.volume = new_vol
        self.min_level = max(0, min(new.min_level, new_vol.nlevels - 1))
        self.point = point
        self.subarea_roi = None
        self._wanted.clear()
        self.cache.cancel_except(())  # the old image's queued fetches; the windows ask for the new one's
        self.image_changed.emit(old_vol)
        self.layers_changed.emit()
        self.overlays_changed.emit()
        self.contrast_changed.emit()
        self.point_changed.emit()
        self.message.emit(f"Showing {new.title} ({index + 1} of {len(self.images)})")
        if self.auto_pending:
            QTimer.singleShot(50, self._data_arrived)  # after the windows have painted and reported their levels

    # ------------------------------------------------------------------ lifetime
    def close(self) -> None:
        """Stop timers and data fetching (the viewer showing this session is closing)."""
        for timer in (self._repaint_timer, self._float_timer, self.movie_timer):
            timer.stop()
        self.auto_pending = False
        self.float_on = False
        self.cache.shutdown()
        for img in self.images:
            if hasattr(img.volume, "close"):
                img.volume.close()

    # ------------------------------------------------------------------ windows
    def open(self, kind: str) -> None:
        if self.opener is not None:
            self.opener(kind)
