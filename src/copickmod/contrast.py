"""3dmod-style contrast handling, done in float space.

3dmod converts images to bytes once, at load time, using a "load range"; the Black/White sliders (0-255) then act on
those bytes. Getting that range wrong made the IMOD OME-Zarr plugin unusable (contrast collapsed to 0/2). Here the
data stay float: the load range only defines what the 0-255 slider positions mean, and black/white are float values
applied when a plane is rendered.

Autocontrast reproduces 3dmod (``info_cb.cpp``): the mean maps to grey level 150 and one standard deviation spans 40
grey levels, i.e. ``black = mean - 3.75 sd`` and ``white = mean + 2.625 sd``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

TARGET_MEAN = 150.0
TARGET_SD = 40.0
LOAD_RANGE_SDS = 6.0  # 3dmod numSDsForScaling
BORDER = 0.05  # 3dmod sFloatMatt: ignore a 5% border on each side
MAX_SAMPLES = 1_000_000


@dataclass
class Stats:
    mean: float
    sd: float
    vmin: float
    vmax: float


def _central(data: np.ndarray, border: float) -> np.ndarray:
    sl = []
    for n in data.shape:
        b = int(n * border)
        sl.append(slice(b, n - b) if n - 2 * b > 0 else slice(None))
    return data[tuple(sl)]


def stats(data: np.ndarray, border: float = BORDER) -> Stats:
    """Mean, SD, min and max of the finite values in the central part of ``data``."""
    sub = _central(np.asarray(data), border).ravel()
    if sub.size > MAX_SAMPLES:
        sub = sub[:: sub.size // MAX_SAMPLES + 1]
    sub = sub.astype(np.float64, copy=False)
    sub = sub[np.isfinite(sub)]
    if sub.size == 0:
        return Stats(0.0, 1.0, 0.0, 1.0)
    sd = float(sub.std())
    return Stats(float(sub.mean()), sd if sd > 0 else 1.0, float(sub.min()), float(sub.max()))


def load_range(s: Stats) -> Tuple[float, float]:
    """Data values that slider positions 0 and 255 stand for: mean +/- 6 SD clipped to the data range."""
    lo = max(s.vmin, s.mean - LOAD_RANGE_SDS * s.sd)
    hi = min(s.vmax, s.mean + LOAD_RANGE_SDS * s.sd)
    if not hi > lo:
        lo, hi = s.mean - 0.5, s.mean + 0.5
    return lo, hi


def auto_black_white(s: Stats) -> Tuple[float, float]:
    """3dmod autocontrast: black and white data values for the given statistics."""
    black = s.mean - s.sd * TARGET_MEAN / TARGET_SD
    white = s.mean + s.sd * (255.0 - TARGET_MEAN) / TARGET_SD
    return black, white


def to_uint8(data: np.ndarray, black: float, white: float, invert: bool = False) -> np.ndarray:
    """Map float data to 0-255 bytes using a black/white window."""
    span = white - black
    if span == 0:
        span = 1e-6
    scaled = (np.asarray(data, dtype=np.float32) - np.float32(black)) * np.float32(255.0 / span)
    np.nan_to_num(scaled, copy=False, nan=127.0)
    np.clip(scaled, 0, 255, out=scaled)
    out = scaled.astype(np.uint8)
    if invert:
        np.subtract(255, out, out=out)
    return out


def _nint(value: float) -> int:
    """3dmod's B3DNINT: round half up."""
    return int(math.floor(value + 0.5))


class Contrast:
    """Black/white state with the 3dmod slider semantics (0-255 positions over a load range)."""

    def __init__(self, lo: float, hi: float, black: float, white: float):
        self.lo, self.hi = lo, hi
        self.black, self.white = black, white
        self.invert = False
        self.false_color = False  # F12 / *

    @classmethod
    def from_stats(cls, s: Stats, load: Optional[Tuple[float, float]] = None) -> "Contrast":
        """Autocontrast for these statistics; ``load`` is the slider range (default: mean +/- 6 SD)."""
        lo, hi = load if load is not None and load[1] > load[0] else load_range(s)
        c = cls(lo, hi, 0.0, 1.0)
        c.set_auto(*auto_black_white(s))
        return c

    def set_auto(self, black: float, white: float) -> None:
        """Set autocontrast values the way 3dmod does: whole slider steps within 0-255, at least 4 apart."""
        b = min(255, max(0, _nint(self.to_slider_f(black))))
        w = min(255, max(0, _nint(self.to_slider_f(white))))
        if w - b < 4:
            mid = (w + b) // 2
            b, w = max(0, mid - 2), min(255, mid + 2)
        self.black, self.white = self.from_slider(b), self.from_slider(w)

    def set_floated(self, black: float, white: float) -> None:
        """Set Float's values like imod_info_bwfloat: whole slider steps within 0-255, backed off to 2 apart."""
        b = max(0, min(_nint(self.to_slider_f(black)), 255))
        w = max(b, min(_nint(self.to_slider_f(white)), 255))
        if w < b + 2:
            b, w = max(0, b - 1), min(w + 1, 255)
        self.black, self.white = self.from_slider(b), self.from_slider(w)

    def to_slider(self, value: float) -> int:
        return int(round((value - self.lo) / (self.hi - self.lo) * 255.0))

    def from_slider(self, pos: float) -> float:
        return self.lo + float(pos) / 255.0 * (self.hi - self.lo)

    @property
    def black_slider(self) -> int:
        return self.to_slider(self.black)

    @property
    def white_slider(self) -> int:
        return self.to_slider(self.white)

    def set_sliders(self, black: float, white: float) -> None:
        """Set both values from slider positions, keeping black < white like 3dmod."""
        black = min(max(black, 0.0), 254.0)
        white = min(max(white, 1.0), 255.0)
        if white <= black:
            white = black + 1
        self.black, self.white = self.from_slider(black), self.from_slider(white)

    def step(self, dblack: float, dwhite: float) -> None:
        """Move black and white by slider units (F1-F8). Only clamps when a value would leave 0-255."""
        b = self.to_slider_f(self.black) + dblack
        w = self.to_slider_f(self.white) + dwhite
        if w <= b:  # contrast keys must not cross the sliders
            return
        self.black, self.white = self.from_slider(b), self.from_slider(w)

    def to_slider_f(self, value: float) -> float:
        return (value - self.lo) / (self.hi - self.lo) * 255.0

    def auto(self, data: np.ndarray) -> None:
        """Autocontrast from a displayed plane (Shift+A)."""
        self.set_auto(*auto_black_white(stats(data)))

    def map(self, data: np.ndarray) -> np.ndarray:
        return to_uint8(data, self.black, self.white, self.invert)


# 3dmod's standard false-colour ramp (libcfshr/colormap.c): (red, green, blue, position) control points.
_FALSE_RAMP = [
    (100, 75, 200, -530), (60, 96, 255, -469), (0, 175, 177, -400), (0, 191, 143, -383), (0, 207, 78, -361),
    (90, 255, 60, -305), (191, 255, 0, -259), (239, 255, 0, -240), (255, 255, 0, -229), (255, 175, 0, -172),
    (255, 105, 0, -122), (255, 45, 55, -60), (255, 0, 90, -40), (255, 0, 192, 10),
]  # fmt: skip


def false_color_table() -> np.ndarray:
    """256 x 3 uint8 table, converted from the ramp as cmapConvertRamp does."""
    ramp = np.asarray(_FALSE_RAMP, dtype=float)
    pos = ramp[0, 3] + np.arange(256) * (ramp[-1, 3] - ramp[0, 3]) / 255.0
    return np.stack([np.interp(pos, ramp[:, 3], ramp[:, c]) for c in range(3)], axis=1).astype(np.uint8)


FALSE_COLOR = false_color_table()


def gray_to_rgb(gray: np.ndarray, false_color: bool) -> np.ndarray:
    """uint8 grey levels to (..., 3) RGB, through the false-colour table if requested."""
    if false_color:
        return FALSE_COLOR[gray]
    return np.repeat(gray[..., None], 3, axis=-1)
