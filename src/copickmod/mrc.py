"""MRC files (``.mrc``, ``.rec``, ``.st``, ...) as volumes that the windows can show next to copick tomograms.

The file is memory-mapped, never read whole. It is cut into chunks like an OME-Zarr level, so the chunk cache, the
pyramid-level rule and the coarse fill-in work unchanged. Files have no pyramid, so coarser levels are made by
taking every 2nd, 4th, ... voxel (cheap, and only used while zoomed out until the finer level has loaded).

Coordinates follow the MRC2014 standard (as ``mrcfile`` reads them): voxel ``i`` is at ``origin + i * pixel``
angstroms. Copick tomograms start at 0 with the same rule, so a reconstruction of the same run lines up with its
picks.
"""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np

from copickmod.volume import Level, MultiscaleVolume, _uid_counter

MRC_EXTENSIONS = (".mrc", ".rec", ".st", ".ali", ".preali", ".map", ".mrcs", ".mrc.gz", ".rec.gz")
CHUNKS = (32, 256, 256)
COARSEST = 256  # stop making coarser levels once both X and Y are this small


def is_mrc_path(path: str) -> bool:
    """True for an existing file whose name looks like an MRC image."""
    return str(path).lower().endswith(MRC_EXTENSIONS) and os.path.isfile(path)


def dropped_paths(event) -> list:
    """MRC files in a drag-and-drop event."""
    mime = event.mimeData()
    if mime is None or not mime.hasUrls():
        return []
    paths = [url.toLocalFile() for url in mime.urls() if url.isLocalFile()]
    return [p for p in paths if is_mrc_path(p)]


class _Strided:
    """Every ``step``-th voxel of a 3-D array, sliced like an array (a coarse level of an MRC file)."""

    def __init__(self, data, step: int):
        self.data = data
        self.step = step
        self.shape = tuple(-(-n // step) for n in data.shape)
        self.dtype = data.dtype
        self.ndim = 3
        self.chunks = CHUNKS

    def __getitem__(self, key: Tuple[slice, ...]) -> np.ndarray:
        step = self.step
        full = tuple(slice((s.start or 0) * step, (s.stop if s.stop is not None else n) * step, step)
                     for s, n in zip(key, self.shape))  # fmt: skip
        return self.data[full]


class MrcVolume(MultiscaleVolume):
    """A memory-mapped MRC file with the :class:`MultiscaleVolume` interface."""

    def __init__(self, path: str, name: str = ""):
        import mrcfile

        self.uid = next(_uid_counter)
        self.path = os.path.abspath(path)
        self.name = name or os.path.basename(path)
        self.group = None
        self._mrc = mrcfile.mmap(path, mode="r", permissive=True)
        data = self._mrc.data
        if data is None:
            raise ValueError(f"{path}: no image data")
        if data.ndim == 2:
            data = data[np.newaxis]
        if data.ndim != 3:
            raise ValueError(f"{path}: {data.ndim}-dimensional data (a stack of volumes) cannot be shown")
        if np.iscomplexobj(data):
            raise ValueError(f"{path}: complex (FFT) data cannot be shown")
        vs = self._mrc.voxel_size
        scale = np.array([float(vs.z), float(vs.y), float(vs.x)])
        scale[~(scale > 0)] = scale[2] if scale[2] > 0 else 1.0  # unset pixel sizes: use X's, or 1
        origin = self._mrc.header.origin
        translation = np.array([float(origin.z), float(origin.y), float(origin.x)])
        translation[~np.isfinite(translation)] = 0.0
        self.levels = [Level(array=data, path="0", shape=tuple(int(n) for n in data.shape), chunks=CHUNKS,
                             scale=scale, translation=translation)]  # fmt: skip
        step = 2
        while max(data.shape[1], data.shape[2]) / step >= COARSEST and min(data.shape[1], data.shape[2]) // step >= 1:
            coarse = _Strided(data, step)
            scale_k = scale * np.array([step if data.shape[0] >= step else 1, step, step])
            if data.shape[0] < step:  # thin files (single images): keep every section
                coarse = _StridedXY(data, step)
            self.levels.append(Level(array=coarse, path=str(len(self.levels)), shape=coarse.shape, chunks=CHUNKS,
                                     scale=scale_k, translation=translation))  # fmt: skip
            step *= 2
        self._lead = ()
        self._ranged = False
        self.dtype = data.dtype.newbyteorder("=")
        # 3dmod's default slider range for files: the header minimum and maximum when they are valid
        # (imodview.cpp; the mean +/- 6 SD rule is its fallback, and what OME-Zarr data, without a header, get).
        h = self._mrc.header
        lo, hi = float(h.dmin), float(h.dmax)
        self.header_range = (lo, hi) if np.isfinite(lo) and np.isfinite(hi) and hi > lo else None

    def read_chunk(self, level: int, cidx: Tuple[int, int, int]) -> np.ndarray:
        lv = self.levels[level]
        sl = tuple(slice(c * n, min((c + 1) * n, s)) for c, n, s in zip(cidx, lv.chunks, lv.shape))
        return np.array(lv.array[sl], dtype=self.dtype)  # a copy: read the file here, in the fetch thread

    def read_level(self, level: int) -> np.ndarray:
        return np.array(self.levels[level].array[(slice(None),) * 3], dtype=self.dtype)

    def close(self) -> None:
        self._mrc.close()


class _StridedXY(_Strided):
    """Like :class:`_Strided`, but keeping every section (for files with fewer sections than the step)."""

    def __init__(self, data, step: int):
        super().__init__(data, step)
        self.shape = (data.shape[0], *self.shape[1:])

    def __getitem__(self, key: Tuple[slice, ...]) -> np.ndarray:
        step = self.step
        z, y, x = key
        return self.data[z, slice((y.start or 0) * step, (y.stop or self.shape[1]) * step, step),
                         slice((x.start or 0) * step, (x.stop or self.shape[2]) * step, step)]  # fmt: skip
