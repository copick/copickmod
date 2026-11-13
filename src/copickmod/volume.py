"""Lazy, chunk-cached access to OME-Zarr multiscale volumes.

Coordinate conventions used throughout copickmod:

* Arrays are indexed ``[z, y, x]``; axis numbers 0, 1, 2 mean z, y, x.
* *World* coordinates are continuous, corner-based level-0 voxel units of the tomogram: voxel ``i`` spans
  ``[i, i + 1)`` and its centre is ``i + 0.5``. Every volume (tomogram or segmentation) maps world coordinates to its
  own level indices through physical angstroms, so volumes with different voxel sizes line up.
* A *plane* is named by the axis held fixed: ``0`` is an XY plane (rows y, cols x), ``1`` an XZ plane (rows z,
  cols x) and ``2`` a YZ plane (rows y, cols z), matching the panels of 3dmod's XYZ window.
"""

from __future__ import annotations

import contextlib
import itertools
import logging
import threading
from collections import OrderedDict
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import zarr

log = logging.getLogger(__name__)

#: (row axis, column axis) of the plane with the given fixed axis.
PLANE_AXES: Dict[int, Tuple[int, int]] = {0: (1, 2), 1: (0, 2), 2: (1, 0)}

ChunkKey = Tuple[int, int, int, int, int]
_uid_counter = itertools.count()


@dataclass
class Level:
    """One pyramid level."""

    array: zarr.Array
    path: str
    shape: Tuple[int, int, int]
    chunks: Tuple[int, int, int]
    scale: np.ndarray  # zyx, angstrom per voxel
    translation: np.ndarray  # zyx, angstrom


def _level_transforms(dataset: dict, ndim: int) -> Tuple[np.ndarray, np.ndarray]:
    scale = np.ones(3)
    translation = np.zeros(3)
    for transform in dataset.get("coordinateTransformations", []):
        if transform.get("type") == "scale":
            scale = np.asarray(transform["scale"][-3:], dtype=float)
        elif transform.get("type") == "translation":
            translation = np.asarray(transform["translation"][-3:], dtype=float)
    return scale, translation


class MultiscaleVolume:
    """A read-only OME-Zarr multiscale image opened lazily.

    Args:
        store: A zarr store/mapping (e.g. ``CopickTomogram.zarr()``), a path, or an opened zarr group.
        name: Display name.
        fallback_voxel_size: Angstrom per voxel at level 0 if the metadata has no scale.
        lead_index: Index used for leading non-spatial dimensions (e.g. 0 picks the label channel of a panoptic
            segmentation).
    """

    def __init__(self, store, name: str = "", fallback_voxel_size: float = 1.0, lead_index: int = 0):
        self.uid = next(_uid_counter)
        self.name = name
        group = store if isinstance(store, zarr.hierarchy.Group) else zarr.open_group(store, mode="r")
        self.group = group
        multiscales = group.attrs.get("multiscales")
        if multiscales:
            datasets = multiscales[0]["datasets"]
        else:
            keys = sorted((k for k in group.array_keys() if k.isdigit()), key=int)
            if not keys:
                raise ValueError(f"{name or store}: no multiscales metadata and no numbered arrays")
            datasets = [
                {"path": k, "coordinateTransformations": [{"type": "scale", "scale": [fallback_voxel_size * 2**i] * 3}]}
                for i, k in enumerate(keys)
            ]
        self.levels: list[Level] = []
        lead: Optional[Tuple[int, ...]] = None
        for dataset in datasets:
            array = group[dataset["path"]]
            if array.ndim < 3:
                raise ValueError(f"{name}: level {dataset['path']} has {array.ndim} dimensions, need 3")
            lead = (lead_index,) * (array.ndim - 3)
            scale, translation = _level_transforms(dataset, array.ndim)
            self.levels.append(
                Level(
                    array=array,
                    path=dataset["path"],
                    shape=tuple(int(s) for s in array.shape[-3:]),
                    chunks=tuple(int(c) for c in array.chunks[-3:]),
                    scale=scale,
                    translation=translation,
                ),
            )
        self._lead = lead or ()
        self.dtype = np.dtype(self.levels[0].array.dtype)
        self._ranged: Optional[bool] = None  # None: untried; parallel ranged reads for remote stores

    # ------------------------------------------------------------------ geometry
    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.levels[0].shape

    @property
    def voxel_size(self) -> float:
        """Level-0 voxel size in angstrom (x axis)."""
        return float(self.levels[0].scale[2])

    @property
    def nlevels(self) -> int:
        return len(self.levels)

    def factor(self, level: int) -> np.ndarray:
        """Size of one voxel of ``level`` in level-0 voxels, zyx."""
        return self.levels[level].scale / self.levels[0].scale

    def level_index(self, level: int, world_ref: "MultiscaleVolume", world: np.ndarray, axis: int) -> np.ndarray:
        """Map world coordinates (of ``world_ref``) along ``axis`` to continuous corner coordinates of ``level``."""
        ref0 = world_ref.levels[0]
        lv = self.levels[level]
        phys = np.asarray(world, dtype=float) * ref0.scale[axis] + ref0.translation[axis]
        return (phys - lv.translation[axis]) / lv.scale[axis]

    def world_coord(self, level: int, world_ref: "MultiscaleVolume", index: np.ndarray, axis: int) -> np.ndarray:
        """Inverse of :meth:`level_index`: corner coordinates of ``level`` to world coordinates of ``world_ref``."""
        ref0 = world_ref.levels[0]
        lv = self.levels[level]
        phys = np.asarray(index, dtype=float) * lv.scale[axis] + lv.translation[axis]
        return (phys - ref0.translation[axis]) / ref0.scale[axis]

    def chunk_bytes(self, level: int) -> int:
        return int(np.prod(self.levels[level].chunks)) * max(self.dtype.itemsize, 4)

    # ------------------------------------------------------------------ reading
    def read_chunk(self, level: int, cidx: Tuple[int, int, int]) -> np.ndarray:
        """Read and decode one chunk (clipped to the array bounds)."""
        lv = self.levels[level]
        sl = tuple(slice(c * n, min((c + 1) * n, s)) for c, n, s in zip(cidx, lv.chunks, lv.shape))
        if not self._lead and self._ranged is not False:
            try:
                block = _read_chunk_ranged(lv.array, cidx)
            except Exception as exc:  # noqa: BLE001 - fall back to plain zarr reads for good
                log.debug("Parallel ranged reads unavailable for %s (%s); using zarr", self.name, exc)
                self._ranged = False
            else:
                if block is not None:
                    self._ranged = True
                    return np.ascontiguousarray(block[tuple(slice(0, s.stop - s.start) for s in sl)])
                self._ranged = False
        return np.ascontiguousarray(lv.array[self._lead + sl])

    def read_plane(self, level: int, z: int) -> np.ndarray:
        """Read one whole XY plane of a level synchronously (no cache), chunk by chunk."""
        lv = self.levels[level]
        out = np.zeros(lv.shape[1:], dtype=self.dtype)
        cz = z // lv.chunks[0]
        for cy in range(-(-lv.shape[1] // lv.chunks[1])):
            for cx in range(-(-lv.shape[2] // lv.chunks[2])):
                block = self.read_chunk(level, (cz, cy, cx))
                y0, x0 = cy * lv.chunks[1], cx * lv.chunks[2]
                out[y0 : y0 + block.shape[1], x0 : x0 + block.shape[2]] = block[z - cz * lv.chunks[0]]
        return out

    def read_level(self, level: int) -> np.ndarray:
        """Read a whole level eagerly (used for small coarse levels)."""
        return np.asarray(self.levels[level].array[self._lead + (slice(None),) * 3])

    def plane(
        self,
        cache: "ChunkCache",
        level: int,
        axis: int,
        index: int,
        rows: Optional[Tuple[int, int]] = None,
        cols: Optional[Tuple[int, int]] = None,
        request: bool = True,
        wanted: Optional[set] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Assemble a plane (or a rectangular part of it) from cached chunks.

        Args:
            cache: Chunk cache to read from.
            level: Pyramid level.
            axis: Fixed axis (0 = XY, 1 = XZ, 2 = YZ plane).
            index: Index along the fixed axis at this level.
            rows, cols: Half-open index ranges along the plane's row/column axes (default: all).
            request: Queue background fetches for chunks that are not cached.
            wanted: If given, keys of the requested (missing) chunks are added to this set.

        Returns:
            ``(data, missing)``: data has the volume dtype; ``missing`` is a boolean mask of pixels whose chunk is
            not cached yet, or ``None`` when the plane is complete.
        """
        lv = self.levels[level]
        row_ax, col_ax = PLANE_AXES[axis]
        r0, r1 = _clip_range(rows, lv.shape[row_ax])
        c0, c1 = _clip_range(cols, lv.shape[col_ax])
        out = np.zeros((r1 - r0, c1 - c0), dtype=self.dtype)
        if not (0 <= index < lv.shape[axis]) or r1 <= r0 or c1 <= c0:
            return out, np.ones(out.shape, dtype=bool)
        missing = None
        fixed_chunk = index // lv.chunks[axis]
        local_fixed = index - fixed_chunk * lv.chunks[axis]
        rn, cn = lv.chunks[row_ax], lv.chunks[col_ax]
        for rc in range(r0 // rn, (r1 - 1) // rn + 1):
            for cc in range(c0 // cn, (c1 - 1) // cn + 1):
                cidx = [0, 0, 0]
                cidx[axis], cidx[row_ax], cidx[col_ax] = fixed_chunk, rc, cc
                cidx = tuple(cidx)
                # Intersection of this chunk with the requested window, in level indices.
                ra, rb = max(r0, rc * rn), min(r1, (rc + 1) * rn)
                ca, cb = max(c0, cc * cn), min(c1, (cc + 1) * cn)
                block = cache.get(self, level, cidx)
                if block is None:
                    if missing is None:
                        missing = np.zeros(out.shape, dtype=bool)
                    missing[ra - r0 : rb - r0, ca - c0 : cb - c0] = True
                    if request:
                        cache.request(self, level, cidx)
                        if wanted is not None:
                            wanted.add(cache.key(self, level, cidx))
                    continue
                sheet = np.take(block, local_fixed, axis=axis)  # remaining axes in increasing order
                if row_ax > col_ax:
                    sheet = sheet.T
                out[ra - r0 : rb - r0, ca - c0 : cb - c0] = sheet[
                    ra - rc * rn : rb - rc * rn,
                    ca - cc * cn : cb - cc * cn,
                ]
        return out, missing

    def request_plane(
        self,
        cache: "ChunkCache",
        level: int,
        axis: int,
        index: int,
        rows: Optional[Tuple[int, int]] = None,
        cols: Optional[Tuple[int, int]] = None,
        wanted: Optional[set] = None,
    ) -> None:
        """Queue fetches for the chunks of a plane without assembling it (read-ahead during movies).

        Chunks that are already cached are marked as recently used, so they are not evicted before they are shown.
        """
        lv = self.levels[level]
        row_ax, col_ax = PLANE_AXES[axis]
        r0, r1 = _clip_range(rows, lv.shape[row_ax])
        c0, c1 = _clip_range(cols, lv.shape[col_ax])
        if not (0 <= index < lv.shape[axis]) or r1 <= r0 or c1 <= c0:
            return
        rn, cn = lv.chunks[row_ax], lv.chunks[col_ax]
        for rc in range(r0 // rn, (r1 - 1) // rn + 1):
            for cc in range(c0 // cn, (c1 - 1) // cn + 1):
                cidx = [0, 0, 0]
                cidx[axis], cidx[row_ax], cidx[col_ax] = index // lv.chunks[axis], rc, cc
                if cache.get(self, level, cidx) is None:
                    cache.request(self, level, cidx)
                    if wanted is not None:
                        wanted.add(cache.key(self, level, cidx))

    def gather(
        self,
        cache: "ChunkCache",
        level: int,
        zi: np.ndarray,
        yi: np.ndarray,
        xi: np.ndarray,
        request: bool = True,
        wanted: Optional[set] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Look up voxel values at integer indices of ``level``.

        Returns ``(values float32, ok bool)``; ``ok`` is False outside the volume or where the chunk is not cached.
        """
        lv = self.levels[level]
        zi, yi, xi = (np.asarray(a, dtype=np.int64).ravel() for a in (zi, yi, xi))
        values = np.zeros(zi.shape, dtype=np.float32)
        ok = (zi >= 0) & (zi < lv.shape[0]) & (yi >= 0) & (yi < lv.shape[1]) & (xi >= 0) & (xi < lv.shape[2])
        if not ok.any():
            return values, ok
        idx = np.nonzero(ok)[0]
        cz, cy, cx = zi[idx] // lv.chunks[0], yi[idx] // lv.chunks[1], xi[idx] // lv.chunks[2]
        ncy = -(-lv.shape[1] // lv.chunks[1])
        ncx = -(-lv.shape[2] // lv.chunks[2])
        cid = (cz * ncy + cy) * ncx + cx
        for chunk_id, group in _group_by(cid):
            sel = idx[group]
            cidx = (int(chunk_id // (ncy * ncx)), int(chunk_id // ncx % ncy), int(chunk_id % ncx))
            block = cache.get(self, level, cidx)
            if block is None:
                ok[sel] = False
                if request:
                    cache.request(self, level, cidx)
                    if wanted is not None:
                        wanted.add(cache.key(self, level, cidx))
                continue
            values[sel] = block[
                zi[sel] - cidx[0] * lv.chunks[0],
                yi[sel] - cidx[1] * lv.chunks[1],
                xi[sel] - cidx[2] * lv.chunks[2],
            ]
        return values, ok

    def sample(
        self,
        cache: "ChunkCache",
        level: int,
        coords: np.ndarray,
        linear: bool = True,
        request: bool = True,
        wanted: Optional[set] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample at continuous corner coordinates of ``level``; ``coords`` has shape ``(3, ...)`` in zyx order.

        Uses trilinear interpolation between voxel centres when ``linear`` and nearest neighbour otherwise.
        Returns ``(values, ok)`` with the shape of ``coords[0]``.
        """
        shape = coords.shape[1:]
        c = coords.reshape(3, -1)
        if not linear:
            vals, ok = self.gather(cache, level, *np.floor(c).astype(np.int64), request=request, wanted=wanted)
            return vals.reshape(shape), ok.reshape(shape)
        c = c - 0.5  # voxel centres -> integer positions
        base = np.floor(c).astype(np.int64)
        frac = (c - base).astype(np.float32)
        lv = self.levels[level]
        hi = np.asarray(lv.shape)[:, None] - 1
        acc = np.zeros(c.shape[1], dtype=np.float32)
        ok_all = np.ones(c.shape[1], dtype=bool)
        for dz, dy, dx in itertools.product((0, 1), repeat=3):
            idx = np.clip(base + np.array([[dz], [dy], [dx]]), 0, hi)  # clamp at the edges
            w = (frac[0] if dz else 1 - frac[0]) * (frac[1] if dy else 1 - frac[1]) * (frac[2] if dx else 1 - frac[2])
            vals, ok = self.gather(cache, level, idx[0], idx[1], idx[2], request=request, wanted=wanted)
            acc += w * vals
            ok_all &= ok
        inside = np.all((c > -1) & (c < np.asarray(lv.shape)[:, None]), axis=0)
        return acc.reshape(shape), (ok_all & inside).reshape(shape)


# ---------------------------------------------------------------------------------------------- remote reads
#: Remote chunks are fetched as parallel byte ranges of this size. Single S3 streams can be slow (a few MB/s), and
#: float tomogram chunks hardly compress, so a 256^3 chunk is ~60 MB: one GET can take 10+ s, ranges well under 1 s.
RANGE_PART = 2 * 1024**2
S3_POOL = 64
_fs_lock = threading.Lock()
_wide_fs: Dict[int, object] = {}


def _remote_fs(store):
    """The fsspec filesystem behind a remote zarr v2 FSStore (with a wider S3 connection pool), else None."""
    fs = getattr(store, "fs", None)
    if fs is None or not hasattr(store, "map") or not getattr(fs, "async_impl", False):
        return None
    protocol = fs.protocol if isinstance(fs.protocol, str) else fs.protocol[0]
    if protocol in ("file", "local", "memory"):
        return None
    if protocol not in ("s3", "s3a"):
        return fs
    with _fs_lock:
        wide = _wide_fs.get(id(fs))
        if wide is None:
            opts = dict(fs.storage_options)
            config = dict(opts.get("config_kwargs") or {})
            config["max_pool_connections"] = max(S3_POOL, int(config.get("max_pool_connections", 0)))
            opts["config_kwargs"] = config
            wide = type(fs)(**opts)
            _wide_fs[id(fs)] = wide
        return wide


def _read_chunk_ranged(array: zarr.Array, cidx: Sequence[int]) -> Optional[np.ndarray]:
    """Fetch one full chunk of a zarr v2 array with parallel range requests and decode it.

    Returns None when the store is not a remote fsspec store (the caller then uses plain zarr reads).
    """
    store = array.store
    fs = _remote_fs(store)
    if fs is None:
        return None
    from numcodecs.compat import ensure_ndarray

    key = store._normalize_key(array._chunk_key(tuple(int(c) for c in cidx)))
    path = store.map._key_to_str(key)
    try:
        size = fs.size(path)
    except FileNotFoundError:
        size = None
    if not size:
        return np.full(array.chunks, array.fill_value if array.fill_value is not None else 0, dtype=array.dtype)
    if size <= RANGE_PART * 3 // 2:
        raw = fs.cat_file(path)
    else:
        starts = list(range(0, size, RANGE_PART))
        ends = [min(start + RANGE_PART, size) for start in starts]
        raw = b"".join(fs.cat_ranges([path] * len(starts), starts, ends))
    buf = array._compressor.decode(raw) if array._compressor is not None else raw
    for f in reversed(array._filters or []):
        buf = f.decode(buf)
    return ensure_ndarray(buf).view(array.dtype).reshape(array.chunks, order=array.order)


def _group_by(ids: np.ndarray):
    """Yield ``(id, positions)`` for each distinct value of a non-negative int array.

    Uses one comparison pass per id when there are few distinct ids (the common case: a slice touches a handful of
    chunks), which is much cheaper than sorting a million samples.
    """
    if ids.size == 0:
        return
    counts = np.bincount(ids)
    present = np.flatnonzero(counts)
    if len(present) <= 32:
        if len(present) == 1:
            yield int(present[0]), np.arange(ids.size)
            return
        for k in present:
            yield int(k), np.flatnonzero(ids == k)
        return
    order = np.argsort(ids, kind="stable")
    bounds = np.flatnonzero(np.diff(ids[order])) + 1
    for group in np.split(order, bounds):
        yield int(ids[group[0]]), group


def _clip_range(rng: Optional[Tuple[int, int]], n: int) -> Tuple[int, int]:
    if rng is None:
        return 0, n
    a, b = rng
    return max(0, int(a)), min(n, int(b))


class ChunkCache:
    """LRU cache of decoded chunks with a byte budget and a background fetch pool.

    ``on_chunk(key)`` is called from a worker thread whenever a requested chunk has been stored.
    """

    def __init__(self, budget_bytes: int = 2 * 1024**3, workers: int = 8):
        self.budget = int(budget_bytes)
        self._lru: "OrderedDict[ChunkKey, np.ndarray]" = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()
        self._pending: Dict[ChunkKey, Future] = {}
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="cmod-fetch")
        self.on_chunk: Optional[Callable[[ChunkKey], None]] = None
        self.on_error: Optional[Callable[[ChunkKey, BaseException], None]] = None
        self._failed: set = set()
        self._closed = False

    @staticmethod
    def key(vol: MultiscaleVolume, level: int, cidx: Sequence[int]) -> ChunkKey:
        return (vol.uid, level, int(cidx[0]), int(cidx[1]), int(cidx[2]))

    @property
    def nbytes(self) -> int:
        return self._bytes

    def __len__(self) -> int:
        return len(self._lru)

    def get(self, vol: MultiscaleVolume, level: int, cidx: Sequence[int]) -> Optional[np.ndarray]:
        key = self.key(vol, level, cidx)
        with self._lock:
            block = self._lru.get(key)
            if block is not None:
                self._lru.move_to_end(key)
            return block

    def put(self, key: ChunkKey, block: np.ndarray) -> None:
        with self._lock:
            if key in self._lru:
                return
            self._lru[key] = block
            self._bytes += block.nbytes
            while self._bytes > self.budget and len(self._lru) > 1:
                _, old = self._lru.popitem(last=False)
                self._bytes -= old.nbytes

    def request(self, vol: MultiscaleVolume, level: int, cidx: Sequence[int]) -> Optional[Future]:
        """Queue a background fetch unless the chunk is cached, pending or previously failed."""
        key = self.key(vol, level, cidx)
        with self._lock:
            if self._closed or key in self._lru or key in self._failed:
                return None
            fut = self._pending.get(key)
            if fut is not None:
                return fut
            fut = self._pool.submit(self._fetch, vol, level, tuple(cidx), key)
            self._pending[key] = fut
            return fut

    def _fetch(self, vol: MultiscaleVolume, level: int, cidx: Tuple[int, int, int], key: ChunkKey) -> None:
        try:
            block = vol.read_chunk(level, cidx)
        except BaseException as exc:  # noqa: BLE001 - report every failure to the UI
            log.warning("Failed to read %s level %d chunk %s: %s", vol.name, level, cidx, exc)
            with self._lock:
                self._pending.pop(key, None)
                self._failed.add(key)
            if self.on_error is not None:
                self.on_error(key, exc)
            return
        self.put(key, block)
        with self._lock:
            self._pending.pop(key, None)
        if self.on_chunk is not None:
            self.on_chunk(key)

    def fetch(self, vol: MultiscaleVolume, level: int, cidx: Sequence[int]) -> np.ndarray:
        """Return a chunk, reading it synchronously if needed."""
        block = self.get(vol, level, cidx)
        if block is not None:
            return block
        fut = self.request(vol, level, cidx)
        if fut is not None:
            with contextlib.suppress(CancelledError):
                fut.result()
        block = self.get(vol, level, cidx)
        if block is None:  # failed or evicted immediately (tiny budget)
            block = vol.read_chunk(level, cidx)
        return block

    def pending(self) -> int:
        with self._lock:
            return len(self._pending)

    def cancel_except(self, keep: Iterable[ChunkKey]) -> int:
        """Cancel queued (not yet running) fetches whose keys are not in ``keep``."""
        keep = set(keep)
        cancelled = 0
        with self._lock:
            for key, fut in list(self._pending.items()):
                if key not in keep and fut.cancel():
                    del self._pending[key]
                    cancelled += 1
        return cancelled

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Block until no fetches are pending (for tests and snapshots)."""
        import time

        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            with self._lock:
                futs = list(self._pending.values())
            if not futs:
                return True
            for fut in futs:
                remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
                with contextlib.suppress(Exception):  # failures are reported via on_error
                    fut.result(timeout=remaining)
                if deadline is not None and time.monotonic() >= deadline:
                    return False

    def shutdown(self) -> None:
        """Stop fetching; later requests are ignored (a closed viewer's timers may still fire)."""
        with self._lock:
            self._closed = True
        self.on_chunk = self.on_error = None
        self._pool.shutdown(wait=False, cancel_futures=True)


def parse_size(text: str) -> int:
    """Parse a size such as ``2G``, ``512M`` or ``1500000000`` into bytes."""
    t = str(text).strip().upper().rstrip("B")
    mult = 1
    if t and t[-1] in "KMGT":
        mult = 1024 ** ("KMGT".index(t[-1]) + 1)
        t = t[:-1]
    return int(float(t) * mult)
