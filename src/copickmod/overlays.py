"""Picks, filaments and segmentations drawn over the tomogram.

All geometry is in world coordinates (level-0 tomogram voxels, corner based, zyx order; see ``volume.py``).
Point and filament layers are drawn like 3dmod scattered-point objects: a sphere of the object's radius shows up as
a circle of radius ``sqrt(r^2 - d^2)`` on every plane within ``r`` of its centre.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from copickmod.volume import PLANE_AXES, ChunkCache, MultiscaleVolume

log = logging.getLogger(__name__)

DEFAULT_RADIUS_A = 50.0
SEG_ALPHA = 255


@dataclass
class PointLayer:
    """Picks or filaments of one copick object."""

    name: str
    kind: str  # "picks" or "filaments"
    rgba: Tuple[int, int, int, int]
    radius: float  # world voxels
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))  # (N, 3) world zyx
    polylines: List[np.ndarray] = field(default_factory=list)  # filaments: (M, 3) world zyx each
    visible: bool = True
    source: object = None  # the copick object it was made from

    def __len__(self) -> int:
        return len(self.points) if self.kind == "picks" else len(self.polylines)


@dataclass
class SegLayer:
    """A segmentation volume with a label palette."""

    name: str
    volume: MultiscaleVolume
    lut: Optional[np.ndarray]  # (K, 4) uint8; None means hashed colours per instance id
    rgba: Tuple[int, int, int, int]
    visible: bool = True
    source: object = None


# ---------------------------------------------------------------------------------------------- construction
def angstrom_to_world(pos_xyz: np.ndarray, ref: MultiscaleVolume) -> np.ndarray:
    """Copick positions (angstrom, xyz) to world coordinates (zyx). A position on a voxel centre lands on ``i+0.5``."""
    pos = np.asarray(pos_xyz, dtype=float).reshape(-1, 3)[:, ::-1]
    lv = ref.levels[0]
    return (pos - lv.translation) / lv.scale + 0.5


def object_style(root, name: str, voxel_size: float) -> Tuple[Tuple[int, int, int, int], float]:
    obj = root.get_object(name)
    rgba = tuple(int(c) for c in obj.color) if obj is not None and obj.color is not None else (255, 255, 0, 255)
    if len(rgba) == 3:
        rgba = (*rgba, 255)
    radius_a = obj.radius if obj is not None and obj.radius else DEFAULT_RADIUS_A
    return rgba, max(float(radius_a) / voxel_size, 1.0)


def picks_layer(root, picks, ref: MultiscaleVolume, name: str) -> PointLayer:
    rgba, radius = object_style(root, picks.pickable_object_name, ref.voxel_size)
    pos = picks.full_positions()
    pts = angstrom_to_world(pos, ref) if len(pos) else np.zeros((0, 3))
    return PointLayer(name=name, kind="picks", rgba=rgba, radius=radius, points=pts, source=picks)


def filaments_layer(root, filaments, ref: MultiscaleVolume, name: str) -> PointLayer:
    rgba, radius = object_style(root, filaments.pickable_object_name, ref.voxel_size)
    lines = [angstrom_to_world(p, ref) for p in filaments.numpy() if len(p)]
    pts = np.concatenate(lines) if lines else np.zeros((0, 3))
    return PointLayer(
        name=name, kind="filaments", rgba=rgba, radius=radius, points=pts, polylines=lines, source=filaments,
    )  # fmt: skip


def label_lut(root) -> np.ndarray:
    """Multilabel palette: label -> pickable object colour, transparent background."""
    objs = list(root.pickable_objects)
    top = max([int(o.label) for o in objs if o.label is not None] + [1])
    lut = np.zeros((top + 1, 4), dtype=np.uint8)
    for o in objs:
        if o.label is not None and o.color is not None:
            c = list(o.color) + [255] * (4 - len(o.color))
            lut[int(o.label)] = c[:4]
    lut[0] = 0
    return lut


def segmentation_layer(root, seg, name: str) -> SegLayer:
    kind = getattr(seg, "segmentation_type", "multilabel" if seg.is_multilabel else "binary")
    vol = MultiscaleVolume(seg.zarr(), name=name, fallback_voxel_size=float(seg.voxel_size), lead_index=0)
    obj = root.get_object(seg.name)
    rgba = tuple(int(c) for c in obj.color) if obj is not None and obj.color is not None else (255, 128, 0, 255)
    if len(rgba) == 3:
        rgba = (*rgba, 255)
    if kind in ("multilabel", "panoptic"):
        lut = label_lut(root)
    elif kind == "instance":
        lut = None
    else:
        lut = np.zeros((2, 4), dtype=np.uint8)
        lut[1] = rgba
    return SegLayer(name=name, volume=vol, lut=lut, rgba=rgba, source=seg)


# ---------------------------------------------------------------------------------------------- drawing helpers
def sphere_sections(
    points: np.ndarray,
    radius: float,
    axis: int,
    plane_world: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Circles where spheres around ``points`` cut the plane ``world[axis] == plane_world``.

    Returns ``(rows, cols, radii, on_plane)`` in world units along the plane's row/column axes; ``on_plane`` marks
    points whose centre lies in the displayed voxel slab.
    """
    if len(points) == 0:
        e = np.zeros(0)
        return e, e, e, e.astype(bool)
    d = points[:, axis] - plane_world
    sel = np.abs(d) < radius
    row_ax, col_ax = PLANE_AXES[axis]
    radii = np.sqrt(np.maximum(radius**2 - d[sel] ** 2, 0.0))
    return points[sel, row_ax], points[sel, col_ax], radii, np.abs(d[sel]) <= 0.5


def slab_segments(polylines: Sequence[np.ndarray], radius: float, axis: int, plane_world: float) -> List[np.ndarray]:
    """Parts of polylines within ``radius`` of an axis-aligned plane, as (K, 2) row/col arrays."""
    row_ax, col_ax = PLANE_AXES[axis]
    out = []
    for line in polylines:
        near = np.abs(line[:, axis] - plane_world) < radius
        if not near.any():
            continue
        # Split into runs of consecutive near points.
        idx = np.flatnonzero(near)
        runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
        for run in runs:
            out.append(line[run][:, (row_ax, col_ax)])
    return out


def _pick_level(vol: MultiscaleVolume, display_scale_a: float) -> int:
    """Coarsest level whose voxels are not larger than the displayed pixels (with 1.5x slack)."""
    best = 0
    for i, lv in enumerate(vol.levels):
        if lv.scale[2] <= display_scale_a * 1.5 + 1e-6:
            best = i
    return best


def colorize(layer: SegLayer, values: np.ndarray) -> np.ndarray:
    """Label values -> RGBA uint8."""
    vals = np.asarray(values)
    if not np.issubdtype(vals.dtype, np.integer):
        vals = np.nan_to_num(vals).astype(np.int64)
    if layer.lut is not None:
        lut = layer.lut
        if len(lut) == 2:  # binary: anything non-zero is the object
            return lut[(vals > 0).astype(np.intp)]
        return lut[np.clip(vals, 0, len(lut) - 1).astype(np.intp)]
    h = (vals.astype(np.uint64) * np.uint64(2654435761)) & np.uint64(0xFFFFFF)
    rgba = np.empty(vals.shape + (4,), dtype=np.uint8)
    rgba[..., 0] = (h >> np.uint64(16)) & np.uint64(255)
    rgba[..., 1] = (h >> np.uint64(8)) & np.uint64(255)
    rgba[..., 2] = h & np.uint64(255)
    rgba[..., 3] = np.where(vals > 0, SEG_ALPHA, 0)
    return rgba


def seg_plane_rgba(
    layer: SegLayer,
    cache: ChunkCache,
    ref: MultiscaleVolume,
    axis: int,
    plane_world: float,
    row_world: np.ndarray,
    col_world: np.ndarray,
    display_scale_a: float,
) -> Tuple[Optional[np.ndarray], bool]:
    """RGBA image of a segmentation on an axis-aligned plane.

    ``row_world``/``col_world`` are world coordinates of the displayed pixel centres. Returns ``(rgba or None,
    complete)``; ``None`` means nothing to draw.
    """
    vol = layer.volume
    level = _pick_level(vol, display_scale_a)
    lv = vol.levels[level]
    row_ax, col_ax = PLANE_AXES[axis]
    fixed = int(np.floor(vol.level_index(level, ref, plane_world, axis)))
    if not 0 <= fixed < lv.shape[axis]:
        return None, True
    ri = np.floor(vol.level_index(level, ref, row_world, row_ax)).astype(np.int64)
    ci = np.floor(vol.level_index(level, ref, col_world, col_ax)).astype(np.int64)
    rok = (ri >= 0) & (ri < lv.shape[row_ax])
    cok = (ci >= 0) & (ci < lv.shape[col_ax])
    if not rok.any() or not cok.any():
        return None, True
    r0, r1 = int(ri[rok].min()), int(ri[rok].max()) + 1
    c0, c1 = int(ci[cok].min()), int(ci[cok].max()) + 1
    data, missing = vol.plane(cache, level, axis, fixed, rows=(r0, r1), cols=(c0, c1))
    rr = np.clip(ri - r0, 0, r1 - r0 - 1)
    cc = np.clip(ci - c0, 0, c1 - c0 - 1)
    sub = data[np.ix_(rr, cc)]
    rgba = colorize(layer, sub)
    outside = ~(rok[:, None] & cok[None, :])
    if missing is not None:
        outside |= missing[np.ix_(rr, cc)]
    rgba[outside] = 0
    return rgba, missing is None


def seg_points_rgba(
    layer: SegLayer,
    cache: ChunkCache,
    ref: MultiscaleVolume,
    coords_world: np.ndarray,
    display_scale_a: float,
) -> np.ndarray:
    """RGBA of a segmentation sampled (nearest) at arbitrary world coordinates ``(3, H, W)`` (Slicer)."""
    vol = layer.volume
    level = _pick_level(vol, display_scale_a)
    lc = np.stack([vol.level_index(level, ref, coords_world[a], a) for a in range(3)])
    vals, ok = vol.sample(cache, level, lc, linear=False)
    rgba = colorize(layer, vals.astype(np.int64))
    rgba[~ok] = 0
    return rgba


# ---------------------------------------------------------------------------------------------- 3D geometry
@dataclass
class MeshLayer:
    """A copick mesh (triangles), shown in the Model View only."""

    name: str
    rgba: Tuple[int, int, int, int]
    vertices: np.ndarray  # (V, 3) world xyz, float32
    normals: np.ndarray  # (V, 3) xyz, float32
    faces: np.ndarray  # (F, 3) uint32, counter-clockwise seen from outside
    visible: bool = True
    kind: str = "mesh"
    source: object = None

    def __len__(self) -> int:
        return len(self.faces)


def _as_trimesh(geom):
    import trimesh

    if isinstance(geom, trimesh.Scene):
        parts = geom.dump()
        return trimesh.util.concatenate(parts) if parts else trimesh.Trimesh()
    return geom


def mesh_layer(root, mesh, ref: MultiscaleVolume, name: str) -> MeshLayer:
    rgba, _ = object_style(root, mesh.pickable_object_name, ref.voxel_size)
    tm = _as_trimesh(mesh.mesh)
    world = angstrom_to_world(np.asarray(tm.vertices, dtype=float), ref)  # zyx
    return MeshLayer(
        name=name,
        rgba=rgba,
        vertices=np.ascontiguousarray(world[:, ::-1], dtype=np.float32),
        normals=np.ascontiguousarray(np.asarray(tm.vertex_normals, dtype=np.float32)),
        faces=np.ascontiguousarray(np.asarray(tm.faces, dtype=np.uint32)),
        source=mesh,
    )


def reframe(layer, old: MultiscaleVolume, new: MultiscaleVolume) -> None:
    """Move a picks, filaments or mesh layer from ``old``'s world coordinates to ``new``'s, through angstroms.

    Used when a ZaP switches to another tomogram of the run (other voxel size) or to an MRC file.
    """
    o, n = old.levels[0], new.levels[0]
    if old is new or (np.allclose(o.scale, n.scale) and np.allclose(o.translation, n.translation)):
        return

    def convert(world_zyx: np.ndarray) -> np.ndarray:
        return ((np.asarray(world_zyx, dtype=float) - 0.5) * o.scale + o.translation - n.translation) / n.scale + 0.5

    if isinstance(layer, PointLayer):
        if len(layer.points):
            layer.points = convert(layer.points)
        layer.polylines = [convert(p) for p in layer.polylines]
        layer.radius *= float(o.scale[2] / n.scale[2])
    elif isinstance(layer, MeshLayer) and len(layer.vertices):
        layer.vertices = convert(np.asarray(layer.vertices)[:, ::-1])[:, ::-1].astype(np.float32)  # xyz


def isosurface_level(vol: MultiscaleVolume, max_voxels: int) -> int:
    """Finest pyramid level small enough for marching cubes."""
    for level in range(vol.nlevels):
        if int(np.prod(vol.levels[level].shape)) <= max_voxels:
            return level
    return vol.nlevels - 1


def seg_isosurfaces(
    layer: SegLayer,
    ref: MultiscaleVolume,
    max_voxels: int = 8_000_000,
    max_faces: int = 400_000,
) -> List[MeshLayer]:
    """Surfaces of a segmentation (one per label, in the label's colour) for the Model View.

    Uses marching cubes on the finest pyramid level with at most ``max_voxels`` voxels, with a coarser step when a
    surface would have more than ``max_faces`` triangles (software OpenGL draws about 10 million triangles a second).
    """
    from skimage.measure import marching_cubes

    vol = layer.volume
    level = isosurface_level(vol, max_voxels)
    data = vol.read_level(level)
    if layer.lut is None or len(layer.lut) == 2:  # binary or instance: one surface
        masks = [(layer.rgba, data > 0)]
    else:
        masks = []
        for value in np.unique(data):
            value = int(value)
            if 0 < value < len(layer.lut) and layer.lut[value][3] > 0:
                masks.append((tuple(int(c) for c in layer.lut[value]), data == value))
    out = []
    for rgba, mask in masks:
        if not mask.any():
            continue
        padded = np.pad(mask, 1).astype(np.float32)
        for step in (1, 2, 3, 4):
            verts, faces, normals, _ = marching_cubes(padded, 0.5, step_size=step)
            if len(faces) <= max_faces:
                break
        verts -= 1.0  # undo the padding; integer positions are voxel centres
        world = np.stack([vol.world_coord(level, ref, verts[:, a] + 0.5, a) for a in range(3)], axis=1)
        out.append(
            MeshLayer(
                name=f"{layer.name} (label {len(out) + 1})" if len(masks) > 1 else layer.name,
                rgba=rgba,
                vertices=np.ascontiguousarray(world[:, ::-1], dtype=np.float32),
                normals=np.ascontiguousarray(normals[:, ::-1], dtype=np.float32),  # outward
                faces=np.ascontiguousarray(faces, dtype=np.uint32),  # counter-clockwise seen from outside in xyz
            ),
        )
    return out


def sphere_template(quality: int) -> Tuple[np.ndarray, np.ndarray]:
    """Unit icosphere (vertices, faces) for drawing picks; quality 0-3."""
    import trimesh

    s = trimesh.creation.icosphere(subdivisions=int(np.clip(quality, 0, 3)))
    return np.asarray(s.vertices, dtype=np.float32), np.asarray(s.faces, dtype=np.uint32)


def sphere_batch(points_zyx: np.ndarray, radius: float, quality: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All spheres of a picks layer merged into one triangle mesh: (vertices xyz, normals, faces)."""
    tv, tf = sphere_template(quality)
    centers = np.asarray(points_zyx, dtype=np.float32)[:, ::-1]
    n = len(centers)
    verts = (tv[None, :, :] * np.float32(radius) + centers[:, None, :]).reshape(-1, 3)
    normals = np.broadcast_to(tv, (n,) + tv.shape).reshape(-1, 3)
    faces = (tf[None, :, :] + (np.arange(n, dtype=np.uint32) * len(tv))[:, None, None]).reshape(-1, 3)
    return np.ascontiguousarray(verts), np.ascontiguousarray(normals), np.ascontiguousarray(faces)


def make_layer(root, obj, kind: str, ref: MultiscaleVolume, name: str):
    """Overlay layer for a copick object of the given kind (picks, filaments, segmentation, mesh)."""
    if kind == "picks":
        return picks_layer(root, obj, ref, name)
    if kind == "filaments":
        return filaments_layer(root, obj, ref, name)
    if kind == "segmentation":
        return segmentation_layer(root, obj, name)
    if kind == "mesh":
        return mesh_layer(root, obj, ref, name)
    raise ValueError(kind)
