"""Geometry of 3dmod's Slicer window.

3dmod applies the rotation angles to the *volume* in the order Z, Y, X (forward matrix ``Rx @ Ry @ Rz`` with standard
right-handed rotations, see ``SlicerFuncs::setForwardMatrix`` and ``libimod/imat.c``) and shows the XY plane of the
rotated volume. A slice pixel ``(u, v)`` (u to the right, v up) therefore sits at ``centre + u * xn + v * yn`` in the
volume, where ``xn, yn, zn`` are the columns of the inverse (transpose) matrix (``SlicerFuncs::transStep``).
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


def _rx(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def forward_matrix(angles_xyz: Sequence[float]) -> np.ndarray:
    """Rotation applied to the volume (xyz coordinates) for 3dmod angles ``(x, y, z)`` in degrees."""
    ax, ay, az = np.radians(np.asarray(angles_xyz, dtype=float))
    return _rx(ax) @ _ry(ay) @ _rz(az)


def slice_axes(angles_xyz: Sequence[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Volume-space (zyx) unit vectors of the slice's u (right), v (up) and normal directions."""
    inv = forward_matrix(angles_xyz).T
    xn, yn, zn = inv[:, 0], inv[:, 1], inv[:, 2]
    return xn[::-1].copy(), yn[::-1].copy(), zn[::-1].copy()


def plane_coords(
    center_zyx: Sequence[float],
    angles_xyz: Sequence[float],
    width: int,
    height: int,
    step: float,
    offset_uv: Sequence[float] = (0.0, 0.0),
    normal_offset: float = 0.0,
) -> np.ndarray:
    """World (zyx) coordinates of a ``height x width`` slice image, shape ``(3, height, width)``.

    Row 0 is the top of the image (largest v). ``step`` is the world distance between samples, ``offset_uv`` the
    in-plane position (world units) of the image centre relative to ``center_zyx``, and ``normal_offset`` a shift
    along the slice normal (used for thick slices).
    """
    eu, ev, en = slice_axes(angles_xyz)
    u = (np.arange(width) - (width - 1) / 2.0) * step + offset_uv[0]
    v = ((height - 1) / 2.0 - np.arange(height)) * step + offset_uv[1]
    c = np.asarray(center_zyx, dtype=float) + normal_offset * en
    return c[:, None, None] + eu[:, None, None] * u[None, None, :] + ev[:, None, None] * v[None, :, None]


def project(points_zyx: np.ndarray, center_zyx: Sequence[float], angles_xyz: Sequence[float]):
    """In-plane ``(u, v)`` and signed normal distance ``d`` of world points relative to the slice."""
    eu, ev, en = slice_axes(angles_xyz)
    rel = np.asarray(points_zyx, dtype=float) - np.asarray(center_zyx, dtype=float)
    return rel @ eu, rel @ ev, rel @ en
