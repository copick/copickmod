"""Opening a copick project and turning ``cmod RUN URI...`` arguments into copick objects."""

from __future__ import annotations

import difflib
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

TOMOGRAM, PICKS, FILAMENTS, SEGMENTATION, MESH = "tomogram", "picks", "filaments", "segmentation", "mesh"


class CmodError(Exception):
    """A user-facing error: printed without a traceback."""


def open_root(config: Optional[str]):
    """Open the copick root described by ``config``."""
    if not config:
        raise CmodError(
            "No copick config given. Set one with\n"
            "    export CFG=/path/to/copick_config.json\n"
            "(a bare `CFG=...` line is not exported to programs), or pass -c/--config.",
        )
    if "://" not in config and not os.path.isfile(config):
        raise CmodError(f"Copick config not found: {config}")
    import copick

    try:
        return copick.from_file(config)
    except Exception as exc:  # noqa: BLE001 - surface any parsing/connection problem plainly
        raise CmodError(f"Could not open copick config {config}: {exc}") from exc


def _run_hint(root, name: str) -> str:
    names = [r.name for r in root.runs]
    if name in names:
        return ""
    close = difflib.get_close_matches(name, names, n=5, cutoff=0.5)
    if close:
        return " Did you mean: " + ", ".join(close) + "?"
    if names:
        return " Runs: " + ", ".join(names[:10]) + (", ..." if len(names) > 10 else "")
    return ""


def get_run(root, name: str):
    """Return the run called ``name`` or raise with suggestions.

    Data-portal projects also accept portal run IDs outside the project's datasets, as copick does.
    """
    run = root.get_run(name)
    if run is not None:
        return run
    raise CmodError(f"Run {name!r} not found in the project ({len(root.runs)} runs).{_run_hint(root, name)}")


def classify_uri(uri: str) -> str:
    """Guess the copick object type from the shape of a URI.

    ``tomo@vs`` or ``tomo`` is a tomogram; ``name:user/session@vs`` (or with ``?multilabel=...`` flags) is a
    segmentation; ``name:user/session`` or ``name`` with a colon is picks (also searched as filaments).
    """
    body = uri[3:] if uri.startswith("re:") else uri
    if ":" not in body:
        return TOMOGRAM
    head, tail = body.split(":", 1)
    if "@" in head:
        raise CmodError(f"{uri!r} looks like a feature URI (tomo@voxel:feature); features cannot be displayed.")
    if "@" in tail or "?" in tail:
        return SEGMENTATION
    return PICKS


def uri_of(obj) -> str:
    from copick.util.uri import serialize_copick_uri

    try:
        return serialize_copick_uri(obj)
    except Exception:  # noqa: BLE001 - fall back to something readable
        name = getattr(obj, "pickable_object_name", None) or getattr(obj, "name", "?")
        return f"{name}:{getattr(obj, 'user_id', '?')}/{getattr(obj, 'session_id', '?')}"


def available(run) -> Dict[str, List[str]]:
    """URIs of everything in a run that cmod can show, by type."""
    out: Dict[str, List[str]] = {TOMOGRAM: [], PICKS: [], FILAMENTS: [], SEGMENTATION: [], MESH: []}
    for vs in run.voxel_spacings:
        out[TOMOGRAM].extend(uri_of(t) for t in vs.tomograms)
    out[PICKS] = sorted(uri_of(p) for p in run.picks)
    out[FILAMENTS] = sorted(uri_of(f) for f in getattr(run, "filaments", []))
    out[SEGMENTATION] = sorted(uri_of(s) for s in run.segmentations)
    out[MESH] = sorted(uri_of(m) for m in getattr(run, "meshes", []))
    return out


def format_available(run, types: Sequence[str] = (TOMOGRAM, PICKS, FILAMENTS, SEGMENTATION, MESH)) -> str:
    avail = available(run)
    lines = []
    for t in types:
        title = {MESH: "meshes (Model View only)"}.get(t, t if t.endswith("s") else f"{t}s")
        lines.append(f"{title} ({len(avail[t])}):")
        lines.extend(f"    {u}" for u in avail[t])
    return "\n".join(lines)


def _resolve(root, run, uri: str, object_type: str) -> list:
    from copick.util.uri import resolve_copick_objects

    try:
        return resolve_copick_objects(uri, root, object_type, run.name)
    except ValueError as exc:
        raise CmodError(f"Bad {object_type} URI {uri!r}: {exc}") from exc


@dataclass
class Selection:
    """Everything one ``cmod`` invocation will show."""

    run: object  # None when only MRC files are opened
    tomogram: object  # the one shown first; None when only MRC files are opened
    picks: list = field(default_factory=list)
    filaments: list = field(default_factory=list)
    segmentations: list = field(default_factory=list)
    meshes: list = field(default_factory=list)
    more_tomograms: list = field(default_factory=list)  # further tomograms in the same ZaP (Left/Right switch)
    files: list = field(default_factory=list)  # MRC files in the same ZaP


def resolve(
    root,
    run,
    uris: Sequence[str] = (),
    picks_uris: Sequence[str] = (),
    seg_uris: Sequence[str] = (),
) -> Selection:
    """Resolve positional URIs (type auto-detected) plus explicitly typed overlay URIs."""
    tomo_uris: List[str] = []
    picks_uris = list(picks_uris)
    seg_uris = list(seg_uris)
    for uri in uris:
        kind = classify_uri(uri)
        {TOMOGRAM: tomo_uris, PICKS: picks_uris, SEGMENTATION: seg_uris}[kind].append(uri)

    more: list = []
    if tomo_uris:
        # Several tomogram URIs (or a glob matching several) open in the same ZaP, which switches between them.
        tomos: list = []
        for uri in tomo_uris:
            found = _resolve(root, run, uri, TOMOGRAM)
            if not found:
                raise CmodError(f"No tomogram matches {uri!r} in run {run.name}.\n" + format_available(run, [TOMOGRAM]))
            found = sorted(found, key=lambda t: (float(t.voxel_spacing.voxel_size), t.tomo_type))
            tomos.extend(t for t in found if all(uri_of(t) != uri_of(u) for u in tomos))
        tomogram, more = tomos[0], tomos[1:]
    else:
        tomos = [t for vs in run.voxel_spacings for t in vs.tomograms]
        if len(tomos) != 1:
            if not tomos:
                hint = _run_hint(root, run.name)
                note = f" (it is not part of this project.{hint})" if hint else ""
                raise CmodError(f"Run {run.name} has no tomograms{note}")
            raise CmodError(
                f"Run {run.name} has {len(tomos)} tomograms; choose one:\n" + format_available(run, [TOMOGRAM]),
            )
        tomogram = tomos[0]

    sel = Selection(run=run, tomogram=tomogram, more_tomograms=more)
    for uri in picks_uris:
        picks = _resolve(root, run, uri, PICKS)
        filaments = _resolve(root, run, uri, FILAMENTS)
        meshes = _resolve(root, run, uri, MESH)
        if not picks and not filaments and not meshes:
            raise CmodError(
                f"No picks, filaments or meshes match {uri!r} in run {run.name}.\n"
                + format_available(run, [PICKS, FILAMENTS, MESH]),
            )
        sel.picks.extend(p for p in picks if p not in sel.picks)
        sel.filaments.extend(f for f in filaments if f not in sel.filaments)
        sel.meshes.extend(m for m in meshes if m not in sel.meshes)
    for uri in seg_uris:
        segs = _resolve(root, run, uri, SEGMENTATION)
        if not segs:
            raise CmodError(
                f"No segmentation matches {uri!r} in run {run.name}.\n" + format_available(run, [SEGMENTATION]),
            )
        sel.segmentations.extend(s for s in segs if s not in sel.segmentations)
    return sel


def run_tomograms(run) -> list:
    """(voxel size, tomogram) pairs of a run, in copick order."""
    return [(float(vs.voxel_size), t) for vs in run.voxel_spacings for t in vs.tomograms]


def choose_tomogram(run, prefer_type: Optional[str] = None, prefer_voxel: Optional[float] = None):
    """The tomogram to open for a run (copick-web's rule): the one with the preferred type and voxel size if the
    run has it, else a denoised, then wbp, tomogram at the coarsest voxel size, else the first one there."""
    tomos = run_tomograms(run)
    if not tomos:
        return None
    if prefer_type is not None and prefer_voxel is not None:
        for vs, t in tomos:
            if t.tomo_type == prefer_type and abs(vs - prefer_voxel) < 1e-3:
                return t
    coarsest = max(vs for vs, _ in tomos)
    at = [t for vs, t in tomos if abs(vs - coarsest) < 1e-6]
    for key in ("denoised", "wbp"):
        for t in at:
            if key in t.tomo_type.lower():
                return t
    return at[0]


def object_kind(obj) -> str:
    """picks, filaments, segmentation or mesh for a copick object."""
    name = type(obj).__name__
    if "Filaments" in name:
        return FILAMENTS
    if "Segmentation" in name:
        return SEGMENTATION
    if "Mesh" in name:
        return MESH
    return PICKS
