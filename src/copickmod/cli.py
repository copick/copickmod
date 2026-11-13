"""``cmod``: open a copick tomogram like ``imod file.mrc``."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from typing import List, Optional, Sequence

import click

from copickmod import __version__
from copickmod.project import CmodError, format_available, get_run, open_root, resolve

log = logging.getLogger("copickmod")

EPILOG = """\b
Examples:
  export CFG=/path/to/copick_config.json
  cmod TS_001 wbp@10.0                       open a tomogram
  cmod TS_001 wbp@10.0 'ribosome:*/*'        ... with all ribosome picks
  cmod TS_001 'denoised*@10' 'membrane:*/*@10.0' -xyz
  cmod TS_001 wbp@10.0 'denoised*@10.0'      two tomograms in one ZaP (Left/Right switch)
  cmod TS_001 wbp@10.0 my_rec.mrc            ... with an MRC file of the run
  cmod tomo.rec other.mrc                    MRC files only (no copick project needed)
  cmod -l TS_001                             list what the run contains
  cmod -l                                    list the runs
  cmod                                       browse the project (run browser only)

Overlay URIs need a colon: 'object:user/session' for picks and filaments,
'name:user/session@voxel_size' for segmentations (add ?multilabel=true for
multilabel ones). Globs (*, ?) work everywhere; prefix 're:' for regexes.
Like 3dmod, cmod returns to the shell immediately; use -W to wait.
"""


def _parse_triple(text: Optional[str], cast, name: str):
    if text is None:
        return None
    try:
        parts = [cast(v) for v in text.replace(" ", "").split(",")]
    except ValueError as exc:
        raise click.BadParameter(f"{name} must be three comma-separated numbers") from exc
    if len(parts) != 3:
        raise click.BadParameter(f"{name} must be three comma-separated numbers")
    return parts


def _child_args(params: dict) -> List[str]:
    """Rebuild the command line for the detached viewer process."""
    args: List[str] = []
    if params["config"]:
        args += ["-c", os.path.abspath(params["config"]) if os.path.exists(params["config"]) else params["config"]]
    for flag, key in (
        ("-b", "browser"),
        ("-xyz", "open_xyz"),
        ("-S", "open_slicer"),
        ("-V", "open_model"),
        ("-Z", "open_zap"),
        ("--debug", "debug"),
    ):
        if params[key]:
            args.append(flag)
    args += ["-C", str(params["cache_size"]), "-B", str(params["binning"])]
    for flag, key in (("-iz", "init_zoom"), ("-ic", "init_center"), ("-ia", "init_angles")):
        if params[key] is not None:
            args += [flag, str(params[key])]
    for uri in params["picks_uris"]:
        args += ["--picks", uri]
    for uri in params["seg_uris"]:
        args += ["--seg", uri]
    args += ["-W"]
    positional = [p for p in [params["run_name"], *params["uris"], *params.get("files", [])] if p]
    if positional:
        args += ["--", *positional]
    return args


def _split_files(params: dict) -> List[str]:
    """Take MRC file paths out of the positional arguments; returns them (the run name may be one)."""
    from copickmod.mrc import MRC_EXTENSIONS, is_mrc_path

    args = [a for a in [params["run_name"], *params["uris"]] if a]
    for a in args:
        if a.lower().endswith(MRC_EXTENSIONS) and not os.path.isfile(a):
            raise CmodError(f"No such file: {a}")
    files = [a for a in args if is_mrc_path(a)]
    if params["run_name"] and is_mrc_path(params["run_name"]):
        others = [a for a in params["uris"] if not is_mrc_path(a)]
        if others or params["picks_uris"] or params["seg_uris"]:
            raise CmodError(
                f"Overlay URIs need a copick run before the files, e.g. cmod RUN TOMO_URI {others[0] if others else 'URI'} file.mrc",
            )
        params["run_name"] = None
    params["uris"] = tuple(a for a in params["uris"] if not is_mrc_path(a))
    return files


def _check_config(config: Optional[str]) -> None:
    if not config or ("://" not in config and not os.path.isfile(config)):
        open_root(config)  # raises the explanatory CmodError


def _detach(params: dict) -> int:
    """Run the viewer as a detached process, like 3dmod, and return once it has validated the arguments.

    The child resolves the project and URIs itself (so this is only done once) and writes ``ok`` to a pipe when
    done; errors are printed by the child to the shared terminal and reported here through the exit code.
    """
    read_fd, write_fd = os.pipe()
    cmd = [sys.executable, "-m", "copickmod", "--ready-fd", str(write_fd), *_child_args(params)]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        pass_fds=(write_fd,),
    )
    os.close(write_fd)
    with os.fdopen(read_fd, "rb") as pipe:
        status = pipe.read()  # returns at "ok" + close, or at EOF if the child dies
    if status.strip() == b"ok":
        return 0
    try:
        return proc.wait(timeout=10) or 1
    except subprocess.TimeoutExpired:
        return 1


def _signal_ready(fd: Optional[int]) -> None:
    if fd is None:
        return
    try:
        os.write(fd, b"ok")
        os.close(fd)
    except OSError:
        pass


def _setup_logging(debug: bool) -> None:
    """Own handler for cmod's messages; copick reconfigures the root logger when it is imported."""
    logger = logging.getLogger("copickmod")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("cmod: %(relativeCreated)7.0fms %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
    logger.propagate = False
    logger.setLevel(logging.DEBUG if debug else logging.WARNING)


def _have_display() -> bool:
    if os.environ.get("QT_QPA_PLATFORM"):
        return True
    if sys.platform == "darwin" or sys.platform.startswith("win"):
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


@click.command(
    name="cmod",
    epilog=EPILOG,
    context_settings={"help_option_names": ["-h", "--help"], "max_content_width": 100},
    short_help="Open copick tomograms in a 3dmod-style viewer.",
)
@click.argument("run_name", required=False)
@click.argument("uris", nargs=-1)
@click.option(
    "-c",
    "--config",
    envvar=["CFG", "COPICK_CONFIG"],
    show_envvar=True,
    metavar="PATH",
    help="Copick config file (defaults to $CFG, then $COPICK_CONFIG).",
)
@click.option("-l", "--list", "list_only", is_flag=True, help="List the run's tomograms and overlays (or all runs).")
@click.option("-xyz", "open_xyz", is_flag=True, help="Open the XYZ window first.")
@click.option("-S", "open_slicer", is_flag=True, help="Open the Slicer window first.")
@click.option("-V", "open_model", is_flag=True, help="Open the Model View (3D) window first.")
@click.option("-Z", "open_zap", is_flag=True, help="Also open a ZaP window (with -xyz, -S or -V).")
@click.option("-C", "cache_size", default="2G", show_default=True, help="Chunk cache size (e.g. 500M, 4G).")
@click.option("-B", "binning", type=int, default=1, show_default=True, help="Never display finer than this binning.")
@click.option("-iz", "init_zoom", type=float, default=None, help="Initial zoom.")
@click.option("-ic", "init_center", default=None, metavar="X,Y,Z", help="Initial current point (1-based voxels).")
@click.option("-ia", "init_angles", default=None, metavar="X,Y,Z", help="Initial Slicer angles in degrees.")
@click.option("-W", "foreground", is_flag=True, help="Stay in the foreground instead of returning to the shell.")
@click.option("-b", "--browser", "browser", is_flag=True, help="Also open the run browser (cmod alone opens only it).")
@click.option("--picks", "picks_uris", multiple=True, metavar="URI", help="Picks/filaments URI (explicit type).")
@click.option("--seg", "seg_uris", multiple=True, metavar="URI", help="Segmentation URI (explicit type).")
@click.option(
    "--snapshot",
    type=click.Path(dir_okay=False),
    default=None,
    help="Render, save <stem>_<window>.png for every window once loaded, and exit (implies -W).",
)
@click.option("--debug", is_flag=True, help="Verbose logging.")
@click.option("--ready-fd", "ready_fd", type=int, default=None, hidden=True)
@click.version_option(__version__, "--version", prog_name="cmod")
def cmod(**params) -> None:
    """Open a copick tomogram, with optional picks and segmentations, in a 3dmod-style viewer.

    RUN_NAME is a copick run; the first URI selects the tomogram (tomo_type@voxel_size, e.g. wbp@10.0; may be
    omitted if the run has a single tomogram) and further URIs add overlays. More tomogram URIs, and MRC files,
    open in the same ZaP, which switches between them with Left/Right (or 1/2). With only MRC files, no copick
    project is needed.
    """
    _setup_logging(params["debug"])
    try:
        _main(params)
    except CmodError as exc:
        click.echo(f"cmod: {exc}", err=True)
        sys.exit(1)


def _main(params: dict) -> None:
    from copickmod.app import ViewOptions
    from copickmod.volume import parse_size

    try:
        cache_bytes = parse_size(params["cache_size"])
    except ValueError as exc:
        raise CmodError(f"Bad cache size {params['cache_size']!r}") from exc
    center = _parse_triple(params["init_center"], int, "-ic")
    angles = _parse_triple(params["init_angles"], float, "-ia")

    if params["list_only"]:
        root = open_root(params["config"])
        if not params["run_name"]:
            names = sorted(r.name for r in root.runs)
            click.echo(f"runs ({len(names)}):")
            for n in names:
                click.echo(f"    {n}")
            return
        click.echo(format_available(get_run(root, params["run_name"])))
        return

    files = _split_files(params)
    params["files"] = [os.path.abspath(f) for f in files]
    if not params["run_name"] and not files and params["snapshot"]:
        raise CmodError("--snapshot needs a run or a file. Usage: cmod RUN_NAME [TOMO_URI] [OVERLAY_URI...]")
    if not _have_display() and not params["snapshot"]:
        raise CmodError(
            "No display found ($DISPLAY is not set). Run cmod in a desktop session, or connect with ssh -X/-Y.",
        )
    files_only = bool(files) and not params["run_name"]

    if not params["foreground"] and not params["snapshot"]:
        if not files_only:
            _check_config(params["config"])  # cheap checks before spawning the viewer
        sys.exit(_detach(params))

    # Validate everything before any window opens; a detached viewer then tells the waiting shell command.
    t0 = time.monotonic()
    from copickmod.mrc import MrcVolume
    from copickmod.project import Selection

    for path in files:
        try:
            MrcVolume(path).close()
        except Exception as exc:  # noqa: BLE001 - any reason the file cannot be shown
            raise CmodError(f"Cannot open {path}: {exc}") from exc
    if files_only:  # MRC files need no project; with one, the run browser works too
        try:
            root = open_root(params["config"]) if params["config"] else None
        except CmodError as exc:
            log.warning("Not using the copick project: %s", exc)
            root = None
    else:
        root = open_root(params["config"])
    sel = None
    if params["run_name"]:  # without a run or files, only the run browser opens
        run = get_run(root, params["run_name"])
        sel = resolve(root, run, params["uris"], params["picks_uris"], params["seg_uris"])
        sel.files = list(params["files"])
    elif files:
        sel = Selection(run=None, tomogram=None, files=list(params["files"]))
    log.debug("Opened project and resolved URIs in %.2fs", time.monotonic() - t0)
    _signal_ready(params["ready_fd"])

    import signal

    signal.signal(signal.SIGINT, signal.SIG_DFL)  # let Ctrl+C stop a foreground viewer
    opts = ViewOptions(
        open_xyz=params["open_xyz"],
        open_slicer=params["open_slicer"],
        open_zap=params["open_zap"],
        open_model=params["open_model"],
        cache_bytes=cache_bytes,
        binning=max(1, params["binning"]),
        zoom=params["init_zoom"],
        center_xyz=center,
        angles_xyz=angles,
        snapshot=params["snapshot"],
        browser=params["browser"],
    )
    from copickmod.app import run_viewer

    sys.exit(run_viewer(root, sel, opts))


def main(argv: Optional[Sequence[str]] = None) -> None:
    cmod.main(args=list(argv) if argv is not None else None, prog_name="cmod")


if __name__ == "__main__":
    main()
