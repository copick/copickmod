# copickmod: `cmod`

[![PyPI - Version](https://img.shields.io/pypi/v/copickmod.svg)](https://pypi.org/project/copickmod)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/copickmod.svg)](https://pypi.org/project/copickmod)
[![tests](https://github.com/copick/copickmod/actions/workflows/test.yml/badge.svg)](https://github.com/copick/copickmod/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

<img width="2044" height="1406" alt="cmod-windows" src="https://github.com/user-attachments/assets/1af81874-2fbf-4e77-bb41-ca942a2f773c" />

`cmod` opens a tomogram from a [copick](https://github.com/copick/copick) project in a window that works like
IMOD's 3dmod. It replaces patching OME-Zarr support into IMOD:

```bash
export CFG=/path/to/copick_config.json
cmod TS_001 wbp@10.0                              # like: imod TS_001.rec
cmod TS_001 wbp@10.0 'ribosome:*/*' 'membrane:*/*@10.0'
cmod TS_001 wbp@10.0 'denoised*@10.0' my_rec.mrc  # three images in one ZaP: Left/Right switch
cmod tomo.rec                                     # MRC files alone need no copick project
```

It is pure Python/Qt (PyQt5), reads data lazily from local disk or the CryoET Data Portal (S3), and is view-only.

## Install

```bash
pip install copickmod                   # once released on PyPI
uv tool install copickmod               # or as a tool: puts `cmod` on your PATH (~/.local/bin)
uv tool install -e /path/to/copickmod   # from a checkout
```

## Usage

```
cmod RUN_NAME [TOMOGRAM_URI ...] [FILE.mrc ...] [OVERLAY_URI ...]
cmod FILE.mrc [FILE.mrc ...]
```

| Argument / option | Meaning |
|---|---|
| `RUN_NAME` | A copick run. |
| `TOMOGRAM_URI` | `tomo_type@voxel_size`, e.g. `wbp@10.0` or `'denoised*@10'`. You can leave it out if the run has only one tomogram. Several URIs, or a glob matching several tomograms, open together in one ZaP, the first one shown. |
| `FILE.mrc` | MRC files (`.mrc`, `.rec`, `.st`, `.ali`, `.map`, ...) open in the same ZaP. They line up with the run's overlays through their pixel size and origin. With only files, no copick config is needed. |
| `OVERLAY_URI` | Picks, filaments or meshes: `object:user/session`, e.g. `'ribosome:*/*'`; meshes appear in the Model View only. Segmentations: `name:user/session@voxel_size`, with `?multilabel=true` or `?instance=true` for those types. The type is detected from the shape of the URI. `--picks URI` and `--seg URI` force it. |
| `-c PATH` | Copick config. Defaults to `$CFG`, then `$COPICK_CONFIG`. A bare `CFG=...` line on its own is not exported to programs, so use `export CFG=...` (or `CFG=... cmod ...`). |
| `-l` | List the tomograms, picks, filaments, segmentations and meshes of a run, or every run when no run is given. |
| `-xyz`, `-S`, `-V`, `-Z` | Open the XYZ, Slicer or Model View (3D) window first; `-Z` also opens a ZaP window (as in 3dmod). |
| `-C 2G` | Size of the chunk cache in memory. |
| `-B N` | Never display data finer than binning N. |
| `-iz Z` | Initial zoom. |
| `-ic X,Y,Z` | Initial point, 1-based. |
| `-ia X,Y,Z` | Initial Slicer angles. |
| `-b` | Also open the run browser. Running `cmod` with no run opens only the browser. |
| `-W` | Stay in the foreground. By default cmod returns to the shell as soon as the run and URIs are valid, like 3dmod. Errors are still reported with exit code 1. |
| `--snapshot out.png` | Render every window once its data has loaded, save `out_<window>.png`, and exit. Works headless with `QT_QPA_PLATFORM=offscreen`. |

URIs follow copick's syntax: globs (`*`, `?`) work everywhere, and a `re:` prefix makes the URI a regex.

`cmod` is also registered as a copick CLI plugin, so `copick cmod ...` works too.

## Windows

The windows follow 3dmod's behaviour and look:
- its Qt style ("Windows" on Linux, Fusion on macOS; set `CMOD_STYLE` to use another);
- its default colours;
- its tool bars and tool buttons, with look-alike icons drawn by cmod;
- its mouse and keyboard handling.

- **Several tomograms per ZaP:** a ZaP can hold several tomograms and MRC files and shows one at a time, like
  3dmod's times.
  - Switch with Left/Right (Edit › Options can turn that off), with 1 and 2 (3dmod's keys), or with the row
    under the tool bar that appears once there is more than one.
  - Switching keeps the place, zoom and field of view in ångströms, so tomograms with different voxel sizes line
    up. Each tomogram keeps its own contrast.
  - Add MRC files with File › Open Image File (Ctrl+O) or by dropping them onto the ZaP or Info window.
- **Several runs:** each run gets its own Info and ZaP windows. A new run's windows open at the same size and zoom,
  one title bar down and to the right of the current ones (on top of them if there is no room). cmod remembers
  where the Info and ZaP windows were for the next start.
- **Info window** (the main window, laid out like 3dmod's):
  - **Object / Contour / Point** step through the overlays. Picks and filament points can be selected, which jumps to
    them.
  - **X / Y / Z** show the current point.
  - **Black / White** sliders use 3dmod's 0–255 positions; the data value is shown as a tooltip.
  - **Float** keeps the mean and SD of the display constant as you change section. As in 3dmod it is off at first,
    and cmod remembers how you left it.
  - **Subarea** bases Float and **Auto** on the area visible in the ZaP.
  - **Show** (on by default) moves to the chosen point. The pin keeps the window on top, and the raise button
    brings all cmod windows forward.
  - **Edit › Object List** turns overlays on and off and sets the segmentation opacity.
  - Closing an Info window closes its ZaP and other windows; closing the last one quits cmod (unless the run
    browser is open).
- **ZaP** pans like 3dmod:
  - An image that fits the window stays centred.
  - A larger one can be dragged until at most 1/6 of the window shows past its edge.
  - The ZaP does not chase the current point; the centring toggle or K keeps it centred.
  - **Tool bar:**
    - **lock** keeps the window on its own section;
    - **rubber band** (Shift+B): Shift+R fits the window to the band, Ctrl+Shift+R zooms the band to fill the
      window, and snapshots save only the band;
    - **arrow** draws a pointer.
    - The insert and lasso buttons are model-mode tools, so they stay disabled.
- **XYZ** is laid out like 3dmod's:
  - The XY, XZ and YZ panels and a Z square share the window. Drag the blue box where the gutters cross to change
    the split.
  - Dragging a panel pans it and the panels that share its axes.
  - Click or drag in the top gutter to set X, in the right gutter to set Y, and in the top-right square to set Z.
  - **Lock** keeps the window at its own location, and **Sum** averages that many sections in each panel.
- **Slicer:**
  - An oblique slice through a centre of rotation, with angles applied to the volume in the order Z, Y, X (as in
    3dmod and rotatevol).
  - Dragging moves the centre. The slice follows Z changes and single X/Y steps made elsewhere, but not clicks on new
    points (3dmod's "new" mode).
  - K switches to classic mode, which keeps the slice centred on the current point.
- **Model View** (3D, like 3dmod's; open with V, Image › Model View or `-V`):
  - **What it shows:** picks as spheres, filaments as lines, and copick meshes. Segmentations appear as surfaces;
    marching cubes runs on a coarse pyramid level in the background. The image planes through the current point
    are textured with the current contrast; Z, Shift+Y and Shift+X toggle them.
  - **Mouse:** first-button drag moves the model, second-button drag rotates it, and Shift + third-button drag
    rotates it around the viewing axis. The wheel zooms. A third-button click makes the nearest pick the current
    point.
  - **Keys:** the keypad rotates (Keypad Enter or U for continuous rotation), and T/B/F/K/L/R show the standard
    views.
  - **Requirements:** it uses fixed-function OpenGL, so it works with software rendering (Mesa llvmpipe) in VNC
    sessions without a GPU.

- **Run browser** (not in 3dmod; modelled on copick-web). Open it with Ctrl+B, File › Run Browser, `-b`, or by
  running `cmod` with no run.
  - **Runs:**
    - The tree goes run, then voxel spacing, then tomogram, with a search box.
    - Click a tomogram to open it. By default, another tomogram of a run that is open goes into that run's ZaP,
      and a tomogram of another run gets new Info and ZaP windows. Edit › Options (or File › Options in the
      browser) can send each case the other way, for example to show several runs in one ZaP.
    - The browser lists the objects of the run in the window you used last.
    - Double-click a run to open its preferred tomogram. The right-click menu has Run Info, which shows portal
      links, paths, contents and tomogram sizes.
  - **Gallery:** central-slice thumbnails of all runs.
  - **Objects of the open run:** picks, segmentations, filaments and meshes.
    - Check boxes show and hide overlays, loading them the first time.
    - Each tab has a search box and a "Shown only" filter. Pick counts are loaded in the background.
    - Filaments expand into a list; click one to go to it.
- **Snapshots:**
  - Ctrl+S, Shift+S and Ctrl+Shift+S save the window as TIFF, PNG or JPEG (only the rubber-band area if one is
    drawn), named `zap000.png`, `xyz000.png`, `slicer000.png`, `modv0000.png` as 3dmod names them.
  - The **Snapshot tool** (Shift+H, or File › Snapshot Tool) works like 3dmod's Grab with Note. It lets you:
    - pick a window and area;
    - render at 2× to 6× with finer data;
    - turn overlays and the current-point marker on or off;
    - add a scale bar and a note printed under the image;
    - then save or copy to the clipboard.

More 3dmod hot keys:
- **Objects:** O/P previous/next object, C/Shift+C contour, `[`/`]` point, `{`/`}` first/last point, E unselect,
  Shift+PageUp/PageDown next/previous section with points of the current object.
- **Display:** T overlays, Shift+T current-point marker, Ctrl+T current object.
- **Image:** Shift+F finds the brightest nearby pixel. F9/F10 switch between four stored contrast settings, and
  F12 or `*` toggles false colour.
- **Movie:** `#`/`$` (or a middle/right click in the ZaP) start a movie through Z, and either stops it. As in
  3dmod, it runs at 20 frames/s, `,`/`.` step through 3dmod's rates (0.2–200 frames/s), and at the first or last
  section it turns around. The next block of data is read ahead while it plays.
- **Windows:** Ctrl+R raises all windows, Shift+U opens the Model View.

The mouse wheel does nothing in the 2D windows, as in 3dmod. Help › Hot keys in any window lists all keys.

Picks are drawn as 3dmod draws spheres: a circle of radius √(r² − d²) on every section within the object's radius.
Filaments are drawn as polylines and segmentations as coloured labels, using each object's copick colour.

## How data are read

- **Pyramid levels.** Each window shows the coarsest pyramid level that needs at most 1.5× upsampling at its zoom,
  which is 3dmod's rule. Coarser levels fill in while finer ones load.
- **Chunk cache.** Chunks are cached decoded in an LRU limited by `-C`, so stepping through Z never decompresses the
  same chunk twice. Chunks are fetched by 8 background threads, and queued fetches nobody needs any more are
  cancelled.
- **Remote reads.** On S3 or HTTP, each chunk is downloaded as parallel 2 MB byte ranges (with a 64-connection S3
  pool). Single S3 streams can be slow, and float tomogram chunks hardly compress (about 60 MB per 256³ chunk), so
  this is 5–10× faster than plain zarr reads.
- **Contrast.** Contrast is done in float space, never by converting the data to bytes at load time. The initial
  black/white levels are 3dmod's autocontrast on the displayed section (without a 5% border): the mean maps to
  grey 150 and one SD to 40 grey levels. Values are rounded to whole slider steps and kept within 0–255, as 3dmod
  does.
  - The slider range is the file's header minimum and maximum for MRC files (3dmod's default). OME-Zarr data have
    no header, so they use 3dmod's fallback, the mean ± 6 SD.
- **MRC files** are memory-mapped and read in chunks through the same cache. Coarser levels for zoomed-out views
  use every 2nd, 4th, ... voxel.

## Development

```bash
uv sync --extra test --extra dev   # .venv with cmod, pytest, black, ruff and pre-commit
uv run pre-commit install          # black, ruff and the usual checks before each commit
uv run pytest                      # Qt tests run offscreen; the Model View test needs xvfb-run
```

Pull request titles follow [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, ...).
Merging to `main` lets release-please open a release PR; merging that publishes the package on PyPI.

## License

MIT, see [LICENSE](LICENSE).
