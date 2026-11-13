"""3dmod hot keys shared by all windows (imod_input.cpp). Windows handle their own keys first."""

from __future__ import annotations

from qtpy.QtCore import Qt

from copickmod.state import Session

CONTRAST_STEP = 3  # 3dmod: F1-F8 move the sliders by 3

HELP_TEXT = """\
cmod hot keys and mouse (3dmod conventions)

All windows
  PageUp / PageDown      next / previous section (in XYZ: through the planes of the panel under the mouse)
  Keypad * / Keypad /    up / down 10 sections
  Home / End             last / first section
  Shift+Home, Insert     middle section
  Arrow keys             move the current point by one pixel; with several tomograms in a ZaP, Left/Right
                         switch between them (Edit > Options)
  1 / 2                  previous / next tomogram of the window (3dmod's times)
  Ctrl+O                 open MRC files into this window (or drop them onto the ZaP or Info window)
  F1 / F2                black level down / up
  F3 / F4                white level down / up
  F5 / F6                contrast down / up
  F7 / F8                brightness down / up
  F9 / F10               contrast setting 1 / cycle through the four stored settings
  F11 (&)                invert contrast           F12 (*)   false color
  Shift+A                autocontrast on the current section (or Subarea)
  F                      print the value at the current point
  Shift+F                find the brightest pixel within 10 pixels
  O / P                  previous / next object (overlay)
  C / Shift+C            previous / next contour (filament)
  [ < / ] >              previous / next point (pick)    { ( / } )   first / last point
  E                      unselect the current point
  Shift+PageUp/Down      next / previous section with points of the current object
  T                      toggle all overlays       Shift+T   toggle the current point marker
  Ctrl+T                 toggle the current object
  Ctrl+S / Shift+S / Ctrl+Shift+S   snapshot of the window as TIFF / PNG / JPEG
  Shift+H                Snapshot tool (window, area, resolution, scale bar, note; copy or save)
  # / $                  movie forward / backward through Z, turning at the ends (either key stops it)
  , / .                  slower / faster movie (3dmod's rates: 20 frames/s at first)
  Z                      open a ZaP window
  Ctrl+X                 open the XYZ window
  \\                      open the Slicer
  V, Shift+U             open the Model View (3D)
  Ctrl+R                 raise all cmod windows
  Ctrl+B                 run browser and object table (not in 3dmod)
  Ctrl+Q                 quit
The mouse wheel does nothing, as in 3dmod.

ZaP window
  - / = (+)              zoom down / up            Shift+R   resize window to fit the image (or the band)
  K                      keep the current point centered (toggle)
  Shift+B                rubber band on/off        Ctrl+Shift+R  zoom the band to fill the window
  Lock button: keep this window on its section (PageUp/Down then step only this window).
  Q                      distance from the current point to the mouse
  Ctrl+S / Shift+S       save a PNG snapshot       Shift+I   window/image info
  Keypad arrows          move the image 5 pixels   Esc       close
  Left click: set the current point. Left drag: move the image (an image that fits the window stays centered).
  Middle / right click: movie forward / backward.

XYZ window
  - / = (+)              zoom down / up            K         center the panels on the current point
  R                      toggle high-quality       Shift+R   resize window to fit the image
  Sum: average that many sections in each panel.  Lock: keep this window at its own location.
  Left click in a panel sets its two coordinates; click or drag in the gutters to set X (top gutter),
  Y (right gutter) or Z (top-right square). Drag a panel to move the images; drag the blue box where the
  gutters cross to change how the window is shared between the panels.

Model View (3D)
  First (left) drag      move the model            Second (middle) drag  rotate the model
  Shift + right drag     rotate around the viewing axis
  Right click            make the nearest pick or filament point the current point
  Mouse wheel            zoom                      - / = and _ / +       zoom in small / big steps
  Keypad 2/8, 4/6, 3/9   rotate around X, Y, Z     Keypad Enter, 5, U    start / stop continuous rotation
  7/J, Y/I, 6/H          rotate (keypad substitutes)   , / .            smaller / bigger rotation step
  T B F K L R            top, bottom, front, back, left, right views
  Arrows, PageUp/Down    move the model            Z, Shift+Y, Shift+X   toggle the Z / Y / X image planes
  P                      toggle the current point  G / Shift+G           better / coarser spheres
  Shift+L                Object List               Shift+S, Ctrl+S       PNG snapshot
  Esc, Q                 close

Slicer
  Keypad 4/6, 2/8, 1/3   change the last adjusted angle by 0.1, 0.5, 15 degrees
  Keypad 0 (Insert)      set the last adjusted angle to 0
  _ / +                  thinner / thicker slice (averaged planes)
  - / =                  zoom down / up            K         keep centered on the current point (classic mode)
  Arrow keys             move the image a pixel    PageUp/Down  move along the slice normal
  Left click: set the current point (the slice stays put). Left drag: move the slice center.
"""


def _arrows_switch() -> bool:
    from copickmod import settings

    return bool(settings.get("keys/arrows_switch"))


def handle_global(session: Session, event, axis: int = 0) -> bool:
    """Handle a key press shared by all windows. ``axis`` is the axis PageUp/PageDown move along."""
    key = event.key()
    mods = event.modifiers()
    shift = bool(mods & Qt.ShiftModifier)
    ctrl = bool(mods & Qt.ControlModifier)
    keypad = bool(mods & Qt.KeypadModifier)
    s = session
    n = s.volume.shape[axis]

    if key in (Qt.Key_PageUp, Qt.Key_PageDown) and shift and axis == 0:
        s.navigate("next_section" if key == Qt.Key_PageUp else "prev_section")
    elif key == Qt.Key_PageUp:
        s.move(axis, 1)
    elif key == Qt.Key_PageDown:
        s.move(axis, -1)
    elif key == Qt.Key_Asterisk and keypad:
        s.move(axis, 10)
    elif key == Qt.Key_Slash and keypad:
        s.move(axis, -10)
    elif key == Qt.Key_Home and shift:
        s.set_axis(axis, n // 2)
    elif key == Qt.Key_Home:
        s.set_axis(axis, n - 1)
    elif key == Qt.Key_End:
        s.set_axis(axis, 0)
    elif key == Qt.Key_Insert and not keypad:
        s.set_axis(axis, n // 2)
    elif key in (Qt.Key_Left, Qt.Key_Right) and not keypad and not ctrl and len(s.images) > 1 and _arrows_switch():
        s.step_image(-1 if key == Qt.Key_Left else 1)  # cmod: switch between the tomograms of the window
    elif key in (Qt.Key_Left, Qt.Key_Right) and not keypad:
        s.move(2, -1 if key == Qt.Key_Left else 1)
    elif key in (Qt.Key_1, Qt.Key_2) and not keypad and not ctrl:
        if not s.step_image(-1 if key == Qt.Key_1 else 1):  # 3dmod: previous / next time
            s.message.emit("Only one tomogram is open in this window (File > Open Image File, or the run browser)")
    elif key == Qt.Key_O and ctrl:
        s.open("open_files")
    elif key in (Qt.Key_Up, Qt.Key_Down) and not keypad:
        s.move(1, 1 if key == Qt.Key_Up else -1)
    elif key == Qt.Key_F1:
        s.step_contrast(-CONTRAST_STEP, 0)
    elif key == Qt.Key_F2:
        s.step_contrast(CONTRAST_STEP, 0)
    elif key == Qt.Key_F3:
        s.step_contrast(0, -CONTRAST_STEP)
    elif key == Qt.Key_F4:
        s.step_contrast(0, CONTRAST_STEP)
    elif key == Qt.Key_F5:  # less contrast: widen the window
        s.step_contrast(-CONTRAST_STEP, CONTRAST_STEP)
    elif key == Qt.Key_F6:
        s.step_contrast(CONTRAST_STEP, -CONTRAST_STEP)
    elif key == Qt.Key_F7:  # darker: move the window up
        s.step_contrast(CONTRAST_STEP, CONTRAST_STEP)
    elif key == Qt.Key_F8:
        s.step_contrast(-CONTRAST_STEP, -CONTRAST_STEP)
    elif key in (Qt.Key_F11, Qt.Key_Ampersand):
        s.toggle_invert()
    elif key == Qt.Key_F12 or (key == Qt.Key_Asterisk and not keypad):
        s.toggle_false_color()
    elif key == Qt.Key_F9:
        s.select_ramp(0)
    elif key == Qt.Key_F10:
        s.select_ramp(s.ramp_index + 1)
    elif key == Qt.Key_F and shift and not ctrl:
        y, x, value = s.find_max_nearby()
        s.message.emit(f"Maximum within 10 pixels is {value:g} at ({x + 1}, {y + 1}, {s.point[0] + 1})")
    # Overlays ("model"): 3dmod's object / contour / point stepping and display toggles.
    elif key == Qt.Key_O and not shift and not ctrl:
        s.navigate("prev_object")
    elif key == Qt.Key_P and not shift and not ctrl:
        s.navigate("next_object")
    elif key == Qt.Key_C and not ctrl:
        s.navigate("next_contour" if shift else "prev_contour")
    elif key in (Qt.Key_BracketLeft, Qt.Key_Less) and not ctrl:
        s.navigate("prev_point")
    elif key in (Qt.Key_BracketRight, Qt.Key_Greater) and not ctrl:
        s.navigate("next_point")
    elif key in (Qt.Key_BraceLeft, Qt.Key_ParenLeft):
        s.navigate("first_point")
    elif key in (Qt.Key_BraceRight, Qt.Key_ParenRight):
        s.navigate("last_point")
    elif key == Qt.Key_E and not shift and not ctrl:
        s.navigate("unselect_point")
    elif key == Qt.Key_T and ctrl:
        s.navigate("toggle_object")
    elif key == Qt.Key_T and shift:
        s.toggle_markers()
    elif key == Qt.Key_T:
        s.toggle_model()
    elif key == Qt.Key_M and not shift and not ctrl:
        s.message.emit("cmod is view-only: there is no model mode")
    elif key == Qt.Key_A and shift and not ctrl:
        s.autocontrast()
    elif key == Qt.Key_F and not shift and not ctrl:
        z, y, x = s.point
        s.message.emit(f"Value at ({x + 1}, {y + 1}, {z + 1}) = {s.value_at():g}")
    elif key == Qt.Key_NumberSign:
        s.toggle_movie(1, axis)
    elif key == Qt.Key_Dollar:
        s.toggle_movie(-1, axis)
    elif key == Qt.Key_Comma:
        s.movie_speed(faster=False)
    elif key == Qt.Key_Period:
        s.movie_speed(faster=True)
    elif key == Qt.Key_Z and not shift and not ctrl:
        s.open("zap")
    elif key == Qt.Key_R and ctrl and not shift:
        s.open("raise")
    elif key == Qt.Key_U and shift:
        s.open("model")
    elif key == Qt.Key_H and shift:
        s.open("snapshot")
    elif key == Qt.Key_B and ctrl:
        s.open("browser")
    elif key == Qt.Key_X and ctrl:
        s.open("xyz")
    elif key == Qt.Key_Backslash:
        s.open("slicer")
    elif key == Qt.Key_V and not shift and not ctrl:
        s.open("model")
    elif key == Qt.Key_Q and ctrl:
        s.open("quit")
    else:
        return False
    return True
