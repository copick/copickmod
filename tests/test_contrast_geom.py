import numpy as np
import pytest

from copickmod import contrast, slicer_geom


def test_autocontrast_matches_3dmod_targets():
    rng = np.random.default_rng(0)
    data = rng.normal(5.0, 2.0, (200, 200)).astype(np.float32)
    data[0, 0] = np.nan
    s = contrast.stats(data)
    assert s.mean == pytest.approx(5.0, abs=0.05)
    assert s.sd == pytest.approx(2.0, abs=0.05)
    black, white = contrast.auto_black_white(s)
    assert black == pytest.approx(s.mean - 3.75 * s.sd)
    assert white == pytest.approx(s.mean + 2.625 * s.sd)
    out = contrast.to_uint8(np.array([s.mean, s.mean + s.sd]), black, white)
    assert abs(int(out[0]) - 150) <= 1  # the mean maps to grey 150
    assert abs(int(out[1]) - int(out[0]) - 40) <= 1  # one SD spans 40 grey levels


def test_border_is_ignored():
    data = np.zeros((100, 100), dtype=np.float32)
    data[:3] = 1e6  # within the 5% border
    assert contrast.stats(data).vmax == 0.0


def test_slider_semantics_like_3dmod():
    s = contrast.Stats(mean=0.0, sd=1.0, vmin=-100, vmax=100)
    c = contrast.Contrast.from_stats(s)
    assert (c.lo, c.hi) == (-6.0, 6.0)
    assert c.black_slider == 48 and c.white_slider == 183  # typical 3dmod autocontrast positions
    c.set_sliders(200, 100)  # black pushed past white -> white is moved along
    assert c.black_slider == 200 and c.white_slider == 201
    c.step(-3, -3)
    assert c.black_slider == 197
    c.invert = True
    assert c.map(np.array([c.black]))[0] == 255


def test_constant_data_is_safe():
    c = contrast.Contrast.from_stats(contrast.stats(np.ones((10, 10))))
    assert c.hi > c.lo
    assert c.map(np.ones((2, 2))).dtype == np.uint8


def test_slicer_identity_is_xy():
    eu, ev, en = slicer_geom.slice_axes((0, 0, 0))
    np.testing.assert_allclose(eu, [0, 0, 1])  # zyx: u along x
    np.testing.assert_allclose(ev, [0, 1, 0])  # v along y
    np.testing.assert_allclose(en, [1, 0, 0])
    c = slicer_geom.plane_coords((5.5, 6.5, 7.5), (0, 0, 0), width=3, height=2, step=1.0)
    assert c.shape == (3, 2, 3)
    np.testing.assert_allclose(c[0], 5.5)  # constant z
    np.testing.assert_allclose(c[2][0], [6.5, 7.5, 8.5])  # x increases to the right
    assert c[1][0, 0] > c[1][1, 0]  # row 0 is the top (larger y)


def test_slicer_x90_shows_xz_and_y90_shows_yz():
    eu, ev, en = slicer_geom.slice_axes((90, 0, 0))
    np.testing.assert_allclose(eu, [0, 0, 1], atol=1e-12)
    assert abs(ev[0]) == pytest.approx(1)  # v runs along z
    eu, ev, en = slicer_geom.slice_axes((0, 90, 0))
    assert abs(eu[0]) == pytest.approx(1)  # u runs along z
    np.testing.assert_allclose(ev, [0, 1, 0], atol=1e-12)


def test_slicer_axes_orthonormal_and_project_roundtrip():
    angles = (17.0, -33.0, 120.0)
    eu, ev, en = slicer_geom.slice_axes(angles)
    m = np.stack([eu, ev, en])
    np.testing.assert_allclose(m @ m.T, np.eye(3), atol=1e-12)
    c = slicer_geom.plane_coords((10, 20, 30), angles, width=5, height=4, step=2.0, normal_offset=1.5)
    pts = c.reshape(3, -1).T
    u, v, d = slicer_geom.project(pts, (10, 20, 30), angles)
    np.testing.assert_allclose(d, 1.5)
    np.testing.assert_allclose(sorted(set(np.round(u, 6))), [-4, -2, 0, 2, 4])


def test_image_offset_matches_3dmod_rules():
    from copickmod.canvas import axis_center, image_offset, trans_factor

    # Fits: centred, the pan offset is ignored.
    assert image_offset(500, 200, 1.0, 0) == (150, 0, 0)
    assert image_offset(500, 200, 1.0, 77)[:2] == (150, 0)
    assert axis_center(500, 200, 1.0, 77)[0] == 100.0
    # Larger than the window: the offset pans, the first drawn pixel moves.
    woff, doff, off = image_offset(400, 1000, 1.0, 0)
    assert (woff, doff, off) == (0, 300, 0)
    assert image_offset(400, 1000, 1.0, 100)[1] == 200
    # Panned past the start: at most 1/6 of the window shows before the image, and the offset is clamped.
    woff, doff, off = image_offset(400, 1000, 1.0, 10_000)
    assert (woff, doff) == (400 // 6, 0)
    assert off < 10_000 and image_offset(400, 1000, 1.0, off)[:2] == (woff, 0)
    # Panned past the end: the image still covers at least 5/6 of the window.
    woff, doff, off = image_offset(400, 1000, 1.0, -10_000)
    assert woff == 0 and 1000 - doff >= int(400 * 0.8333333)
    # Drag rate: follows the mouse below zoom 1, one image pixel per screen pixel above.
    assert trans_factor(0.5) == 2.0 and trans_factor(3.0) == 1.0


def test_zoom_text_like_3dmod():
    from copickmod.style import zoom_text

    assert zoom_text(0.75) == "0.75"
    assert zoom_text(1.0) == "1.00"
    assert zoom_text(0.3333) == "0.3333"
    assert zoom_text(0.1667) == "0.1667"


def test_auto_rounds_like_3dmod():
    from copickmod.contrast import Contrast, Stats

    c = Contrast.from_stats(Stats(mean=0.0, sd=1.0, vmin=-100.0, vmax=100.0))
    # Slider range mean +/- 6 SD: black = -3.75 SD at slider 47.8 -> 48, white = +2.625 SD at 183.3 -> 183.
    assert (c.black_slider, c.white_slider) == (48, 183)
    assert c.black == pytest.approx(c.from_slider(48))
    narrow = Contrast(0.0, 1.0, 0.0, 1.0)
    narrow.set_auto(0.5, 0.505)  # less than 4 steps apart: spread to 4 around the middle
    assert narrow.white_slider - narrow.black_slider == 4
    narrow.set_floated(2.0, 3.0)  # beyond the range: clamped, backed off to stay 2 apart
    assert (narrow.black_slider, narrow.white_slider) == (254, 255)
    header = Contrast.from_stats(Stats(mean=0.0, sd=1.0, vmin=-100.0, vmax=100.0), load=(-50.0, 50.0))
    assert (header.lo, header.hi) == (-50.0, 50.0)  # MRC header min/max as the slider range (3dmod default)
