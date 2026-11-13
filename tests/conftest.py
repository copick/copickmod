import json
import os

import numpy as np
import pytest
import zarr

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(autouse=True, scope="session")
def _windows_style(qapp):
    """Test with the Qt style cmod uses (3dmod's)."""
    from copickmod.style import apply_app_style

    apply_app_style(qapp)


@pytest.fixture(autouse=True, scope="session")
def _settings_dir(tmp_path_factory):
    """Keep the user's cmod settings out of the tests."""
    from qtpy.QtCore import QSettings

    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(tmp_path_factory.mktemp("settings")))


@pytest.fixture(autouse=True)
def _default_settings(_settings_dir):
    from copickmod import settings

    settings._store().clear()
    settings.save_geometry = True
    yield
    settings._store().clear()


def write_mrc(path, data, voxel=10.0, origin=(0.0, 0.0, 0.0)):
    """A small MRC file (data zyx)."""
    import mrcfile

    with mrcfile.new(str(path), overwrite=True) as mrc:
        mrc.set_data(np.asarray(data, dtype=np.float32))
        mrc.voxel_size = voxel
        mrc.header.origin = origin
    return str(path)


SHAPE = (40, 70, 90)  # z, y, x
VOXEL = 10.0


def synthetic_volume(shape=SHAPE, seed=0) -> np.ndarray:
    """Smooth-ish structured data: a ramp plus noise, so every voxel is distinguishable."""
    rng = np.random.default_rng(seed)
    z, y, x = np.indices(shape, dtype=np.float32)
    return (x + 100 * y + 10000 * z + rng.normal(0, 0.1, shape)).astype(np.float32)


def write_ome_zarr(path, levels, voxel=VOXEL, chunks=(16, 16, 16)):
    """Write a zarr v2 OME-Zarr group with the given level arrays (each 2x coarser)."""
    group = zarr.open_group(str(path), mode="w")
    datasets = []
    for i, data in enumerate(levels):
        group.create_dataset(str(i), data=data, chunks=chunks)
        s = voxel * 2**i
        datasets.append({"path": str(i), "coordinateTransformations": [{"type": "scale", "scale": [s, s, s]}]})
    group.attrs["multiscales"] = [
        {
            "version": "0.4",
            "axes": [{"name": a, "type": "space", "unit": "angstrom"} for a in "zyx"],
            "datasets": datasets,
        },
    ]
    return group


@pytest.fixture
def pyramid():
    data = synthetic_volume()
    return [data, data[::2, ::2, ::2].copy(), data[::4, ::4, ::4].copy()]


@pytest.fixture
def ome_group(tmp_path, pyramid):
    return write_ome_zarr(tmp_path / "vol.zarr", pyramid)


OBJECTS = [
    {"name": "ribosome", "is_particle": True, "label": 1, "color": [0, 255, 0, 255], "radius": 60.0},
    {"name": "membrane", "is_particle": False, "label": 2, "color": [255, 0, 255, 255], "radius": 50.0},
    {"name": "actin", "is_particle": False, "label": 3, "color": [230, 120, 30, 255], "radius": 35.0},
]

PICKS_XYZ_A = np.array([[100.0, 200.0, 150.0], [455.0, 305.0, 205.0], [800.0, 600.0, 300.0]])


@pytest.fixture
def copick_config(tmp_path):
    """A filesystem copick project with one run, two tomograms, picks, filaments and segmentations."""
    import copick

    cfg = {
        "name": "cmod test",
        "description": "synthetic",
        "version": "1.32.0",
        "config_type": "filesystem",
        "pickable_objects": OBJECTS,
        "overlay_root": "local://" + str(tmp_path / "overlay"),
        "overlay_fs_args": {"auto_mkdir": True},
    }
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(cfg))
    root = copick.from_file(str(cfg_path))
    run = root.new_run("TS_001")
    vs = run.new_voxel_spacing(VOXEL)
    data = synthetic_volume()
    vs.new_tomogram("wbp").from_numpy(data, levels=3)
    vs.new_tomogram("denoised").from_numpy(data * 2, levels=3)
    run.new_picks("ribosome", "1", "alice").from_numpy(PICKS_XYZ_A)
    labels = np.zeros(data.shape, dtype=np.uint8)
    labels[10:20, 20:40, 30:60] = 2
    run.new_segmentation(VOXEL, "membrane", "1", False, "alice").from_numpy((labels > 0).astype(np.uint8))
    run.new_segmentation(VOXEL, "labels", "1", True, "alice").from_numpy(labels)
    fil = run.new_filaments("actin", "1", "alice") if hasattr(run, "new_filaments") else None
    if fil is not None:
        fil.from_numpy([np.array([[50.0, 50.0, 50.0], [150.0, 60.0, 55.0], [250.0, 70.0, 60.0]])])
    import trimesh

    sphere = trimesh.creation.icosphere(subdivisions=2, radius=100.0)
    sphere.apply_translation([450.0, 350.0, 200.0])  # angstrom, xyz
    mesh = run.new_mesh("membrane", "1", "alice")
    mesh.mesh = sphere
    mesh.store()
    run2 = root.new_run("TS_002")
    run2.new_voxel_spacing(VOXEL).new_tomogram("wbp").from_numpy(synthetic_volume(seed=1), levels=2)
    root2 = copick.from_file(str(cfg_path))  # fresh root: nothing cached in memory
    os.environ.pop("CFG", None)
    os.environ.pop("COPICK_CONFIG", None)
    return str(cfg_path), root2
