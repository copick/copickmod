import os
import subprocess
import sys
import time

import pytest
from click.testing import CliRunner

from copickmod.cli import cmod


@pytest.fixture
def runner():
    return CliRunner()


def test_missing_config(runner, monkeypatch):
    monkeypatch.delenv("CFG", raising=False)
    monkeypatch.delenv("COPICK_CONFIG", raising=False)
    res = runner.invoke(cmod, ["TS_001"])
    assert res.exit_code == 1
    assert "export CFG=" in res.output


def test_list_via_cfg_env(runner, copick_config):
    cfg, _ = copick_config
    res = runner.invoke(cmod, ["-l", "TS_001"], env={"CFG": cfg})
    assert res.exit_code == 0, res.output
    assert "wbp@10.0" in res.output and "ribosome:alice/1" in res.output
    res = runner.invoke(cmod, ["-l"], env={"COPICK_CONFIG": cfg})
    assert "TS_001" in res.output


def test_errors(runner, copick_config, monkeypatch):
    cfg, _ = copick_config
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    res = runner.invoke(cmod, ["-c", cfg, "-W", "TS_001"])
    assert res.exit_code == 1 and "has 2 tomograms" in res.output and "denoised@10.0" in res.output
    res = runner.invoke(cmod, ["-c", cfg, "-W", "TS_01", "wbp"])
    assert res.exit_code == 1 and "Did you mean: TS_001" in res.output
    res = runner.invoke(cmod, ["-c", cfg, "-W", "TS_001", "wbp", "nothing:*/*"])
    assert res.exit_code == 1 and "No picks, filaments or meshes" in res.output
    res = runner.invoke(cmod, ["-c", cfg, "-W", "-ic", "1,2", "TS_001", "wbp"])
    assert res.exit_code != 0 and "three comma-separated" in res.output


def test_no_display(runner, copick_config, monkeypatch):
    cfg, _ = copick_config
    for var in ("DISPLAY", "WAYLAND_DISPLAY", "QT_QPA_PLATFORM"):
        monkeypatch.delenv(var, raising=False)
    res = runner.invoke(cmod, ["-c", cfg, "TS_001", "wbp"])
    assert res.exit_code == 1 and "No display" in res.output


def _cmod(args, cfg, tmp_path, **kw):
    """Run cmod in a subprocess. Output goes to files: a detached viewer keeps inherited pipes open."""
    env = dict(
        os.environ,
        QT_QPA_PLATFORM="offscreen",
        CFG=cfg,
        XDG_CONFIG_HOME=str(tmp_path),
    )  # not the user's settings
    out, err = tmp_path / "out.txt", tmp_path / "err.txt"
    with open(out, "w") as fo, open(err, "w") as fe:
        proc = subprocess.run([sys.executable, "-m", "copickmod", *args], env=env, stdout=fo, stderr=fe, **kw)
    proc.stdout, proc.stderr = out.read_text(), err.read_text()
    return proc


def test_detached_error_is_reported(copick_config, tmp_path):
    cfg, _ = copick_config
    proc = _cmod(["TS_404", "wbp"], cfg, tmp_path, timeout=120)
    assert proc.returncode == 1
    assert "not found" in proc.stderr


def test_detach_returns_while_viewer_runs(copick_config, tmp_path):
    cfg, _ = copick_config
    t0 = time.monotonic()
    try:
        proc = _cmod(["TS_001", "wbp"], cfg, tmp_path, timeout=120)
        assert proc.returncode == 0, proc.stderr
        assert time.monotonic() - t0 < 60
        # The detached viewer is still alive.
        alive = subprocess.run(["pgrep", "-f", f"copickmod --ready-fd .* -c {cfg}"], capture_output=True, text=True)
        assert alive.stdout.strip(), "viewer process not running"
    finally:
        subprocess.run(["pkill", "-f", f"copickmod --ready-fd .* -c {cfg}"], check=False)
