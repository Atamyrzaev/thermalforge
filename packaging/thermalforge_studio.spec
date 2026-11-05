# ThermalForge Studio :: PyInstaller build.
#
# Produces a folder containing ThermalForge Studio.exe and everything it needs --
# Python, NumPy, SciPy, the solver and the app's static files. A machine that
# has never seen Python runs the exe and gets the browser app.
#
#     python -m PyInstaller packaging/thermalforge_studio.spec --noconfirm
#
# One-DIR rather than one-file on purpose: a one-file build unpacks ~120 MB
# to a temporary directory on every launch, which costs seconds of staring at
# nothing at a trade show. The folder starts in about a second.

from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

ROOT = Path(SPECPATH).parent

# The app is served from disk, so the whole static tree and the measured
# validation artifacts travel with it.
datas = [
    (str(ROOT / "studio" / "static"), "studio/static"),
    (str(ROOT / "configs" / "config.json"), "configs"),
    (str(ROOT / "validation"), "validation"),
]

hiddenimports = collect_submodules("scipy.sparse.linalg") + [
    "scipy.sparse.csgraph",
    "studio.engine",
    "studio.scenarios",
    "studio.server",
]

a = Analysis(
    [str(ROOT / "run_studio.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    # Nothing in the app draws with matplotlib or runs tests; leaving them in
    # doubles the download for no benefit.
    excludes=["matplotlib", "pytest", "tkinter", "PIL", "IPython", "notebook"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ThermalForge Studio",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,          # the window doubles as the server log; closing it stops the app
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="ThermalForge Studio",
)
