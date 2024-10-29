# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['DroneDetection.py'],
    pathex=['<script path>'],
    binaries=[],
    datas=[('best.pt', '.'), ('kw_logo.png', '.')],
    hiddenimports=["PyQt5.QtWidgets", "PyQt5.QtCore", "PyQt5.QtGui", "sahi"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['opencv-python', 'torch'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='DroneDetection',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['kw_logo.ico'],
)
