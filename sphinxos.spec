# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for SphinxOS

Builds a standalone executable for SphinxOS including:
- Core quantum-spacetime kernel
- Economic simulator
- PoX automation tools
- NPTC framework

Usage:
    pyinstaller sphinxos.spec

Produces:
    dist/sphinxos.exe (Windows)
    dist/sphinxos.app (macOS)
    dist/sphinxos (Linux)
"""

block_cipher = None

# Analysis: Collect all Python modules
a = Analysis(
    ['sphinx_os/economics/simulator.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('contracts/*.clar', 'contracts'),
        ('docs/security/*.md', 'docs/security'),
        ('README.md', '.'),
        ('ECONOMICS.md', '.'),
    ],
    hiddenimports=[
        'sphinx_os',
        'sphinx_os.economics',
        'sphinx_os.economics.yield_calculator',
        'sphinx_os.economics.simulator',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'numpy',  # Optional dependency
        'scipy',  # Optional dependency
        'matplotlib',  # Optional dependency
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='sphinxos',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add icon path here if available
)

# macOS app bundle
if sys.platform == 'darwin':
    app = BUNDLE(
        exe,
        name='SphinxOS.app',
        icon=None,
        bundle_identifier='ai.sphinxos.app',
        info_plist={
            'CFBundleName': 'SphinxOS',
            'CFBundleDisplayName': 'SphinxOS',
            'CFBundleShortVersionString': '1.0.0',
            'CFBundleVersion': '1.0.0',
            'NSHighResolutionCapable': True,
        },
    )
