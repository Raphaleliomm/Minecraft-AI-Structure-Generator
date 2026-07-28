#!/usr/bin/env python
"""
Portable Distribution Builder – Minecraft Structure Generator
============================================================

Erzeugt eine portable Windows-Distribution mit EMBEDDED Python und
allen Abhängigkeiten. Kein separates Installieren von Python oder
Bibliotheken nötig. Einfach entpacken und starten.

Verwendung:
    python build_portable.py

Das Ergebnis liegt im Ordner 'dist/MinecraftGenerator_Portable/'.
Diesen Ordner kann man als .zip packen und ins Internet stellen.
"""

import os
import sys
import shutil
import subprocess
import json
import zipfile
import urllib.request
import tarfile
import io
import tempfile
import struct
from pathlib import Path
from typing import Optional

# ─── Konfiguration ──────────────────────────────────────────────────────────

DIST_NAME = "MinecraftStructureGenerator_Portable"
DIST_DIR = Path(__file__).parent / "dist" / DIST_NAME
TEMP_DIR = Path(tempfile.gettempdir()) / "minecraft_gen_build"
PROJECT_ROOT = Path(__file__).parent

# Welche Python-Version embedded werden soll (muss zur installierten passen)
# Wird automatisch ermittelt, kann aber überschrieben werden.
EMBEDDED_PYTHON_VERSION: Optional[str] = None  # z.B. "3.12.4"

# Pakete, die in die portable Distribution müssen
REQUIRED_PACKAGES = [
    "torch",
    "nbtlib",
    "customtkinter",
    "Pillow",
    "numpy",
    "darkdetect",   # customtkinter dependency
    "packaging",     # torch dependency
    "sympy",         # torch dependency
    "networkx",      # torch dependency
    "jinja2",        # torch dependency
    "mpmath",        # sympy dependency
    "filelock",      # torch dependency
    "typing_extensions",
    "fsspec",        # torch dependency
]

# Dateien/Ordner aus dem Projekt, die KOPIERT werden müssen
PROJECT_FILES_TO_COPY = [
    "gui_app.py",
    "start_gui.bat",
    "dataset.py",
    "model.py",
    "generate.py",
    "train.py",
    "config.json",
    "README.md",
    "requirements.txt",
    "augment_data.py",
    "filter_dataset.py",
    "kaggle_export.py",
    "start_app.bat",
]

PROJECT_DIRS_TO_COPY = [
    "app",
    "tests",
]

# Trainingsdaten-Verzeichnisse (für Training und Evaluation in der portablen Version)
TRAINING_DATA_DIRS = [
    "Trainingsdaten good thoroughly analyzed",
    "Trainingsdaten zu gross vorerst ausgelagert",
]

# Dateien/Ordner, die aus dem Projekt NICHT in die Distribution sollen
EXCLUDE_PATTERNS = {
    "__pycache__", "*.pyc", ".git", ".gitignore",
    "node_modules", ".vscode", ".idea",
    "build_portable.py",
}

# ─── Hilfsfunktionen ────────────────────────────────────────────────────────


def get_python_version() -> str:
    """Liefert die aktuelle Python-Version als '3.x.y'."""
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def get_arch() -> str:
    """Gibt 'win32' oder 'win_amd64' zurück."""
    return "win_amd64" if struct.calcsize("P") * 8 == 64 else "win32"


# Fallback-Python-Versionen für embedded (absteigende Verfügbarkeit)
# Python 3.14.5 ist evtl. zu neu auf python.org; versuche ältere stabile
FALLBACK_PYTHON_VERSIONS = ["3.13.3", "3.12.9", "3.12.8", "3.12.7", "3.12.4"]


def get_embedded_python_url(version: str, arch: str) -> str:
    """Baut den Download-Link für die embeddable Python Distribution."""
    major_minor = ".".join(version.split(".")[:2])
    return (
        f"https://www.python.org/ftp/python/{version}/"
        f"python-{version}-embed-{arch}.zip"
    )


def get_pip_url() -> str:
    """Liefert die URL zum Herunterladen von get-pip.py."""
    return "https://bootstrap.pypa.io/get-pip.py"


def download_file(url: str, target_path: Path) -> None:
    """Lädt eine Datei herunter und zeigt Fortschritt."""
    print(f"  ⬇️  {url.split('/')[-1]} ...")
    try:
        urllib.request.urlretrieve(url, target_path)
    except Exception as e:
        print(f"  ❌ Fehler beim Download: {e}")
        raise


def ensure_dir(path: Path) -> None:
    """Erstellt ein Verzeichnis, falls es nicht existiert."""
    path.mkdir(parents=True, exist_ok=True)


def copy_project_files() -> None:
    """Kopiert alle relevanten Projekt-Dateien in die Distribution."""
    print("\n📂 Kopiere Projektdateien ...")
    
    for filename in PROJECT_FILES_TO_COPY:
        src = PROJECT_ROOT / filename
        dst = DIST_DIR / filename
        if src.exists():
            shutil.copy2(src, dst)
            print(f"   ✓ {filename}")
        else:
            print(f"   ⚠️  Fehlt: {filename}")

    for dirname in PROJECT_DIRS_TO_COPY:
        src = PROJECT_ROOT / dirname
        dst = DIST_DIR / dirname
        if src.exists():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(
                src, dst,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
            print(f"   ✓ {dirname}/")
        else:
            print(f"   ⚠️  Fehlt: {dirname}/")

    # Kopiere den runs/ Ordner (trainierte Modelle) falls vorhanden
    src_runs = PROJECT_ROOT / "runs"
    dst_runs = DIST_DIR / "runs"
    if src_runs.exists():
        if dst_runs.exists():
            shutil.rmtree(dst_runs)
        shutil.copytree(
            src_runs, dst_runs,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.log"),
        )
        print(f"   ✓ runs/ (trainierte Modelle)")
    else:
        print(f"   ⚠️  Kein runs/ Ordner – trainiere zuerst Modelle!")

    # Kopiere Trainingsdaten-Verzeichnisse
    for train_dir in TRAINING_DATA_DIRS:
        src_train = PROJECT_ROOT / train_dir
        dst_train = DIST_DIR / train_dir
        if src_train.exists():
            if dst_train.exists():
                shutil.rmtree(dst_train)
            print(f"   📦 Kopiere {train_dir}/ ...")
            shutil.copytree(
                src_train, dst_train,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
            # Zähle .schem-Dateien
            schem_count = len(list(dst_train.rglob("*.schem")))
            print(f"     ✓ {train_dir}/ ({schem_count} .schem-Dateien)")
        else:
            print(f"   ⚠️  Fehlt: {train_dir}/")


def _download_and_extract_python(version: str, arch: str, python_dir: Path) -> Optional[Path]:
    """Lade eine bestimmte Python-Version herunter und entpacke sie.
    Gibt den Pfad zur python.exe zurück, oder None bei Fehler."""
    url = get_embedded_python_url(version, arch)
    zip_path = TEMP_DIR / f"python-{version}-embed-{arch}.zip"
    
    if not zip_path.exists():
        print(f"\n   📥 Lade embedded Python herunter ({version}) ...")
        try:
            download_file(url, zip_path)
        except Exception as e:
            print(f"   ⚠️  Konnte Python {version} nicht herunterladen: {e}")
            return None
    else:
        print(f"   📦 Verwende gecachte Python: {zip_path}")
    
    # Leeres Zielverzeichnis
    if python_dir.exists():
        shutil.rmtree(python_dir)
    ensure_dir(python_dir)
    
    # Entpacken
    print("   📦 Entpacke Python ...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(python_dir)
    
    # Fix: Entferne die _pth-Datei, damit site-packages geladen werden
    for pth_file in python_dir.glob("*._pth"):
        print(f"   🔧 Entferne {pth_file.name} (Site-Packages aktivieren)")
        pth_file.rename(pth_file.with_suffix("._pth.disabled"))
    
    python_exe = python_dir / "python.exe"
    if not python_exe.exists():
        print(f"   ❌ python.exe nicht gefunden in {python_dir}")
        return None
    
    return python_exe


def _install_pip_and_packages(python_exe: Path) -> bool:
    """Installiere pip und alle benötigten Pakete in das embedded Python.
    Gibt True bei Erfolg zurück."""
    # Installiere pip
    print("\n   📥 Installiere pip ...")
    pip_path = TEMP_DIR / "get-pip.py"
    if not pip_path.exists():
        download_file(get_pip_url(), pip_path)
    
    result = subprocess.run(
        [str(python_exe), str(pip_path), "--no-warn-script-location"],
        check=False, capture_output=True,
        cwd=python_exe.parent,
    )
    if result.returncode != 0:
        print("   ⚠️  pip-Installation fehlgeschlagen")
        return False
    
    # Installiere Pakete
    _install_packages(python_exe)
    return True


def setup_embedded_python() -> None:
    """
    Lädt die embeddable Python Distribution herunter und richtet sie ein.
    Versucht zuerst die exakte Version, dann Fallback-Versionen.
    Falls alles fehlschlägt: System-Python als Basis.
    """
    arch = get_arch()
    
    # Liste der zu probierenden Versionen
    versions_to_try = []
    primary_version = EMBEDDED_PYTHON_VERSION or get_python_version()
    versions_to_try.append(primary_version)
    # Fallback-Versionen, falls die primäre nicht verfügbar ist
    for fb in FALLBACK_PYTHON_VERSIONS:
        if fb != primary_version:
            versions_to_try.append(fb)
    
    python_dir = DIST_DIR / "python"
    
    for version in versions_to_try:
        print(f"\n🐍 Versuche embedded Python {version} ({arch}) ...")
        python_exe = _download_and_extract_python(version, arch, python_dir)
        if python_exe is None:
            continue
        
        if _install_pip_and_packages(python_exe):
            print(f"\n   ✅ Embedded Python {version} erfolgreich eingerichtet!")
            return
        else:
            print(f"   ⚠️  Paketinstallation in {version} fehlgeschlagen")
    
    # Alles fehlgeschlagen -> Fallback auf System-Python
    print(f"\n   🤔 Konnte keine embedded Python-Version herunterladen.")
    print(f"   🤔 Verwende installiertes Python als Basis ...")
    _use_installed_python(python_dir)


def _use_installed_python(python_dir: Path) -> None:
    """Fallback: Verwende die installierte Python-Installation als Basis.
    
    Kopiert python.exe + benötigte DLLs in das portable Verzeichnis
    und installiert alle benötigten Pakete per pip.
    """
    print("\n   🔧 Verwende installiertes Python als Basis ...")
    
    python_exe = Path(sys.executable)
    
    # Stelle sicher, dass das Zielverzeichnis existiert
    ensure_dir(python_dir)
    
    # Kopiere python.exe
    print("   📄 Kopiere python.exe ...")
    shutil.copy2(str(python_exe), str(python_dir / "python.exe"))
    
    # Kopiere notwendige DLLs vom Python-Root
    python_root = python_exe.parent
    for dll_pattern in ["python*.dll", "vcruntime*.dll", "libcrypto*.dll", "libssl*.dll"]:
        for dll in python_root.glob(dll_pattern):
            shutil.copy2(str(dll), str(python_dir / dll.name))
    
    # Erstelle Zielverzeichnisse
    target_sp = python_dir / "Lib" / "site-packages"
    ensure_dir(target_sp)
    
    # install.py – kleine Hilfskomponente, die `pip install` macht
    # und eine .pth-Datei anlegt, damit die importierten Pakete gefunden werden.
    install_script = python_dir / "_install_packages.py"
    
    script_content = f'''#!/usr/bin/env python
"""Installiert Pakete in ein benutzerdefiniertes target-Verzeichnis."""
import subprocess
import sys
import os

PACKAGES = {json.dumps(REQUIRED_PACKAGES, indent=4)}
TARGET = r"{target_sp}"

def main():
    # Installiere pip selbst, falls nicht vorhanden
    pip_available = True
    try:
        import pip
    except ImportError:
        pip_available = False
    
    for pkg in PACKAGES:
        print(f"  Installiere {{pkg}} ...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install",
             "--target", TARGET,
             "--upgrade",
             pkg],
            check=False, capture_output=True, text=True,
        )
        if result.returncode != 0:
            # Letzte Zeile der Fehlerausgabe
            err_lines = [l for l in result.stderr.splitlines() if l.strip()]
            err_msg = err_lines[-1] if err_lines else "Unbekannter Fehler"
            print(f"  ⚠️  {{pkg}}: {{err_msg}}")
        else:
            print(f"  ✓ {{pkg}}")
    
    print("\\n✅ Paketinstallation abgeschlossen")

if __name__ == "__main__":
    main()
'''
    install_script.write_text(script_content, encoding="utf-8")
    
    # Führe das Installationsskript mit dem SYSTEM-Python aus
    # (damit pip verfügbar ist)
    print("   📦 Installiere Pakete (mit Abhängigkeiten) ...")
    result = subprocess.run(
        [str(python_exe), str(install_script)],
        check=False, capture_output=True, text=True,
        cwd=python_dir,
    )
    print(result.stdout)
    if result.stderr:
        # Zeige nur relevante Fehler
        for line in result.stderr.splitlines():
            if "error" in line.lower() or "warning" in line.lower():
                print(f"   ⚠️  {line}")
    
    # Erstelle .pth-Datei
    _fix_pth_files(python_dir)
    
    # Erstelle eine kleine python_config.py, die die Pfade setzt
    # und von run_portable.py importiert werden kann
    config_script = python_dir / "_python_config.py"
    config_content = f'''# Portable Python Configuration
# This file is generated by build_portable.py
import sys
import os
from pathlib import Path

BASE = Path(__file__).parent.resolve()
SITE_PACKAGES = BASE / "Lib" / "site-packages"
PROJECT_ROOT = BASE.parent

# Add paths if not already present
for p in [SITE_PACKAGES, PROJECT_ROOT]:
    p_str = str(p)
    if p.exists() and p_str not in sys.path:
        sys.path.insert(0, p_str)
'''
    config_script.write_text(config_content, encoding="utf-8")
    print(f"   ✓ _python_config.py")
    
    # Aufräumen
    install_script.unlink()
    print(f"   🧹 Installationsskript aufgeräumt")


def _fix_pth_files(python_dir: Path) -> None:
    """Erstellt .pth-Dateien für die korrekten Import-Pfade."""
    pth_path = python_dir / "minecraft_app._pth"
    with open(pth_path, "w") as f:
        f.write("import site\n")
        f.write("Lib/site-packages\n")
        f.write("../../\n")
        f.write("../\n")
    print(f"   🔧 Erstelle {pth_path.name}")


def _install_packages(python_exe: Path) -> None:
    """Installiert alle benötigten Pakete in das embedded Python."""
    print("\n   📦 Installiere Pakete ...")
    
    for pkg in REQUIRED_PACKAGES:
        print(f"     → {pkg}")
        result = subprocess.run(
            [str(python_exe), "-m", "pip", "install",
             "--no-warn-script-location",
             pkg],
            check=False, capture_output=True, text=True,
            cwd=python_exe.parent,
        )
        if result.returncode != 0:
            print(f"       ⚠️  {result.stderr.splitlines()[-1] if result.stderr else 'Fehler'}")
    
    # Überprüfe Installation
    print("\n   🔍 Überprüfe Installation ...")
    result = subprocess.run(
        [str(python_exe), "-m", "pip", "list", "--format=json"],
        check=False, capture_output=True, text=True,
        cwd=python_exe.parent,
    )
    if result.returncode == 0:
        packages = json.loads(result.stdout)
        print(f"     ✓ {len(packages)} Pakete installiert")
        for pkg in packages:
            print(f"       • {pkg['name']}=={pkg['version']}")


def create_launcher_bat() -> None:
    """Erstellt die Haupt-Startdatei für die portable Distribution."""
    print("\n🪟 Erstelle Startdatei (Batch) ...")
    
    launcher_content = r"""@echo off
title Minecraft Structure Generator (Portable)
chcp 65001 >nul

REM ─── Portable Minecraft Structure Generator ─────────────────────────────
REM Alle Pfade sind RELATIV. Diese Batch-Datei kann von überall ausgeführt
REM werden, solange die Verzeichnisstruktur erhalten bleibt.
REM ─────────────────────────────────────────────────────────────────────────

setlocal enabledelayedexpansion

REM Basis-Verzeichnis = Verzeichnis dieser Batch-Datei
set "BASE=%~dp0"
cd /d "%BASE%"

REM Python und Bibliotheken im embedded/python Ordner
set "PYTHON_DIR=%BASE%python"

REM Prüfe, ob embedded Python existiert
if exist "%PYTHON_DIR%\python.exe" (
    set "PYTHON=%PYTHON_DIR%\python.exe"
) else (
    REM Fallback: System-Python
    where python >nul 2>&1
    if !errorlevel! equ 0 (
        echo [INFO] Verwende System-Python ...
        set "PYTHON=python"
    ) else (
        echo [FEHLER] Weder embedded noch System-Python gefunden!
        echo.
        echo Bitte laden Sie die portable Distribution vollstaendig herunter
        echo oder installieren Sie Python von https://www.python.org/
        echo.
        pause
        exit /b 1
    )
)

REM Prüfe auf CUDA-Verfügbarkeit
echo.
echo == Minecraft Structure Generator - Portable Edition ==
echo.
echo Starte Anwendung ...

REM Starte die GUI
"%PYTHON%" gui_app.py

REM Falls die GUI abstürzt, Fenster offen halten
if !errorlevel! neq 0 (
    echo.
    echo [FEHLER] Die Anwendung wurde mit Fehler !errorlevel! beendet.
    echo Bitte pruefen Sie die Konsole auf Fehlermeldungen.
    pause
)
"""
    
    launcher_path = DIST_DIR / "start_minecraft_generator.bat"
    launcher_path.write_text(launcher_content, encoding="utf-8")
    print(f"   ✓ {launcher_path.name}")


def create_python_loader_script() -> None:
    """
    Erstellt eine Python-Startskript-Datei, die sicherstellt, dass
    alle Import-Pfade korrekt gesetzt sind, bevor die eigentliche
    App geladen wird. Dies vermeidet Import-Probleme mit embedded Python.
    """
    loader_content = r'''"""
Minecraft Structure Generator – Portable Loader
===============================================
Stellt sicher, dass alle Import-Pfade korrekt gesetzt sind,
bevor die eigentliche App geladen wird. Unterstützt sowohl
embedded Python als auch System-Python.

Dieses Skript wird von start_minecraft_generator.bat aufgerufen
und ersetzt den direkten Aufruf von gui_app.py.
"""
import os
import sys
import site
from pathlib import Path

def setup_paths():
    """Stelle sicher, dass alle benötigten Pfade im sys.path sind."""
    base = Path(__file__).parent.resolve()
    
    # Wichtige Pfade, die durchsucht werden müssen
    paths_to_add = [
        base,                          # Hauptverzeichnis
        base / "Lib" / "site-packages", # embedded Python Lib
        base.parent,                    # (falls Python in Unterordner)
    ]
    
    for p in paths_to_add:
        p_str = str(p)
        if p.exists() and p_str not in sys.path:
            sys.path.insert(0, p_str)
    
    # Stelle sicher, dass das Projekt-Root in sys.path ist
    for candidate in [base, base.parent]:
        # Prüfe, ob gui_app.py in diesem Verzeichnis ist
        if (candidate / "gui_app.py").exists():
            c_str = str(candidate)
            if c_str not in sys.path:
                sys.path.insert(0, c_str)

def main():
    setup_paths()
    
    # Importe erst NACH Pfad-Setup
    try:
        from gui_app import main
        main()
    except ImportError as e:
        print(f"\n[FEHLER] Konnte App nicht laden: {e}")
        print(f"\nAktuelle sys.path:")
        for i, p in enumerate(sys.path):
            print(f"  {i}: {p}")
        print("\nInstallierte Pakete:")
        try:
            import pkg_resources
            for pkg in sorted(pkg_resources.working_set, key=lambda x: x.key):
                print(f"  {pkg.key}=={pkg.version}")
        except ImportError:
            print("  (pkg_resources nicht verfügbar)")
        input("\nDrücke Enter zum Beenden...")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
    loader_path = DIST_DIR / "run_portable.py"
    loader_path.write_text(loader_content, encoding="utf-8")
    print(f"   ✓ {loader_path.name}")


def create_readme() -> None:
    """Erstellt eine README für die portable Distribution."""
    readme_content = f"""# Minecraft Structure Generator – Portable Edition

## ⚡ Schnellstart
1. **Entpacken** Sie das gesamte Archiv
2. **Doppelklick** auf `start_minecraft_generator.bat`
3. **Warten** bis das Fenster erscheint (beim ersten Start werden
   ggf. Komponenten vorbereitet)
4. **Prompt eingeben** und auf *Generieren* klicken

## 📦 Systemvoraussetzungen
- **Betriebssystem:** Windows 10 oder höher (64-Bit)
- **Speicher:** ~4 GB freier Festplattenspeicher (nach Entpacken)
- **RAM:** 8 GB empfohlen
- **GPU (optional):** NVIDIA Grafikkarte mit CUDA-Unterstützung
  für schnellere Generierung

## 🧠 Enthaltene Komponenten
- **Python {get_python_version()}** (portabel, ohne System-Installation)
- **PyTorch** (GPU-beschleunigt falls CUDA verfügbar)
- **CustomTkinter** (moderne GUI)
- **Alle weiteren Abhängigkeiten**

## 📁 Verzeichnisstruktur
```
MinecraftStructureGenerator_Portable/
├── start_minecraft_generator.bat   ← STARTEN
├── run_portable.py                 ← Python-Loader (automatisch)
├── python/                         ← Embedded Python
│   ├── python.exe
│   └── ...
├── gui_app.py                      ← Hauptanwendung
├── dataset.py, model.py, ...       ← Quellcode
├── app/                            ← App-Module
├── runs/                           ← Trainierte Modelle
└── exports/                        ← Generierte Bauwerke
```

## 🎮 Verwendung
1. Wählen Sie ein Modell aus (Dropdown oben rechts)
2. Geben Sie eine Baubeschreibung ein (Englisch)
   - Z.B. "small medieval wooden cottage with stone foundation"
3. Temperatur und Top-K nach Wunsch anpassen
4. Auf **Generieren** klicken
5. Ergebnis als `.schem`-Datei speichern
   (mit WorldEdit oder Litematica in Minecraft laden)

## ⚙️ Erweiterte Optionen
- **Temperatur** (0.1–1.5): Höher = kreativer/unvorhersehbarer
- **Top-K** (5–100): Begrenzt die Auswahl auf die K wahrscheinlichsten Blöcke
- **Modell-Größe:** 16×16×16, 32×32×32 oder 48×48×48

## 🏗️ Bauwerke in Minecraft laden
1. Minecraft starten
2. WorldEdit oder Litematica installieren
3. `.schem`-Datei in den entsprechenden Ordner kopieren
4. Mit `//schem load dateiname.schem` laden

## 🐛 Fehlerbehebung
- **"python.exe nicht gefunden"** → Distribution nicht vollständig entpackt
- **"Kein Modell gefunden"** → runs/ Ordner fehlt (trainierte Modelle)
- **GUI startet nicht** → Terminal öffnen und `start_minecraft_generator.bat`
  darin ausführen, um Fehlermeldungen zu sehen

## 📄 Lizenz
MIT – siehe LICENSE-Datei im Hauptprojekt
"""
    
    readme_path = DIST_DIR / "README_PORTABLE.txt"
    readme_path.write_text(readme_content, encoding="utf-8")
    print(f"   ✓ {readme_path.name}")


def create_main_launcher() -> None:
    """Erstellt die verbesserte Batch-Datei, die den Python-Loader nutzt."""
    print("\n🪟 Erstelle Haupt-Startdatei ...")
    
    launcher_content = r"""@echo off
title Minecraft Structure Generator (Portable)
chcp 65001 >nul

REM ============================================================
REM  Minecraft Structure Generator – Portable Edition
REM  Alle Pfade sind relativ – kann von USB-Stick laufen
REM ============================================================

setlocal enabledelayedexpansion

REM ---- Basis-Verzeichnis ermitteln ----
set "BASE=%~dp0"
cd /d "%BASE%"

echo.
echo == Minecraft Structure Generator - Portable Edition ==
echo.

REM ---- Python finden ----
set "PYTHON="
if exist "%BASE%python\python.exe" (
    set "PYTHON=%BASE%python\python.exe"
    echo [OK] Embedded Python gefunden
) else (
    where python >nul 2>&1
    if not errorlevel 1 (
        set "PYTHON=python"
        echo [INFO] Verwende System-Python
    )
)

if not defined PYTHON (
    echo [FEHLER] Kein Python gefunden!
    echo.
    echo Bitte stellen Sie sicher, dass die portable Distribution
    echo vollstaendig entpackt wurde.
    echo.
    pause
    exit /b 1
)

REM ---- Umgebungsvariablen fuer portable Python ----
if exist "%BASE%python\" (
    set "PYTHONHOME=%BASE%python"
    set "PYTHONPATH=%BASE%python\Lib\site-packages;%BASE%"
)

REM ---- Python-Optimierungen ----
set "PYTHONDONTWRITEBYTECODE=1"
set "PYTHONIOENCODING=utf-8"
set "OMP_NUM_THREADS=4"

REM ---- Starte Anwendung ----
echo [START] Lade Anwendung ...
echo.

"%PYTHON%" run_portable.py

if errorlevel 1 (
    echo.
    echo [FEHLER] Die Anwendung wurde unerwartet beendet (Code: !errorlevel!).
    echo Druecke eine Taste, um das Fenster zu schliessen.
    pause
)
"""
    
    launcher_path = DIST_DIR / "start_minecraft_generator.bat"
    launcher_path.write_text(launcher_content, encoding="utf-8")
    print(f"   ✓ {launcher_path.name}")


def create_build_info() -> None:
    """Erstellt eine build_info.json mit Metadaten zur Distribution."""
    info = {
        "app_name": "Minecraft Structure Generator",
        "version": "2.1.0",
        "build_date": __import__("datetime").datetime.now().isoformat(),
        "python_version": get_python_version(),
        "architecture": get_arch(),
        "embedded_python": True,
        "cuda_support": __import__("torch").cuda.is_available(),
        "requires_gpu": False,
        "min_windows_version": "10",
        "dist_size_mb": 0,  # wird später gesetzt
    }
    
    info_path = DIST_DIR / "build_info.json"
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"   ✓ build_info.json")


def verify_distribution() -> list[str]:
    """Überprüft die Vollständigkeit der Distribution und gibt Warnungen zurück."""
    warnings = []
    
    # Prüfe essentielle Dateien
    essential_files = [
        "start_minecraft_generator.bat",
        "run_portable.py",
        "gui_app.py",
        "dataset.py",
        "model.py",
    ]
    
    for fname in essential_files:
        if not (DIST_DIR / fname).exists():
            warnings.append(f"❌ Fehlt: {fname}")
    
    # Prüfe Python
    python_exe = DIST_DIR / "python" / "python.exe"
    if not python_exe.exists():
        warnings.append("⚠️  Embedded Python nicht gefunden (python/python.exe)")
    
    # Prüfe runs/ Ordner
    runs_dir = DIST_DIR / "runs"
    if not runs_dir.exists() or not any(runs_dir.iterdir()):
        warnings.append("⚠️  Keine trainierten Modelle gefunden (runs/)")
    
    # Prüfe auf Modell-Checkpoints
    checkpoints = list(DIST_DIR.rglob("model.pt"))
    if not checkpoints:
        warnings.append("⚠️  Keine model.pt Checkpoints gefunden")
    
    # Berechne Größe
    total_size = sum(
        f.stat().st_size for f in DIST_DIR.rglob("*")
        if f.is_file()
    )
    size_mb = total_size / (1024 * 1024)
    warnings.append(f"📊 Gesamtgröße: {size_mb:.1f} MB")
    
    return warnings


def cleanup_temp() -> None:
    """Räumt temporäre Dateien auf."""
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
        print("\n🧹 Temporäre Dateien bereinigt")


def build() -> None:
    """Hauptfunktion: Baut die portable Distribution."""
    print("=" * 60)
    print("  Minecraft Structure Generator – Portable Builder")
    print("=" * 60)
    print(f"\n📦 Ziel: {DIST_DIR}")
    
    # Alte Distribution entfernen
    if DIST_DIR.exists():
        print("\n🗑️  Entferne alte Distribution ...")
        shutil.rmtree(DIST_DIR)
    
    # Verzeichnisse erstellen
    ensure_dir(DIST_DIR)
    ensure_dir(DIST_DIR / "exports")
    
    # 1. Projektdateien kopieren
    copy_project_files()
    
    # 2. Embedded Python einrichten
    print("\n🏗️  Richte portable Python-Umgebung ein ...")
    setup_embedded_python()
    
    # 3. Launcher erstellen
    create_main_launcher()
    create_python_loader_script()
    create_readme()
    create_build_info()
    
    # 4. Verifikation
    print("\n" + "=" * 60)
    print("  ✅ Distribution erstellt!")
    print("=" * 60)
    
    warnings = verify_distribution()
    for warning in warnings:
        if warning.startswith("📊"):
            print(warning)
        elif warning.startswith("❌"):
            print(f"\n{warning}")
        elif warning.startswith("⚠️"):
            print(f"\n{warning}")
    
    print(f"\n📂 Ausgabe: {DIST_DIR}")
    print("\n👉 Zum Starten: Doppelklick auf")
    print(f"   {DIST_DIR / 'start_minecraft_generator.bat'}")
    print()


if __name__ == "__main__":
    build()