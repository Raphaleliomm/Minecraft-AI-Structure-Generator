@echo off
title Minecraft Structure Generator v2.0
cd /d "%~dp0"
echo ============================================
echo   Minecraft Structure Generator v2.0
echo   Starte Anwendung...
echo ============================================
echo.
python -m app.main_app
if %errorlevel% neq 0 (
    echo.
    echo Fehler beim Starten. Installiere Abhaengigkeiten...
    pip install -r requirements.txt
    echo.
    python -m app.main_app
)
pause