@echo off
title Minecraft Trainingsdaten Scraper
cd /d "%~dp0"
echo ============================================
echo   Minecraft Scraper - alle Quellen parallel
echo   Resume-faehig; Abbruch mit Strg+C
echo ============================================
echo.
python -u scrape_structures.py --source all --workers 3 --delay 0.5 --jitter 0.2 --max-file-mb 0 --max-voxels 0
echo.
echo Lauf beendet. Neustart setzt fort.
pause
