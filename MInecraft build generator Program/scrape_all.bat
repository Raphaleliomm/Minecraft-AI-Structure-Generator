@echo off
title PhantomMarket Trainingsdaten-Scraper
cd /d "%~dp0"
echo ============================================
echo   PhantomMarket Scraper - Volllauf
echo   Build-Beschreibung = Prompt
echo   Ziel: ..\Trainingsdaten\phantommarket
echo   Abbruch mit Strg+C - Neustart macht weiter (Resume)
echo ============================================
echo.
python -u scrape_phantommarket.py --delay 1.0 --jitter 0.4 --max-file-mb 600
echo.
echo Lauf beendet (oder abgebrochen). Ein Neustart setzt fort,
echo bereits geladene Builds werden uebersprungen.
pause