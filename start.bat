@echo off
title Medanta Hospital - Cash & Service Counter Operations Monitor
echo ======================================================================
echo   Medanta Hospital - Cash & Service Counter Operations Monitor
echo   Opening at: http://localhost:8195
echo ======================================================================
cd /d "%~dp0backend"
python counter_tracker.py
pause
