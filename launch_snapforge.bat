@echo off
title SnapForge - Snapdragon AI Optimization Studio
cd /d "%~dp0"
set PYTHONPATH=%~dp0
"C:\Users\Vatsal Kumar\AppData\Local\Python\pythoncore-3.14-64\python.exe" run.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo SnapForge exited with error code %ERRORLEVEL%.
    pause
)
