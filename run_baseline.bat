@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Afruz PXRD environment is missing.
    echo Run: powershell -ExecutionPolicy Bypass -File .\bootstrap_windows.ps1
    exit /b 1
)

".venv\Scripts\python.exe" tools\capture_baseline.py
exit /b %errorlevel%

