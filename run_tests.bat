@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Afruz PXRD development environment is missing.
    echo Run: powershell -ExecutionPolicy Bypass -File .\bootstrap_windows.ps1 -Development
    exit /b 1
)

set QT_QPA_PLATFORM=offscreen
set MPLBACKEND=Agg
".venv\Scripts\python.exe" -m pytest --cov=afruz_pxrd --cov-report=term-missing --cov-report=html:test_artifacts/coverage --junitxml=test_artifacts/pytest-results.xml %*
exit /b %errorlevel%
