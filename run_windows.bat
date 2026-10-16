@echo off
setlocal
cd /d "%~dp0"

set "AFRUZ_VENV_PYTHON=.venv\Scripts\python.exe"

if not exist "%AFRUZ_VENV_PYTHON%" (
    echo DIFFRUX is not configured yet.
    echo.
    echo Create the locked Python 3.12 environment with:
    echo   powershell -ExecutionPolicy Bypass -File .\bootstrap_windows.ps1
    exit /b 1
)

if not exist "main.py" (
    echo DIFFRUX cannot start because main.py is missing from:
    echo   %CD%
    echo Restore the complete application folder and try again.
    exit /b 1
)

"%AFRUZ_VENV_PYTHON%" -c "import sys; assert sys.version_info[:2] == (3, 12), 'DIFFRUX requires Python 3.12'; import PySide6, pyqtgraph, numpy, scipy, pandas, openpyxl, matplotlib, reportlab; from afruz_pxrd.version import APP_RELEASE; print(f'Starting {APP_RELEASE}...')"
if errorlevel 1 (
    echo.
    echo DIFFRUX startup checks failed. Repair the locked environment with:
    echo   powershell -ExecutionPolicy Bypass -File .\bootstrap_windows.ps1
    exit /b 2
)

"%AFRUZ_VENV_PYTHON%" main.py
set "AFRUZ_EXIT_CODE=%errorlevel%"
if not "%AFRUZ_EXIT_CODE%"=="0" (
    echo.
    echo DIFFRUX exited with code %AFRUZ_EXIT_CODE%.
    echo If the problem repeats, run .\run_tests.bat and preserve the terminal output.
)
exit /b %AFRUZ_EXIT_CODE%
