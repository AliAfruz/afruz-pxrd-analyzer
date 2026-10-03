param(
    [switch]$Development
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$lockFile = Join-Path $projectRoot "requirements-lock.txt"
$developmentLockFile = Join-Path $projectRoot "requirements-dev-lock.txt"

function Test-Python312 {
    param([string]$Executable)
    if (-not $Executable -or -not (Test-Path -LiteralPath $Executable)) {
        return $false
    }
    try {
        $version = & $Executable -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
        return $version -eq "3.12"
    }
    catch {
        return $false
    }
}

if (-not (Test-Path -LiteralPath $lockFile)) {
    throw "Dependency lock file is missing: $lockFile"
}
if ($Development -and -not (Test-Path -LiteralPath $developmentLockFile)) {
    throw "Development dependency lock file is missing: $developmentLockFile"
}

if (-not (Test-Path -LiteralPath $venvPython)) {
    $candidates = @()
    if ($env:AFRUZ_PYTHON) {
        $candidates += $env:AFRUZ_PYTHON
    }
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCommand) {
        $candidates += $pythonCommand.Source
    }
    $candidates += @(
        (Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"),
        "C:\Program Files\Python312\python.exe",
        "C:\Python312\python.exe",
        (Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe")
    )

    $basePython = $null
    foreach ($candidate in $candidates | Select-Object -Unique) {
        if (Test-Python312 -Executable $candidate) {
            $basePython = $candidate
            break
        }
    }
    if (-not $basePython) {
        throw "Python 3.12 was not found. Install Python 3.12 or set AFRUZ_PYTHON to its python.exe path."
    }

    Write-Host "Creating Python 3.12 environment with $basePython"
    & $basePython -m venv (Join-Path $projectRoot ".venv")
}

$selectedLockFile = if ($Development) { $developmentLockFile } else { $lockFile }
$environmentLabel = if ($Development) { "development/test" } else { "runtime" }
Write-Host "Installing the locked Afruz PXRD $environmentLabel environment..."
& $venvPython -m pip install --disable-pip-version-check -r $selectedLockFile
if ($LASTEXITCODE -ne 0) {
    throw "Dependency installation failed with exit code $LASTEXITCODE."
}

if ($Development) {
    Write-Host "Development environment ready. Run tests with: .\run_tests.bat"
}
else {
    Write-Host "Environment ready. Start the application with: .\run_windows.bat"
}
