# One-time setup for Windows PowerShell: creates .venv and installs dependencies.
# Run from the project folder:   powershell -ExecutionPolicy Bypass -File .\setup.ps1
# (-ExecutionPolicy Bypass applies to this one command only; it does not change system settings.)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "Python was not found. Install Python 3.12 from https://www.python.org/downloads/ and tick 'Add python.exe to PATH'." -ForegroundColor Red
    exit 1
}

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Creating virtual environment (.venv)..." -ForegroundColor Cyan
    python -m venv .venv
}

# Calling the venv's python directly means no activation (and no execution-policy change) is needed.
$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
Write-Host "Upgrading pip..." -ForegroundColor Cyan
& $venvPython -m pip install --upgrade pip
Write-Host "Installing dependencies from requirements.txt..." -ForegroundColor Cyan
& $venvPython -m pip install -r requirements.txt

Write-Host "Checking imports..." -ForegroundColor Cyan
& $venvPython -c "import streamlit, pandas, numpy, plotly; print('streamlit', streamlit.__version__, '| pandas', pandas.__version__, '| plotly', plotly.__version__)"

Write-Host ""
Write-Host "Setup complete. Start the dashboard with:  powershell -ExecutionPolicy Bypass -File .\run.ps1" -ForegroundColor Green
