# Starts the dashboard using the project's virtual environment.
# Run from the project folder:   powershell -ExecutionPolicy Bypass -File .\run.ps1

Set-Location -Path $PSScriptRoot
$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "Virtual environment not found. Run setup.ps1 first." -ForegroundColor Red
    exit 1
}
& $venvPython -m streamlit run app.py
