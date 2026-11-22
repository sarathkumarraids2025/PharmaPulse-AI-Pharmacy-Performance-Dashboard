# Runs the automated tests with the project's virtual environment.
Set-Location -Path $PSScriptRoot
$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) { Write-Host "Run setup.ps1 first." -ForegroundColor Red; exit 1 }
& $venvPython -m unittest discover -s tests -v
