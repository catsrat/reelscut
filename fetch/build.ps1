# Build fetch\dist\ReelscutFetch.exe (Windows, Python 3.10+ on PATH).
# GitHub builds and publishes it automatically: .github/workflows/fetch.yml
# (Exit codes are checked by hand: Windows PowerShell 5 turns any stderr
# output of pip/PyInstaller into an error, even harmless notices.)
Set-Location $PSScriptRoot
if (-not (Test-Path .venv-build)) {
    python -m venv .venv-build
    if ($LASTEXITCODE) { exit $LASTEXITCODE }
}
.\.venv-build\Scripts\python -m pip install -q --disable-pip-version-check -r requirements-build.txt
if ($LASTEXITCODE) { exit $LASTEXITCODE }
.\.venv-build\Scripts\pyinstaller --noconfirm --clean --log-level WARN --onefile --windowed `
    --name ReelscutFetch --icon icon.ico --add-data "icon.ico;." reelscut_fetch.py
if ($LASTEXITCODE) { exit $LASTEXITCODE }
Get-Item dist\ReelscutFetch.exe | Select-Object Name, Length
