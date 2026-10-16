param([string]$PythonPath)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$python = $PythonPath
if (!$python -and (Get-Command py -ErrorAction SilentlyContinue)) {
    $candidate = & py -3.12 -c "import sys; print(sys.executable)" 2>$null
    if ($LASTEXITCODE -eq 0) { $python = ($candidate | Select-Object -Last 1).Trim() }
}
if (!$python -and (Get-Command python -ErrorAction SilentlyContinue)) {
    $candidate = & python -c "import sys; print(sys.executable)" 2>$null
    if ($LASTEXITCODE -eq 0) { $python = ($candidate | Select-Object -Last 1).Trim() }
}
if (!$python) { throw 'Python 3.12 was not found. Install it or pass -PythonPath.' }
& $python -c "import sys; assert sys.version_info[:2] == (3, 12), 'Use Python 3.12 for these pinned dependencies.'"
if ($LASTEXITCODE -ne 0) { throw 'The selected Python must be version 3.12.' }

$venvPython = Join-Path $root '.venv\Scripts\python.exe'
if (!(Test-Path -LiteralPath $venvPython)) {
    & $python -m venv (Join-Path $root '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the virtual environment.' }
}
& $venvPython -c "import sys; assert sys.version_info[:2] == (3, 12)"
if ($LASTEXITCODE -ne 0) { throw 'The existing .venv does not use Python 3.12.' }
& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'Could not update pip.' }
& $venvPython -m pip install -r (Join-Path $root 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }

if (!(Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Warning 'Install FFmpeg and add its bin directory to PATH before recording audio.'
}
if (!(Get-Command ollama -ErrorAction SilentlyContinue)) {
    Write-Warning 'Install and start Ollama before using chat replies.'
}
Write-Host 'Python dependencies installed.'
Write-Host 'Next: ollama pull gemma4:12b'
Write-Host 'Then run start_windows.bat. See README.md for enrollment and training.'
