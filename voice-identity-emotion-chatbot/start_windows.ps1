param(
    [ValidateSet('all', 'showcase', 'training')][string]$Mode = 'all',
    [string]$WorkspaceRoot
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$python = Join-Path $root '.venv\Scripts\python.exe'
if (!(Test-Path -LiteralPath $python)) { throw 'Run setup_windows.bat first.' }
if (!(Get-Command ffmpeg -ErrorAction SilentlyContinue) -and
    !(Test-Path -LiteralPath (Join-Path $root 'tools\ffmpeg.exe'))) {
    throw 'FFmpeg is missing. Add its bin directory to PATH or read tools/README.md.'
}

$ports = switch ($Mode) {
    'all' { 7860; 7862 }
    'showcase' { 7862 }
    'training' { 7860 }
}
foreach ($port in $ports) {
    $listener = [Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback, $port)
    try {
        $listener.Start()
    } catch {
        throw "Port $port is unavailable. Close the existing server before starting another. No process was stopped."
    } finally {
        $listener.Stop()
    }
}

if (!$WorkspaceRoot) { $WorkspaceRoot = $env:VOICE_PROJECT_WORKSPACE }
if (!$WorkspaceRoot) { $WorkspaceRoot = Join-Path $root 'workspace' }
$env:VOICE_PROJECT_WORKSPACE = [IO.Path]::GetFullPath($WorkspaceRoot)
$env:PYTHONPATH = Join-Path $root 'src'
$env:MPLCONFIGDIR = Join-Path $root '.matplotlib'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUNBUFFERED = '1'
$env:GRADIO_ANALYTICS_ENABLED = 'False'
$entry = switch ($Mode) {
    'all' { 'run_competition_demo.py' }
    'showcase' { 'run_showcase_demo.py' }
    'training' { 'run_personal_demo.py' }
}

Write-Host "Local data: $env:VOICE_PROJECT_WORKSPACE"
Write-Host 'Showcase (when enabled): http://127.0.0.1:7862/'
Write-Host 'Training (when enabled): http://127.0.0.1:7860/'
Write-Host 'Keep this console open. Press Ctrl+C to stop the server.'
Push-Location $root
try {
    & $python (Join-Path $root "scripts\$entry")
    if ($LASTEXITCODE -ne 0) { throw "Server exited with code $LASTEXITCODE." }
} finally {
    Pop-Location
}
