$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
foreach ($relative in @('setup_windows.ps1', 'start_windows.ps1')) {
    $tokens = $null
    $parseErrors = $null
    [void][Management.Automation.Language.Parser]::ParseFile(
        (Join-Path $root $relative), [ref]$tokens, [ref]$parseErrors
    )
    if ($parseErrors.Count -gt 0) { throw ($parseErrors | Out-String) }
}

# A busy port must fail before attempting to run Python or stop another process.
$sandbox = Join-Path ([IO.Path]::GetTempPath()) ('voice-launch-test-' + [guid]::NewGuid())
New-Item -ItemType Directory -Path (Join-Path $sandbox '.venv\Scripts') -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $sandbox 'tools') -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $root 'start_windows.ps1') -Destination $sandbox
New-Item -ItemType File -Path (Join-Path $sandbox '.venv\Scripts\python.exe') | Out-Null
New-Item -ItemType File -Path (Join-Path $sandbox 'tools\ffmpeg.exe') | Out-Null
$listener = [Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback, 7862)
try {
    try { $listener.Start() } catch {
        Write-Host 'Port 7862 already busy; checking the existing listener is not stopped.'
    }
    $rejected = $false
    try { & (Join-Path $sandbox 'start_windows.ps1') -Mode showcase } catch {
        if ($_.Exception.Message -notmatch 'Port 7862 is unavailable') { throw }
        $rejected = $true
    }
    if (!$rejected) { throw 'A busy port was not rejected.' }
} finally {
    $listener.Stop()
    $resolved = [IO.Path]::GetFullPath($sandbox)
    $temp = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\') + '\'
    if (!$resolved.StartsWith($temp, [StringComparison]::OrdinalIgnoreCase) -or
        [IO.Path]::GetFileName($resolved) -notlike 'voice-launch-test-*') {
        throw 'Refusing to remove a test directory outside the temp workspace.'
    }
    Remove-Item -LiteralPath $resolved -Recurse -Force
}
Write-Host 'Launch scripts parsed; busy port rejected without stopping processes.'
