function Test-ProjectServerProcess {
    param($Process, $Parent, [string]$PythonPath)

    if (!$Process -or [IO.Path]::GetFileName($Process.ExecutablePath) -ne 'python.exe') {
        return $false
    }
    $projectLauncher = $Process.ExecutablePath -eq $PythonPath -or
        ($Parent -and $Parent.ExecutablePath -eq $PythonPath)
    $demoCommand = '(?:^|[\\/\s"])run_(?:competition|personal|showcase|integrated)_demo\.py(?:["\s]|$)'
    $runsDemo = $Process.CommandLine -match $demoCommand -or
        ($Parent -and $Parent.CommandLine -match $demoCommand)
    return [bool]($projectLauncher -and $runsDemo)
}
