function Test-ProjectServerProcess {
    param($Process, $Parent, [string]$PythonPath, [switch]$AllowAugmentation)

    if (!$Process -or [IO.Path]::GetFileName($Process.ExecutablePath) -ne 'python.exe') {
        return $false
    }
    $projectLauncher = $Process.ExecutablePath -eq $PythonPath -or
        ($Parent -and $Parent.ExecutablePath -eq $PythonPath)
    $demoCommand = '(?:^|[\\/\s"])run_(?:competition|personal|showcase|integrated)_demo\.py(?:["\s]|$)'
    $runsDemo = $Process.CommandLine -match $demoCommand -or
        ($Parent -and $Parent.CommandLine -match $demoCommand)
    if ($AllowAugmentation) {
        $augmentationCommand = '(?:^|[\\/\s"])audio_augmentation_tool\.py(?:["\s]|$)'
        $runsDemo = $runsDemo -or $Process.CommandLine -match $augmentationCommand -or
            ($Parent -and $Parent.CommandLine -match $augmentationCommand)
    }
    return [bool]($projectLauncher -and $runsDemo)
}
