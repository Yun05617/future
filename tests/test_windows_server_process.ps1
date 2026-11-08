. (Join-Path $PSScriptRoot '..\scripts\windows_server_process.ps1')
$pythonPath = 'C:\project\.venv\Scripts\python.exe'
$launcher = [pscustomobject]@{
    ExecutablePath = $pythonPath
    CommandLine = '"C:\project\.venv\Scripts\python.exe" scripts\run_competition_demo.py'
}
$child = [pscustomobject]@{
    ExecutablePath = 'C:\Python312\python.exe'
    CommandLine = '"C:\project\.venv\Scripts\python.exe" scripts\run_competition_demo.py'
}
if (!(Test-ProjectServerProcess $child $launcher $pythonPath)) {
    throw 'The virtual-environment child server must be recognized.'
}
if (!(Test-ProjectServerProcess $launcher $null $pythonPath)) {
    throw 'The direct project server must be recognized.'
}
if (Test-ProjectServerProcess $child $null $pythonPath) {
    throw 'A system Python without the project parent must not be stopped.'
}
$unrelated = [pscustomobject]@{ ExecutablePath = $pythonPath; CommandLine = 'python.exe unrelated.py' }
if (Test-ProjectServerProcess $unrelated $null $pythonPath) {
    throw 'An unrelated program in the same environment must not be stopped.'
}
$augmentation = [pscustomobject]@{ ExecutablePath = $pythonPath; CommandLine = 'python.exe scripts\audio_augmentation_tool.py --no-browser' }
if (Test-ProjectServerProcess $augmentation $null $pythonPath) {
    throw 'The augmentation exception must be opt-in.'
}
if (!(Test-ProjectServerProcess $augmentation $null $pythonPath -AllowAugmentation)) {
    throw 'The dedicated augmentation tool must be recognized on its own port.'
}
if (Test-ProjectServerProcess $unrelated $null $pythonPath -AllowAugmentation) {
    throw 'The augmentation exception must not allow unrelated Python scripts.'
}
Write-Host '7 server-ownership checks passed.'
