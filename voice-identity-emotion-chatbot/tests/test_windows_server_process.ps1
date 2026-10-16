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
Write-Host '4 server-ownership checks passed.'
