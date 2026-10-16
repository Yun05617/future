@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_windows.ps1"
set "RESULT=%ERRORLEVEL%"
if not "%RESULT%"=="0" echo Setup failed. Read the error above.
pause
exit /b %RESULT%
