@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_windows.ps1"
set "RESULT=%ERRORLEVEL%"
if not "%RESULT%"=="0" echo Startup failed. Read the error above.
pause
exit /b %RESULT%
