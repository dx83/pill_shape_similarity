@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" crop_pills.py %*
    exit /b %errorlevel%
)

echo Python environment is not installed. Run setup.cmd first. 1>&2
exit /b 1
