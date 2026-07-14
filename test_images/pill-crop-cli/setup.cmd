@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel% equ 0 (
    set "PYTHON=py -3"
) else (
    set "PYTHON=python"
)

echo Creating Python environment...
%PYTHON% -m venv .venv || exit /b 1

echo Installing dependencies...
".venv\Scripts\python.exe" -m pip install --upgrade pip || exit /b 1
".venv\Scripts\python.exe" -m pip install -r requirements.txt || exit /b 1

echo.
echo Setup complete. Run run.cmd --help for usage.
