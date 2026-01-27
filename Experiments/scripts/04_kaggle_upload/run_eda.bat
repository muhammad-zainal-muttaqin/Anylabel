@echo off
REM Run simple_eda.py with venv environment
cd /d "%~dp0"
set SCRIPT_DIR=%~dp0
set PROJECT_DIR=%SCRIPT_DIR%..\..
set VENV_PYTHON=%PROJECT_DIR%\venv\Scripts\python.exe

if not exist "%VENV_PYTHON%" (
    echo ERROR: Python not found at %VENV_PYTHON%
    pause
    exit /b 1
)

"%VENV_PYTHON%" simple_eda.py
echo.
echo [Selesai] Script EDA telah dijalankan
pause
