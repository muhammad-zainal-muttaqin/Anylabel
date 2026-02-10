@echo off
setlocal
cd /d "%~dp0\.."

echo ========================================
echo   Labeling Progress Checker
echo ========================================
echo.

set "PYTHON_EXE="
if exist "python\python.exe" (
    set "PYTHON_EXE=python\python.exe"
) else (
    where python >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Python tidak ditemukan.
        echo         Jalankan INSTALL.bat dulu.
        pause
        exit /b 1
    )
    set "PYTHON_EXE=python"
)

set "TARGET_DIR=Dataset"
set /p TARGET_DIR="Folder target [Enter=Dataset]: "
if "%TARGET_DIR%"=="" set "TARGET_DIR=Dataset"

echo.
%PYTHON_EXE% _internal\scripts\verify_labels.py --images "%TARGET_DIR%" --labels "%TARGET_DIR%" --recursive
echo.
%PYTHON_EXE% _internal\scripts\label_stats.py --labels "%TARGET_DIR%" --recursive
echo.
pause
