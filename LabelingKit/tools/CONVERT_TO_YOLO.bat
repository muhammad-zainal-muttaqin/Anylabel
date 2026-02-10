@echo off
setlocal
cd /d "%~dp0\.."

echo ============================================
echo   Convert LabelMe JSON to YOLO TXT
echo ============================================
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

set "INPUT_DIR=Dataset"
set /p INPUT_DIR="Folder input JSON [Enter=Dataset]: "
if "%INPUT_DIR%"=="" set "INPUT_DIR=Dataset"

set "OUTPUT_DIR=output\yolo"
set /p OUTPUT_DIR="Folder output YOLO [Enter=output\yolo]: "
if "%OUTPUT_DIR%"=="" set "OUTPUT_DIR=output\yolo"

echo.
%PYTHON_EXE% _internal\scripts\convert_json_to_yolo.py --input "%INPUT_DIR%" --output "%OUTPUT_DIR%" --recursive --preserve_structure
echo.
pause
