@echo off
setlocal enabledelayedexpansion

echo ============================================
echo   Build LabelingKit.zip
echo ============================================
echo.

set "SOURCE_DIR=LabelingKit"
set "OUTPUT_ZIP=LabelingKit.zip"

:: Check if LabelingKit folder exists
if not exist "%SOURCE_DIR%" (
    echo [ERROR] LabelingKit folder not found
    pause
    exit /b 1
)

:: Remove old zip if exists
if exist "%OUTPUT_ZIP%" (
    echo Removing old %OUTPUT_ZIP%...
    del "%OUTPUT_ZIP%"
)

:: Remove folders that shouldn't be in the zip (will be created on setup)
if exist "%SOURCE_DIR%\venv" (
    echo Removing venv from package...
    rmdir /s /q "%SOURCE_DIR%\venv"
)
if exist "%SOURCE_DIR%\python" (
    echo Removing portable python from package...
    rmdir /s /q "%SOURCE_DIR%\python"
)
if exist "%SOURCE_DIR%\output" (
    echo Removing output from package...
    rmdir /s /q "%SOURCE_DIR%\output"
)

echo.
echo Creating ZIP archive...
echo.

:: Use PowerShell to create zip
powershell -Command "Compress-Archive -Path '%SOURCE_DIR%\*' -DestinationPath '%OUTPUT_ZIP%' -Force"

if exist "%OUTPUT_ZIP%" (
    echo.
    echo ============================================
    echo   SUCCESS!
    echo ============================================
    echo.
    echo Created: %OUTPUT_ZIP%
    echo.
    for %%A in ("%OUTPUT_ZIP%") do echo Size: %%~zA bytes
    echo.
    echo Transfer this ZIP to another computer and:
    echo   1. Extract the ZIP
    echo   2. Run 1_INSTALL.bat
    echo   3. Run 2_START_LABELING.bat
    echo   4. Optional: AUTO_LABEL_ALL (opsional).bat
    echo.
) else (
    echo [ERROR] Failed to create ZIP
)

pause
