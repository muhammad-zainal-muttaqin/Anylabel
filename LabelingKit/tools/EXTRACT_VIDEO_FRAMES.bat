@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0\.."

echo ============================================
echo   Extract Video to Image Frames
echo ============================================
echo.

set "PYTHON_EXE="
if exist "python\python.exe" (
    set "PYTHON_EXE=python\python.exe"
) else (
    where python >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Python tidak ditemukan.
        echo         Jalankan 1_INSTALL.bat dulu.
        pause
        exit /b 1
    )
    set "PYTHON_EXE=python"
)

set "INPUT_PATH=Dataset\Video"
set /p INPUT_PATH="Input video/file folder [Enter=Dataset\Video]: "
if "%INPUT_PATH%"=="" set "INPUT_PATH=Dataset\Video"

set "OUTPUT_BASE=Dataset\Video"
set /p OUTPUT_BASE="Output base folder [Enter=Dataset\Video]: "
if "%OUTPUT_BASE%"=="" set "OUTPUT_BASE=Dataset\Video"

set "FPS=2"
set /p FPS="Target FPS extraction [Enter=2]: "
if "%FPS%"=="" set "FPS=2"

set "IMG_EXT=jpg"
set /p IMG_EXT="Image extension (jpg/png/webp) [Enter=jpg]: "
if "%IMG_EXT%"=="" set "IMG_EXT=jpg"

set "QUALITY=95"
set /p QUALITY="Image quality 1-100 [Enter=95]: "
if "%QUALITY%"=="" set "QUALITY=95"

echo.
choice /C YN /N /M "Scan recursive subfolder? (Y/N): "
if errorlevel 2 (set "RECURSIVE=") else (set "RECURSIVE=--recursive")

echo.
choice /C YN /N /M "Overwrite images lama per video? (Y/N): "
if errorlevel 2 (set "OVERWRITE=") else (set "OVERWRITE=--overwrite")

echo.
echo Menjalankan ekstraksi frame...
echo Input  : %INPUT_PATH%
echo Output : %OUTPUT_BASE%
echo FPS    : %FPS%
echo Ext    : %IMG_EXT%
echo.

%PYTHON_EXE% _internal\scripts\extract_video_frames.py --input "%INPUT_PATH%" --output "%OUTPUT_BASE%" --fps %FPS% --image_ext %IMG_EXT% --quality %QUALITY% %RECURSIVE% %OVERWRITE%

echo.
echo Selesai. Buka folder hasil: Dataset\Video\[nama_video]\images
echo Lalu jalankan 2_START_LABELING.bat untuk labeling/review.
echo.
pause
