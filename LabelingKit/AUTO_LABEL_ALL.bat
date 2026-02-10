@echo off
setlocal
cd /d "%~dp0"

echo.
echo ========================================
echo   Batch Auto-Label (YOLO)
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

%PYTHON_EXE% -c "import ultralytics" >nul 2>&1
if errorlevel 1 (
    echo [INFO] ultralytics belum terpasang.
    set /p INSTALL_ULTRA="Install ultralytics sekarang? (Y/N): "
    if /I "%INSTALL_ULTRA%"=="Y" (
        echo [INFO] Installing lightweight ultralytics...
        %PYTHON_EXE% -m pip install -qU ultralytics --no-deps
        if errorlevel 1 (
            echo [ERROR] Gagal install ultralytics.
            pause
            exit /b 1
        )
        %PYTHON_EXE% -c "import ultralytics" >nul 2>&1
        if errorlevel 1 (
            echo [ERROR] ultralytics terpasang tapi belum bisa dipakai.
            echo         Coba install dependency minimal ini:
            echo         %PYTHON_EXE% -m pip install torch torchvision
            pause
            exit /b 1
        )
    ) else (
        echo Dibatalkan.
        pause
        exit /b 0
    )
)

set "DEFAULT_IMG=Dataset"
set "IMG_DIR="
if not "%~1"=="" (
    set "IMG_DIR=%~1"
) else (
    set /p IMG_DIR="Folder gambar (Enter untuk default Dataset): "
)
if "%IMG_DIR%"=="" set "IMG_DIR=%DEFAULT_IMG%"

set "DEFAULT_MODEL=models\ffb_autolabel_stage1_seed42_best.pt"
if not exist "%DEFAULT_MODEL%" (
    if exist "ffb_autolabel_stage1_seed42_best.pt" (
        set "DEFAULT_MODEL=ffb_autolabel_stage1_seed42_best.pt"
    )
)
if exist "%DEFAULT_MODEL%" (
    set /p MODEL_PATH="Path model YOLO (.pt) [Enter = %DEFAULT_MODEL%]: "
    if "%MODEL_PATH%"=="" set "MODEL_PATH=%DEFAULT_MODEL%"
) else (
    set /p MODEL_PATH="Path model YOLO (.pt): "
)

if "%MODEL_PATH%"=="" (
    echo [ERROR] Path model tidak boleh kosong.
    pause
    exit /b 1
)

set /p CONF="Confidence (default 0.25): "
if "%CONF%"=="" set "CONF=0.25"

echo.
echo Menjalankan batch auto-label...
echo Folder : %IMG_DIR%
echo Model  : %MODEL_PATH%
echo Conf   : %CONF%
echo.

%PYTHON_EXE% _internal\scripts\batch_auto_label.py --images_dir "%IMG_DIR%" --model "%MODEL_PATH%" --conf %CONF% --recursive

echo.
echo Selesai. Buka folder di AnyLabeling untuk review hasil auto-label.
echo.
pause
