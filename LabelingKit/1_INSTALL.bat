@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo.
echo ========================================
echo   FFB Labeling Kit - Instalasi
echo ========================================
echo.
echo Proses ini akan memakan waktu 5-10 menit.
echo Pastikan komputer terhubung ke internet.
echo.
pause

set "PYTHON_DIR=python"
set "GET_PIP_URL=https://bootstrap.pypa.io/get-pip.py"
set "USE_PORTABLE=0"
set "REQUIREMENTS_FILE=_internal\requirements.txt"

:: ========================================
:: STEP 1: Setup Python
:: ========================================
echo.
echo [1/3] Menyiapkan Python...

:: Check if portable Python already setup
if exist "%PYTHON_DIR%\python.exe" (
    echo       Python portable sudah ada.
    set "USE_PORTABLE=1"
    set "PYTHON_EXE=%PYTHON_DIR%\python.exe"
    goto :install_deps
)

:: Check for existing zip file
set "PYTHON_ZIP="
for %%f in (python-*-embed-amd64.zip) do set "PYTHON_ZIP=%%f"

if defined PYTHON_ZIP (
    echo       Mengekstrak %PYTHON_ZIP%...
    if not exist "%PYTHON_DIR%" mkdir "%PYTHON_DIR%"
    powershell -Command "Expand-Archive -Path '%PYTHON_ZIP%' -DestinationPath '%PYTHON_DIR%' -Force"

    :: Enable pip support
    for %%f in (%PYTHON_DIR%\python*._pth) do (
        powershell -Command "(Get-Content '%%f') -replace '#import site', 'import site' | Set-Content '%%f'"
    )

    :: Install pip
    echo       Menginstall pip...
    powershell -Command "& {$ProgressPreference='SilentlyContinue'; Invoke-WebRequest -Uri '%GET_PIP_URL%' -OutFile '%PYTHON_DIR%\get-pip.py'}" 2>nul
    %PYTHON_DIR%\python.exe %PYTHON_DIR%\get-pip.py --quiet 2>nul
    del "%PYTHON_DIR%\get-pip.py" 2>nul

    set "USE_PORTABLE=1"
    set "PYTHON_EXE=%PYTHON_DIR%\python.exe"
    goto :install_deps
)

:: Try system Python
python --version >nul 2>&1
if not errorlevel 1 (
    echo       Menggunakan Python sistem...
    set "PYTHON_EXE=python"
    goto :install_deps
)

:: No Python found
echo [ERROR] Python tidak ditemukan!
echo.
echo Download Python dari:
echo   https://www.python.org/ftp/python/3.12.10/python-3.12.10-embed-amd64.zip
echo.
echo Simpan file ZIP di folder ini, lalu jalankan 1_INSTALL.bat lagi.
echo.
pause
exit /b 1

:install_deps
:: ========================================
:: STEP 2: Install Dependencies
:: ========================================
echo [2/3] Menginstall dependencies...
echo       (AnyLabeling, OpenCV, dll - mohon tunggu...)

if "%USE_PORTABLE%"=="1" (
    %PYTHON_DIR%\python.exe -m pip install --upgrade pip --quiet 2>nul
    %PYTHON_DIR%\python.exe -m pip install -r "%REQUIREMENTS_FILE%" --quiet
) else (
    python -m pip install --upgrade pip --quiet 2>nul
    pip install -r "%REQUIREMENTS_FILE%" --quiet
)

if errorlevel 1 (
    echo [ERROR] Gagal install dependencies
    echo         Pastikan koneksi internet stabil.
    pause
    exit /b 1
)

echo       Menerapkan hotfix AnyLabeling...
%PYTHON_EXE% _internal\scripts\fix_anylabeling_colormap.py >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Hotfix AnyLabeling gagal dijalankan.
    echo          Instalasi tetap lanjut, coba 2_START_LABELING.bat dulu.
)

:: Verify anylabeling module is importable
%PYTHON_EXE% -c "import anylabeling" >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Module anylabeling tidak ditemukan.
    echo          Coba jalankan ulang 1_INSTALL.bat.
)

:: ========================================
:: STEP 3: Create Output Folder
:: ========================================
echo [3/3] Menyiapkan folder output...
if not exist "output" mkdir "output"

echo.
echo ========================================
echo   Instalasi Selesai!
echo ========================================
echo.
echo Langkah selanjutnya:
echo   1. Klik 2x: 2_START_LABELING.bat
echo   2. Pilih folder kelompok kamu di Dataset\
echo.
echo Tools opsional:
echo   - tools\CHECK_PROGRESS.bat
echo   - tools\CONVERT_TO_YOLO.bat
echo   - tools\EXTRACT_VIDEO_FRAMES.bat
echo.
pause
