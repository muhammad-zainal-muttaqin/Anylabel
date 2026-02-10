@echo off
setlocal enabledelayedexpansion

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

:: ========================================
:: STEP 1: Setup Python
:: ========================================
echo.
echo [1/4] Menyiapkan Python...

if exist "%PYTHON_DIR%\python.exe" (
    echo       Python sudah ada, skip...
    goto :setup_venv
)

:: Check for existing zip file
set "PYTHON_ZIP="
for %%f in (python-*-embed-amd64.zip) do set "PYTHON_ZIP=%%f"

if not defined PYTHON_ZIP (
    :: Try to use system Python
    python --version >nul 2>&1
    if not errorlevel 1 (
        echo       Menggunakan Python sistem...
        goto :setup_venv_system
    )
    
    echo [ERROR] Python tidak ditemukan!
    echo.
    echo Download Python dari:
    echo   https://www.python.org/ftp/python/3.12.10/python-3.12.10-embed-amd64.zip
    echo.
    echo Simpan file ZIP di folder ini, lalu jalankan INSTALL.bat lagi.
    echo.
    pause
    exit /b 1
)

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

:setup_venv
:: ========================================
:: STEP 2: Create Virtual Environment
:: ========================================
echo [2/4] Membuat virtual environment...

if exist "venv\Scripts\python.exe" (
    echo       venv sudah ada, skip...
    goto :install_deps
)

%PYTHON_DIR%\python.exe -m venv venv
if errorlevel 1 (
    echo [ERROR] Gagal membuat virtual environment
    pause
    exit /b 1
)
goto :install_deps

:setup_venv_system
:: Using system Python
echo [2/4] Membuat virtual environment...
if exist "venv\Scripts\python.exe" (
    echo       venv sudah ada, skip...
    goto :install_deps
)
python -m venv venv
if errorlevel 1 (
    echo [ERROR] Gagal membuat virtual environment
    pause
    exit /b 1
)

:install_deps
:: ========================================
:: STEP 3: Install Dependencies
:: ========================================
echo [3/4] Menginstall dependencies...
echo       (AnyLabeling, OpenCV, dll - mohon tunggu...)

call venv\Scripts\activate.bat
python -m pip install --upgrade pip --quiet 2>nul
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo [ERROR] Gagal install dependencies
    pause
    exit /b 1
)

:: ========================================
:: STEP 4: Create Output Folder
:: ========================================
echo [4/4] Menyiapkan folder output...

if not exist "output" mkdir "output"

:: ========================================
:: DONE
:: ========================================
echo.
echo ========================================
echo   Instalasi Selesai!
echo ========================================
echo.
echo Langkah selanjutnya:
echo   1. Klik 2x: START.bat
echo   2. Pilih folder kelompok kamu di Dataset\
echo.
pause
