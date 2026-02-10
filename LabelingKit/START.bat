@echo off
setlocal
cd /d "%~dp0"

echo.
echo Membuka AnyLabeling...
echo.

:: Apply startup hotfix (safe to run multiple times)
if exist "python\python.exe" (
    "python\python.exe" _internal\scripts\fix_anylabeling_colormap.py >nul 2>&1
) else (
    python _internal\scripts\fix_anylabeling_colormap.py >nul 2>&1
)

:: Preferred: portable AnyLabeling executable
if exist "python\Scripts\anylabeling.exe" (
    "python\Scripts\anylabeling.exe"
    if errorlevel 1 goto :launch_failed
    exit /b 0
)

:: Fallback: portable python module
if exist "python\python.exe" (
    "python\python.exe" -m anylabeling
    if errorlevel 1 goto :launch_failed
    exit /b 0
)

:: Fallback: system command
where anylabeling >nul 2>&1
if not errorlevel 1 (
    anylabeling
    if errorlevel 1 goto :launch_failed
    exit /b 0
)

:: Last fallback: system python module
python -m anylabeling
if errorlevel 1 goto :launch_failed
exit /b 0

:launch_failed
echo.
echo [ERROR] AnyLabeling gagal dibuka.
echo Kemungkinan penyebab:
echo   1. INSTALL.bat belum selesai/berhasil
echo   2. Dependency belum terpasang sempurna
echo   3. Path folder dipindahkan saat proses install
echo.
echo Jalankan ulang INSTALL.bat, lalu coba START.bat lagi.
echo.
pause
exit /b 1
