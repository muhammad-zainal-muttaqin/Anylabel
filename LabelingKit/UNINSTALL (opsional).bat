@echo off
setlocal
cd /d "%~dp0"

echo.
echo ========================================
echo   FFB Labeling Kit - Uninstall
echo ========================================
echo.
echo Ini akan menghapus komponen instalasi:
echo   - python\
echo   - venv\
echo   - output\
echo   - __pycache__ / *.pyc
echo.
echo Data yang TIDAK dihapus:
echo   - Dataset\
echo   - file label (.json/.txt) milik user
echo.
choice /C YN /N /M "Lanjut uninstall? (Y/N): "
if errorlevel 2 (
    echo.
    echo Dibatalkan.
    pause
    exit /b 0
)

echo.
echo Menghapus file/folder...

if exist "python" (
    rmdir /s /q "python"
    echo   [OK] python\
) else (
    echo   [SKIP] python\ tidak ada
)

if exist "venv" (
    rmdir /s /q "venv"
    echo   [OK] venv\
) else (
    echo   [SKIP] venv\ tidak ada
)

if exist "output" (
    rmdir /s /q "output"
    echo   [OK] output\
) else (
    echo   [SKIP] output\ tidak ada
)

for /r %%F in (*.pyc) do del /q "%%F" >nul 2>&1
for /d /r %%D in (__pycache__) do rmdir /s /q "%%D" >nul 2>&1
echo   [OK] cache dibersihkan

echo.
echo ========================================
echo   Uninstall selesai
echo ========================================
echo.
echo Jika ingin pakai lagi, jalankan 1_INSTALL.bat
echo.
pause
