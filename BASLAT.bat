@echo off
setlocal EnableExtensions
cd /d "%~dp0"

if /I "%~1"=="/check" (
    echo BASLAT.bat syntax OK
    exit /b 0
)

title Iron Polcy v7
set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
set "PYTHONW_EXE=%~dp0.venv\Scripts\pythonw.exe"

if exist "%~dp0KISAYOL_OLUSTUR.ps1" (
    powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "%~dp0KISAYOL_OLUSTUR.ps1" >nul 2>&1
)

if not exist "%PYTHON_EXE%" (
    cls
    echo Iron Polcy v7 ilk kez hazirlaniyor.
    echo Gerekli paketlerin kurulumu birkac dakika surebilir.
    echo.
    call "%~dp0KUR.bat" /auto
    if errorlevel 1 goto :setup_error
)

if not exist "%PYTHON_EXE%" goto :setup_error

"%PYTHON_EXE%" -c "import numpy, pygame, gymnasium, stable_baselines3, torch, cloudpickle, matplotlib" >nul 2>&1
if errorlevel 1 (
    echo.
    echo Eksik paketler kuruluyor...
    call "%~dp0KUR.bat" /auto
    if errorlevel 1 goto :setup_error
)

if exist "%PYTHONW_EXE%" (
    start "" "%PYTHONW_EXE%" "%~dp0launcher_v7.py"
    exit /b 0
)

start "" "%PYTHON_EXE%" "%~dp0launcher_v7.py"
exit /b 0

:setup_error
echo.
echo HATA: Iron Polcy v7 kurulumu tamamlanamadi.
echo KUR.bat dosyasini elle calistirip ekrandaki hatayi kontrol edin.
pause
exit /b 1
