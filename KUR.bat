@echo off
setlocal EnableExtensions
cd /d "%~dp0"

if /I "%~1"=="/check" (
    echo KUR.bat syntax OK
    exit /b 0
)

title Iron Polcy v7 Kurulum
set "AUTO_MODE=0"
if /I "%~1"=="/auto" set "AUTO_MODE=1"
set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
    echo Projeye ozel Python ortami olusturuluyor...
    where py.exe >nul 2>&1
    if not errorlevel 1 (
        py.exe -3.13 -m venv "%~dp0.venv" >nul 2>&1
        if not exist "%PYTHON_EXE%" py.exe -3.12 -m venv "%~dp0.venv" >nul 2>&1
        if not exist "%PYTHON_EXE%" py.exe -3.11 -m venv "%~dp0.venv" >nul 2>&1
        if not exist "%PYTHON_EXE%" py.exe -3.10 -m venv "%~dp0.venv" >nul 2>&1
    )
)

if not exist "%PYTHON_EXE%" (
    where python.exe >nul 2>&1
    if not errorlevel 1 (
        python.exe -c "import sys; raise SystemExit(0 if (3, 10) ^<= sys.version_info ^< (3, 14) else 1)" >nul 2>&1
        if not errorlevel 1 python.exe -m venv "%~dp0.venv"
    )
)

if not exist "%PYTHON_EXE%" (
    where python3.exe >nul 2>&1
    if not errorlevel 1 (
        python3.exe -c "import sys; raise SystemExit(0 if (3, 10) ^<= sys.version_info ^< (3, 14) else 1)" >nul 2>&1
        if not errorlevel 1 python3.exe -m venv "%~dp0.venv"
    )
)

if not exist "%PYTHON_EXE%" (
    echo.
    echo HATA: Uyumlu Python bulunamadi.
    echo Python 3.10 - 3.13 surumlerinden birini python.org adresinden kurun.
    goto :install_error
)

echo Paket yoneticisi guncelleniyor...
"%PYTHON_EXE%" -m pip install --upgrade pip
if errorlevel 1 goto :install_error

echo Iron Polcy v7 paketleri kuruluyor. Bu islem birkac dakika surebilir...
"%PYTHON_EXE%" -m pip install -r "%~dp0requirements.txt"
if errorlevel 1 goto :install_error

echo.
echo Kurulum tamamlandi.
if exist "%~dp0KISAYOL_OLUSTUR.ps1" (
    powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "%~dp0KISAYOL_OLUSTUR.ps1" >nul 2>&1
)
if "%AUTO_MODE%"=="0" (
    echo Oyunu acmak icin BASLAT.bat dosyasini calistirin.
    pause
)
exit /b 0

:install_error
echo.
echo HATA: Kurulum tamamlanamadi.
echo Internet baglantisini ve yukaridaki hata mesajini kontrol edin.
pause
exit /b 1
