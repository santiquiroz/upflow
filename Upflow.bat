@echo off
setlocal
title Upflow

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\upflow-launcher.ps1"
set "exitCode=%errorlevel%"

if not "%exitCode%"=="0" (
    echo.
    echo Upflow termino con un error. Revisa el mensaje de arriba.
    pause
)

endlocal
exit /b %exitCode%
