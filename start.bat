@echo off
rem strata-slim on Windows: prepare what is missing, write the config, start the server.
rem
rem   start.bat                  prepare, then serve
rem   start.bat prepare          only download / pack / MTP, then stop
rem   start.bat print-config     write and print the config, do not start
rem
rem The settings are in .env beside this file (copy example.env to .env and edit it).
setlocal
set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"
set "PYTHONDONTWRITEBYTECODE=1"
cd /d "%~dp0"

if not exist ".env" (
    echo.
    echo  No settings yet: copy example.env to .env, edit it, then run start.bat again.
    echo.
    pause
    exit /b 1
)

"%~dp0python\python.exe" "%~dp0launch.py" %*
set "CODE=%ERRORLEVEL%"
if not "%CODE%"=="0" pause
exit /b %CODE%
