@echo off
REM EarthPulse - one-command start for Windows
REM
REM   Double-click this file, or run:  run.bat
REM
REM Creates a virtual environment, installs dependencies the first time,
REM then starts the server on http://localhost:8000

cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
  echo ERROR: Python not found on PATH.
  echo Install Python 3.10+ from https://python.org
  echo IMPORTANT: tick "Add Python to PATH" during installation.
  pause
  exit /b 1
)

if not exist ".venv" (
  echo ==^> Creating virtual environment ^(.venv^)
  python -m venv .venv
  if errorlevel 1 goto :fail
)

call .venv\Scripts\activate.bat

if not exist ".venv\.deps-installed" (
  echo ==^> Installing dependencies ^(first run only, ~1-2 minutes^)
  python -m pip install --upgrade pip --quiet
  python -m pip install -r requirements.txt
  if errorlevel 1 goto :fail
  type nul > .venv\.deps-installed
)

echo.
echo   EarthPulse is starting...
echo   Open http://localhost:8000 in your browser
echo   Press Ctrl+C to stop
echo.

cd backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
goto :eof

:fail
echo.
echo Setup failed. See the error above.
pause
exit /b 1
