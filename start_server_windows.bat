@echo off
setlocal ENABLEDELAYEDEXPANSION

:: =============================================
:: PKC Benchmark - Windows Server Start Script
:: Keeps this console open; shows all server logs
:: Opens benchmark_canvas.html automatically
:: =============================================

:: 1) Set console to UTF-8 for logs
chcp 65001 >nul
title PKC Benchmark Server

:: 2) Go to script directory
cd /d "%~dp0"
echo [INFO] Working dir: %CD%

:: 3a) [NEW] Attempt to activate Conda environment first
echo [INFO] Checking for Conda (Anaconda) environment...
where conda >nul 2>&1
if %errorlevel% == 0 (
  echo [INFO] Found Conda. Activating base environment...
  :: Try activating 'base' environment. This might require prior 'conda init'.
  call conda activate base
  if !errorlevel! neq 0 (
    echo [WARN] 'conda activate base' failed. Ensure Conda is initialized for cmd.
  ) else (
    echo [INFO] Conda 'base' environment activated.
  )
) else (
  echo [INFO] Conda not found in PATH. Checking for local venv...
)

:: 3b) [Original] Activate virtual environment if it exists
if exist ".venv\Scripts\activate.bat" (
  echo [INFO] Activating local .venv environment...
  call ".venv\Scripts\activate.bat"
) else if exist "venv\Scripts\activate.bat" (
  echo [INFO] Activating local venv environment...
  call "venv\Scripts\activate.bat"
)

:: 4) Check Python
where python >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Python was not found in PATH. Install Python or open this script from a terminal with Python available.
  pause
  exit /b 1
)
echo [INFO] Using Python found at:
where python

:: 5) Open the UI (HTML) in default browser after a short delay (non-blocking)
::    Using PowerShell so this returns immediately and does not hide server logs.
start "" powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "Start-Sleep -Seconds 7; Start-Process (Resolve-Path '.\benchmark_canvas.html')" ^
  1>nul 2>nul

:: 6) Start FastAPI server (blocking in this window)
set "HOST=127.0.0.1"
set "PORT=8000"
echo [INFO] Starting server on http://%HOST%:%PORT% ...
echo [INFO] Press Ctrl+C to stop. This window will remain open.

:: Prefer running via uvicorn if available; otherwise fall back to direct python
python -c "import uvicorn" 1>nul 2>nul
if %errorlevel%==0 (
  python -m uvicorn benchmark_server:app --host %HOST% --port %PORT% --log-level info
) else (
  echo [WARN] 'uvicorn' module not found; attempting "python benchmark_server.py"...
  python benchmark_server.py
)

:: [REMOVED] Section 7 (pycache cleanup) was removed as requested.

echo.
echo [INFO] Server process exited. Press any key to close this window.
pause >nul
endlocal