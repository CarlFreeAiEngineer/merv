@echo off
if not exist "%~dp0bin\llama.cpp\llama-server.exe" (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\ensure-llama-server.ps1" || exit /b 1
)
"%~dp0bin\uv.exe" run "%~dp0serve.py" %*
