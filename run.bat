@echo off
cd /d "%~dp0"
rem serve.py downloads the bundled GPU-capable llama.cpp build itself if missing.
"%~dp0bin\uv.exe" run "%~dp0serve.py" %*
