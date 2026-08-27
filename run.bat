@echo off
set "UV_CMD=uv"
where uv >nul 2>nul
if %errorlevel% neq 0 (
    if exist "%USERPROFILE%\.local\bin\uv.exe" (
        set "UV_CMD=%USERPROFILE%\.local\bin\uv.exe"
    ) else (
        echo uv is not installed or not in PATH. Please install uv from astral.sh
        exit /b 1
    )
)

set PYTHONPATH=src
%UV_CMD% run python -m uncertain_lang.cli %*
