@echo off
title Instagram AI Detector — Train All Models
color 0A

echo ============================================================
echo   Instagram AI Detector — Training All Models
echo ============================================================
echo.

REM ── Step 1: Install dependencies ─────────────────────────────
echo [1/4] Installing required packages...
python -m pip install -r requirements.txt --quiet
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: pip install failed. Check your Python / pip setup.
    pause & exit /b 1
)
echo       Done.
echo.

REM ── Step 2: Train Fake Account model ─────────────────────────
echo [2/4] Training Fake Account Detection model...
python train_fake_account.py
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: train_fake_account.py failed.
    pause & exit /b 1
)
echo.

REM ── Step 3: Train Bot Detection model ────────────────────────
echo [3/4] Training Bot Detection model...
python train_bot_model.py
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: train_bot_model.py failed.
    pause & exit /b 1
)
echo.

REM ── Step 4: Train Image Detection model ──────────────────────
echo [4/4] Training Fake Image Detection model (CNN — may take longer)...
python train_image_model.py
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: train_image_model.py failed.
    pause & exit /b 1
)
echo.

echo ============================================================
echo   ALL MODELS TRAINED SUCCESSFULLY!
echo   Now run:  streamlit run app.py
echo ============================================================
pause
