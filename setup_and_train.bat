@echo off
echo ============================================================
echo   Instagram AI Detector - Complete Setup Script
echo ============================================================
echo.

cd /d "F:\DEEP LEARNING PROJECT\Instagram_AI_Detector"

echo [1/5] Installing Python dependencies...
pip install streamlit pandas numpy scikit-learn joblib matplotlib plotly fpdf2 reportlab opencv-python Pillow imbalanced-learn seaborn
echo.

echo [2/5] Generating synthetic datasets...
python generate_datasets.py
echo.

echo [3/5] Training Fake Account Detection model...
python train_fake_account.py
echo.

echo [4/5] Training Bot Detection model...
python train_bot_model.py
echo.

echo [5/5] Setup complete! To train the image model (requires TensorFlow):
echo       python train_image_model.py
echo.
echo ============================================================
echo   Run the app with:  streamlit run app.py
echo ============================================================
pause
