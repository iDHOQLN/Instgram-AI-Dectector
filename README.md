# 🔍 Instagram AI Detector

### AI-Powered Fake Account, Bot & AI-Generated Image Detection System

[![Python](https://img.shields.io/badge/Python-3.13+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](https://opensource.org/licenses/MIT)

> 🎓 B.Tech Final Year Project — Artificial Intelligence & Data Science

---

## 📌 About the Project

**Instagram AI Detector** is an end-to-end AI-powered web application that detects suspicious activity on Instagram through **three intelligent modules**:

| Module | Algorithm | Output |
|--------|-----------|--------|
| 👤 **Fake Account Detection** | Random Forest + Logistic Regression + GBM | Real / Fake + Confidence |
| 🤖 **Bot Detection** | Random Forest + Isolation Forest | Human / Bot + Confidence |
| 🖼️ **Fake Image Detection** | MobileNetV2 CNN (Transfer Learning) | Real / AI-Gen / Edited |
| 📊 **Analytics Dashboard** | Plotly Interactive Charts | Trends, KPIs, Distributions |
| 📜 **History & Reports** | FPDF2 / CSV | Downloadable PDF & CSV |

---

## 🎯 Problem Statement

Instagram has over **2 billion monthly users**, but the platform is widely exploited by:

- 🚫 **Fake accounts** — used for scams, fraud, and follower manipulation
- 🤖 **Bot accounts** — auto-like, auto-comment, spam automation
- 🖼️ **AI-generated profile images** — used to impersonate real people

This system **automates detection** using ML & Deep Learning to protect users and maintain platform integrity.

---

## 🛠️ Technology Stack

| Layer | Technology |
|-------|-----------|
| Web Framework | Streamlit (multi-page) |
| Machine Learning | Scikit-learn (Random Forest, LR, GBM, Isolation Forest) |
| Deep Learning | TensorFlow / Keras — MobileNetV2 CNN |
| Image Processing | OpenCV, Pillow |
| Visualization | Plotly |
| Reports | FPDF2 |
| Language | Python 3.13+ |

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/Instagram_AI_Detector.git
cd Instagram_AI_Detector
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train All Models (one-time setup)
```bash
python train_fake_account.py
python train_bot_model.py
python train_image_model.py
```
> 💡 Or run `train_all_models.bat` on Windows for one-click training.

### 4. Launch the App
```bash
streamlit run app.py
```

### 5. Login
| Username | Password |
|----------|----------|
| `admin` | `admin123` |
| `demo` | `demo123` |
| `analyst` | `analyst@123` |

---

## 📊 Model Performance

| Model | Accuracy | F1-Score | ROC-AUC |
|-------|----------|----------|---------|
| Fake Account (Random Forest) | ~97% | ~0.97 | ~0.99 |
| Bot Detection (Random Forest) | ~97% | ~0.97 | ~0.99 |
| Image Detection (MobileNetV2) | ~85%+ | ~0.85 | — |

*Trained on synthetic data. Performance improves with real Instagram datasets.*

---

## 📁 Project Structure

```
Instagram_AI_Detector/
├── app.py                    ← Main Streamlit entry point
├── login.py                  ← Authentication module
├── predict.py                ← Unified prediction engine
├── generate_datasets.py      ← Synthetic dataset generator
├── train_fake_account.py     ← Fake account model training
├── train_bot_model.py        ← Bot detection model training
├── train_image_model.py      ← CNN image model training
├── train_all_models.bat      ← One-click training script (Windows)
│
├── config/
│   ├── config.py             ← Constants, paths, feature lists
│   ├── settings.py           ← Streamlit config + CSS loader
│   └── style.css             ← Dark theme stylesheet
│
├── pages/
│   ├── Fake_Account.py       ← Fake account detection page
│   ├── Bot_Detection.py      ← Bot detection page
│   ├── Fake_Image.py         ← Image detection page
│   ├── Dashboard.py          ← Analytics dashboard
│   ├── History.py            ← Prediction history & reports
│   └── About.py              ← Project info page
│
├── utils/
│   ├── preprocessing.py      ← Feature engineering
│   ├── helper.py             ← Session & history utilities
│   ├── visualization.py      ← Plotly chart builders
│   └── report_generator.py   ← PDF & CSV generation
│
└── assets/
    └── style.css             ← Dark theme CSS
```

---

## 👨‍💻 Developer

- **Project:** B.Tech Final Year Project
- **Domain:** Artificial Intelligence & Data Science
- **Version:** 1.0.0

---

*Instagram AI Detector — Keeping Social Media Safe with AI*
