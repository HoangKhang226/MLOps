# 📡 Telco Customer Churn Prediction: End-to-End MLOps Pipeline

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Serving-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B.svg)](https://streamlit.io/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange.svg)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Deployment-2496ED.svg)](https://www.docker.com/)

## 📝 Overview
This project delivers a production-ready MLOps system for predicting customer churn in the telecommunications industry. Using the **Telco Customer Churn** dataset, we implement a robust 6-stage pipeline that automates everything from data ingestion to model serving via a modern web interface.

The solution leverages **Random Forest** and **XGBoost** models, optimized through automated hyperparameter tuning and managed experiments.

---

## 🔥 Key Features
- ✅ **Full MLOps Lifecycle**: Modular architecture following enterprise standards.
- 🛠️ **Automated Preprocessing**: Intelligent handling of categorical data, missing values, and feature engineering.
- ⚖️ **Class Balance**: Integration of **SMOTE** to handle imbalanced churn data.
- 📊 **Metric Tracking**: Centralized logging of parameters and metrics using **MLflow**.
- 🚀 **Dual Serving Layer**:
    - **FastAPI**: High-performance REST API for model inference.
    - **Streamlit**: Intuitive, interactive dashboard for end-users.
- 🐳 **Containerized**: Fully Dockerized for seamless deployment across environments.

---

## 🏗️ Pipeline Architecture (6 Stages)

1.  **Data Ingestion**: Securely fetches and extracts the raw dataset.
2.  **Data Validation**: Ensures data integrity against a predefined schema.
3.  **Data Transformation**: Handles encoding, scaling, and generates derived features like `TotalServices`.
4.  **Model Trainer**: Hyperparameter optimization using GridSearchCV.
5.  **Model Evaluation**: Comprehensive metrics analysis and MLflow registration.
6.  **Serving & UX**: Orchestrates the API and User Interface.

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/HoangKhang226/MLOP.git
cd MLOP
```

### 2. Configure Environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Method 1: Local Execution (Recommended for Dev)

1.  **Run Pipeline**: Execute the entire training flow.
    ```bash
    python main.py
    ```
2.  **Start Backend (API)**:
    ```bash
    uvicorn app:app --reload
    ```
3.  **Start Frontend (UI)**:
    ```bash
    streamlit run streamlit_app.py
    ```

### Method 2: Dockerize (Production Mode)
```bash
# Build the image
docker build -t churn-app .

# Run the container (Maps API to 8000 and UI to 8501)
docker run -p 8000:8000 -p 8501:8501 churn-app
```

---

## 📂 Project Structure
```text
.
├── app.py                # FastAPI Backend
├── streamlit_app.py      # Streamlit Frontend
├── main.py               # Main Pipeline Entry
├── Dockerfile            # Container configuration
├── config/               # YAML configuration files
├── research/             # Experimental Notebooks
└── src/mlProject/
    ├── components/       # Core Logic per Stage
    ├── pipeline/         # Stage Orchestrators
    └── config/           # Configuration Manager
```

---

## 📊 Evaluation Insights
Model performance is tracked globally. Key metrics achieved:
- **Accuracy**: ~78%
- **ROC-AUC**: 0.85
- **Recall (Churn)**: optimized for early risk detection.

---
*Developed with ❤️ by Antigravity AI assistant.*