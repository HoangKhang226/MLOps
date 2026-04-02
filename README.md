# 📡 Telco Customer Churn Prediction

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://hub.docker.com/r/hoangkhang226/mlop-churn)

## 🌟 Vision
Customer churn is one of the most critical challenges in the telecom industry. This project isn't just about training a model; it's about building a **scalable, reproducible, and production-ready MLOps ecosystem**. 

By transforming raw data into actionable insights through a standardized 6-stage pipeline, we provide a blueprint for turning machine learning research into a living, breathing software product.

---

## 🚀 Professional Features
- **🏗️ 6-Stage Modular Pipeline**: Follows the `Cookiecutter` inspired MLOps architecture for clean separation of concerns.
- **🧬 Advanced Feature Engineering**: Automated data cleaning, SMOTE-based oversampling for imbalanced classes, and generation of complex engagement features.
- **📈 MLflow Tracking & Registry**: Every experiment is logged. Metrics, parameters, and model versions are stored centrally for full auditability.
- **⚡ Dual-Service Architecture**: 
    - **Backend (FastAPI)**: Optimized for speed and low-latency inference.
    - **Frontend (Streamlit)**: A premium, dark-themed dashboard designed for business users to simulate "What-if" scenarios.
- **🐳 Cloud-Ready Containerization**: Standardized Docker environment ensuring "it works on my machine" everywhere.

---

## 🛠️ Tech Stack & Tools
| Category | Tools |
| :--- | :--- |
| **Language** | Python 3.12 |
| **Machine Learning** | Scikit-Learn, XGBoost, Imbalanced-Learn (SMOTE) |
| **MLOps & Tracking** | MLflow |
| **API Framework** | FastAPI |
| **Dashboard / UI** | Streamlit |
| **Containerization** | Docker |
| **Data Handling** | Pandas, Joblib |

---

## 🏗️ The 6-Stage Lifecycle
> Each stage is a self-contained module located in `src/mlProject/pipeline`.

1.  **Data Ingestion**: Automated retrieval and extraction of the Telco Churn dataset.
2.  **Data Validation**: Strict schema enforcement to prevent "training-serving skew".
3.  **Data Transformation**: Orchestration of one-hot encoding, feature scaling, and class balancing.
4.  **Model Trainer**: Automated hyperparameter optimization (Grid Search) for Random Forest and XGBoost.
5.  **Model Evaluation**: Deep dive into accuracy, recall, and ROC-AUC metrics; logs artifacts to MLflow.
6.  **Prediction Service**: The UI & API bridge that serves the model to the world.

---

## ⚡ Quick Start

### 🔌 Local Development
```bash
# Clone & Enter
git clone https://github.com/HoangKhang226/MLOP.git && cd MLOP

# Setup Environment
python -m venv venv
source venv/Scripts/activate  # Or venv/bin/activate
pip install -r requirements.txt

# Run the Magic
python main.py
```

### 🐳 Docker (Instant Deployment)
One command to run the entire stack:
```bash
docker run -p 8000:8000 -p 8501:8501 hoangkhang226/mlop-churn:latest
```

---

## 📊 Performance Benchmark
The model is tuned to prioritize **Recall**, ensuring we capture as many potential churners as possible.
- **Accuracy**: 78%
- **ROC-AUC**: 0.85 🚀
- **F1-Score**: Optimized via GridSearchCV

---

## 📂 Project Navigation
```text
.
├── app.py                # FastAPI Service
├── streamlit_app.py      # Streamlit Interactive UI
├── main.py               # The Orchestrator
├── Dockerfile            # The Blueprint
└── src/mlProject/
    ├── components/       # Technical Implementation 
    ├── pipeline/         # Workflow Stages
    └── config/           # Centralized Configuration
```

---
## 🤝 Contribution
Eager to improve this pipeline? Feel free to open a PR or reach out!

**Maintained with passion by [HoangKhang226](https://github.com/HoangKhang226)** 
*Built during the Antigravity AI Coding Session.*
