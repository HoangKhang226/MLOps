import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import os
from src.mlProject import logger


class PredictionPipeline:
    def __init__(self):
        self.model_path = Path("artifacts/model_trainer/model.joblib")
        self.model = joblib.load(self.model_path)

    def predict(self, data):
        """
        data: Dict containing raw user input
        Example:
        {
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 1,
            "PhoneService": "No",
            "MultipleLines": "No phone service",
            "InternetService": "DSL",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 29.85,
            "TotalCharges": 29.85
        }
        """
        try:
            df = pd.DataFrame([data])
            logger.info(f"Input data: {data}")

            # 1. Binary Mapping
            binary_map = {"No": 0, "Yes": 1}
            gender_map = {"Female": 0, "Male": 1}

            for col in ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
                if col in df.columns:
                    df[col] = df[col].map(binary_map)

            if "gender" in df.columns:
                df["gender"] = df["gender"].map(gender_map)

            # 2. Reduce Triple Values (e.g., No internet service -> No)
            triple_features = [
                "MultipleLines",
                "OnlineSecurity",
                "OnlineBackup",
                "DeviceProtection",
                "TechSupport",
                "StreamingTV",
                "StreamingMovies",
            ]
            triple_map = {
                "No": 0,
                "No internet service": 0,
                "No phone service": 0,
                "Yes": 1,
            }
            for col in triple_features:
                if col in df.columns:
                    df[col] = df[col].replace(triple_map)

            # 3. Ordinal Encoding
            contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
            if "Contract" in df.columns:
                df["Contract"] = df["Contract"].map(contract_map)

            # 4. One-Hot Encoding (Manual to match training order)
            # InternetService: DSL, Fiber optic, No
            df["IS_DSL"] = 1 if data.get("InternetService") == "DSL" else 0
            df["IS_Fiber optic"] = (
                1 if data.get("InternetService") == "Fiber optic" else 0
            )
            df["IS_No"] = 1 if data.get("InternetService") == "No" else 0

            # PaymentMethod: Bank transfer (automatic), Credit card (automatic), Electronic check, Mailed check
            pm = data.get("PaymentMethod")
            df["PM_Bank transfer (automatic)"] = (
                1 if pm == "Bank transfer (automatic)" else 0
            )
            df["PM_Credit card (automatic)"] = (
                1 if pm == "Credit card (automatic)" else 0
            )
            df["PM_Electronic check"] = 1 if pm == "Electronic check" else 0
            df["PM_Mailed check"] = 1 if pm == "Mailed check" else 0

            # 5. Feature Engineering: TotalServices
            addon_cols = [
                "PhoneService",
                "MultipleLines",
                "OnlineSecurity",
                "OnlineBackup",
                "DeviceProtection",
                "TechSupport",
                "StreamingTV",
                "StreamingMovies",
            ]
            # MultipleLines is special in the original data, but we already reduced it to binary 0/1
            # We need to ensure we use the transformed values
            df["TotalServices"] = df[addon_cols].sum(axis=1)

            # 6. Final Column Selection and Ordering (Must match train.csv exactly)
            final_columns = [
                "gender",
                "SeniorCitizen",
                "Partner",
                "Dependents",
                "tenure",
                "PhoneService",
                "MultipleLines",
                "OnlineSecurity",
                "OnlineBackup",
                "DeviceProtection",
                "TechSupport",
                "StreamingTV",
                "StreamingMovies",
                "Contract",
                "PaperlessBilling",
                "MonthlyCharges",
                "TotalCharges",
                "IS_DSL",
                "IS_Fiber optic",
                "IS_No",
                "PM_Bank transfer (automatic)",
                "PM_Credit card (automatic)",
                "PM_Electronic check",
                "PM_Mailed check",
                "TotalServices",
            ]

            df = df[final_columns]

            # Predict
            prediction = self.model.predict(df)
            probability = self.model.predict_proba(df)[0][1]

            return prediction[0], probability

        except Exception as e:
            logger.exception(f"Error in prediction pipeline: {e}")
            raise e
