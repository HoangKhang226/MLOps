import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os
import mlflow
import mlflow.sklearn
from src.mlProject import logger
from src.mlProject.entity.config_entity import ModelEvaluationConfig
from urllib.parse import urlparse


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def save_results(self, metrics, cm, fpr, tpr, roc_auc):
        # Save metrics.json
        with open(self.config.metric_file_name, "w") as f:
            json.dump(metrics, f, indent=4)

        # Save Confusion Matrix plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.ylabel("Actual Label")
        plt.xlabel("Predicted Label")
        cm_path = os.path.join(self.config.root_dir, "confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()

        # Save ROC Curve plot
        plt.figure(figsize=(8, 6))
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=2,
            label=f"ROC curve (area = {roc_auc:.2f})",
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC)")
        plt.legend(loc="lower right")
        roc_path = os.path.join(self.config.root_dir, "roc_curve.png")
        plt.savefig(roc_path)
        plt.close()

        logger.info(f"Evaluation artifacts saved in {self.config.root_dir}")
        return cm_path, roc_path

    def initiate_model_evaluation(self):
        try:
            test_data = pd.read_csv(self.config.test_data_path)
            model = joblib.load(self.config.model_path)

            X_test = test_data.drop([self.config.target_column], axis=1)
            y_test = test_data[self.config.target_column]

            mlflow.set_registry_uri(self.config.mlflow_uri)
            logger.info(f"Connected to MLflow Tracking URI: {self.config.mlflow_uri}")
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            with mlflow.start_run(run_name="Evaluation_Stage"):
                y_pred = model.predict(X_test)

                # For ROC Curve, we need probabilities
                try:
                    y_prob = model.predict_proba(X_test)[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, y_prob)
                    roc_auc = auc(fpr, tpr)
                except:
                    logger.warning(
                        "Model does not support predict_proba, using 0s for ROC-AUC"
                    )
                    roc_auc = 0.0
                    fpr, tpr = [0, 1], [0, 1]

                metrics = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred),
                    "recall": recall_score(y_test, y_pred),
                    "f1_score": f1_score(y_test, y_pred),
                    "roc_auc": roc_auc,
                }

                cm = confusion_matrix(y_test, y_pred)

                # Log metrics to MLflow
                for key, value in metrics.items():
                    mlflow.log_metric(key, value)

                # Save and log artifacts
                cm_path, roc_path = self.save_results(metrics, cm, fpr, tpr, roc_auc)

                mlflow.log_artifact(cm_path)
                mlflow.log_artifact(roc_path)
                mlflow.log_artifact(self.config.metric_file_name)

                # Log params from model training (optional but good)
                mlflow.log_params(self.config.all_params)

                if tracking_url_type_store != "file":
                    # Register the model if it's not a local file store (optional)
                    mlflow.sklearn.log_model(
                        model, "model", registered_model_name="ChurnModel"
                    )
                else:
                    mlflow.sklearn.log_model(model, "model")

            logger.info(
                "MLflow Evaluation run completed and model registered if applicable."
            )

        except Exception as e:
            logger.exception(f"Error in Model Evaluation: {e}")
            raise e
