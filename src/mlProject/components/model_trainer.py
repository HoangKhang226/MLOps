import pandas as pd
import os
from src.mlProject import logger
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import mlflow
import mlflow.sklearn
from src.mlProject.entity.config_entity import ModelTrainerConfig
import json


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def split_data(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        X_train = train_data.drop([self.config.target_column], axis=1)
        X_test = test_data.drop([self.config.target_column], axis=1)
        y_train = train_data[self.config.target_column]
        y_test = test_data[self.config.target_column]

        return X_train, X_test, y_train, y_test

    def hyperparameter_tuning(self, model, param_grid, X_train, y_train):
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, scoring='f1')
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_, grid_search.best_params_

    def train_and_log(self, model_name, model, params, X_train, X_test, y_train, y_test):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        logger.info(f"Connected to MLflow Tracking URI: {self.config.mlflow_uri}")
        
        with mlflow.start_run(run_name=model_name):
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            
            # Metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            try:
                auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
            except:
                auc = 0.0

            # Logging
            mlflow.log_params(params)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("roc_auc", auc)

            # Register model as "ChurnModel"
            mlflow.sklearn.log_model(model, "model", registered_model_name="ChurnModel")

            logger.info(f"Model {model_name} logged and registered as ChurnModel with F1: {f1}")
            
            return {
                "model_name": model_name,
                "model_obj": model,
                "metrics": {"accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1, "roc_auc": auc}
            }

    def compare_models(self, results):
        best_model_info = max(results, key=lambda x: x['metrics']['f1_score'])
        logger.info(f"Best Model Selected: {best_model_info['model_name']} with F1: {best_model_info['metrics']['f1_score']}")
        return best_model_info

    def save_model(self, best_model_info):
        joblib.dump(best_model_info['model_obj'], os.path.join(self.config.root_dir, self.config.model_name))
        
        with open(os.path.join(self.config.root_dir, "metrics.json"), "w") as f:
            json.dump(best_model_info['metrics'], f, indent=4)
        
        logger.info(f"Best model saved to {self.config.root_dir}")

    def initiate_model_trainer(self):
        X_train, X_test, y_train, y_test = self.split_data()

        results = []

        # 1. Random Forest
        logger.info("Tuning Random Forest...")
        rf = RandomForestClassifier(random_state=42)
        best_rf, best_rf_params = self.hyperparameter_tuning(rf, self.config.rf_params, X_train, y_train)
        rf_res = self.train_and_log("RandomForest", best_rf, best_rf_params, X_train, X_test, y_train, y_test)
        results.append(rf_res)

        # 2. XGBoost
        logger.info("Tuning XGBoost...")
        # Note: XGBClassifier expects 0/1 for binary targets
        xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        best_xgb, best_xgb_params = self.hyperparameter_tuning(xgb, self.config.xgboost_params, X_train, y_train)
        xgb_res = self.train_and_log("XGBoost", best_xgb, best_xgb_params, X_train, X_test, y_train, y_test)
        results.append(xgb_res)

        # Comparison
        best_model_info = self.compare_models(results)
        self.save_model(best_model_info)
