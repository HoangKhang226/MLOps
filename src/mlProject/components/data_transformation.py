import os
import pandas as pd
from src.mlProject import logger
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from src.mlProject.entity.config_entity import DataTransformationConfig
from pathlib import Path


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def bin_col(self, df, columns, zero_val="No", one_val="Yes"):
        """Direct binary mapping for one or more columns."""
        if isinstance(columns, str):
            columns = [columns]
        binary_map = {zero_val: 0, one_val: 1}
        for col in columns:
            if col in df.columns:
                df[col] = df[col].map(binary_map)
                logger.info(f"Mapped '{col}': {zero_val} -> 0, {one_val} -> 1")
        return df

    def reduce_triple_values(self, df, columns):
        """Reduces triple categories (e.g. No service -> No) into binary."""
        if isinstance(columns, str):
            columns = [columns]
        mapping = {"No": 0, "No internet service": 0, "No phone service": 0, "Yes": 1}
        for col in columns:
            if col in df.columns:
                df[col] = df[col].replace(mapping)
                logger.info(f"Reduced triple values for '{col}'")
        return df

    def encode_ordinal(self, df, column, mapping_dict):
        """Ordinal encoding using provided dictionary."""
        if column in df.columns:
            df[column] = df[column].map(mapping_dict)
            logger.info(f"Ordinal encoded '{column}'")
        return df

    def encode_onehot(self, df, column, prefix):
        """One-hot encoding for categorical variables."""
        if column in df.columns:
            df = pd.get_dummies(df, columns=[column], prefix=prefix)
            logger.info(f"One-hot encoded '{column}' with prefix '{prefix}'")
        return df

    def create_engagement_features(
        self, df, addon_columns, new_col_name="TotalServices"
    ):
        """Feature Engineering: Engagement score by summing services."""
        valid_cols = [col for col in addon_columns if col in df.columns]
        if valid_cols:
            df[new_col_name] = df[valid_cols].sum(axis=1)
            logger.info(f"Created '{new_col_name}' from {valid_cols}")
        return df

    def fill_null(self, df, column, strategy="constant", fill_value=0):
        """Handles missing values."""
        if column in df.columns:
            if strategy == "constant":
                df[column] = df[column].fillna(fill_value)
            logger.info(f"Filled nulls in '{column}' using {strategy}")
        return df

    def smote_oversampling(self, df, target_column):
        """Applies SMOTE to training data (features should be numeric)."""
        logger.info(f"Applying SMOTE to balance class: {target_column}")
        X = df.drop(columns=[target_column])
        y = df[target_column]
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X, y)
        df_balanced = pd.concat(
            [pd.DataFrame(X_res), pd.Series(y_res, name=target_column)], axis=1
        )
        logger.info(
            f"SMOTE applied. New distribution: {y_res.value_counts().to_dict()}"
        )
        return df_balanced

    def initiate_data_transformation(self):
        try:
            df = pd.read_csv(self.config.data_path)

            # Implementation logic using methods
            df = self.bin_col(
                df,
                ["Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"],
            )
            df = self.bin_col(df, "gender", zero_val="Female", one_val="Male")

            triple_features = [
                "MultipleLines",
                "OnlineSecurity",
                "OnlineBackup",
                "DeviceProtection",
                "TechSupport",
                "StreamingTV",
                "StreamingMovies",
            ]
            df = self.reduce_triple_values(df, triple_features)

            contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
            df = self.encode_ordinal(df, "Contract", contract_map)

            df = self.encode_onehot(df, "InternetService", "IS")
            df = self.encode_onehot(df, "PaymentMethod", "PM")

            if "TotalCharges" in df.columns:
                df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
                df = self.fill_null(
                    df, "TotalCharges", strategy="constant", fill_value=0
                )

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
            df = self.create_engagement_features(df, addon_cols)

            if "customerID" in df.columns:
                df = df.drop(columns=["customerID"])

            # Split Train/Test
            train, test = train_test_split(
                df, test_size=self.config.test_size, random_state=42
            )

            # SMOTE only for Training set
            if self.config.oversampling:
                train = self.smote_oversampling(train, "Churn")

            # Save artifacts
            train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
            test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)

            logger.info("Transformation complete. Train and Test data saved.")
            logger.info(f"Train Shape: {train.shape}, Test Shape: {test.shape}")

        except Exception as e:
            raise e
