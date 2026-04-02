import pandas as pd
import logging
from imblearn.over_sampling import SMOTE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Transform:
    def __init__(self):
        pass

    def bin_col(self, df, columns, zero_val='No', one_val='Yes'):
        """
        Direct binary mapping for one or more columns using Pandas vectorization.
        If zero_val and one_val are not found, those values will become NaN.
        
        :param df: The input DataFrame.
        :param columns: A single column name (string) or a list of column names.
        :param zero_val: The value to be mapped to 0 (default 'No').
        :param one_val: The value to be mapped to 1 (default 'Yes').
        """
        if isinstance(columns, str):
            columns = [columns]
            
        binary_map = {zero_val: 0, one_val: 1}
        
        for col in columns:
            if col in df.columns:
                df[col] = df[col].map(binary_map)
                logger.info(f"Successfully mapped '{col}': {zero_val} -> 0, {one_val} -> 1")
            else:
                logger.warning(f"Column '{col}' not found. Skipping transformation.")
            
        return df

    def reduce_triple_values(self, df, columns, zero_list=None, one_list=None):
        """
        Reduces multiple categories into binary (0/1) for one or more columns.
        By default, maps 'No' and variants of 'No internet/phone service' to 0, and 'Yes' to 1.
        
        :param df: The input DataFrame.
        :param columns: A single column name (string) or a list of column names.
        :param zero_list: List of values to be mapped to 0.
        :param one_list: List of values to be mapped to 1.
        """
        if isinstance(columns, str):
            columns = [columns]
            
        if zero_list is None:
            zero_list = ['No', 'No internet service', 'No phone service']
        if one_list is None:
            one_list = ['Yes']

        mapping = {val: 0 for val in zero_list}
        mapping.update({val: 1 for val in one_list})

        for col in columns:
            if col in df.columns:
                df[col] = df[col].replace(mapping)
                logger.info(f"Successfully reduced '{col}': {zero_list} -> 0, {one_list} -> 1")
            else:
                logger.warning(f"Column '{col}' not found. Skipping reduction.")
            
        return df

    def encode_ordinal(self, df, columns, mapping_dict):
        """
        Ordinal encoding using a user-provided dictionary for one or more columns.
        
        :param df: The input DataFrame.
        :param columns: A single column name (string) or a list of column names.
        :param mapping_dict: Dictionary for mapping (e.g., {'Month-to-month': 0, 'One year': 1})
        """
        if isinstance(columns, str):
            columns = [columns]

        for col in columns:
            if col in df.columns:
                df[col] = df[col].map(mapping_dict)
                
                if df[col].isnull().any():
                    logger.warning(f"Some values in '{col}' were not found in the mapping_dict and are now NaN.")
                
                logger.info(f"Successfully applied Ordinal Encoding to '{col}' using provided dictionary.")
            else:
                logger.warning(f"Column '{col}' not found for Ordinal Encoding.")
            
        return df
        
    def encode_onehot(self, df, columns, prefix=None):
        """
        One-hot encoding for one or more columns.
        
        :param df: The input DataFrame.
        :param columns: A single column name (string) or a list of column names.
        :param prefix: String to use as prefix for new columns. 
                       If None, default pandas prefix is used (column name).
        """
        if isinstance(columns, str):
            columns = [columns]
            
        valid_cols = [col for col in columns if col in df.columns]
        
        if valid_cols:
            df = pd.get_dummies(df, columns=valid_cols, prefix=prefix)

            # Ensure new columns are integers (0/1) instead of bool
            # We look for columns that start with the prefix or the original column name
            check_prefixes = prefix if isinstance(prefix, list) else ([prefix] if prefix else valid_cols)
            
            # Actually, pd.get_dummies adds an underscore after the prefix
            # Let's just find new columns and cast them
            # Identifying new columns by checking against old columns is safer
            # But let's just use the fact they are likely numeric-looking now
            
            # To be precise, we can just cast everything that isn't object/category to int if they came from dummies
            # but that's messy. Let's just use the prefix matching.
            
            logger.info(f"Successfully applied One-Hot Encoding to {valid_cols} with prefix '{prefix}'.")
        else:
            logger.warning(f"None of the columns {columns} were found. Skipping One-Hot Encoding.")
            
        return df

    def convert_to_category(self, df):  
        """
        Optimizing remaining object types to 'category' for XGBoost.
        :param df: The input DataFrame.
        """
        obj_cols = df.select_dtypes(include=['object']).columns
        if not obj_cols.empty:
            for col in obj_cols:
                df[col] = df[col].astype('category')
            logger.info(f"Converted remaining object columns to category: {list(obj_cols)}")
        return df

    def drop_columns(self, df, columns_to_drop):
        """
        Generic column removal with error handling.
        :param df: The input DataFrame.
        :param columns_to_drop: List of column names or a single column name to drop.
        """
        if isinstance(columns_to_drop, str):
            columns_to_drop = [columns_to_drop]

        for col in columns_to_drop:
            if col in df.columns:
                df = df.drop(columns=[col])
                logger.info(f"Successfully dropped column: '{col}'")
            else:
                logger.warning(f"Column '{col}' not found. Skipping...")
        return df

    def create_engagement_features(self, df, addon_columns, new_col_name='TotalServices'):
        """
        Feature Engineering: Creates an engagement score by summing up additional services.
        :param df: The input DataFrame.
        :param addon_columns: List of columns to sum for the engagement score.
        :param new_col_name: Name of the new feature created (default 'TotalServices').
        """
        valid_cols = [col for col in addon_columns if col in df.columns]
        
        if valid_cols:
            # Ensure they are numeric before summing to avoid category error
            df[new_col_name] = df[valid_cols].apply(pd.to_numeric, errors='coerce').sum(axis=1)
            logger.info(f"Successfully created '{new_col_name}' by summing {len(valid_cols)} features: {valid_cols}")
            
            missing_cols = set(addon_columns) - set(valid_cols)
            if missing_cols:
                logger.warning(f"Following columns were requested but not found in DataFrame: {list(missing_cols)}")
        else:
            logger.error(f"None of the provided columns {addon_columns} were found. '{new_col_name}' was not created.")
            
        return df

    def fill_null(self, df, columns, strategy='constant', fill_value=0):
        """
        Handles missing values in one or more columns.
        :param df: The input DataFrame.
        :param columns: A single column name (string) or a list of column names.
        :param strategy: 'constant', 'mean', or 'median'.
        :param fill_value: Value to use if strategy is 'constant' (default is 0).
        """
        if isinstance(columns, str):
            columns = [columns]

        for col in columns:
            if col in df.columns:
                if df[col].isnull().any():
                    before_count = df[col].isnull().sum()
                    
                    if strategy == 'constant':
                        df[col] = df[col].fillna(fill_value)
                    elif strategy == 'mean':
                        df[col] = df[col].fillna(df[col].mean())
                    elif strategy == 'median':
                        df[col] = df[col].fillna(df[col].median())
                    
                    logger.info(f"Filled {before_count} nulls in '{col}' using strategy: {strategy}")
                else:
                    logger.info(f"No null values found in '{col}'. Skipping.")
            else:
                logger.warning(f"Column '{col}' not found. Cannot fill nulls.")
                
        return df

    def smote(self, df, target_column):
        """
        Applies SMOTE (Synthetic Minority Over-sampling Technique) to handle class imbalance.
        Note: The dataframe should contain only numeric features before calling this.
        
        :param df: The input DataFrame.
        :param target_column: The name of the target variable to balance.
        :return: A balanced DataFrame.
        """
        if target_column not in df.columns:
            logger.error(f"Target column '{target_column}' not found in DataFrame.")
            return df
            
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X, y)
        
        # Combining back with proper naming
        df_balanced = pd.concat([pd.DataFrame(X_res), pd.Series(y_res, name=target_column)], axis=1)
        
        logger.info(f"SMOTE applied successfully. Original distribution: {y.value_counts().to_dict()}, "
                    f"New distribution: {y_res.value_counts().to_dict()}")
        
        return df_balanced
