"""Data Preprocessing, Imputation, Encoding, and Feature Scaling."""
from dataclasses import dataclass
from typing import List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "loan_status"
ID_COLUMNS = ["customer_id"]

NUMERIC_FEATURES = [
    "age",
    "annual_income",
    "credit_score",
    "debt_to_income",
    "loan_amount",
    "employment_years",
    "num_open_accounts",
]

CATEGORICAL_FEATURES = [
    "education_level",
    "home_ownership",
    "loan_intent",
]


class PipelinePreprocessor:
    """Production-grade preprocessor pipeline with imputation, encoding, and scaling."""

    def __init__(self):
        self.num_cols = NUMERIC_FEATURES
        self.cat_cols = CATEGORICAL_FEATURES
        self.target_col = TARGET_COLUMN
        self.transformer: Optional[ColumnTransformer] = None
        self.is_fitted = False
        self.transformed_feature_names: List[str] = []

    def fit(self, df: pd.DataFrame) -> "PipelinePreprocessor":
        """Fit preprocessor pipeline on reference data."""
        # Available columns in df
        num_present = [c for c in self.num_cols if c in df.columns]
        cat_present = [c for c in self.cat_cols if c in df.columns]

        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])

        transformers = []
        if num_present:
            transformers.append(("num", num_pipeline, num_present))
        if cat_present:
            transformers.append(("cat", cat_pipeline, cat_present))

        self.transformer = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
        )
        self.transformer.fit(df)
        self.is_fitted = True

        # Extract feature names
        names = []
        if num_present:
            names.extend(num_present)
        if cat_present:
            cat_step = self.transformer.named_transformers_["cat"].named_steps["encoder"]
            cat_encoded_names = list(cat_step.get_feature_names_out(cat_present))
            names.extend(cat_encoded_names)
        self.transformed_feature_names = names

        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Transform dataframe into feature matrix X and target y.
        Returns: (X, y) where y is None if target column is absent.
        """
        if not self.is_fitted or self.transformer is None:
            raise ValueError("Preprocessor must be fitted before transforming.")

        X = self.transformer.transform(df)

        y = None
        if self.target_col in df.columns:
            y = df[self.target_col].to_numpy()

        return X, y

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)

    def save(self, filepath: str) -> None:
        """Persist fitted preprocessor."""
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "PipelinePreprocessor":
        """Load persisted preprocessor."""
        return joblib.load(filepath)
