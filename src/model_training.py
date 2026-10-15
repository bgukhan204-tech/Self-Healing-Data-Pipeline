"""Model Training, Evaluation, and Performance Metrics."""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from config.settings import get_settings
from src.preprocessing import PipelinePreprocessor, TARGET_COLUMN


@dataclass
class ModelEvaluationReport:
    """Standardized model evaluation metrics and validation report."""
    accuracy: float
    precision_score: float
    recall_score: float
    f1_score: float
    roc_auc: float
    confusion_matrix: List[List[int]]
    sample_count: int
    evaluated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "precision_score": self.precision_score,
            "recall_score": self.recall_score,
            "f1_score": self.f1_score,
            "roc_auc": self.roc_auc,
            "confusion_matrix": self.confusion_matrix,
            "sample_count": self.sample_count,
            "evaluated_at": self.evaluated_at,
        }


def train_model(
    df: pd.DataFrame,
    algorithm: str = "RandomForest",
    random_seed: int = 42,
    degrade_intentionally: bool = False,
) -> Tuple[Any, PipelinePreprocessor, ModelEvaluationReport, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Train a classification model with train/validation split.
    
    If degrade_intentionally is True (used by Failure Simulation Lab),
    a constrained weak model is trained to test candidate rejection & rollback.
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in training dataset.")

    # 1. Stratified split
    train_df, val_df = train_test_split(
        df,
        test_size=0.25,
        random_state=random_seed,
        stratify=df[TARGET_COLUMN],
    )

    # 2. Fit preprocessor on training split
    preprocessor = PipelinePreprocessor()
    X_train, y_train = preprocessor.fit_transform(train_df)
    X_val, y_val = preprocessor.transform(val_df)

    if y_train is None or y_val is None:
        raise ValueError("Target array cannot be None.")

    # 3. Model instantiation
    if degrade_intentionally:
        # Deliberately underfit with 1 shallow tree for simulation testing
        model = RandomForestClassifier(
            n_estimators=1,
            max_depth=1,
            random_state=random_seed,
            max_features=1,
        )
    elif algorithm == "GradientBoosting":
        model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            random_state=random_seed,
        )
    else:  # Default RandomForest
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=4,
            class_weight="balanced",
            random_state=random_seed,
            n_jobs=-1,
        )

    # 4. Fit model (handle class imbalance for GradientBoosting with sample weights)
    if algorithm == "GradientBoosting" and not degrade_intentionally:
        from sklearn.utils.class_weight import compute_sample_weight
        sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
        model.fit(X_train, y_train, sample_weight=sample_weights)
    else:
        model.fit(X_train, y_train)

    # 5. Evaluate on validation split
    y_pred = model.predict(X_val)
    
    auc = 0.5
    try:
        if len(np.unique(y_val)) > 1:
            y_prob = model.predict_proba(X_val)[:, 1]
            auc = float(roc_auc_score(y_val, y_prob))
    except Exception:
        auc = 0.5

    acc = float(accuracy_score(y_val, y_pred))
    prec = float(precision_score(y_val, y_pred, average="weighted", zero_division=0))
    rec = float(recall_score(y_val, y_pred, average="weighted", zero_division=0))
    f1 = float(f1_score(y_val, y_pred, average="weighted", zero_division=0))
    cm = confusion_matrix(y_val, y_pred, labels=[0, 1]).tolist()

    report = ModelEvaluationReport(
        accuracy=round(acc, 4),
        precision_score=round(prec, 4),
        recall_score=round(rec, 4),
        f1_score=round(f1, 4),
        roc_auc=round(auc, 4),
        confusion_matrix=cm,
        sample_count=len(val_df),
    )

    return model, preprocessor, report, (train_df, val_df)


def evaluate_existing_model(
    model_artifact: Dict[str, Any],
    eval_df: pd.DataFrame,
) -> ModelEvaluationReport:
    """Evaluate an existing model artifact dictionary on an evaluation dataframe."""
    model = model_artifact["model"]
    preprocessor: PipelinePreprocessor = model_artifact["preprocessor"]

    X_val, y_val = preprocessor.transform(eval_df)
    if y_val is None:
        raise ValueError(f"Evaluation dataframe must contain '{TARGET_COLUMN}' target.")

    y_pred = model.predict(X_val)
    auc = 0.5
    try:
        if len(np.unique(y_val)) > 1:
            y_prob = model.predict_proba(X_val)[:, 1]
            auc = float(roc_auc_score(y_val, y_prob))
    except Exception:
        auc = 0.5

    acc = float(accuracy_score(y_val, y_pred))
    prec = float(precision_score(y_val, y_pred, average="weighted", zero_division=0))
    rec = float(recall_score(y_val, y_pred, average="weighted", zero_division=0))
    f1 = float(f1_score(y_val, y_pred, average="weighted", zero_division=0))
    cm = confusion_matrix(y_val, y_pred, labels=[0, 1]).tolist()

    return ModelEvaluationReport(
        accuracy=round(acc, 4),
        precision_score=round(prec, 4),
        recall_score=round(rec, 4),
        f1_score=round(f1, 4),
        roc_auc=round(auc, 4),
        confusion_matrix=cm,
        sample_count=len(eval_df),
    )
