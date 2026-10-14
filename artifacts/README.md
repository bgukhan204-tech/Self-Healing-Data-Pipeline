# Artifacts Directory

This folder stores serialized machine learning artifacts, fitted pipelines, and model metadata.

## Subdirectories

- **`models/`**: Serialized model bundles containing:
  - Scikit-learn classification model artifact (`.joblib`)
  - Fitted preprocessor with imputation, one-hot encoding, and standard scaling
  - Model evaluation report (Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix)
  - Training metadata and promotion timestamps
- **`cache/`**: Temporary preprocessor and detector caches.
