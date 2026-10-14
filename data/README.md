# Data Directory

This folder contains reference baselines, sample demonstration datasets, uploads, and quarantined records for the Self-Healing Data Pipeline.

## Files and Subdirectories

- **`sample_reference.csv`**: Baseline golden dataset (1,200 records). Used to establish reference feature distributions, fit the baseline Isolation Forest anomaly detector, and establish the initial validated model (`v1.0.0`).
- **`sample_clean_batch.csv`**: Standard operational production batch (400 records) exhibiting healthy statistical behavior and normal transaction patterns.
- **`sample_corrupted_batch.csv`**: Problematic batch (400 records) containing injected missing fields, duplicate transactions, and out-of-range sensor/credit values (e.g., negative incomes, impossible ages) designed to trigger data validation and quarantine isolation.
- **`sample_drifted_batch.csv`**: Real-world demographic and economic distribution shift batch (500 records) demonstrating Kolmogorov-Smirnov and Chi-square statistical drift.
- **`quarantine/`**: Isolated rows that failed data quality constraints, stored with timestamp and full reason logs for auditable post-mortem inspection.
- **`uploads/`**: Temporarily stored user-uploaded CSV batches.
