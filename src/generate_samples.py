"""Generate sample datasets for reference, clean processing, corruption, and drift."""
from pathlib import Path
from src.ingestion import generate_synthetic_dataset

def generate_all_samples():
    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating reference dataset...")
    ref_df = generate_synthetic_dataset(
        n_samples=1200,
        random_seed=42,
        drift_factor=0.0,
        outlier_ratio=0.0,
        missing_ratio=0.0,
    )
    ref_df.to_csv(data_dir / "sample_reference.csv", index=False)
    print(f"Saved {len(ref_df)} rows to sample_reference.csv")

    print("Generating clean batch...")
    clean_df = generate_synthetic_dataset(
        n_samples=400,
        random_seed=101,
        drift_factor=0.0,
        outlier_ratio=0.01, # Tiny natural anomaly rate
        missing_ratio=0.0,
    )
    clean_df.to_csv(data_dir / "sample_clean_batch.csv", index=False)
    print(f"Saved {len(clean_df)} rows to sample_clean_batch.csv")

    print("Generating corrupted batch (outliers + missing)...")
    corrupt_df = generate_synthetic_dataset(
        n_samples=400,
        random_seed=202,
        drift_factor=0.0,
        outlier_ratio=0.10, # 10% severe outliers
        missing_ratio=0.12, # 12% missing cells
    )
    corrupt_df.to_csv(data_dir / "sample_corrupted_batch.csv", index=False)
    print(f"Saved {len(corrupt_df)} rows to sample_corrupted_batch.csv")

    print("Generating drifted batch (significant distribution shift)...")
    drift_df = generate_synthetic_dataset(
        n_samples=500,
        random_seed=303,
        drift_factor=0.75, # Heavy distribution shift in age, debt-to-income, credit score
        outlier_ratio=0.02,
        missing_ratio=0.01,
    )
    drift_df.to_csv(data_dir / "sample_drifted_batch.csv", index=False)
    print(f"Saved {len(drift_df)} rows to sample_drifted_batch.csv")

    print("All sample datasets generated successfully!")

if __name__ == "__main__":
    generate_all_samples()
