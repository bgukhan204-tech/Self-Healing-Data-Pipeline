"""Data Ingestion, Synthetic Generation, and Safe File Loading."""
import io
import os
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
import pandas as pd
from config.settings import get_settings


def generate_synthetic_dataset(
    n_samples: int = 1000,
    random_seed: int = 42,
    drift_factor: float = 0.0,
    outlier_ratio: float = 0.0,
    missing_ratio: float = 0.0,
) -> pd.DataFrame:
    """
    Generate a realistic credit/financial transaction dataset.
    
    Features:
    - customer_id: Unique string ID
    - age: Integer (18-75)
    - annual_income: Float ($20k-$200k)
    - credit_score: Integer (300-850)
    - debt_to_income: Float (0.05-0.90)
    - loan_amount: Float ($1k-$50k)
    - employment_years: Integer (0-35)
    - num_open_accounts: Integer (1-20)
    - education_level: Categorical ('High School', 'Bachelor', 'Master', 'Doctorate')
    - home_ownership: Categorical ('RENT', 'MORTGAGE', 'OWN')
    - loan_intent: Categorical ('PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION')
    - loan_status: Target (0 = Good / Repaid, 1 = Default / High Risk)
    """
    np.random.seed(random_seed)
    
    customer_ids = [f"CUST-{100000 + i}" for i in range(n_samples)]
    
    # Base distributions with optional drift shift
    age = np.random.normal(loc=38.0 + (drift_factor * 12.0), scale=10.0, size=n_samples)
    age = np.clip(np.round(age), 18, 80).astype(int)
    
    # Income: log-normal distribution shifted by drift
    income_base = np.random.lognormal(mean=10.8 - (drift_factor * 0.4), sigma=0.5, size=n_samples)
    annual_income = np.round(np.clip(income_base, 15000, 250000), 2)
    
    # Credit score: normal distribution centered at 680, degraded if drift
    credit_base = np.random.normal(loc=680 - (drift_factor * 80), scale=65, size=n_samples)
    credit_score = np.clip(np.round(credit_base), 300, 850).astype(int)
    
    # Debt to income ratio: beta distribution, increases if drift
    dti_base = np.random.beta(a=2.0 + (drift_factor * 2.5), b=5.0, size=n_samples)
    debt_to_income = np.round(np.clip(dti_base * 0.9, 0.02, 0.95), 4)
    
    # Loan amount
    loan_amount = np.round(np.random.gamma(shape=3.0, scale=4000, size=n_samples) + 2000, 2)
    loan_amount = np.clip(loan_amount, 1000, 50000)
    
    # Employment years
    emp_years = np.clip(np.round(np.random.exponential(scale=6.0, size=n_samples)), 0, 40).astype(int)
    
    # Number of open accounts
    num_open_accounts = np.clip(np.random.poisson(lam=6.0, size=n_samples), 1, 25).astype(int)
    
    # Categoricals
    educations = np.random.choice(
        ['High School', 'Bachelor', 'Master', 'Doctorate'],
        size=n_samples,
        p=[0.35, 0.45, 0.15, 0.05] if drift_factor == 0.0 else [0.55, 0.30, 0.10, 0.05]
    )
    
    home_ownerships = np.random.choice(
        ['RENT', 'MORTGAGE', 'OWN'],
        size=n_samples,
        p=[0.45, 0.45, 0.10] if drift_factor == 0.0 else [0.65, 0.28, 0.07]
    )
    
    loan_intents = np.random.choice(
        ['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'],
        size=n_samples,
        p=[0.20, 0.20, 0.15, 0.15, 0.15, 0.15]
    )
    
    # Realistic probability of default based on normalized features (Logit model)
    norm_income = (annual_income - 60000.0) / 30000.0
    norm_credit = (credit_score - 660.0) / 60.0
    norm_dti = (debt_to_income - 0.35) / 0.15
    norm_emp = (emp_years - 6.0) / 4.0

    z = (
        - 0.4
        - 1.0 * norm_credit
        + 1.1 * norm_dti
        - 0.6 * norm_income
        - 0.3 * norm_emp
        + (0.6 if drift_factor > 0 else 0.0)
    )
    prob_default = 1.0 / (1.0 + np.exp(-z))
    loan_status = (np.random.uniform(0, 1, size=n_samples) < prob_default).astype(int)
    
    df = pd.DataFrame({
        'customer_id': customer_ids,
        'age': age,
        'annual_income': annual_income,
        'credit_score': credit_score,
        'debt_to_income': debt_to_income,
        'loan_amount': loan_amount,
        'employment_years': emp_years,
        'num_open_accounts': num_open_accounts,
        'education_level': educations,
        'home_ownership': home_ownerships,
        'loan_intent': loan_intents,
        'loan_status': loan_status,
    })
    
    # Inject Outliers if requested
    if outlier_ratio > 0.0:
        n_outliers = max(1, int(n_samples * outlier_ratio))
        outlier_indices = np.random.choice(n_samples, size=n_outliers, replace=False)
        for idx in outlier_indices:
            # Extreme unrealistic values
            coin = np.random.randint(0, 3)
            if coin == 0:
                df.at[idx, 'annual_income'] = float(np.random.choice([1500000.0, 3000000.0])) # Extreme wealth
                df.at[idx, 'loan_amount'] = 450000.0
            elif coin == 1:
                df.at[idx, 'age'] = int(np.random.choice([145, 180])) # Impossible human age
            else:
                df.at[idx, 'debt_to_income'] = 12.5 # Huge debt-to-income
                df.at[idx, 'credit_score'] = 150 # Abnormally low
    
    # Inject Missing values if requested
    if missing_ratio > 0.0:
        n_missing = max(1, int(n_samples * missing_ratio))
        missing_indices = np.random.choice(n_samples, size=n_missing, replace=False)
        cols_for_nulls = ['annual_income', 'credit_score', 'debt_to_income', 'employment_years']
        for idx in missing_indices:
            col = np.random.choice(cols_for_nulls)
            df.at[idx, col] = np.nan

    return df


def load_csv_safely(
    file_source: Union[str, Path, io.BytesIO],
    max_size_mb: Optional[int] = None,
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Safely load a CSV from a file path or file-like buffer with validation.
    
    Returns:
        (DataFrame, None) on success
        (None, error_message) on failure
    """
    settings = get_settings()
    max_mb = max_size_mb or settings.max_upload_size_mb
    
    # 1. Size checking if it's a file buffer
    if hasattr(file_source, "getbuffer"):
        size_bytes = file_source.getbuffer().nbytes
        if size_bytes > max_mb * 1024 * 1024:
            return None, f"File size ({size_bytes / (1024*1024):.2f} MB) exceeds maximum allowed limit ({max_mb} MB)."
        if size_bytes == 0:
            return None, "Uploaded file is completely empty (0 bytes)."
    
    # 2. Path checking if it's a file path
    elif isinstance(file_source, (str, Path)):
        p = Path(file_source)
        if not p.exists():
            return None, f"File not found at path: {p}"
        if p.stat().st_size == 0:
            return None, "Target CSV file is empty (0 bytes)."
        if p.stat().st_size > max_mb * 1024 * 1024:
            return None, f"File exceeds maximum allowed size ({max_mb} MB)."

    # 3. Read CSV with encoding fallback
    try:
        if hasattr(file_source, "seek"):
            file_source.seek(0)
        df = pd.read_csv(file_source, encoding="utf-8")
    except UnicodeDecodeError:
        try:
            if hasattr(file_source, "seek"):
                file_source.seek(0)
            df = pd.read_csv(file_source, encoding="latin1")
        except Exception as e:
            return None, f"Failed to parse CSV due to encoding error: {str(e)}"
    except Exception as e:
        return None, f"CSV parsing error: {str(e)}"
        
    if df.empty:
        return None, "Parsed CSV contains no rows or columns."
        
    return df, None


def save_uploaded_batch(df: pd.DataFrame, original_filename: str) -> Path:
    """Persist an uploaded dataset safely with timestamp."""
    settings = get_settings()
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    clean_name = Path(original_filename).name.replace(" ", "_")
    target_path = settings.uploads_dir / clean_name
    df.to_csv(target_path, index=False)
    return target_path
