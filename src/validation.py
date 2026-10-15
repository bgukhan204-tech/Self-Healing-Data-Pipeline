"""Data Validation, Schema Verification, Quality Scoring, and Quarantine Isolation."""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from config.settings import get_settings


# Domain validation rules for our Credit/Financial Pipeline
REQUIRED_COLUMNS = [
    "customer_id",
    "age",
    "annual_income",
    "credit_score",
    "debt_to_income",
    "loan_amount",
    "employment_years",
    "num_open_accounts",
    "education_level",
    "home_ownership",
    "loan_intent",
]

COLUMN_CONSTRAINTS = {
    "age": {"type": "numeric", "min": 18, "max": 100},
    "annual_income": {"type": "numeric", "min": 0, "max": 1000000},
    "credit_score": {"type": "numeric", "min": 300, "max": 850},
    "debt_to_income": {"type": "numeric", "min": 0.0, "max": 2.5},
    "loan_amount": {"type": "numeric", "min": 500, "max": 100000},
    "employment_years": {"type": "numeric", "min": 0, "max": 50},
    "num_open_accounts": {"type": "numeric", "min": 0, "max": 40},
    "education_level": {
        "type": "categorical",
        "allowed": {"High School", "Bachelor", "Master", "Doctorate"}
    },
    "home_ownership": {
        "type": "categorical",
        "allowed": {"RENT", "MORTGAGE", "OWN"}
    },
    "loan_intent": {
        "type": "categorical",
        "allowed": {"PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"}
    },
    "loan_status": {"type": "numeric", "min": 0, "max": 1}
}


@dataclass
class ValidationReport:
    """Detailed summary of dataset validation results."""
    is_valid: bool
    quality_score: float
    total_records: int
    valid_records_count: int
    quarantined_records_count: int
    duplicate_count: int
    total_missing_cells: int
    flagged_reasons: Dict[int, List[str]] # row_index -> reasons
    valid_df: pd.DataFrame
    quarantined_df: pd.DataFrame
    column_summary: pd.DataFrame
    quarantine_file_path: Optional[str] = None
    quarantine_reasons_df: Optional[pd.DataFrame] = None
    missing_required_columns: List[str] = field(default_factory=list)


def validate_dataset(
    df: pd.DataFrame,
    dataset_name: str = "batch",
    quarantine_on_severe_issues: bool = True,
    required_columns: Optional[List[str]] = None,
    custom_constraints: Optional[Dict[str, Any]] = None,
) -> ValidationReport:
    """
    Validate dataset against schema and constraints, scoring overall quality
    and isolating quarantined records.
    """
    if df is None or df.empty:
        empty_summary = pd.DataFrame(columns=["Column", "Type", "Missing", "Missing %", "Unique"])
        return ValidationReport(
            is_valid=False,
            quality_score=0.0,
            total_records=0,
            valid_records_count=0,
            quarantined_records_count=0,
            duplicate_count=0,
            total_missing_cells=0,
            flagged_reasons={},
            valid_df=pd.DataFrame(),
            quarantined_df=pd.DataFrame(),
            column_summary=empty_summary,
            missing_required_columns=list(required_columns or REQUIRED_COLUMNS),
        )

    # Reset index to ensure 0..N-1 contiguous integer indexing
    df = df.reset_index(drop=True)
    total_records = len(df)
    total_cells = df.size
    
    # Schema check: verify expected required columns
    expected_cols = list(required_columns if required_columns is not None else REQUIRED_COLUMNS)
    missing_required = [c for c in expected_cols if c not in df.columns]
    
    # Active constraints mapping
    active_constraints = dict(COLUMN_CONSTRAINTS)
    if custom_constraints:
        active_constraints.update(custom_constraints)

    # 1. Missing values check
    missing_counts = df.isnull().sum()
    total_missing_cells = int(missing_counts.sum())
    missing_cell_ratio = total_missing_cells / total_cells if total_cells > 0 else 0.0

    # 2. Duplicate records check
    duplicate_mask = df.duplicated()
    duplicate_count = int(duplicate_mask.sum())
    duplicate_ratio = duplicate_count / total_records if total_records > 0 else 0.0

    # 3. Constraint & Range checking
    flagged_reasons: Dict[int, List[str]] = {}
    quarantined_indices: Set[int] = set()

    for idx, row in df.iterrows():
        reasons: List[str] = []
        is_quarantine_level = False

        # Duplicate flag
        if duplicate_mask.iloc[idx]:
            reasons.append("Exact duplicate row")
            is_quarantine_level = True

        for col, rules in active_constraints.items():
            if col not in df.columns:
                continue
            
            val = row[col]
            if pd.isna(val):
                reasons.append(f"Missing value in '{col}'")
                # Missing in critical identifiers or key features
                if col in ["customer_id", "loan_status", "annual_income", "credit_score"]:
                    is_quarantine_level = True
                continue

            rule_type = rules.get("type")
            if rule_type == "numeric":
                try:
                    num_val = float(val)
                    if np.isinf(num_val) or np.isnan(num_val):
                        reasons.append(f"'{col}' contains invalid or infinite numerical value: '{val}'")
                        is_quarantine_level = True
                    else:
                        min_val = rules.get("min")
                        max_val = rules.get("max")
                        if min_val is not None and num_val < min_val:
                            reasons.append(f"'{col}' value ({num_val}) below minimum ({min_val})")
                            is_quarantine_level = True
                        elif max_val is not None and num_val > max_val:
                            reasons.append(f"'{col}' value ({num_val}) above maximum ({max_val})")
                            is_quarantine_level = True
                except (ValueError, TypeError):
                    reasons.append(f"'{col}' could not be converted to float: '{val}'")
                    is_quarantine_level = True

            elif rule_type == "categorical":
                allowed = rules.get("allowed", set())
                if str(val) not in allowed:
                    reasons.append(f"Invalid category '{val}' for '{col}' (Allowed: {', '.join(allowed)})")

        if reasons:
            flagged_reasons[idx] = reasons
            if is_quarantine_level:
                quarantined_indices.add(idx)

    # 4. Partition into valid and quarantined
    if quarantine_on_severe_issues and quarantined_indices:
        quarantined_mask = df.index.isin(quarantined_indices)
        quarantined_df = df[quarantined_mask].copy()
        valid_df = df[~quarantined_mask].copy()
    else:
        quarantined_df = pd.DataFrame(columns=df.columns)
        valid_df = df.copy()

    # 5. Calculate Data Quality Score (0-100)
    # Deductions:
    # - Up to 40 points for missing required schema columns
    # - Up to 30 points for missing cells
    # - Up to 20 points for duplicate rows
    # - Up to 35 points for invalid / out-of-range cells
    invalid_cells_ratio = len(flagged_reasons) / total_records if total_records > 0 else 0.0
    
    score = 100.0
    if missing_required:
        score -= min(60.0, len(missing_required) * 12.0)
    score -= min(30.0, missing_cell_ratio * 150.0)
    score -= min(20.0, duplicate_ratio * 100.0)
    score -= min(35.0, invalid_cells_ratio * 60.0)
    quality_score = float(np.clip(round(score, 1), 0.0, 100.0))

    # 6. Build column summary table
    summary_rows = []
    for col in df.columns:
        null_count = int(df[col].isnull().sum())
        null_pct = round((null_count / total_records) * 100.0, 2)
        n_unique = int(df[col].nunique(dropna=True))
        dtype_str = str(df[col].dtype)
        
        min_repr = "-"
        max_repr = "-"
        if pd.api.types.is_numeric_dtype(df[col]):
            non_null = df[col].dropna()
            if not non_null.empty:
                min_repr = f"{non_null.min():.2f}"
                max_repr = f"{non_null.max():.2f}"

        summary_rows.append({
            "Column": col,
            "Type": dtype_str,
            "Missing": null_count,
            "Missing %": null_pct,
            "Unique": n_unique,
            "Min": min_repr,
            "Max": max_repr,
        })
    column_summary = pd.DataFrame(summary_rows)

    # 7. Build reasons dataframe for reporting & export
    reasons_list = []
    for idx, reasons in flagged_reasons.items():
        row_id = df.at[idx, "customer_id"] if "customer_id" in df.columns else str(idx)
        reasons_list.append({
            "row_index": idx,
            "record_id": str(row_id),
            "is_quarantined": idx in quarantined_indices,
            "issue_count": len(reasons),
            "reasons": " | ".join(reasons)
        })
    quarantine_reasons_df = pd.DataFrame(reasons_list)

    # 8. Save quarantined records if any
    quarantine_file_path = None
    if not quarantined_df.empty:
        settings = get_settings()
        settings.quarantine_dir.mkdir(parents=True, exist_ok=True)
        now_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        clean_name = dataset_name.replace(".csv", "").replace(" ", "_")
        q_path = settings.quarantine_dir / f"quarantined_{clean_name}_{now_str}.csv"
        
        # Attach reasons to export
        export_q = quarantined_df.copy()
        export_q["quarantine_reasons"] = export_q.index.map(
            lambda i: " | ".join(flagged_reasons.get(i, ["Quarantined"]))
        )
        export_q.to_csv(q_path, index=True)
        quarantine_file_path = str(q_path)

    is_valid = (
        len(missing_required) == 0
        and quality_score >= 60.0
        and len(valid_df) > 0
    )

    return ValidationReport(
        is_valid=is_valid,
        quality_score=quality_score,
        total_records=total_records,
        valid_records_count=len(valid_df),
        quarantined_records_count=len(quarantined_df),
        duplicate_count=duplicate_count,
        total_missing_cells=total_missing_cells,
        flagged_reasons=flagged_reasons,
        valid_df=valid_df,
        quarantined_df=quarantined_df,
        column_summary=column_summary,
        quarantine_file_path=quarantine_file_path,
        quarantine_reasons_df=quarantine_reasons_df,
        missing_required_columns=missing_required,
    )
