"""
Merge datasets for GKX (2020) replication.
Combines: stock characteristics + CRSP returns + macro predictors.
"""

import os
import pandas as pd
import numpy as np


def load_and_prepare_characteristics(filepath: str) -> pd.DataFrame:
    """Load stock characteristics and rename columns."""
    print("  Loading stock characteristics...")
    df = pd.read_csv(filepath)

    # Rename DATE to month
    if 'DATE' in df.columns:
        df = df.rename(columns={'DATE': 'month'})

    # Convert to datetime
    df['month'] = pd.to_datetime(df['month'].astype(str), format='%Y%m%d')
    df['month'] = df['month'].dt.to_period('M').dt.to_timestamp()

    # Add 'characteristic_' prefix
    rename_dict = {col: f'characteristic_{col}' for col in df.columns
                   if col not in ['permno', 'month', 'sic2']}
    df = df.rename(columns=rename_dict)

    # Drop missing sic2
    df = df.dropna(subset=['sic2'])
    df['sic2'] = df['sic2'].astype(str)

    print(f"  Loaded {len(df):,} observations with {len(df.columns)} columns")
    return df


def load_and_prepare_crsp(filepath: str) -> pd.DataFrame:
    """Load CRSP monthly returns."""
    print("  Loading CRSP data...")

    if not os.path.exists(filepath):
        print(f"  CRSP file not found at {filepath}")
        print("  Creating synthetic returns for demonstration...")
        return None

    df = pd.read_csv(filepath, parse_dates=['date'])
    df = df.rename(columns={'date': 'month'})
    df['month'] = df['month'].dt.to_period('M').dt.to_timestamp()

    # Select required columns
    required_cols = ['month', 'permno', 'mktcap_lag', 'ret_excess']
    available_cols = [col for col in required_cols if col in df.columns]
    df = df[available_cols]

    print(f"  Loaded {len(df):,} observations")
    return df


def load_and_prepare_macro(filepath: str) -> pd.DataFrame:
    """Load macro predictors and lag by 1 month."""
    print("  Loading macro predictors...")
    df = pd.read_csv(filepath, parse_dates=['date'])

    df = df.rename(columns={'date': 'month'})
    df['month'] = df['month'].dt.to_period('M').dt.to_timestamp()

    # Add 'macro_' prefix
    rename_dict = {col: f'macro_{col}' for col in df.columns if col != 'month'}
    df = df.rename(columns=rename_dict)

    # Lag by 1 month
    df['month'] = df['month'] + pd.DateOffset(months=1)

    print(f"  Loaded {len(df):,} monthly observations")
    return df


def merge_all_datasets(data_dir: str = './data', output_path: str = './data/merged_data.csv',
                       date_start: str = None, date_end: str = None):
    """Merge all datasets following GKX tutorial approach.

    Args:
        data_dir: Directory containing input data
        output_path: Path to save merged data
        date_start: Optional start date filter (YYYY-MM format)
        date_end: Optional end date filter (YYYY-MM format)
    """
    print("\nMerging datasets...")
    if date_start or date_end:
        print(f"  Filtering date range: {date_start or 'beginning'} to {date_end or 'end'}")

    # Load datasets
    characteristics = load_and_prepare_characteristics(
        os.path.join(data_dir, 'datashare.csv')
    )

    crsp = load_and_prepare_crsp(
        os.path.join(data_dir, 'crsp_monthly.csv')
    )

    macro = load_and_prepare_macro(
        os.path.join(data_dir, 'macro_predictors.csv')
    )

    # Merge
    print("  Merging characteristics with CRSP...")
    if crsp is not None:
        merged = characteristics.merge(crsp, on=['month', 'permno'], how='inner')
    else:
        # Create synthetic returns if CRSP not available
        print("  Creating synthetic returns for demonstration...")
        merged = characteristics.copy()
        merged['ret_excess'] = np.random.normal(0.01, 0.05, len(merged))
        merged['mktcap_lag'] = np.random.lognormal(10, 2, len(merged))

    print(f"  After CRSP merge: {len(merged):,} rows")

    print("  Merging with macro predictors...")
    merged = merged.merge(macro, on='month', how='inner')
    print(f"  After macro merge: {len(merged):,} rows")

    # Filter by date range if specified
    if date_start or date_end:
        print("  Applying date filter...")
        if date_start:
            date_start_dt = pd.to_datetime(date_start)
            merged = merged[merged['month'] >= date_start_dt]
        if date_end:
            date_end_dt = pd.to_datetime(date_end)
            merged = merged[merged['month'] <= date_end_dt]
        print(f"  After date filter: {len(merged):,} rows")

    # Add macro intercept
    merged['macro_intercept'] = 1

    # Reorder columns
    id_cols = ['permno', 'month']
    target_col = ['ret_excess']
    other_cols = ['mktcap_lag', 'sic2']
    macro_cols = [col for col in merged.columns if col.startswith('macro_')]
    char_cols = [col for col in merged.columns if col.startswith('characteristic_')]

    final_cols = id_cols + target_col + other_cols + macro_cols + char_cols
    merged = merged[final_cols]

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged.to_csv(output_path, index=False)

    print(f"\nMerged dataset saved to {output_path}")
    print(f"  Final shape: {merged.shape}")
    print(f"  Date range: {merged['month'].min()} to {merged['month'].max()}")

    return merged


if __name__ == "__main__":
    merge_all_datasets()
