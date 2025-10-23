"""
Preprocess data for GKX (2020) replication.
Applies: cross-sectional ranking, missing value imputation, interactions, industry dummies.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple


def rank_transform(series: pd.Series) -> pd.Series:
    """Map values to [-1, 1] interval using cross-sectional ranking."""
    if series.isna().all():
        return series

    n_valid = series.notna().sum()

    if n_valid <= 1:
        # Vectorized version - avoid apply
        result = series.copy()
        result[series.notna()] = 0
        return result

    ranks = series.rank(method='average', na_option='keep')
    return 2 * (ranks - 1) / (n_valid - 1) - 1


def preprocess_data(input_path: str, output_path: str):
    """Preprocess merged data following GKX tutorial."""
    print("  Loading merged data...")
    df = pd.read_csv(input_path, parse_dates=['month'])

    # Identify column types
    char_cols = [col for col in df.columns if col.startswith('characteristic_')]
    macro_cols = [col for col in df.columns if col.startswith('macro_')]

    print(f"  Found {len(char_cols)} characteristics and {len(macro_cols)} macro predictors")

    # Step 1: Cross-sectional ranking (optimized)
    print(f"  Applying cross-sectional ranking to {len(char_cols)} characteristics...")
    print("    (This step processes 3M+ rows and may take 5-10 minutes)")

    # Sort by month once for faster groupby operations
    df = df.sort_values('month')
    print(f"    Processing {df['month'].nunique()} unique months...")

    # Single vectorized transform is most efficient
    df[char_cols] = df.groupby('month', sort=False)[char_cols].transform(rank_transform)
    print("    ✓ Ranking complete")

    # Step 2: Missing value imputation (optimized)
    print("  Imputing missing values...")
    feature_cols = char_cols + macro_cols

    # Fill with cross-sectional median - single operation
    df[feature_cols] = df.groupby('month', sort=False)[feature_cols].transform(
        lambda x: x.fillna(x.median())
    )
    # Fill remaining with 0
    df[feature_cols] = df[feature_cols].fillna(0)
    print("    ✓ Imputation complete")

    # Step 3: Create industry dummies
    print("  Creating industry dummies...")
    if 'sic2' in df.columns:
        industry_dummies = pd.get_dummies(df['sic2'], prefix='sic2', dtype=int)
        df = pd.concat([df, industry_dummies], axis=1)
        df = df.drop(columns=['sic2'])
        print(f"  Created {len(industry_dummies.columns)} industry dummies")

    # Step 4: Create interaction features (optimized with numpy)
    print(f"  Creating interaction features ({len(char_cols)} × {len(macro_cols)} = {len(char_cols) * len(macro_cols)})...")

    # Extract numpy arrays for faster computation
    char_array = df[char_cols].values
    macro_array = df[macro_cols].values

    # Pre-allocate result array for maximum efficiency
    n_interactions = len(char_cols) * len(macro_cols)
    interaction_data = np.empty((len(df), n_interactions), dtype=np.float64)
    interaction_cols = []

    idx = 0
    for i, macro in enumerate(macro_cols):
        for j, char in enumerate(char_cols):
            interaction_cols.append(f"{char}_x_{macro}")
            interaction_data[:, idx] = char_array[:, j] * macro_array[:, i]
            idx += 1

    # Create DataFrame from pre-allocated array
    interaction_df = pd.DataFrame(
        interaction_data,
        columns=interaction_cols,
        index=df.index
    )
    df = pd.concat([df, interaction_df], axis=1)

    print(f"    ✓ Created {len(interaction_cols)} interaction features")

    # Step 5: Drop original characteristics and macros (keep only interactions + industry dummies)
    print("  Dropping original features (keeping only interactions + industry dummies)...")
    df = df.drop(columns=char_cols + macro_cols)

    # Save
    df.to_csv(output_path, index=False)

    print(f"\nPreprocessed data saved to {output_path}")
    print(f"  Final shape: {df.shape}")
    print(f"  Features: {df.shape[1] - 3}")  # Subtract month, permno, ret_excess

    return df


def create_temporal_splits(
    df: pd.DataFrame,
    date_col: str = 'month',
    train_start: str = '1957-03',
    train_end: str = '1974-12',
    validation_months: int = 12,
    refit_frequency: str = 'annual'
) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Create expanding window train-validation-test splits.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with date column
    date_col : str
        Name of date column
    train_start : str
        Start date for initial training period (YYYY-MM format)
    train_end : str
        End date for initial training period (YYYY-MM format)
    validation_months : int
        Number of months for validation window
    refit_frequency : str
        How often to refit model ('annual' or 'monthly')

    Returns
    -------
    List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]
        List of (train, validation, test) splits
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    splits = []

    # Get unique dates
    unique_dates = sorted(df[date_col].unique())

    # Convert train_start and train_end to datetime
    train_start_dt = pd.to_datetime(train_start)
    train_end_dt = pd.to_datetime(train_end)

    # Find initial training end index
    train_end_idx = None
    for idx, date in enumerate(unique_dates):
        if date >= train_end_dt:
            train_end_idx = idx
            break

    if train_end_idx is None:
        raise ValueError("train_end date not found in dataset")

    # Determine step size based on refit frequency
    step = 12 if refit_frequency == 'annual' else 1

    # Create expanding window splits
    current_train_end_idx = train_end_idx

    # Modified condition: we need at least 1 month after train for validation/test
    while current_train_end_idx + 1 < len(unique_dates):
        # Training set: from start to current_train_end_idx
        train_end_date = unique_dates[current_train_end_idx]
        train_df = df[df[date_col] <= train_end_date].copy()

        # Calculate how many months we have left
        months_remaining = len(unique_dates) - current_train_end_idx - 1

        if months_remaining >= validation_months + 1:
            # Standard case: enough data for validation + test
            val_start_idx = current_train_end_idx + 1
            val_end_idx = val_start_idx + validation_months - 1
            val_start_date = unique_dates[val_start_idx]
            val_end_date = unique_dates[val_end_idx]
            val_df = df[
                (df[date_col] >= val_start_date) &
                (df[date_col] <= val_end_date)
            ].copy()

            # Test set: month after validation
            test_idx = val_end_idx + 1
            test_date = unique_dates[test_idx]
            test_df = df[df[date_col] == test_date].copy()
        elif months_remaining > 1:
            # Edge case: use remaining months split between validation and test
            # Use first month(s) for validation, last month for test
            val_months = max(1, months_remaining - 1)
            val_start_idx = current_train_end_idx + 1
            val_end_idx = val_start_idx + val_months - 1
            val_start_date = unique_dates[val_start_idx]
            val_end_date = unique_dates[val_end_idx]
            val_df = df[
                (df[date_col] >= val_start_date) &
                (df[date_col] <= val_end_date)
            ].copy()

            # Test set: last available month
            test_idx = len(unique_dates) - 1
            test_date = unique_dates[test_idx]
            test_df = df[df[date_col] == test_date].copy()
        else:
            # Only 1 month left: use it as both validation and test
            val_idx = current_train_end_idx + 1
            val_date = unique_dates[val_idx]
            val_df = df[df[date_col] == val_date].copy()
            test_df = val_df.copy()

        if len(test_df) > 0 and len(val_df) > 0:
            splits.append((train_df, val_df, test_df))

        # Move forward by step
        current_train_end_idx += step

        # Break if we've used all available data
        if current_train_end_idx >= len(unique_dates) - 1:
            break

    return splits


if __name__ == "__main__":
    preprocess_data('./data/merged_data.csv', './data/preprocessed_data.csv')
