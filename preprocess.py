"""
Preprocess data for GKX (2020) replication.
Applies: cross-sectional ranking, missing value imputation, interactions, industry dummies.
"""

import pandas as pd
import numpy as np
from typing import List


def rank_transform(series: pd.Series) -> pd.Series:
    """Map values to [-1, 1] interval using cross-sectional ranking."""
    if series.isna().all():
        return series

    ranks = series.rank(method='average', na_option='keep')
    n_valid = series.notna().sum()

    if n_valid <= 1:
        return ranks.apply(lambda x: 0 if pd.notna(x) else np.nan)

    return 2 * (ranks - 1) / (n_valid - 1) - 1


def preprocess_data(input_path: str, output_path: str):
    """Preprocess merged data following GKX tutorial."""
    print("  Loading merged data...")
    df = pd.read_csv(input_path, parse_dates=['month'])

    # Identify column types
    char_cols = [col for col in df.columns if col.startswith('characteristic_')]
    macro_cols = [col for col in df.columns if col.startswith('macro_')]

    print(f"  Found {len(char_cols)} characteristics and {len(macro_cols)} macro predictors")

    # Step 1: Cross-sectional ranking
    print("  Applying cross-sectional ranking...")
    for col in char_cols:
        df[col] = df.groupby('month')[col].transform(rank_transform)

    # Step 2: Missing value imputation
    print("  Imputing missing values...")
    for col in char_cols + macro_cols:
        # Fill with cross-sectional median
        df[col] = df.groupby('month')[col].transform(lambda x: x.fillna(x.median()))
        # Fill remaining with 0
        df[col] = df[col].fillna(0)

    # Step 3: Create industry dummies
    print("  Creating industry dummies...")
    if 'sic2' in df.columns:
        industry_dummies = pd.get_dummies(df['sic2'], prefix='sic2', dtype=int)
        df = pd.concat([df, industry_dummies], axis=1)
        df = df.drop(columns=['sic2'])
        print(f"  Created {len(industry_dummies.columns)} industry dummies")

    # Step 4: Create interaction features
    print("  Creating interaction features...")
    interaction_features = []

    for macro in macro_cols:
        for char in char_cols:
            interaction_name = f"{char}_x_{macro}"
            df[interaction_name] = df[char] * df[macro]
            interaction_features.append(interaction_name)

    print(f"  Created {len(interaction_features)} interaction features")

    # Step 5: Drop original characteristics and macros (keep only interactions + industry dummies)
    print("  Dropping original features (keeping only interactions + industry dummies)...")
    df = df.drop(columns=char_cols + macro_cols)

    # Save
    df.to_csv(output_path, index=False)

    print(f"\nPreprocessed data saved to {output_path}")
    print(f"  Final shape: {df.shape}")
    print(f"  Features: {df.shape[1] - 3}")  # Subtract month, permno, ret_excess

    return df


if __name__ == "__main__":
    preprocess_data('./data/merged_data.csv', './data/preprocessed_data.csv')
