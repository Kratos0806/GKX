"""
Utility functions for GKX (2020) replication.
"""

import numpy as np
import pandas as pd
from typing import List, Optional


def load_data(
    filepath: str,
    date_col: str = 'date',
    parse_dates: bool = True
) -> pd.DataFrame:
    """
    Load data from file.

    Parameters
    ----------
    filepath : str
        Path to data file (CSV, parquet, etc.)
    date_col : str
        Name of date column
    parse_dates : bool
        Whether to parse dates

    Returns
    -------
    pd.DataFrame
        Loaded data
    """
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath, parse_dates=[date_col] if parse_dates else None)
    elif filepath.endswith('.parquet'):
        df = pd.read_parquet(filepath)
        if parse_dates and date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col])
    else:
        raise ValueError(f"Unsupported file format: {filepath}")

    return df


def validate_data(
    df: pd.DataFrame,
    required_cols: List[str],
    date_col: str = 'date'
) -> None:
    """
    Validate that data has required columns and proper structure.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    required_cols : List[str]
        Required column names
    date_col : str
        Date column name

    Raises
    ------
    ValueError
        If validation fails
    """
    # Check required columns
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Check date column is datetime
    if date_col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        raise ValueError(f"Column '{date_col}' must be datetime type")

    # Check for duplicate date-stock combinations
    if 'permno' in df.columns:
        duplicates = df.duplicated(subset=[date_col, 'permno']).sum()
        if duplicates > 0:
            raise ValueError(f"Found {duplicates} duplicate date-stock combinations")


def generate_sample_data(
    start_date: str = '1957-03-01',
    end_date: str = '1990-12-01',
    n_stocks: int = 500,
    n_stock_chars: int = 94,
    n_macro_predictors: int = 8,
    n_industries: int = 74,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic data for testing.

    Parameters
    ----------
    start_date : str
        Start date
    end_date : str
        End date
    n_stocks : int
        Number of stocks
    n_stock_chars : int
        Number of stock characteristics
    n_macro_predictors : int
        Number of macro predictors
    n_industries : int
        Number of industries
    seed : int
        Random seed

    Returns
    -------
    pd.DataFrame
        Synthetic dataset
    """
    np.random.seed(seed)

    dates = pd.date_range(start_date, end_date, freq='MS')

    data = []

    # Generate macro predictors (same for all stocks in a period)
    macro_data = {}
    for date in dates:
        macro_data[date] = {
            f'macro_{i}': np.random.normal(0, 0.5)
            for i in range(n_macro_predictors)
        }

    # Generate stock-level data
    for date in dates:
        for stock_id in range(n_stocks):
            row = {
                'date': date,
                'permno': stock_id,
                'ret_excess': np.random.normal(0.01, 0.05),
                'market_cap': np.random.lognormal(10, 2)
            }

            # Add stock characteristics
            for i in range(n_stock_chars):
                if np.random.random() < 0.1:  # 10% missing
                    row[f'char_{i}'] = np.nan
                else:
                    row[f'char_{i}'] = np.random.normal(0, 1)

            # Add macro predictors
            row.update(macro_data[date])

            # Add industry dummies (one-hot encoded)
            stock_industry = stock_id % n_industries
            for i in range(n_industries):
                row[f'industry_{i}'] = 1 if i == stock_industry else 0

            data.append(row)

    df = pd.DataFrame(data)

    print(f"Generated synthetic data:")
    print(f"  Dates: {df['date'].min()} to {df['date'].max()}")
    print(f"  Stocks: {n_stocks}")
    print(f"  Observations: {len(df):,}")
    print(f"  Stock characteristics: {n_stock_chars}")
    print(f"  Macro predictors: {n_macro_predictors}")
    print(f"  Industries: {n_industries}")

    return df


def print_summary_statistics(df: pd.DataFrame, numeric_cols: Optional[List[str]] = None):
    """
    Print summary statistics for dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    numeric_cols : Optional[List[str]]
        Numeric columns to summarize (if None, uses all numeric)
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    print("\nSummary Statistics:")
    print("=" * 80)
    print(df[numeric_cols].describe())
    print("=" * 80)


def save_results(
    results: dict,
    output_dir: str = './results'
):
    """
    Save results to disk.

    Parameters
    ----------
    results : dict
        Dictionary of results (dataframes, metrics, etc.)
    output_dir : str
        Output directory
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    for name, result in results.items():
        if isinstance(result, pd.DataFrame):
            filepath = os.path.join(output_dir, f'{name}.csv')
            result.to_csv(filepath, index=False)
            print(f"Saved {name} to {filepath}")
        elif isinstance(result, dict):
            import json
            filepath = os.path.join(output_dir, f'{name}.json')
            with open(filepath, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            print(f"Saved {name} to {filepath}")


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format value as percentage."""
    return f"{value * 100:.{decimals}f}%"


def format_sharpe_ratio(value: float, decimals: int = 2) -> str:
    """Format Sharpe ratio."""
    return f"{value:.{decimals}f}"


class ProgressTracker:
    """Simple progress tracker for long-running operations."""

    def __init__(self, total: int, description: str = "Progress"):
        self.total = total
        self.current = 0
        self.description = description

    def update(self, n: int = 1):
        """Update progress."""
        self.current += n
        pct = (self.current / self.total) * 100
        print(f"\r{self.description}: {self.current}/{self.total} ({pct:.1f}%)", end='')

    def finish(self):
        """Finish progress tracking."""
        print()


if __name__ == "__main__":
    # Test data generation
    df = generate_sample_data(
        start_date='1957-03-01',
        end_date='1960-12-01',
        n_stocks=100,
        n_stock_chars=10,
        n_macro_predictors=3,
        n_industries=10
    )

    print("\nData shape:", df.shape)
    print("\nFirst few rows:")
    print(df.head())

    # Test validation
    required_cols = ['date', 'permno', 'ret_excess']
    try:
        validate_data(df, required_cols)
        print("\nData validation passed!")
    except ValueError as e:
        print(f"\nData validation failed: {e}")
