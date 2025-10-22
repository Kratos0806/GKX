"""
Example: Using Real Data for GKX (2020) Replication

This script demonstrates how to use the actual data from the tutorial:
- Stock characteristics from Dacheng Xiu's website
- CRSP returns data
- Macroeconomic predictors

Following the exact tutorial from:
https://www.tidy-finance.org/blog/gu-kelly-xiu-replication/
"""

from main import run_replication
from download_data import (
    download_characteristics_data,
    load_characteristics_data,
    get_column_lists,
    merge_datasets
)
import pandas as pd


def example1_characteristics_only():
    """
    Example 1: Run with just the characteristics data from Xiu's website.

    This downloads and uses the stock characteristics data, which already
    includes 94 characteristics and 74 industry dummies.
    """
    print("=" * 80)
    print("EXAMPLE 1: Using Stock Characteristics from Xiu's Website")
    print("=" * 80)

    # Step 1: Download the data (only runs once)
    print("\nStep 1: Downloading stock characteristics...")
    filepath = download_characteristics_data(output_dir='./data')

    # Step 2: Load the data
    print("\nStep 2: Loading data...")
    df = load_characteristics_data(filepath)

    # Step 3: Auto-detect column types
    print("\nStep 3: Detecting columns...")
    cols = get_column_lists(df)

    # Step 4: Run replication
    # Note: The characteristics data already includes returns if merged
    print("\nStep 4: Running replication...")

    # Check if return data is present
    if 'ret_excess' in df.columns:
        results = run_replication(
            data=df,
            stock_characteristics=cols['stock_chars'],
            macro_predictors=cols['macro_preds'],
            industry_dummies=cols['industries'],
            train_start='1957-03',
            train_end='1974-12',
            n_estimators=300,
            output_dir='./results/real_data',
            verbose=1
        )
        return results
    else:
        print("\n⚠ No return data (ret_excess) found in characteristics file.")
        print("You need to merge with CRSP data. See Example 2.")


def example2_full_data_pipeline():
    """
    Example 2: Full pipeline with characteristics + CRSP + macro data.

    This shows how to merge all three datasets as in the tutorial.
    """
    print("\n\n" + "=" * 80)
    print("EXAMPLE 2: Full Data Pipeline (Characteristics + CRSP + Macro)")
    print("=" * 80)

    # Load characteristics data
    print("\nLoading stock characteristics...")
    char_file = './data/datashare.csv'
    try:
        char_df = load_characteristics_data(char_file)
    except FileNotFoundError:
        print(f"File not found: {char_file}")
        print("Run: python download_data.py")
        return

    # Load CRSP data (if available)
    print("\nChecking for CRSP data...")
    crsp_file = './data/crsp_monthly.csv'
    try:
        crsp_df = pd.read_csv(crsp_file)
        print(f"✓ Loaded CRSP data: {len(crsp_df):,} observations")
    except FileNotFoundError:
        print(f"⚠ CRSP data not found at {crsp_file}")
        print("See download_data.py for instructions to obtain CRSP data")
        crsp_df = None

    # Load macro data (if available)
    print("\nChecking for macro predictors...")
    macro_file = './data/macro_predictors.csv'
    try:
        macro_df = pd.read_csv(macro_file)
        print(f"✓ Loaded macro data: {len(macro_df):,} observations")
    except FileNotFoundError:
        print(f"⚠ Macro data not found at {macro_file}")
        print("See download_data.py for instructions to obtain macro data")
        macro_df = None

    # Merge datasets
    print("\nMerging datasets...")
    merged_df = merge_datasets(
        char_df,
        crsp_df=crsp_df,
        macro_df=macro_df
    )

    # Get column lists
    cols = get_column_lists(merged_df)

    # Run replication
    print("\nRunning replication with merged data...")
    results = run_replication(
        data=merged_df,
        stock_characteristics=cols['stock_chars'],
        macro_predictors=cols['macro_preds'],
        industry_dummies=cols['industries'],
        train_start='1957-03',
        train_end='1974-12',
        n_estimators=300,
        weighting='equal',
        output_dir='./results/full_pipeline',
        verbose=1
    )

    return results


def example3_direct_csv_path():
    """
    Example 3: Simplest approach - just pass the CSV path.

    The main script will auto-detect all column types.
    """
    print("\n\n" + "=" * 80)
    print("EXAMPLE 3: Simplest Approach - Direct CSV Path")
    print("=" * 80)

    # Make sure data is downloaded
    print("\nEnsuring data is downloaded...")
    download_characteristics_data(output_dir='./data')

    # Run replication with auto-detection
    print("\nRunning replication (auto-detecting columns)...")
    results = run_replication(
        data_path='./data/datashare.csv',
        train_start='1957-03',
        train_end='1974-12',
        n_estimators=100,  # Reduced for demo
        output_dir='./results/simple',
        verbose=1
    )

    return results


def example4_custom_date_range():
    """
    Example 4: Custom date range and model configuration.
    """
    print("\n\n" + "=" * 80)
    print("EXAMPLE 4: Custom Configuration")
    print("=" * 80)

    results = run_replication(
        data_path='./data/datashare.csv',
        train_start='1960-01',  # Custom start
        train_end='1980-12',    # Custom end
        validation_months=6,     # 6-month validation window
        refit_frequency='annual',
        n_estimators=500,        # More trees
        mtry_values=[10, 20, 50],  # Custom hyperparameter grid
        min_samples_split_values=[1000, 5000],
        n_quantiles=5,           # Quintiles instead of deciles
        weighting='value',       # Value-weighted portfolios
        output_dir='./results/custom',
        verbose=1
    )

    return results


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                  GKX (2020) REPLICATION WITH REAL DATA                       ║
║                                                                              ║
║  This script demonstrates how to use the actual data from the tutorial.     ║
║  Choose which example to run by uncommenting below.                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

    # Choose which example to run:

    # Example 1: Just characteristics data (easiest, downloads automatically)
    example1_characteristics_only()

    # Example 2: Full pipeline with CRSP + macro data
    # (Requires manual download of CRSP and macro data)
    # example2_full_data_pipeline()

    # Example 3: Simplest - just pass file path
    # example3_direct_csv_path()

    # Example 4: Custom configuration
    # example4_custom_date_range()

    print("\n" + "=" * 80)
    print("✓ DONE! Check the ./results/ directory for output files.")
    print("=" * 80)
