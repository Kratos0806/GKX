"""
Main execution script for Gu, Kelly & Xiu (2020) replication in Python.

This script demonstrates the complete pipeline:
1. Load and preprocess data
2. Create expanding window splits
3. Train Random Forest with hyperparameter tuning
4. Generate out-of-sample predictions
5. Construct portfolios and evaluate performance
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict
import argparse
import time

from data_preprocessing import GKXPreprocessor, create_temporal_splits
from model_training import GKXRandomForest, OutOfSamplePredictor
from portfolio_construction import PortfolioBacktest, PerformanceAnalyzer
from utils import (
    load_data,
    validate_data,
    generate_sample_data,
    save_results,
    format_percentage,
    format_sharpe_ratio
)


def run_replication(
    data: Optional[pd.DataFrame] = None,
    data_path: Optional[str] = None,
    stock_characteristics: Optional[list] = None,
    macro_predictors: Optional[list] = None,
    industry_dummies: Optional[list] = None,
    train_start: str = '1957-03',
    train_end: str = '1974-12',
    validation_months: int = 12,
    refit_frequency: str = 'annual',
    n_estimators: int = 300,
    mtry_values: Optional[list] = None,
    min_samples_split_values: Optional[list] = None,
    n_quantiles: int = 10,
    weighting: str = 'equal',
    output_dir: str = './results',
    verbose: int = 1
) -> Dict:
    """
    Run complete GKX (2020) replication pipeline.

    Parameters
    ----------
    data : Optional[pd.DataFrame]
        Input data (if None, will load from data_path or generate synthetic)
    data_path : Optional[str]
        Path to data file
    stock_characteristics : Optional[list]
        List of stock characteristic column names
    macro_predictors : Optional[list]
        List of macro predictor column names
    industry_dummies : Optional[list]
        List of industry dummy column names
    train_start : str
        Start of initial training period
    train_end : str
        End of initial training period
    validation_months : int
        Number of months for validation window
    refit_frequency : str
        Model refit frequency ('annual' or 'monthly')
    n_estimators : int
        Number of trees in Random Forest
    mtry_values : Optional[list]
        Values for max_features to tune
    min_samples_split_values : Optional[list]
        Values for min_samples_split to tune
    n_quantiles : int
        Number of portfolio quantiles
    weighting : str
        Portfolio weighting scheme ('equal' or 'value')
    output_dir : str
        Directory to save results
    verbose : int
        Verbosity level

    Returns
    -------
    Dict
        Dictionary containing all results
    """
    if verbose:
        print("\n" + "=" * 80)
        print("GU, KELLY & XIU (2020) REPLICATION IN PYTHON")
        print("=" * 80 + "\n")

    # -------------------------------------------------------------------------
    # Step 1: Load or generate data
    # -------------------------------------------------------------------------
    if verbose:
        print("Step 1: Loading data...")

    if data is not None:
        df = data.copy()
    elif data_path is not None:
        df = load_data(data_path)

        # Auto-detect column names from real data if not provided
        if stock_characteristics is None or macro_predictors is None or industry_dummies is None:
            if verbose:
                print("  Auto-detecting column types...")

            # Define the 8 macro predictors (Welch & Goyal 2008)
            known_macros = ['dp', 'ep', 'bm', 'ntis', 'tbl', 'tms', 'dfy', 'svar']

            # Check for categorical industry column (sic2)
            industry_column = None
            detected_industries = []

            if 'sic2' in df.columns:
                # GKX tutorial uses categorical sic2 column
                industry_column = 'sic2'
                if verbose:
                    print(f"    Found categorical industry column: 'sic2' (will be one-hot encoded)")
            else:
                # Identify pre-existing industry dummies (SIC-based)
                detected_industries = [col for col in df.columns if col.lower().startswith('sic')]

            # Stock characteristics are everything else
            exclude_cols = {'date', 'DATE', 'permno', 'PERMNO', 'ret_excess',
                          'mktcap', 'mktcap_lag', 'sic2'} | set(known_macros) | set(detected_industries)
            detected_chars = [col for col in df.columns if col not in exclude_cols]

            # Use detected values if not provided
            if stock_characteristics is None:
                stock_characteristics = detected_chars
            if macro_predictors is None:
                macro_predictors = [m for m in known_macros if m in df.columns]
            if industry_dummies is None:
                industry_dummies = detected_industries

            if verbose:
                print(f"    Detected {len(stock_characteristics)} stock characteristics")
                print(f"    Detected {len(macro_predictors)} macro predictors")
                if industry_column:
                    print(f"    Found categorical industry column: '{industry_column}'")
                else:
                    print(f"    Detected {len(industry_dummies)} pre-existing industry dummies")
    else:
        if verbose:
            print("  No data provided. Generating synthetic data for demonstration...")
            print("  To use real data from GKX (2020), run: python download_data.py")
        df = generate_sample_data(
            start_date=train_start + '-01',
            end_date='1990-12-01',
            n_stocks=200,
            n_stock_chars=20,
            n_macro_predictors=3,
            n_industries=10,
            seed=42
        )

        # Auto-detect column names from generated data
        if stock_characteristics is None:
            stock_characteristics = [col for col in df.columns if col.startswith('char_')]
        if macro_predictors is None:
            macro_predictors = [col for col in df.columns if col.startswith('macro_')]
        if industry_dummies is None:
            industry_dummies = [col for col in df.columns if col.startswith('industry_')]

    if verbose:
        print(f"  Data loaded: {len(df):,} observations")
        print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"  Unique stocks: {df['permno'].nunique()}")

    # Validate data
    required_cols = ['date', 'permno', 'ret_excess'] + \
                    (stock_characteristics or []) + \
                    (macro_predictors or []) + \
                    (industry_dummies or [])
    validate_data(df, required_cols)

    # -------------------------------------------------------------------------
    # Step 2: Preprocess data
    # -------------------------------------------------------------------------
    if verbose:
        print("\nStep 2: Preprocessing data...")

    # Determine industry column handling
    if 'industry_column' in locals() and industry_column is not None:
        # Use categorical industry column (will be one-hot encoded)
        preprocessor = GKXPreprocessor(
            stock_characteristics=stock_characteristics,
            macro_predictors=macro_predictors,
            industry_column=industry_column
        )
    else:
        # Use pre-existing industry dummies
        preprocessor = GKXPreprocessor(
            stock_characteristics=stock_characteristics,
            macro_predictors=macro_predictors,
            industry_dummies=industry_dummies
        )

    df_processed = preprocessor.fit_transform(df)

    feature_cols = preprocessor.get_all_features()

    if verbose:
        print(f"\n  Feature composition:")
        print(f"    - Stock characteristics (before interaction): {len(stock_characteristics)}")
        print(f"    - Macro predictors (before interaction): {len(macro_predictors)} + 1 intercept")
        total_industries = len(industry_dummies) + len(preprocessor.created_industry_dummies)
        print(f"    - Industry dummies: {total_industries}")
        print(f"    - Interaction features: {len(preprocessor.interaction_features)}")
        print(f"    - TOTAL FEATURES: {len(feature_cols)}")

        # Expected for GKX dataset: 94 chars × 9 macros + 74 industries = 920
        if len(stock_characteristics) == 94 and len(macro_predictors) == 8:
            expected_features = 94 * 9 + 74
            if len(feature_cols) == expected_features:
                print(f"    ✓ Feature count matches GKX (2020): {expected_features}")
            else:
                print(f"    ⚠ Expected {expected_features} features for GKX dataset, got {len(feature_cols)}")

    # -------------------------------------------------------------------------
    # Step 3: Create temporal splits
    # -------------------------------------------------------------------------
    if verbose:
        print("\nStep 3: Creating temporal splits...")

    splits = create_temporal_splits(
        df_processed,
        train_start=train_start,
        train_end=train_end,
        validation_months=validation_months,
        refit_frequency=refit_frequency
    )

    if verbose:
        print(f"  Created {len(splits)} train-validation-test splits")
        if len(splits) > 0:
            print(f"  First split:")
            print(f"    Train: {splits[0][0]['date'].min()} to {splits[0][0]['date'].max()}")
            print(f"    Val:   {splits[0][1]['date'].min()} to {splits[0][1]['date'].max()}")
            print(f"    Test:  {splits[0][2]['date'].unique()[0]}")

    # -------------------------------------------------------------------------
    # Step 4: Train models and generate predictions
    # -------------------------------------------------------------------------
    if verbose:
        print("\nStep 4: Training models and generating predictions...")

    start_time = time.time()

    predictor = OutOfSamplePredictor(
        model_class=GKXRandomForest,
        model_params={
            'n_estimators': n_estimators,
            'mtry_values': mtry_values or [3, 5, 10, 20, 30, 50],
            'min_samples_split_values': min_samples_split_values or [5000, 10000],
            'verbose': verbose
        },
        verbose=verbose
    )

    predictions_df = predictor.run_expanding_window(
        splits=splits,
        feature_cols=feature_cols
    )

    elapsed_time = time.time() - start_time

    if verbose:
        print(f"\n  Total training time: {elapsed_time/60:.2f} minutes")
        print(f"  Average time per split: {elapsed_time/len(splits):.2f} seconds")

    # -------------------------------------------------------------------------
    # Step 5: Construct portfolios
    # -------------------------------------------------------------------------
    if verbose:
        print("\nStep 5: Constructing portfolios...")

    backtester = PortfolioBacktest(
        n_quantiles=n_quantiles,
        weighting=weighting,
        long_short=True
    )

    portfolio_returns = backtester.construct_portfolios(
        predictions_df,
        market_cap_col='market_cap' if weighting == 'value' else None
    )

    if verbose:
        print(f"  Portfolio returns calculated for {len(portfolio_returns):,} quantile-periods")

    # -------------------------------------------------------------------------
    # Step 6: Calculate performance metrics
    # -------------------------------------------------------------------------
    if verbose:
        print("\nStep 6: Calculating performance metrics...")

    performance_metrics = backtester.calculate_performance_metrics()

    # -------------------------------------------------------------------------
    # Step 7: Display results
    # -------------------------------------------------------------------------
    if verbose:
        print("\n" + "=" * 80)
        print("PERFORMANCE RESULTS")
        print("=" * 80 + "\n")

        # Format and display metrics
        display_metrics = performance_metrics.copy()
        display_metrics['mean_return'] = display_metrics['mean_return'].apply(
            lambda x: format_percentage(x)
        )
        display_metrics['std_return'] = display_metrics['std_return'].apply(
            lambda x: format_percentage(x)
        )
        display_metrics['sharpe_ratio'] = display_metrics['sharpe_ratio'].apply(
            lambda x: format_sharpe_ratio(x)
        )

        print(display_metrics.to_string(index=False))

        # Highlight long-short performance
        if 'long_short' in performance_metrics['quantile'].values:
            print("\n" + "-" * 80)
            ls_metrics = performance_metrics[
                performance_metrics['quantile'] == 'long_short'
            ].iloc[0]
            print("LONG-SHORT PORTFOLIO (Decile 10 - Decile 1):")
            print(f"  Sharpe Ratio:        {ls_metrics['sharpe_ratio']:.2f}")
            print(f"  Annualized Return:   {format_percentage(ls_metrics['mean_return'])}")
            print(f"  Annualized Std Dev:  {format_percentage(ls_metrics['std_return'])}")
            print(f"  Cumulative Return:   {format_percentage(ls_metrics['cumulative_return'])}")
            print("-" * 80)

        # Summary statistics
        summary_stats = backtester.get_summary_statistics()
        print("\nSUMMARY STATISTICS:")
        print(summary_stats.to_string(index=False))

    # -------------------------------------------------------------------------
    # Step 8: Save results
    # -------------------------------------------------------------------------
    if verbose:
        print(f"\nStep 8: Saving results to {output_dir}...")

    results = {
        'predictions': predictions_df,
        'portfolio_returns': portfolio_returns,
        'performance_metrics': performance_metrics,
        'summary_statistics': summary_stats
    }

    save_results(results, output_dir)

    if verbose:
        print("\n" + "=" * 80)
        print("REPLICATION COMPLETE!")
        print("=" * 80 + "\n")

    return results


def main():
    """Command-line interface for running replication."""
    parser = argparse.ArgumentParser(
        description="GKX (2020) Replication in Python"
    )

    parser.add_argument(
        '--data-path',
        type=str,
        default=None,
        help='Path to input data file (CSV or parquet)'
    )

    parser.add_argument(
        '--train-start',
        type=str,
        default='1957-03',
        help='Start of initial training period (YYYY-MM)'
    )

    parser.add_argument(
        '--train-end',
        type=str,
        default='1974-12',
        help='End of initial training period (YYYY-MM)'
    )

    parser.add_argument(
        '--validation-months',
        type=int,
        default=12,
        help='Number of months for validation window'
    )

    parser.add_argument(
        '--refit-frequency',
        type=str,
        choices=['annual', 'monthly'],
        default='annual',
        help='Model refit frequency'
    )

    parser.add_argument(
        '--n-estimators',
        type=int,
        default=300,
        help='Number of trees in Random Forest'
    )

    parser.add_argument(
        '--n-quantiles',
        type=int,
        default=10,
        help='Number of portfolio quantiles'
    )

    parser.add_argument(
        '--weighting',
        type=str,
        choices=['equal', 'value'],
        default='equal',
        help='Portfolio weighting scheme'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='./results',
        help='Directory to save results'
    )

    parser.add_argument(
        '--verbose',
        type=int,
        default=1,
        help='Verbosity level (0=quiet, 1=normal, 2=detailed)'
    )

    args = parser.parse_args()

    # Run replication
    results = run_replication(
        data_path=args.data_path,
        train_start=args.train_start,
        train_end=args.train_end,
        validation_months=args.validation_months,
        refit_frequency=args.refit_frequency,
        n_estimators=args.n_estimators,
        n_quantiles=args.n_quantiles,
        weighting=args.weighting,
        output_dir=args.output_dir,
        verbose=args.verbose
    )

    return results


if __name__ == "__main__":
    # Example: Run with synthetic data
    results = main()
