"""
GKX (2020) Replication Pipeline
Simple orchestration script that runs the complete workflow.
"""

import argparse
import os
from pathlib import Path
import pandas as pd

# Import pipeline modules
from download import download_all_data
from merge import merge_all_datasets
from preprocess import preprocess_data
from model_training import train_and_predict
from portfolio_construction import construct_portfolios, analyze_performance


def main():
    parser = argparse.ArgumentParser(
        description="GKX (2020) Replication Pipeline"
    )

    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip data download (use existing data)'
    )

    parser.add_argument(
        '--skip-merge',
        action='store_true',
        help='Skip data merging (use existing merged data)'
    )

    parser.add_argument(
        '--skip-preprocess',
        action='store_true',
        help='Skip preprocessing (use existing preprocessed data)'
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        default='./data',
        help='Directory for data files'
    )

    parser.add_argument(
        '--results-dir',
        type=str,
        default='./results',
        help='Directory for results'
    )

    parser.add_argument(
        '--train-start',
        type=str,
        default='2015-01',
        help='Training start date (YYYY-MM)'
    )

    parser.add_argument(
        '--train-end',
        type=str,
        default='2020-12',
        help='Initial training end date (YYYY-MM)'
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
        '--validation-months',
        type=int,
        default=12,
        help='Number of months for validation period'
    )

    parser.add_argument(
        '--test-buffer-months',
        type=int,
        default=12,
        help='Number of additional months to keep after train_end for testing'
    )

    args = parser.parse_args()

    # Create directories
    Path(args.data_dir).mkdir(parents=True, exist_ok=True)
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("GKX (2020) REPLICATION PIPELINE")
    print("=" * 80)

    # Step 1: Download data
    if not args.skip_download:
        print("\n[STEP 1/5] Downloading data...")
        download_all_data(data_dir=args.data_dir)
    else:
        print("\n[STEP 1/5] Skipping download (using existing data)")

    # Step 2: Merge datasets
    merged_path = os.path.join(args.data_dir, 'merged_data.csv')
    if not args.skip_merge:
        print("\n[STEP 2/5] Merging datasets...")
        # Filter to date range needed for training + validation + testing
        # Add buffer before training start (for initial window)
        filter_start = pd.to_datetime(args.train_start) - pd.DateOffset(months=12)
        # Add buffer after training end (validation + test)
        total_buffer = args.validation_months + args.test_buffer_months
        filter_end = pd.to_datetime(args.train_end) + pd.DateOffset(months=total_buffer)

        print(f"  Filtering data from {filter_start.strftime('%Y-%m')} to {filter_end.strftime('%Y-%m')}")
        print(f"  (12 months before train_start + {total_buffer} months after train_end)")
        merge_all_datasets(
            data_dir=args.data_dir,
            output_path=merged_path,
            date_start=filter_start.strftime('%Y-%m'),
            date_end=filter_end.strftime('%Y-%m')
        )
    else:
        print("\n[STEP 2/5] Skipping merge (using existing merged data)")

    # Step 3: Preprocess data
    preprocessed_path = os.path.join(args.data_dir, 'preprocessed_data.csv')
    if not args.skip_preprocess:
        print("\n[STEP 3/5] Preprocessing data...")
        preprocess_data(
            input_path=merged_path,
            output_path=preprocessed_path
        )
    else:
        print("\n[STEP 3/5] Skipping preprocessing (using existing preprocessed data)")

    # Step 4: Train models and generate predictions
    print("\n[STEP 4/5] Training models and generating predictions...")
    predictions_path = train_and_predict(
        data_path=preprocessed_path,
        train_start=args.train_start,
        train_end=args.train_end,
        n_estimators=args.n_estimators,
        results_dir=args.results_dir,
        validation_months=args.validation_months
    )

    # Step 5: Construct portfolios and analyze performance
    print("\n[STEP 5/5] Constructing portfolios and analyzing performance...")
    portfolio_returns = construct_portfolios(
        predictions_path=predictions_path,
        n_quantiles=args.n_quantiles
    )

    performance_metrics = analyze_performance(
        portfolio_returns=portfolio_returns,
        results_dir=args.results_dir
    )

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {args.results_dir}/")

    # Display key results
    if performance_metrics is not None:
        print("\nKey Performance Metrics:")
        ls_metrics = performance_metrics[performance_metrics['quantile'] == 'long_short']
        if len(ls_metrics) > 0:
            sharpe = ls_metrics['sharpe_ratio'].iloc[0]
            annual_ret = ls_metrics['mean_return'].iloc[0] * 12 * 100
            print(f"  Long-Short Sharpe Ratio: {sharpe:.2f}")
            print(f"  Long-Short Annual Return: {annual_ret:.2f}%")


if __name__ == "__main__":
    main()
