"""
Test script to verify the GKX replication installation.

Run this script to check if all dependencies are installed correctly
and the basic functionality works.
"""

import sys
import warnings

# Suppress runtime warnings from numpy for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)


def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")

    required_packages = [
        ('numpy', 'np'),
        ('pandas', 'pd'),
        ('sklearn', None),
        ('scipy', None),
        ('joblib', None),
    ]

    failed = []

    for package_name, alias in required_packages:
        try:
            if alias:
                exec(f"import {package_name} as {alias}")
            else:
                exec(f"import {package_name}")
            print(f"  ✓ {package_name}")
        except ImportError as e:
            print(f"  ✗ {package_name}: {e}")
            failed.append(package_name)

    if failed:
        print(f"\nFailed to import: {', '.join(failed)}")
        print("Please install missing packages with: pip install -r requirements.txt")
        return False

    print("  All imports successful!\n")
    return True


def test_modules():
    """Test that all custom modules can be imported."""
    print("Testing custom modules...")

    modules = [
        'data_preprocessing',
        'model_training',
        'portfolio_construction',
        'utils',
        'main'
    ]

    failed = []

    for module in modules:
        try:
            exec(f"import {module}")
            print(f"  ✓ {module}")
        except Exception as e:
            print(f"  ✗ {module}: {e}")
            failed.append(module)

    if failed:
        print(f"\nFailed to import modules: {', '.join(failed)}")
        return False

    print("  All modules imported successfully!\n")
    return True


def test_basic_functionality():
    """Test basic functionality with minimal synthetic data."""
    print("Testing basic functionality...")

    try:
        from utils import generate_sample_data
        from data_preprocessing import GKXPreprocessor

        # Generate small dataset
        print("  Generating sample data...")
        df = generate_sample_data(
            start_date='1957-03-01',
            end_date='1958-12-01',
            n_stocks=50,
            n_stock_chars=5,
            n_macro_predictors=2,
            n_industries=5,
            seed=42
        )

        # Test preprocessing
        print("  Testing preprocessing...")
        stock_chars = [col for col in df.columns if col.startswith('char_')]
        macro_preds = [col for col in df.columns if col.startswith('macro_')]
        industries = [col for col in df.columns if col.startswith('industry_')]

        preprocessor = GKXPreprocessor(
            stock_characteristics=stock_chars,
            macro_predictors=macro_preds,
            industry_dummies=industries
        )

        df_processed = preprocessor.fit_transform(df)

        print(f"    Data shape: {df_processed.shape}")
        print(f"    Features: {len(preprocessor.get_all_features())}")

        # Test model training
        print("  Testing model training...")
        from model_training import GKXRandomForest
        import pandas as pd
        import numpy as np

        feature_cols = preprocessor.get_all_features()
        X = df_processed[feature_cols].head(100)
        y = df_processed['ret_excess'].head(100)

        model = GKXRandomForest(
            n_estimators=10,
            mtry_values=[3, 5],
            min_samples_split_values=[10],
            verbose=0
        )

        model.fit(
            X, pd.Series(y.values),
            X.head(20), pd.Series(y.head(20).values)
        )

        predictions = model.predict(X.head(10))
        print(f"    Predictions shape: {predictions.shape}")

        # Test portfolio construction
        print("  Testing portfolio construction...")
        from portfolio_construction import PortfolioBacktest

        pred_df = pd.DataFrame({
            'permno': range(100),
            'date': pd.Timestamp('2020-01-01'),
            'predicted_return': np.random.randn(100) * 0.02,
            'actual_return': np.random.randn(100) * 0.02,
            'market_cap': np.random.lognormal(10, 1, 100)
        })

        backtester = PortfolioBacktest(n_quantiles=5, weighting='equal')
        portfolio_returns = backtester.construct_portfolios(pred_df)
        metrics = backtester.calculate_performance_metrics()

        print(f"    Portfolio returns shape: {portfolio_returns.shape}")
        print(f"    Metrics shape: {metrics.shape}")

        print("  ✓ All functionality tests passed!\n")
        return True

    except Exception as e:
        print(f"  ✗ Functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_minimal_pipeline():
    """Test minimal end-to-end pipeline."""
    print("Testing minimal pipeline...")

    try:
        from main import run_replication

        # Run very small replication for testing
        print("  Running minimal replication...")
        results = run_replication(
            train_start='1957-03',
            train_end='1958-12',
            validation_months=6,
            n_estimators=10,  # Minimal for testing
            n_quantiles=5,
            output_dir='./test_results',
            verbose=0
        )

        print(f"    Predictions: {len(results['predictions'])} rows")
        print(f"    Portfolio returns: {len(results['portfolio_returns'])} rows")
        print(f"    Performance metrics: {len(results['performance_metrics'])} rows")

        print("  ✓ Pipeline test passed!\n")
        return True

    except Exception as e:
        print(f"  ✗ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 80)
    print("GKX (2020) REPLICATION - INSTALLATION TEST")
    print("=" * 80 + "\n")

    all_passed = True

    # Test imports
    if not test_imports():
        all_passed = False
        print("\n⚠ Import test failed. Please install dependencies first.")
        print("Run: pip install -r requirements.txt\n")
        return False

    # Test custom modules
    if not test_modules():
        all_passed = False
        print("\n⚠ Module import test failed.\n")
        return False

    # Test basic functionality
    if not test_basic_functionality():
        all_passed = False
        print("\n⚠ Functionality test failed.\n")
        return False

    # Test minimal pipeline (optional - can be slow)
    # Uncomment to test full pipeline
    # if not test_minimal_pipeline():
    #     all_passed = False
    #     print("\n⚠ Pipeline test failed.\n")
    #     return False

    # Summary
    print("=" * 80)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
        print("\nInstallation successful. You're ready to run the replication.")
        print("\nNext steps:")
        print("  1. Run examples: python example.py")
        print("  2. Run main script: python main.py")
        print("  3. Or use the Python API (see README.md)")
    else:
        print("✗ SOME TESTS FAILED")
        print("\nPlease fix the issues above before proceeding.")
    print("=" * 80)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
