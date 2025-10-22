"""
Example usage of the GKX (2020) replication pipeline.

This script demonstrates how to use the replication code with synthetic data.
"""

from main import run_replication
from utils import generate_sample_data


def example_basic_usage():
    """
    Example 1: Basic usage with synthetic data.
    """
    print("=" * 80)
    print("EXAMPLE 1: Basic Usage with Synthetic Data")
    print("=" * 80)

    # The run_replication function will automatically generate synthetic data
    # if no data is provided
    results = run_replication(
        train_start='1957-03',
        train_end='1974-12',
        validation_months=12,
        refit_frequency='annual',
        n_estimators=100,  # Reduced for faster demo
        n_quantiles=10,
        weighting='equal',
        output_dir='./results/example1',
        verbose=1
    )

    return results


def example_custom_data():
    """
    Example 2: Usage with custom generated data.
    """
    print("\n\n" + "=" * 80)
    print("EXAMPLE 2: Custom Data Generation")
    print("=" * 80)

    # Generate custom synthetic data
    df = generate_sample_data(
        start_date='1957-03-01',
        end_date='1980-12-01',
        n_stocks=300,
        n_stock_chars=50,
        n_macro_predictors=5,
        n_industries=20,
        seed=123
    )

    # Define column names
    stock_chars = [col for col in df.columns if col.startswith('char_')]
    macro_preds = [col for col in df.columns if col.startswith('macro_')]
    industries = [col for col in df.columns if col.startswith('industry_')]

    # Run replication with custom data
    results = run_replication(
        data=df,
        stock_characteristics=stock_chars,
        macro_predictors=macro_preds,
        industry_dummies=industries,
        train_start='1957-03',
        train_end='1970-12',
        validation_months=12,
        refit_frequency='annual',
        n_estimators=100,
        mtry_values=[5, 10, 20],  # Custom hyperparameter grid
        min_samples_split_values=[1000, 5000],
        n_quantiles=5,  # Quintiles instead of deciles
        weighting='equal',
        output_dir='./results/example2',
        verbose=1
    )

    return results


def example_value_weighted():
    """
    Example 3: Value-weighted portfolios.
    """
    print("\n\n" + "=" * 80)
    print("EXAMPLE 3: Value-Weighted Portfolios")
    print("=" * 80)

    results = run_replication(
        train_start='1957-03',
        train_end='1974-12',
        validation_months=12,
        refit_frequency='annual',
        n_estimators=100,
        n_quantiles=10,
        weighting='value',  # Value-weighted instead of equal-weighted
        output_dir='./results/example3',
        verbose=1
    )

    return results


def example_analysis():
    """
    Example 4: Detailed analysis of results.
    """
    print("\n\n" + "=" * 80)
    print("EXAMPLE 4: Detailed Results Analysis")
    print("=" * 80)

    # Run replication
    results = run_replication(
        train_start='1957-03',
        train_end='1974-12',
        n_estimators=100,
        output_dir='./results/example4',
        verbose=1
    )

    # Additional analysis
    print("\n" + "-" * 80)
    print("ADDITIONAL ANALYSIS")
    print("-" * 80)

    # Analyze predictions
    predictions = results['predictions']
    print(f"\nPrediction correlation with actual returns:")
    corr = predictions['predicted_return'].corr(predictions['actual_return'])
    print(f"  Correlation: {corr:.4f}")

    # Analyze portfolio returns distribution
    portfolio_returns = results['portfolio_returns']

    print(f"\nPortfolio returns by quantile:")
    quantile_means = portfolio_returns.groupby('quantile')['return'].mean()
    print(quantile_means)

    # Compare performance across quantiles
    performance = results['performance_metrics']

    print(f"\nSharpe ratios by quantile:")
    sharpe_by_quantile = performance.set_index('quantile')['sharpe_ratio']
    print(sharpe_by_quantile)

    # Plot cumulative returns (if matplotlib available)
    try:
        from portfolio_construction import PortfolioBacktest
        import matplotlib.pyplot as plt

        backtester = PortfolioBacktest()
        backtester.portfolio_returns = portfolio_returns

        # Plot deciles 1, 5, 10, and long-short
        quantiles_to_plot = [1, 5, 10, 'long_short']
        backtester.plot_cumulative_returns(
            quantiles_to_plot=quantiles_to_plot,
            figsize=(14, 7)
        )
        plt.savefig('./results/example4/cumulative_returns.png', dpi=300, bbox_inches='tight')
        print("\nCumulative returns plot saved to ./results/example4/cumulative_returns.png")

    except ImportError:
        print("\nMatplotlib not available. Skipping plot.")

    return results


def example_minimal():
    """
    Example 5: Minimal working example.
    """
    print("\n\n" + "=" * 80)
    print("EXAMPLE 5: Minimal Working Example")
    print("=" * 80)

    # Single line to run complete replication
    results = run_replication(n_estimators=50, verbose=1)

    # Access results
    print(f"\nNumber of predictions: {len(results['predictions'])}")
    print(f"Number of models trained: {len(results['predictions']['date'].unique())}")

    # Get long-short Sharpe ratio
    ls_sharpe = results['performance_metrics'][
        results['performance_metrics']['quantile'] == 'long_short'
    ]['sharpe_ratio'].values[0]

    print(f"Long-Short Sharpe Ratio: {ls_sharpe:.2f}")

    return results


if __name__ == "__main__":
    # Run different examples
    # Uncomment the example you want to run:

    # Basic usage
    example_basic_usage()

    # Custom data
    # example_custom_data()

    # Value-weighted portfolios
    # example_value_weighted()

    # Detailed analysis
    # example_analysis()

    # Minimal example
    # example_minimal()
