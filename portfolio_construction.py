"""
Portfolio construction and backtesting module for GKX (2020) replication.

Implements decile portfolio construction and performance evaluation.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Literal


class PortfolioBacktest:
    """
    Portfolio construction and backtesting based on predicted returns.
    """

    def __init__(
        self,
        n_quantiles: int = 10,
        weighting: Literal['equal', 'value'] = 'equal',
        long_short: bool = True
    ):
        """
        Initialize portfolio backtester.

        Parameters
        ----------
        n_quantiles : int
            Number of quantiles for portfolio sorting (10 for deciles)
        weighting : Literal['equal', 'value']
            Weighting scheme for portfolio
        long_short : bool
            Whether to construct long-short portfolio (top - bottom)
        """
        self.n_quantiles = n_quantiles
        self.weighting = weighting
        self.long_short = long_short

        self.portfolio_returns = None
        self.performance_metrics = None

    def construct_portfolios(
        self,
        predictions_df: pd.DataFrame,
        stock_id_col: str = 'permno',
        date_col: str = 'date',
        predicted_col: str = 'predicted_return',
        actual_col: str = 'actual_return',
        market_cap_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Construct quantile portfolios based on predicted returns.

        Parameters
        ----------
        predictions_df : pd.DataFrame
            DataFrame with predictions and actual returns
        stock_id_col : str
            Stock identifier column
        date_col : str
            Date column
        predicted_col : str
            Predicted return column
        actual_col : str
            Actual return column
        market_cap_col : Optional[str]
            Market capitalization column (for value weighting)

        Returns
        -------
        pd.DataFrame
            Portfolio returns by date and quantile
        """
        if self.weighting == 'value' and market_cap_col is None:
            raise ValueError("market_cap_col required for value weighting")

        portfolio_returns = []

        # Group by date
        for date, group in predictions_df.groupby(date_col):
            # Remove any NaN predictions or returns
            group = group.dropna(subset=[predicted_col, actual_col])

            if len(group) == 0:
                continue

            # Assign stocks to quantiles based on predicted returns
            group['quantile'] = pd.qcut(
                group[predicted_col],
                q=self.n_quantiles,
                labels=False,
                duplicates='drop'
            ) + 1  # Quantiles from 1 to n_quantiles

            # Calculate portfolio returns for each quantile
            for quantile in range(1, self.n_quantiles + 1):
                quantile_stocks = group[group['quantile'] == quantile]

                if len(quantile_stocks) == 0:
                    continue

                if self.weighting == 'equal':
                    # Equal-weighted portfolio
                    portfolio_return = quantile_stocks[actual_col].mean()
                else:
                    # Value-weighted portfolio
                    weights = quantile_stocks[market_cap_col] / quantile_stocks[market_cap_col].sum()
                    portfolio_return = (quantile_stocks[actual_col] * weights).sum()

                portfolio_returns.append({
                    date_col: date,
                    'quantile': quantile,
                    'return': portfolio_return,
                    'n_stocks': len(quantile_stocks)
                })

        portfolio_returns_df = pd.DataFrame(portfolio_returns)

        # If long-short, calculate spread
        if self.long_short and len(portfolio_returns_df) > 0:
            long_short_returns = []

            for date in portfolio_returns_df[date_col].unique():
                date_data = portfolio_returns_df[portfolio_returns_df[date_col] == date]

                # Long top quantile, short bottom quantile
                top_return = date_data[date_data['quantile'] == self.n_quantiles]['return'].values
                bottom_return = date_data[date_data['quantile'] == 1]['return'].values

                if len(top_return) > 0 and len(bottom_return) > 0:
                    long_short_returns.append({
                        date_col: date,
                        'quantile': 'long_short',
                        'return': top_return[0] - bottom_return[0],
                        'n_stocks': None
                    })

            if long_short_returns:
                long_short_df = pd.DataFrame(long_short_returns)
                portfolio_returns_df = pd.concat(
                    [portfolio_returns_df, long_short_df],
                    ignore_index=True
                )

        self.portfolio_returns = portfolio_returns_df
        return portfolio_returns_df

    def calculate_performance_metrics(
        self,
        portfolio_returns_df: Optional[pd.DataFrame] = None,
        date_col: str = 'date',
        risk_free_rate: float = 0.0,
        annualization_factor: int = 12
    ) -> pd.DataFrame:
        """
        Calculate performance metrics for portfolios.

        Parameters
        ----------
        portfolio_returns_df : Optional[pd.DataFrame]
            Portfolio returns (uses self.portfolio_returns if None)
        date_col : str
            Date column
        risk_free_rate : float
            Risk-free rate (monthly)
        annualization_factor : int
            Factor to annualize returns (12 for monthly)

        Returns
        -------
        pd.DataFrame
            Performance metrics by quantile
        """
        if portfolio_returns_df is None:
            if self.portfolio_returns is None:
                raise ValueError("No portfolio returns available")
            portfolio_returns_df = self.portfolio_returns

        metrics = []

        # Calculate metrics for each quantile
        quantiles = portfolio_returns_df['quantile'].unique()

        for quantile in quantiles:
            quantile_data = portfolio_returns_df[portfolio_returns_df['quantile'] == quantile]
            returns = quantile_data['return'].values

            # Excess returns (assuming returns are already excess returns)
            excess_returns = returns - risk_free_rate

            # Mean return (annualized)
            mean_return = np.mean(excess_returns) * annualization_factor

            # Standard deviation (annualized)
            std_return = np.std(excess_returns, ddof=1) * np.sqrt(annualization_factor)

            # Sharpe ratio
            sharpe_ratio = mean_return / std_return if std_return > 0 else np.nan

            # Cumulative return
            cumulative_return = np.prod(1 + returns) - 1

            # Number of periods
            n_periods = len(returns)

            metrics.append({
                'quantile': quantile,
                'mean_return': mean_return,
                'std_return': std_return,
                'sharpe_ratio': sharpe_ratio,
                'cumulative_return': cumulative_return,
                'n_periods': n_periods
            })

        metrics_df = pd.DataFrame(metrics)
        self.performance_metrics = metrics_df

        return metrics_df

    def get_summary_statistics(
        self,
        portfolio_returns_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Get summary statistics for all portfolios.

        Parameters
        ----------
        portfolio_returns_df : Optional[pd.DataFrame]
            Portfolio returns (uses self.portfolio_returns if None)

        Returns
        -------
        pd.DataFrame
            Summary statistics
        """
        if portfolio_returns_df is None:
            if self.portfolio_returns is None:
                raise ValueError("No portfolio returns available")
            portfolio_returns_df = self.portfolio_returns

        summary = []

        quantiles = portfolio_returns_df['quantile'].unique()

        for quantile in quantiles:
            quantile_data = portfolio_returns_df[portfolio_returns_df['quantile'] == quantile]
            returns = quantile_data['return'].values

            summary.append({
                'quantile': quantile,
                'mean': np.mean(returns),
                'median': np.median(returns),
                'std': np.std(returns, ddof=1),
                'min': np.min(returns),
                'max': np.max(returns),
                'skewness': self._calculate_skewness(returns),
                'kurtosis': self._calculate_kurtosis(returns)
            })

        return pd.DataFrame(summary)

    @staticmethod
    def _calculate_skewness(returns: np.ndarray) -> float:
        """Calculate skewness of returns."""
        n = len(returns)
        if n < 3:
            return np.nan
        mean = np.mean(returns)
        std = np.std(returns, ddof=1)
        if std == 0:
            return np.nan
        return (n / ((n - 1) * (n - 2))) * np.sum(((returns - mean) / std) ** 3)

    @staticmethod
    def _calculate_kurtosis(returns: np.ndarray) -> float:
        """Calculate excess kurtosis of returns."""
        n = len(returns)
        if n < 4:
            return np.nan
        mean = np.mean(returns)
        std = np.std(returns, ddof=1)
        if std == 0:
            return np.nan
        m4 = np.sum(((returns - mean) / std) ** 4) / n
        return m4 - 3

    def plot_cumulative_returns(
        self,
        portfolio_returns_df: Optional[pd.DataFrame] = None,
        date_col: str = 'date',
        quantiles_to_plot: Optional[List] = None,
        figsize: tuple = (12, 6)
    ):
        """
        Plot cumulative returns for selected quantiles.

        Parameters
        ----------
        portfolio_returns_df : Optional[pd.DataFrame]
            Portfolio returns (uses self.portfolio_returns if None)
        date_col : str
            Date column
        quantiles_to_plot : Optional[List]
            Quantiles to plot (if None, plots all)
        figsize : tuple
            Figure size
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed. Cannot plot.")
            return

        if portfolio_returns_df is None:
            if self.portfolio_returns is None:
                raise ValueError("No portfolio returns available")
            portfolio_returns_df = self.portfolio_returns

        if quantiles_to_plot is None:
            quantiles_to_plot = portfolio_returns_df['quantile'].unique()

        fig, ax = plt.subplots(figsize=figsize)

        for quantile in quantiles_to_plot:
            quantile_data = portfolio_returns_df[
                portfolio_returns_df['quantile'] == quantile
            ].sort_values(date_col)

            returns = quantile_data['return'].values
            cumulative_returns = np.cumprod(1 + returns) - 1

            ax.plot(
                quantile_data[date_col].values,
                cumulative_returns,
                label=f'Quantile {quantile}',
                linewidth=2
            )

        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Cumulative Return', fontsize=12)
        ax.set_title('Portfolio Cumulative Returns', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return fig, ax


class PerformanceAnalyzer:
    """
    Analyze and compare portfolio performance.
    """

    @staticmethod
    def create_performance_table(
        metrics_df: pd.DataFrame,
        format_pct: bool = True
    ) -> pd.DataFrame:
        """
        Create formatted performance table.

        Parameters
        ----------
        metrics_df : pd.DataFrame
            Performance metrics
        format_pct : bool
            Format returns as percentages

        Returns
        -------
        pd.DataFrame
            Formatted table
        """
        table = metrics_df.copy()

        if format_pct:
            table['mean_return'] = table['mean_return'] * 100
            table['std_return'] = table['std_return'] * 100
            table['cumulative_return'] = table['cumulative_return'] * 100

        return table

    @staticmethod
    def compare_strategies(
        strategy_results: Dict[str, pd.DataFrame],
        metric: str = 'sharpe_ratio'
    ) -> pd.DataFrame:
        """
        Compare multiple strategies.

        Parameters
        ----------
        strategy_results : Dict[str, pd.DataFrame]
            Dictionary mapping strategy names to performance metrics
        metric : str
            Metric to compare

        Returns
        -------
        pd.DataFrame
            Comparison table
        """
        comparison = {}

        for strategy_name, metrics_df in strategy_results.items():
            comparison[strategy_name] = metrics_df.set_index('quantile')[metric]

        return pd.DataFrame(comparison)


def construct_portfolios(predictions_path: str, n_quantiles: int = 10):
    """Simplified interface: Construct portfolios from predictions."""
    print("  Loading predictions...")
    predictions_df = pd.read_csv(predictions_path)

    # Handle both 'date' and 'month' column names
    date_col = 'month' if 'month' in predictions_df.columns else 'date'
    predictions_df[date_col] = pd.to_datetime(predictions_df[date_col])

    # Rename to 'date' for compatibility with PortfolioBacktest
    if date_col == 'month':
        predictions_df = predictions_df.rename(columns={'month': 'date'})

    print(f"  Constructing {n_quantiles} quantile portfolios...")
    backtester = PortfolioBacktest(n_quantiles=n_quantiles, weighting='equal', long_short=True)
    portfolio_returns = backtester.construct_portfolios(predictions_df)

    return portfolio_returns


def analyze_performance(portfolio_returns: pd.DataFrame, results_dir: str):
    """Simplified interface: Analyze portfolio performance and save results."""
    import os

    if len(portfolio_returns) == 0:
        print("  Warning: No portfolio returns to analyze!")
        return None

    print("  Calculating performance metrics...")
    backtester = PortfolioBacktest()
    backtester.portfolio_returns = portfolio_returns
    metrics = backtester.calculate_performance_metrics()

    # Save results
    os.makedirs(results_dir, exist_ok=True)
    metrics_path = os.path.join(results_dir, 'performance_metrics.csv')
    metrics.to_csv(metrics_path, index=False)

    print(f"  Saved metrics to {metrics_path}")

    # Display summary
    if 'long_short' in metrics['quantile'].values:
        ls = metrics[metrics['quantile'] == 'long_short'].iloc[0]
        print(f"\n  Long-Short Sharpe Ratio: {ls['sharpe_ratio']:.2f}")
        print(f"  Long-Short Annual Return: {ls['mean_return']*12*100:.2f}%")

    return metrics


if __name__ == "__main__":
    # Example usage with synthetic data
    np.random.seed(42)

    # Create sample predictions
    dates = pd.date_range('2020-01-01', '2020-12-01', freq='MS')
    n_stocks = 100

    predictions = []
    for date in dates:
        for stock_id in range(n_stocks):
            predictions.append({
                'date': date,
                'permno': stock_id,
                'predicted_return': np.random.normal(0.01, 0.05),
                'actual_return': np.random.normal(0.01, 0.05),
                'market_cap': np.random.lognormal(10, 2)
            })

    predictions_df = pd.DataFrame(predictions)

    # Construct portfolios
    backtester = PortfolioBacktest(n_quantiles=10, weighting='equal', long_short=True)
    portfolio_returns = backtester.construct_portfolios(predictions_df)

    print("Portfolio returns shape:", portfolio_returns.shape)
    print("\nSample portfolio returns:")
    print(portfolio_returns.head(15))

    # Calculate performance metrics
    metrics = backtester.calculate_performance_metrics()
    print("\nPerformance metrics:")
    print(metrics)

    # Get summary statistics
    summary = backtester.get_summary_statistics()
    print("\nSummary statistics:")
    print(summary)

    # Highlight long-short performance
    if 'long_short' in metrics['quantile'].values:
        ls_metrics = metrics[metrics['quantile'] == 'long_short'].iloc[0]
        print(f"\nLong-Short Portfolio:")
        print(f"  Sharpe Ratio: {ls_metrics['sharpe_ratio']:.2f}")
        print(f"  Annualized Return: {ls_metrics['mean_return']*100:.2f}%")
        print(f"  Annualized Std Dev: {ls_metrics['std_return']*100:.2f}%")
