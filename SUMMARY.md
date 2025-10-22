# GKX (2020) Python Replication - Summary

## What Has Been Built

This is a complete Python implementation of the core methodology from **"Empirical Asset Pricing via Machine Learning"** by Gu, Kelly, and Xiu (2020), replicating the tutorial from the Tidy Finance blog.

## Key Features

### 1. Data Preprocessing ([data_preprocessing.py](data_preprocessing.py))
- **Cross-sectional ranking**: Ranks stock characteristics within each time period and maps to [-1, 1] interval
- **Missing value imputation**: Two-step process using cross-sectional medians, then zero-filling
- **Feature engineering**: Creates interaction terms between macroeconomic predictors and stock characteristics
- **Temporal splits**: Expanding window train-validation-test splits with annual or monthly refitting

### 2. Machine Learning Models ([model_training.py](model_training.py))
- **Random Forest implementation** with scikit-learn
- **Hyperparameter tuning**: Grid search over `max_features` ∈ {3,5,10,20,30,50} and `min_samples_split` ∈ {5000,10000}
- **Validation-based selection**: Uses RMSE on rolling 12-month validation window
- **Out-of-sample predictions**: Manages expanding window forecasts across multiple time periods

### 3. Portfolio Construction ([portfolio_construction.py](portfolio_construction.py))
- **Decile portfolios**: Sorts stocks into quantiles based on predicted returns
- **Equal and value weighting**: Flexible portfolio weighting schemes
- **Long-short strategies**: Constructs zero-cost portfolios (top decile - bottom decile)
- **Performance metrics**: Sharpe ratios, mean returns, volatility, cumulative returns
- **Visualization**: Cumulative return plots (when matplotlib available)

### 4. Real Data Download ([download_data.py](download_data.py)) ⭐ NEW
- **Automated download**: Fetches actual data from Dacheng Xiu's website (~500MB)
- **Stock characteristics**: 94 characteristics + 74 industry dummies (1957-2021)
- **CRSP instructions**: Detailed guide for obtaining monthly returns
- **Macro instructions**: Guide for obtaining 8 Welch & Goyal (2008) predictors
- **Data merging**: Combines all datasets following tutorial methodology

### 5. Utilities ([utils.py](utils.py))
- **Data loading**: Support for CSV and Parquet formats
- **Data validation**: Ensures data integrity and structure
- **Synthetic data generation**: Creates realistic test datasets for demos
- **Results saving**: Exports to CSV and JSON formats

### 6. Main Pipeline ([main.py](main.py))
- **End-to-end workflow**: Complete replication pipeline from data to results
- **Auto-detection**: Automatically identifies stock chars, macros, and industries from real data
- **Command-line interface**: Easy execution with customizable parameters
- **Progress tracking**: Verbose output for monitoring long-running processes
- **Automatic reporting**: Formatted performance tables and statistics

## Methodology Details

### Data Flow

```
Raw Data
    ↓
Cross-Sectional Ranking (map to [-1,1])
    ↓
Missing Value Imputation (median → 0)
    ↓
Feature Engineering (macro × characteristics)
    ↓
Temporal Splits (expanding window)
    ↓
Model Training + Validation (hyperparameter tuning)
    ↓
Out-of-Sample Predictions
    ↓
Portfolio Construction (decile sorting)
    ↓
Performance Evaluation (Sharpe, returns)
```

### Training Schedule

- **Initial training**: 1957-03 to 1974-12 (18 years)
- **Validation window**: Rolling 12 months
- **Test period**: Next month after validation
- **Refit frequency**: Annual (or monthly if specified)
- **Hyperparameter tuning**: Performed at each refit

### Default Model Configuration

```python
Random Forest:
- n_estimators: 300 trees
- max_features: tuned ∈ {3, 5, 10, 20, 30, 50}
- min_samples_split: tuned ∈ {5000, 10000}
- criterion: MSE (for regression)
- bootstrap: True
```

## Files Created

| File | Lines | Purpose |
|------|-------|---------|
| [requirements.txt](requirements.txt) | 8 | Python dependencies (including requests) |
| [README.md](README.md) | 140+ | Documentation and usage guide |
| [download_data.py](download_data.py) | 350+ | ⭐ Download real data from Xiu's website |
| [data_preprocessing.py](data_preprocessing.py) | 350+ | Data transformation pipeline |
| [model_training.py](model_training.py) | 350+ | ML model training and prediction |
| [portfolio_construction.py](portfolio_construction.py) | 400+ | Portfolio construction and analysis |
| [utils.py](utils.py) | 250+ | Helper functions |
| [main.py](main.py) | 380+ | Main execution script with auto-detection |
| [example.py](example.py) | 250+ | Usage examples (synthetic data) |
| [example_real_data.py](example_real_data.py) | 250+ | ⭐ Usage examples (real data) |
| [test_installation.py](test_installation.py) | 250+ | Installation verification |
| [DATA_GUIDE.md](DATA_GUIDE.md) | 350+ | ⭐ Comprehensive data guide |
| [QUICKSTART.md](QUICKSTART.md) | 200+ | Quick start guide |
| [SUMMARY.md](SUMMARY.md) | 250+ | This file |

**Total: ~3,600+ lines of documented Python code**

## Usage Examples

### Using Real Data (Recommended)
```bash
# Step 1: Download actual GKX data
python download_data.py

# Step 2: Run replication with real data (auto-detects columns)
python main.py --data-path ./data/datashare.csv
```

Or use the dedicated real data example:
```bash
python example_real_data.py
```

### Quick Demo with Synthetic Data
```bash
python main.py
```
Runs with synthetic data, outputs results to `./results/`

### With Custom Data
```python
from main import run_replication

# Auto-detection of columns from real data
results = run_replication(
    data_path='./data/datashare.csv',  # Real data from Xiu
    train_start='1957-03',
    train_end='1974-12'
    # Columns auto-detected!
)

# Or specify manually
results = run_replication(
    data_path='your_data.csv',
    stock_characteristics=['mktcap', 'bm', 'mom', ...],
    macro_predictors=['dp', 'ep', 'bm', ...],
    industry_dummies=['sic1', 'sic2', ...],
    train_start='1957-03',
    train_end='1974-12'
)
```

## Expected Outputs

### Performance Metrics Table
```
quantile  mean_return  std_return  sharpe_ratio  cumulative_return  n_periods
1         5.23%        18.45%      0.28          125.3%            120
2         6.12%        17.89%      0.34          142.8%            120
...
10        12.45%       21.34%      0.58          298.7%            120
long_short 7.22%       16.78%      2.14          187.4%            120
```

### Files Saved
- `predictions.csv`: Stock-level return predictions
- `portfolio_returns.csv`: Time series of portfolio returns
- `performance_metrics.csv`: Summary statistics by quantile
- `summary_statistics.csv`: Distributional properties

## Differences from Original R Tutorial

### Similarities ✓
- ✅ **Same data source**: Uses Dacheng Xiu's datashare.csv
- ✅ **Same core methodology**: Cross-sectional ranking, median imputation, macro interactions
- ✅ **Same hyperparameter grids**: mtry ∈ {3,5,10,20,30,50}, min_n ∈ {5000,10000}
- ✅ **Same performance metrics**: Sharpe ratios, mean returns, decile portfolios
- ✅ **Same expanding window**: Annual refitting with 12-month validation

### Differences
- **Language**: Python instead of R (tidyverse → pandas/numpy)
- **ML library**: scikit-learn instead of tidymodels/ranger
- **Data download**: Automated Python script vs manual R download
- **Column detection**: Auto-detects characteristics, macros, and industries
- **Parallelization**: scikit-learn's joblib instead of future package
- **Bonus feature**: Synthetic data generator for quick testing

## Testing

Run the installation test:
```bash
python test_installation.py
```

This verifies:
- ✓ All dependencies installed
- ✓ Custom modules import correctly
- ✓ Basic functionality works
- ✓ Data preprocessing pipeline functions
- ✓ Model training completes
- ✓ Portfolio construction succeeds

## Performance Considerations

### Speed Optimizations
- Uses scikit-learn's parallel processing (`n_jobs=-1`)
- Processes data in chunks by time period
- Pre-computes feature transformations
- Efficient pandas operations

### Memory Optimizations
- Streams data by temporal splits
- Doesn't load entire history for each prediction
- Can handle large datasets with chunking

### Typical Runtime
- **Synthetic demo** (1957-1990, 200 stocks): ~2-5 minutes
- **Full replication** (1957-2021, 2000+ stocks): ~2-8 hours (depends on hardware)

## Extensions and Modifications

The code is modular and easy to extend:

1. **Add new models**: Inherit from or mimic `GKXRandomForest` class
2. **Different features**: Modify `GKXPreprocessor` parameters
3. **Alternative weighting**: Extend `PortfolioBacktest` class
4. **Custom metrics**: Add to `calculate_performance_metrics()`
5. **Different splits**: Modify `create_temporal_splits()` function

## References

- **Paper**: Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine learning. *Review of Financial Studies*, 33(5), 2223-2273.
- **Original Tutorial**: https://www.tidy-finance.org/blog/gu-kelly-xiu-replication/
- **Python Libraries**:
  - scikit-learn: https://scikit-learn.org/
  - pandas: https://pandas.pydata.org/
  - numpy: https://numpy.org/

## Next Steps

1. **Get real data**: Acquire CRSP, Compustat, and macro data
2. **Format data**: Match the required schema (see README.md)
3. **Run replication**: Use `main.py` with your data
4. **Analyze results**: Compare to published results
5. **Experiment**: Try different models, features, or portfolio constructions

---

**Status**: ✅ Complete and tested
**Last Updated**: October 2025
