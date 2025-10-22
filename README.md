# Gu, Kelly & Xiu (2020) Replication in Python

This project replicates the core methodology from "Empirical Asset Pricing via Machine Learning" by Gu, Kelly, and Xiu (2020).

## Overview

The implementation includes:
- Cross-sectional ranking transformations
- Missing value handling with median imputation
- Feature engineering with macro-characteristic interactions
- Expanding window train-validation-test splits
- Random Forest model with hyperparameter tuning
- Portfolio construction and backtesting
- Performance metrics (Sharpe ratios, returns)

## Project Structure

```
.
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── download_data.py          # ⭐ Download real data from Xiu's website
├── data_preprocessing.py     # Data transformation pipeline
├── model_training.py         # ML model implementation
├── portfolio_construction.py # Portfolio building and backtesting
├── utils.py                  # Helper functions
├── main.py                   # Main execution script
├── example.py                # Usage examples (synthetic data)
├── example_real_data.py      # ⭐ Usage examples (real data)
└── test_installation.py      # Installation verification
```

## Installation

```bash
pip install -r requirements.txt
```

## Data Requirements

The methodology requires the same data as the original tutorial:

### 1. Stock Characteristics (Required)
- **Source**: Dacheng Xiu's website at Chicago Booth
- **URL**: https://dachxiu.chicagobooth.edu/download/datashare.zip
- **Contents**: 94 stock characteristics + 74 industry dummies
- **Download**: Automated via `python download_data.py`

### 2. CRSP Returns (Recommended)
- **Source**: WRDS CRSP database or Tidy Finance SQLite
- **Contents**: Monthly stock returns, market cap
- **Time period**: 1957-03 onwards

### 3. Macro Predictors (Recommended)
- **Source**: Tidy Finance database or Welch & Goyal (2008) updates
- **Contents**: 8 macro variables (dp, ep, bm, ntis, tbl, tms, dfy, svar)

**Total after preprocessing: 920 covariates** (with macro × characteristic interactions)

## Quick Start

### Step 1: Download Real Data (Recommended)

```bash
# Download stock characteristics from Xiu's website (~500MB)
python download_data.py
```

This downloads `datashare.csv` with 94 characteristics + 74 industry dummies.

For full replication, also obtain CRSP and macro data (see instructions printed by the script).

### Step 2: Run Replication

#### Option A: With Real Data (Recommended)

```bash
# Auto-detects columns from Xiu's data
python main.py --data-path ./data/datashare.csv

# Or use the example script
python example_real_data.py
```

#### Option B: Quick Demo with Synthetic Data

```bash
# Generates synthetic data for testing
python main.py
```

#### Option C: Python API

```python
from main import run_replication

# With real data from Xiu's website (auto-detects columns)
results = run_replication(
    data_path='./data/datashare.csv',
    train_start='1957-03',
    train_end='1974-12',
    n_estimators=300,
    verbose=1
)

# Or with custom data
results = run_replication(
    data_path='your_data.csv',
    stock_characteristics=['char1', 'char2', ...],
    macro_predictors=['dp', 'ep', 'bm', ...],
    industry_dummies=['sic1', 'sic2', ...],
    train_start='1957-03',
    train_end='1974-12'
)

# Access results
predictions = results['predictions']
portfolio_returns = results['portfolio_returns']
performance_metrics = results['performance_metrics']
```

## Output

The pipeline produces:
- **predictions.csv**: Out-of-sample return predictions for each stock-date
- **portfolio_returns.csv**: Returns for each quantile portfolio by date
- **performance_metrics.csv**: Sharpe ratios, mean returns, volatility by quantile
- **summary_statistics.csv**: Detailed distributional statistics

## Methodology

### Data Transformations
1. Cross-sectional ranking: Map characteristics to [-1, 1] interval
2. Missing values: Replace with cross-sectional medians, then zeros
3. Feature engineering: Create macro × characteristic interactions

### Model Specification
- **Random Forest**: 300 trees, tuning mtry ∈ {3,5,10,20,30,50} and min_n ∈ {5000,10000}
- **Validation**: Rolling 12-month window
- **Refit frequency**: Annual

### Portfolio Construction
- Decile portfolios based on predicted returns
- Equal-weighted and value-weighted schemes
- Long-short (D10 - D1) zero-cost portfolios

## References

Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine learning. *Review of Financial Studies*, 33(5), 2223-2273.
