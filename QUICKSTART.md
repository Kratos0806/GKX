# Quick Start Guide

## Two Options: Real Data or Synthetic Demo

### 🎯 Option A: With Real Data (Recommended - Follows Tutorial Exactly)

#### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 2. Download Real Data from Xiu's Website
```bash
python download_data.py
```

This downloads the actual stock characteristics data (~500MB) used in the GKX (2020) paper.

#### 3. Run Replication with Real Data
```bash
python main.py --data-path ./data/datashare.csv
```

Or use the dedicated example script:
```bash
python example_real_data.py
```

### 🚀 Option B: Quick Demo with Synthetic Data

#### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 2. Verify Installation
```bash
python test_installation.py
```

You should see: `✓ ALL TESTS PASSED!`

#### 3. Run Demo
```bash
python main.py
```

This generates synthetic data and runs a quick demonstration.

## Example Output

```
================================================================================
GU, KELLY & XIU (2020) REPLICATION IN PYTHON
================================================================================

Step 1: Loading data...
  No data provided. Generating synthetic data for demonstration...
  Data loaded: 66,000 observations
  Date range: 1957-03-01 to 1990-12-01
  Unique stocks: 200

Step 2: Preprocessing data...
  Total features: 70
    - Stock characteristics: 20
    - Macro predictors: 3
    - Industry dummies: 10
    - Interaction features: 60

Step 3: Creating temporal splits...
  Created 16 train-validation-test splits

Step 4: Training models and generating predictions...
[Progress bars and model training details...]

Step 5: Constructing portfolios...
Step 6: Calculating performance metrics...

================================================================================
PERFORMANCE RESULTS
================================================================================

quantile  mean_return  std_return  sharpe_ratio  cumulative_return  n_periods
1         4.23%        19.45%      0.22          89.3%             16
2         5.12%        18.89%      0.27          105.8%            16
3         6.34%        18.12%      0.35          128.4%            16
...
10        11.45%       20.34%      0.56          245.7%            16
long_short 7.22%       15.78%      2.14          147.4%            16

================================================================================
Long-Short Portfolio (Decile 10 - Decile 1):
  Sharpe Ratio:        2.14
  Annualized Return:   7.22%
  Annualized Std Dev:  15.78%
================================================================================
```

## Try Different Configurations

### Example 1: Value-Weighted Portfolios
```bash
python main.py --weighting value
```

### Example 2: More Trees, Fewer Quantiles
```bash
python main.py --n-estimators 500 --n-quantiles 5
```

### Example 3: Monthly Rebalancing
```bash
python main.py --refit-frequency monthly
```

## Using the Python API

Create a file `my_replication.py`:

```python
from main import run_replication

# Run with default synthetic data
results = run_replication(
    train_start='1957-03',
    train_end='1974-12',
    n_estimators=300,
    verbose=1
)

# Access results
predictions = results['predictions']
portfolio_returns = results['portfolio_returns']
performance_metrics = results['performance_metrics']

# Print long-short Sharpe ratio
ls = performance_metrics[performance_metrics['quantile'] == 'long_short']
print(f"Long-Short Sharpe: {ls['sharpe_ratio'].values[0]:.2f}")
```

Then run:
```bash
python my_replication.py
```

## Using Your Own Data

### Data Format Required

Your CSV or Parquet file should have columns:
- `date`: Date in YYYY-MM-DD format
- `permno`: Stock identifier (any integer)
- `ret_excess`: Excess returns (target variable)
- `char_1`, `char_2`, ...: Stock characteristics
- `macro_1`, `macro_2`, ...: Macro predictors (same for all stocks in period)
- `industry_1`, `industry_2`, ...: Industry dummies (0 or 1)

### Example with Real Data

```python
from main import run_replication

results = run_replication(
    data_path='my_stock_data.csv',
    stock_characteristics=[
        'mktcap', 'bm', 'mom12m', 'vol', 'beta',
        'prof', 'inv', 'roe', 'turnover', 'idiovol'
        # ... add all 94 characteristics
    ],
    macro_predictors=[
        'dp', 'ep', 'bm_macro', 'ntis', 'tbl',
        'tms', 'dfy', 'svar'
    ],
    industry_dummies=[
        'ind_1', 'ind_2', ..., 'ind_74'
    ],
    train_start='1957-03',
    train_end='1974-12',
    n_estimators=300,
    weighting='equal',
    output_dir='./my_results'
)
```

## Explore Examples

Run different pre-configured examples:

```bash
python example.py
```

Edit [example.py](example.py) to uncomment different examples:
- Example 1: Basic usage
- Example 2: Custom data generation
- Example 3: Value-weighted portfolios
- Example 4: Detailed analysis with plots
- Example 5: Minimal working example

## Common Issues

### Issue: Import errors
**Solution**: Run `pip install -r requirements.txt`

### Issue: Memory errors with large datasets
**Solution**:
- Reduce the number of stocks
- Use monthly refitting instead of more frequent
- Process in smaller date ranges

### Issue: Training is too slow
**Solution**:
- Reduce `n_estimators` (try 100 instead of 300)
- Reduce hyperparameter grid (fewer mtry values)
- Use fewer features

### Issue: Warnings from numpy
**Solution**: These are harmless. The test script suppresses them. Add to your code:
```python
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
```

## Getting Help

1. **Read the docs**: See [README.md](README.md) for full documentation
2. **Check the summary**: See [SUMMARY.md](SUMMARY.md) for methodology details
3. **Review examples**: Open [example.py](example.py) for usage patterns
4. **Inspect the code**: All modules are well-documented with docstrings

## What's Next?

1. ✅ Run the basic demo (you've done this!)
2. **Customize hyperparameters** - Try different Random Forest settings
3. **Add your own data** - Format your dataset and run real replication
4. **Compare results** - Check against GKX (2020) published results
5. **Extend the code** - Add new models (e.g., Neural Networks, XGBoost)
6. **Experiment** - Try different portfolio constructions or weighting schemes

## Quick Reference Commands

```bash
# Test installation
python test_installation.py

# Run basic demo
python main.py

# Run with custom settings
python main.py --n-estimators 500 --weighting value

# Run examples
python example.py

# Get help on command-line options
python main.py --help
```

---

**You're all set! Happy replicating! 🚀**
