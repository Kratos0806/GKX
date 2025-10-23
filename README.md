# GKX (2020) Replication

Simple Python implementation of Gu, Kelly & Xiu (2020) "Empirical Asset Pricing via Machine Learning".

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run complete pipeline
python main.py
```

## Pipeline

1. **Download** - Downloads stock characteristics, macro predictors, CRSP data
2. **Merge** - Combines all datasets
3. **Preprocess** - Cross-sectional ranking, interactions, industry dummies
4. **Train** - Random Forest with hyperparameter tuning
5. **Evaluate** - Portfolio construction and performance metrics

## Data Sources

- **Stock Characteristics**: [Dacheng Xiu's website](https://dachxiu.chicagobooth.edu/)
- **Macro Predictors**: Welch & Goyal (2008) via Tidy Finance
- **CRSP Data**: Requires WRDS access (optional)

## Usage

```bash
# Run full pipeline
python main.py

# Skip download (use existing data)
python main.py --skip-download

# Custom parameters
python main.py --n-estimators 500 --n-quantiles 10
```

## Individual Scripts

```bash
python download.py    # Download data
python merge.py       # Merge datasets
python preprocess.py  # Preprocess data
```

## WRDS Access (Optional)

Create `.env` file:

```
WRDS_USER=your_username
WRDS_PASSWORD=your_password
```

## Core Modules

- `main.py` - Pipeline orchestration
- `download.py` - Data download
- `merge.py` - Dataset merging
- `preprocess.py` - Data preprocessing
- `data_preprocessing.py` - GKX preprocessing classes
- `model_training.py` - Random Forest implementation
- `portfolio_construction.py` - Portfolio backtesting
- `utils.py` - Utility functions

## Reference

Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine learning. *The Review of Financial Studies*, 33(5), 2223-2273.
