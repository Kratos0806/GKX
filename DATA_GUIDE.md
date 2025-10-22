# Data Guide for GKX (2020) Replication

This guide explains how to obtain and use the exact same data as the original tutorial.

## Overview

The GKX (2020) replication requires three datasets:

1. **Stock Characteristics** (Required) - 94 firm characteristics
2. **CRSP Returns** (Recommended) - Monthly stock returns
3. **Macro Predictors** (Recommended) - 8 macroeconomic variables

## 1. Stock Characteristics Data

### Source
**Dacheng Xiu's website** at University of Chicago Booth School of Business

### URL
https://dachxiu.chicagobooth.edu/download/datashare.zip

### Contents
- **94 stock characteristics**: 61 annual, 13 quarterly, 20 monthly
- **74 industry dummies**: Based on SIC codes
- **Time coverage**: 1957-2021 (extended beyond original 2016 paper)
- **File**: `datashare.csv` (~500MB compressed)

### How to Download

#### Automated (Recommended)
```bash
python download_data.py
```

This script:
1. Downloads the ZIP file from Xiu's website
2. Extracts `datashare.csv`
3. Saves to `./data/datashare.csv`
4. Displays data preview and statistics

#### Manual
1. Visit: https://dachxiu.chicagobooth.edu/download/datashare.zip
2. Download and extract the ZIP file
3. Place `datashare.csv` in `./data/` directory

### Data Format

The CSV file has columns:
- `DATE`: Date (YYYYMM format)
- `permno`: CRSP permanent identifier for the stock
- `sic1`, `sic2`, ..., `sic74`: Industry dummy variables (0 or 1)
- Characteristic columns: Various firm-level predictors

### Example Code
```python
from download_data import download_characteristics_data, load_characteristics_data

# Download
filepath = download_characteristics_data(output_dir='./data')

# Load
df = load_characteristics_data(filepath)
print(f"Loaded {len(df):,} observations")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
```

## 2. CRSP Returns Data

### Source
**CRSP (Center for Research in Security Prices)** via WRDS

### What You Need
- Monthly stock returns (`ret`)
- Excess returns (`ret_excess` = ret - risk-free rate)
- Market capitalization (`mktcap`)
- CRSP permanent number (`permno`)
- Date (`month` or `date`)

### How to Obtain

#### Option A: WRDS Access (Best Quality)
If you have institutional access to WRDS:

```python
import wrds

db = wrds.Connection()
crsp = db.raw_sql('''
    SELECT
        a.permno,
        a.date,
        a.ret,
        a.shrout * a.prc AS mktcap
    FROM crsp.msf AS a
    WHERE a.date >= '1957-03-01'
        AND a.shrcd IN (10, 11)  -- Common stocks only
''')

# Calculate excess returns (subtract risk-free rate)
# Merge with Fama-French factors for risk-free rate
crsp.to_csv('./data/crsp_monthly.csv', index=False)
```

#### Option B: Tidy Finance Database (Easiest)
Download the `tidy_finance_r.sqlite` database from:
https://www.tidy-finance.org/

```python
import sqlite3
import pandas as pd

conn = sqlite3.connect('tidy_finance_r.sqlite')
crsp = pd.read_sql_query('''
    SELECT month, permno, mktcap_lag, ret_excess
    FROM crsp_monthly
    WHERE month >= '1957-03-01'
''', conn)

crsp.to_csv('./data/crsp_monthly.csv', index=False)
conn.close()
```

#### Option C: Manual Download
1. Access WRDS: https://wrds-www.wharton.upenn.edu/
2. Navigate: CRSP → Stock/Security Files → Monthly Stock File
3. Select fields: permno, date, ret, prc, shrout
4. Filter: Common stocks (shrcd 10, 11) from 1957-03 onwards
5. Download as CSV

### Data Format Expected
```
month,permno,mktcap_lag,ret_excess
1957-03-01,10001,1234.56,0.0123
1957-03-01,10002,5678.90,-0.0045
...
```

## 3. Macroeconomic Predictors

### Source
**Welch & Goyal (2008)** macro predictors, updated

### Variables Needed (8 total)
1. `dp`: Dividend-price ratio (log)
2. `ep`: Earnings-price ratio (log)
3. `bm`: Book-to-market ratio
4. `ntis`: Net equity expansion
5. `tbl`: Treasury bill rate (3-month)
6. `tms`: Term spread (10-year - 3-month)
7. `dfy`: Default yield spread (BAA - AAA)
8. `svar`: Stock market variance

### How to Obtain

#### Option A: Tidy Finance Database (Easiest)
```python
import sqlite3
import pandas as pd

conn = sqlite3.connect('tidy_finance_r.sqlite')
macro = pd.read_sql_query('''
    SELECT month, dp, ep, bm, ntis, tbl, tms, dfy, svar
    FROM macro_predictors
''', conn)

macro.to_csv('./data/macro_predictors.csv', index=False)
conn.close()
```

#### Option B: Amit Goyal's Website
Download updated Welch & Goyal data:
- URL: http://www.hec.unil.ch/agoyal/
- File: `PredictorData2021.xlsx`
- Select the 8 variables listed above
- Save as CSV

#### Option C: Construct from Raw Data
Sources for each variable:
- **dp, ep**: S&P 500 data from Shiller's website
- **bm**: Fama-French data library
- **ntis**: Federal Reserve Flow of Funds
- **tbl, tms**: FRED (Federal Reserve Economic Data)
- **dfy**: Moody's bond yields (FRED)
- **svar**: Calculated from CRSP daily returns

### Data Format Expected
```
month,dp,ep,bm,ntis,tbl,tms,dfy,svar
1957-01-01,-3.23,-2.89,0.45,0.012,0.0234,0.0123,0.0089,0.0001
1957-02-01,-3.25,-2.91,0.46,0.013,0.0245,0.0119,0.0087,0.0002
...
```

**Important**: Macro predictors are lagged by one month before merging with stock data.

## Merging Datasets

### Full Pipeline

```python
from download_data import (
    load_characteristics_data,
    merge_datasets
)
import pandas as pd

# 1. Load characteristics
char_df = load_characteristics_data('./data/datashare.csv')

# 2. Load CRSP (if available)
crsp_df = pd.read_csv('./data/crsp_monthly.csv')

# 3. Load macro (if available)
macro_df = pd.read_csv('./data/macro_predictors.csv')

# 4. Merge all datasets
merged = merge_datasets(
    char_df,
    crsp_df=crsp_df,
    macro_df=macro_df
)

# 5. Save merged data
merged.to_csv('./data/merged_data.csv', index=False)
```

### Run Replication with Merged Data

```python
from main import run_replication

results = run_replication(
    data_path='./data/merged_data.csv',
    train_start='1957-03',
    train_end='1974-12',
    n_estimators=300,
    verbose=1
)
```

## Data Quality Checks

After loading data, verify:

```python
import pandas as pd

df = pd.read_csv('./data/merged_data.csv')

# Check date range
print(f"Date range: {df['date'].min()} to {df['date'].max()}")

# Check number of stocks
print(f"Unique stocks: {df['permno'].nunique()}")

# Check for missing returns
print(f"Missing returns: {df['ret_excess'].isna().sum() / len(df) * 100:.2f}%")

# Check characteristics
char_cols = [c for c in df.columns if c not in
             ['date', 'permno', 'ret_excess', 'mktcap']]
print(f"Number of characteristics: {len(char_cols)}")

# Sample data
print(df.head())
```

## Expected Data Sizes

Based on the original paper:
- **Observations**: ~3 million stock-month pairs
- **Stocks**: ~20,000 unique firms
- **Time period**: 1957-2021 (768 months)
- **Columns**: ~170+ before interaction terms
- **Final features**: 920 (after macro × characteristic interactions)

## Troubleshooting

### Issue: Can't download from Xiu's website
**Solution**: Download manually and place in `./data/` directory

### Issue: Don't have WRDS access
**Solution**: Use Tidy Finance database or focus on characteristics-only replication

### Issue: Missing macro predictors
**Solution**: The code works without macro data (just won't create interaction features)

### Issue: Different column names
**Solution**: Rename columns to match expected format:
```python
df = df.rename(columns={
    'DATE': 'date',
    'PERMNO': 'permno',
    'RET': 'ret_excess'
})
```

## Minimum Data Requirements

To run the replication, you minimally need:
1. ✅ Stock characteristics from Xiu's website (download_data.py)
2. ✅ Return data (ret_excess column)
3. ⚠️ Optional: Macro predictors (for full 920 features)

The characteristics file from Xiu may already include returns if it's a merged dataset. Check with:
```python
'ret_excess' in df.columns
```

## Next Steps

Once you have the data:

1. **Verify data is loaded correctly**
   ```bash
   python -c "from download_data import load_characteristics_data; load_characteristics_data('./data/datashare.csv')"
   ```

2. **Run the replication**
   ```bash
   python main.py --data-path ./data/datashare.csv
   ```

3. **Check results**
   ```bash
   ls -lh ./results/
   ```

## References

- **GKX Paper**: Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine learning. *Review of Financial Studies*, 33(5), 2223-2273.
- **Xiu's Data**: https://dachxiu.chicagobooth.edu/
- **WRDS**: https://wrds-www.wharton.upenn.edu/
- **Tidy Finance**: https://www.tidy-finance.org/
- **Welch & Goyal**: http://www.hec.unil.ch/agoyal/

---

**Need help?** See [README.md](README.md) for full documentation or run `python download_data.py` for detailed instructions.
