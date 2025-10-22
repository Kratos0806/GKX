"""
Download and prepare data for GKX (2020) replication.

This script downloads the actual data used in the tutorial:
1. Stock characteristics from Dacheng Xiu's website
2. Instructions for obtaining CRSP data
3. Instructions for macro predictors

Following the exact approach from:
https://www.tidy-finance.org/blog/gu-kelly-xiu-replication/
"""

import os
import zipfile
import requests
import pandas as pd
from io import BytesIO
from typing import Optional


def download_characteristics_data(
    output_dir: str = './data',
    force_download: bool = False
) -> str:
    """
    Download stock characteristics data from Dacheng Xiu's website.

    This downloads the datashare.zip file containing 94 stock characteristics
    plus 74 industry dummies, exactly as used in the GKX (2020) paper.

    Parameters
    ----------
    output_dir : str
        Directory to save the data
    force_download : bool
        If True, download even if file exists

    Returns
    -------
    str
        Path to the downloaded CSV file
    """
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, 'datashare.csv')

    # Check if file already exists
    if os.path.exists(output_file) and not force_download:
        print(f"Data already exists at {output_file}")
        print("Use force_download=True to re-download")
        return output_file

    print("Downloading stock characteristics data from Dacheng Xiu's website...")
    print("This may take several minutes (file is ~500MB)...")

    url = "https://dachxiu.chicagobooth.edu/download/datashare.zip"

    try:
        # Download the zip file
        response = requests.get(url, stream=True, timeout=1200)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        print(f"Download size: {total_size / 1024 / 1024:.1f} MB")

        # Read zip content
        zip_content = BytesIO()
        downloaded = 0

        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                zip_content.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = (downloaded / total_size) * 100
                    print(f"\rProgress: {pct:.1f}%", end='')

        print("\n\nExtracting datashare.csv from zip archive...")

        # Extract the CSV file
        zip_content.seek(0)
        with zipfile.ZipFile(zip_content, 'r') as zip_ref:
            # Extract datashare.csv
            zip_ref.extract('datashare.csv', output_dir)

        print(f"✓ Data successfully downloaded to {output_file}")

        # Print info about the data
        print("\nLoading data to check...")
        df = pd.read_csv(output_file, nrows=5)
        print(f"\nData preview (first 5 rows):")
        print(df.head())
        print(f"\nColumns ({len(df.columns)}): {list(df.columns[:10])}...")

        return output_file

    except Exception as e:
        print(f"\n✗ Error downloading data: {e}")
        print("\nAlternative: Download manually from:")
        print("https://dachxiu.chicagobooth.edu/download/datashare.zip")
        print(f"Then extract datashare.csv to {output_dir}/")
        raise


def load_characteristics_data(
    filepath: str = './data/datashare.csv',
    date_col: str = 'DATE',
    stock_id_col: str = 'permno'
) -> pd.DataFrame:
    """
    Load and prepare characteristics data.

    Parameters
    ----------
    filepath : str
        Path to datashare.csv
    date_col : str
        Name of date column (will be renamed to 'date')
    stock_id_col : str
        Name of stock identifier column

    Returns
    -------
    pd.DataFrame
        Loaded characteristics data
    """
    print(f"Loading characteristics data from {filepath}...")

    df = pd.read_csv(filepath)

    # Rename date column to match our code
    if date_col in df.columns and date_col != 'date':
        df = df.rename(columns={date_col: 'date'})

    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])

    print(f"✓ Loaded {len(df):,} observations")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Unique stocks: {df[stock_id_col].nunique():,}")
    print(f"  Columns: {len(df.columns)}")

    return df


def get_column_lists(df: pd.DataFrame) -> dict:
    """
    Automatically detect stock characteristics, macro predictors, and industry dummies.

    Parameters
    ----------
    df : pd.DataFrame
        Loaded characteristics dataframe

    Returns
    -------
    dict
        Dictionary with 'stock_chars', 'macro_preds', 'industries' lists
    """
    # Define the 8 macro predictors (Welch & Goyal 2008)
    macro_predictors = ['dp', 'ep', 'bm', 'ntis', 'tbl', 'tms', 'dfy', 'svar']

    # Identify industry dummies (SIC-based, typically named like 'sic1', 'sic2', etc.)
    industry_dummies = [col for col in df.columns if col.lower().startswith('sic')]

    # Stock characteristics are everything else except date, permno, returns, and macros
    exclude_cols = {'date', 'permno', 'ret_excess', 'mktcap', 'mktcap_lag'} | set(macro_predictors) | set(industry_dummies)

    stock_characteristics = [col for col in df.columns if col not in exclude_cols]

    print("\nDetected columns:")
    print(f"  Stock characteristics: {len(stock_characteristics)}")
    print(f"  Macro predictors: {len(macro_predictors)}")
    print(f"  Industry dummies: {len(industry_dummies)}")

    return {
        'stock_chars': stock_characteristics,
        'macro_preds': [m for m in macro_predictors if m in df.columns],
        'industries': industry_dummies
    }


def prepare_crsp_instructions():
    """Print instructions for obtaining CRSP data."""
    print("""
================================================================================
CRSP DATA INSTRUCTIONS
================================================================================

The GKX replication requires monthly stock returns from CRSP.

Option 1: WRDS Access (Recommended)
------------------------------------
If you have access to WRDS (Wharton Research Data Services):

1. Connect to WRDS database
2. Query CRSP Monthly Stock File (crsp.msf)
3. Download fields: permno, date, ret, mktcap
4. Filter: Common stocks on NYSE/AMEX/NASDAQ
5. Time period: 1957-03 onwards

Python code example:
```python
import wrds
db = wrds.Connection()
crsp = db.raw_sql('''
    SELECT permno, date, ret, mktcap
    FROM crsp.msf
    WHERE date >= '1957-03-01'
''')
crsp.to_csv('data/crsp_monthly.csv', index=False)
```

Option 2: Tidy Finance Database
--------------------------------
Download the tidy_finance_r.sqlite database from:
https://www.tidy-finance.org/

Then extract CRSP data:
```python
import sqlite3
conn = sqlite3.connect('tidy_finance_r.sqlite')
crsp = pd.read_sql_query(
    "SELECT month, permno, mktcap_lag, ret_excess FROM crsp_monthly",
    conn
)
crsp.to_csv('data/crsp_monthly.csv', index=False)
```

Option 3: Manual Download
--------------------------
Visit WRDS website: https://wrds-www.wharton.upenn.edu/
Navigate to: CRSP > Stock/Security Files > Monthly Stock File
Download the required fields manually

================================================================================
""")


def prepare_macro_instructions():
    """Print instructions for obtaining macro predictors."""
    print("""
================================================================================
MACROECONOMIC PREDICTORS INSTRUCTIONS
================================================================================

The GKX replication uses 8 macro predictors following Welch & Goyal (2008).

Option 1: Tidy Finance Database (Easiest)
------------------------------------------
Download tidy_finance_r.sqlite from https://www.tidy-finance.org/

Extract macro data:
```python
import sqlite3
conn = sqlite3.connect('tidy_finance_r.sqlite')
macro = pd.read_sql_query(
    '''SELECT month, dp, ep, bm, ntis, tbl, tms, dfy, svar
       FROM macro_predictors''',
    conn
)
macro.to_csv('data/macro_predictors.csv', index=False)
```

Option 2: Amit Goyal's Website
-------------------------------
Download from: http://www.hec.unil.ch/agoyal/
File: PredictorData2021.xlsx
(Updated version of Welch & Goyal 2008 predictors)

Option 3: Construct Manually
-----------------------------
Variables needed:
- dp:   Dividend-price ratio (S&P 500)
- ep:   Earnings-price ratio (S&P 500)
- bm:   Book-to-market ratio
- ntis: Net equity expansion
- tbl:  Treasury bill rate (3-month)
- tms:  Term spread (10-year - 3-month)
- dfy:  Default yield spread (BAA - AAA)
- svar: Stock variance

Sources: FRED, Shiller's website, Bloomberg

================================================================================
""")


def merge_datasets(
    characteristics_df: pd.DataFrame,
    crsp_df: Optional[pd.DataFrame] = None,
    macro_df: Optional[pd.DataFrame] = None,
    date_col: str = 'date',
    stock_id_col: str = 'permno'
) -> pd.DataFrame:
    """
    Merge characteristics, CRSP, and macro data.

    Parameters
    ----------
    characteristics_df : pd.DataFrame
        Stock characteristics from Xiu's website
    crsp_df : Optional[pd.DataFrame]
        CRSP monthly returns (if available)
    macro_df : Optional[pd.DataFrame]
        Macroeconomic predictors (if available)
    date_col : str
        Date column name
    stock_id_col : str
        Stock identifier column

    Returns
    -------
    pd.DataFrame
        Merged dataset ready for replication
    """
    merged = characteristics_df.copy()

    if crsp_df is not None:
        print("Merging CRSP data...")
        crsp_df[date_col] = pd.to_datetime(crsp_df[date_col])
        merged = merged.merge(
            crsp_df,
            on=[date_col, stock_id_col],
            how='left'
        )
        print(f"  After CRSP merge: {len(merged):,} rows")

    if macro_df is not None:
        print("Merging macro data...")
        macro_df[date_col] = pd.to_datetime(macro_df[date_col])
        # Lag macro predictors by one month
        macro_df[date_col] = macro_df[date_col] + pd.DateOffset(months=1)
        merged = merged.merge(
            macro_df,
            on=[date_col],
            how='left'
        )
        print(f"  After macro merge: {len(merged):,} rows")

    return merged


if __name__ == "__main__":
    print("=" * 80)
    print("GKX (2020) DATA DOWNLOAD SCRIPT")
    print("=" * 80)

    # Download characteristics data
    try:
        filepath = download_characteristics_data()
        df = load_characteristics_data(filepath)
        cols = get_column_lists(df)

        print("\n" + "=" * 80)
        print("✓ Stock characteristics data ready!")
        print("=" * 80)

    except Exception as e:
        print(f"\nCould not download characteristics data: {e}")

    # Print instructions for other datasets
    print("\n")
    prepare_crsp_instructions()

    print("\n")
    prepare_macro_instructions()

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("""
1. You now have stock characteristics data in ./data/datashare.csv

2. Obtain CRSP data (see instructions above) and save to ./data/crsp_monthly.csv

3. Obtain macro predictors (see instructions above) and save to ./data/macro_predictors.csv

4. Run the replication:
   python main.py --data-path ./data/datashare.csv

Or use the merge script to combine all datasets first.
""")
