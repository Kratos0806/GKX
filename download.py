"""
Download data for GKX (2020) replication.
Downloads: stock characteristics, CRSP data, macro predictors.
"""

import os
import zipfile
import requests
import pandas as pd
from io import BytesIO
from typing import Optional
from dotenv import load_dotenv


def download_characteristics(data_dir: str = './data') -> str:
    """Download stock characteristics from Dacheng Xiu's website."""
    os.makedirs(data_dir, exist_ok=True)
    output_file = os.path.join(data_dir, 'datashare.csv')

    if os.path.exists(output_file):
        print(f"  Stock characteristics already exist at {output_file}")
        return output_file

    print("  Downloading stock characteristics (94 characteristics + 74 industry dummies)...")
    url = "https://dachxiu.chicagobooth.edu/download/datashare.zip"

    response = requests.get(url, stream=True, timeout=1200)
    response.raise_for_status()

    print("  Extracting from zip archive...")
    zip_content = BytesIO(response.content)
    with zipfile.ZipFile(zip_content, 'r') as zip_ref:
        zip_ref.extract('datashare.csv', data_dir)

    print(f"  Saved to {output_file}")
    return output_file


def download_macro_predictors(data_dir: str = './data') -> str:
    """Download macro predictors from Google Drive (Tidy Finance source)."""
    os.makedirs(data_dir, exist_ok=True)
    output_file = os.path.join(data_dir, 'macro_predictors.csv')

    if os.path.exists(output_file):
        print(f"  Macro predictors already exist at {output_file}")
        return output_file

    print("  Downloading macro predictors (Welch & Goyal 2008)...")
    sheet_id = "1bM7vCWd3WOt95Sf9qjLPZjoiafgF_8EG"
    sheet_name = "macro_predictors.xlsx"
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"

    macro_predictors = pd.read_csv(url, thousands=",")
    macro_predictors['date'] = pd.to_datetime(macro_predictors['yyyymm'], format='%Y%m')
    macro_predictors['dp'] = (macro_predictors['D12'] / macro_predictors['Index']).apply(lambda x: x if x > 0 else pd.NA).apply(lambda x: pd.NA if pd.isna(x) else pd.NA if x <= 0 else pd.Series([x]).apply(lambda y: y).iloc[0])

    # Simplified calculation
    import numpy as np
    macro_predictors = macro_predictors.assign(
        dp=lambda x: np.log(x['D12']) - np.log(x['Index']),
        dy=lambda x: np.log(x['D12']) - np.log(x['Index'].shift(1)),
        ep=lambda x: np.log(x['E12']) - np.log(x['Index']),
        de=lambda x: np.log(x['D12']) - np.log(x['E12']),
        tms=lambda x: x['lty'] - x['tbl'],
        dfy=lambda x: x['BAA'] - x['AAA']
    )

    macro_predictors = macro_predictors.rename(columns={'b/m': 'bm'})
    macro_predictors = macro_predictors[['date', 'dp', 'ep', 'bm', 'ntis', 'tbl', 'tms', 'dfy', 'svar']].dropna()

    macro_predictors.to_csv(output_file, index=False)
    print(f"  Saved to {output_file}")
    return output_file


def download_crsp_data(data_dir: str = './data', wrds_user: Optional[str] = None,
                       wrds_password: Optional[str] = None) -> Optional[str]:
    """Download CRSP data from WRDS (requires credentials)."""
    load_dotenv()

    user = wrds_user or os.getenv('WRDS_USER')
    password = wrds_password or os.getenv('WRDS_PASSWORD')

    if not user or not password:
        print("  WRDS credentials not found - skipping CRSP download")
        print("  Set WRDS_USER and WRDS_PASSWORD in .env file to download CRSP data")
        return None

    try:
        from sqlalchemy import create_engine

        print("  Connecting to WRDS...")
        connection_string = f"postgresql+psycopg2://{user}:{password}@wrds-pgdata.wharton.upenn.edu:9737/wrds"
        wrds = create_engine(connection_string, pool_pre_ping=True)

        output_file = os.path.join(data_dir, 'crsp_monthly.csv')

        if os.path.exists(output_file):
            print(f"  CRSP data already exists at {output_file}")
            return output_file

        print("  Downloading CRSP monthly data...")
        query = """
            SELECT msf.permno, date_trunc('month', msf.mthcaldt)::date AS date,
                   msf.mthret AS ret, msf.shrout * 1000 AS shrout,
                   msf.mthprc AS altprc
            FROM crsp.msf_v2 AS msf
            INNER JOIN crsp.stksecurityinfohist AS ssih
            ON msf.permno = ssih.permno
            WHERE msf.mthcaldt BETWEEN '01/01/1960' AND '12/31/2024'
                AND ssih.primaryexch IN ('N', 'A', 'Q')
            LIMIT 1000000
        """

        crsp = pd.read_sql_query(query, wrds, parse_dates=['date'])
        crsp.to_csv(output_file, index=False)
        print(f"  Saved to {output_file}")
        return output_file

    except Exception as e:
        print(f"  Could not download CRSP data: {e}")
        return None


def download_all_data(data_dir: str = './data'):
    """Download all required datasets."""
    print("\nDownloading all datasets...")

    # 1. Stock characteristics (always available)
    download_characteristics(data_dir)

    # 2. Macro predictors (always available)
    download_macro_predictors(data_dir)

    # 3. CRSP data (requires WRDS access - optional)
    download_crsp_data(data_dir)

    print("\nDownload complete!")


if __name__ == "__main__":
    download_all_data()
