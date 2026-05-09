"""
market_data.py
--------------
Pulls all market input features for the FOMC statement prediction project.

Data sources (all free / public):
  - FRED API  : 2yr/10yr Treasury yields, MOVE index, inflation breakevens,
                credit spreads, Fed funds futures implied rate
  - yfinance  : VIX, S&P 500, DXY (dollar index)

HOW TO RUN:
    Create a .env file in the project root:
        FRED_API_KEY=your_key_here
    Then run:
        python src/market_data.py

FRED API key (free): https://fred.stlouisfed.org/docs/api/api_key.html
"""

import os
import argparse
import pandas as pd
import yfinance as yf
from fredapi import Fred
from dotenv import load_dotenv

#Paths
ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_MARKET = os.path.join(ROOT, "data", "raw", "market")
os.makedirs(RAW_MARKET, exist_ok=True)

#FRED series to download
# Each entry: (FRED series ID, output filename, column rename)
FRED_SERIES = [
    ("DGS2",    "treasury_2yr.csv",         "treasury_2yr"),
    ("DGS10",   "treasury_10yr.csv",        "treasury_10yr"),
    ("MOVE",    "move_index.csv",           "move_index"),        # bond volatility
    ("T5YIE",   "inflation_breakeven.csv",  "breakeven_5yr"),     # 5yr inflation expectations
    ("BAMLH0A0HYM2", "hy_spread.csv",       "hy_spread"),         # high-yield credit spread
    ("FEDFUNDS","fed_funds_rate.csv",       "fed_funds_rate"),
    ("T10Y2Y",  "yield_curve_slope.csv",    "yield_curve_slope"), # 10yr - 2yr spread
]

#yfinance tickers to download
YFINANCE_TICKERS = {
    "^VIX":   ("vix.csv",    "vix"),          # equity fear gauge
    "^GSPC":  ("sp500.csv",  "sp500"),        # S&P 500
    "DX-Y.NYB": ("dxy.csv", "dxy"),           # US dollar index
}


def download_fred_series(fred: Fred, start: str, end: str):
    """Downloads each FRED series and saves to CSV."""
    print("[market_data] Downloading FRED series...")
    for series_id, filename, col_name in FRED_SERIES:
        fpath = os.path.join(RAW_MARKET, filename)
        if os.path.exists(fpath):
            print(f"  [skip] {filename} already exists")
            continue
        try:
            data = fred.get_series(series_id, observation_start=start, observation_end=end)
            df   = data.reset_index()
            df.columns = ["date", col_name]
            df.to_csv(fpath, index=False)
            print(f"  [ok] {filename}  ({len(df)} rows)")
        except Exception as e:
            print(f"  [!] {series_id} failed: {e}")


def download_yfinance_data(start: str, end: str):
    """Downloads each yfinance ticker and saves Close prices to CSV."""
    print("[market_data] Downloading yfinance data...")
    for ticker, (filename, col_name) in YFINANCE_TICKERS.items():
        fpath = os.path.join(RAW_MARKET, filename)
        if os.path.exists(fpath):
            print(f"  [skip] {filename} already exists")
            continue
        try:
            df = yf.download(ticker, start=start, end=end, progress=False)
            if df.empty:
                print(f"  [!] No data returned for {ticker}")
                continue
            out = df[["Close"]].rename(columns={"Close": col_name})
            out.index.name = "date"
            out.reset_index().to_csv(fpath, index=False)
            print(f"  [ok] {filename}  ({len(out)} rows)")
        except Exception as e:
            print(f"  [!] {ticker} failed: {e}")


def main():
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--start",    type=str, default="1993-01-01",
                        help="Start date for market data (default: 1993-01-01)")
    parser.add_argument("--end",      type=str, default="2024-12-31",
                        help="End date for market data (default: 2024-12-31)")
    parser.add_argument("--fred-key", type=str, default=None,
                        help="FRED API key (or set FRED_API_KEY in .env)")
    args = parser.parse_args()

    fred_key = args.fred_key or os.getenv("FRED_API_KEY")
    if not fred_key:
        raise ValueError(
            "FRED API key not found. "
            "Either pass --fred-key or create a .env file with FRED_API_KEY=..."
        )

    fred = Fred(api_key=fred_key)
    download_fred_series(fred, args.start, args.end)
    download_yfinance_data(args.start, args.end)

    print(f"\n[market_data] Done. All files saved to {RAW_MARKET}")


if __name__ == "__main__":
    main()
