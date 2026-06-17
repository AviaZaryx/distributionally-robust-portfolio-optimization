"""
Fetch OHLCV stock data from Yahoo Finance for tickers NOT already in
combined_all_stocks_cleaned.csv, in the exact same format and date range.

Output schema (matches original exactly):
    Date, Open, High, Low, Close, Adj Close, Volume, Ticker

- Date range: 2010-01-04 to 2023-12-29 (same as original, ~3,522 trading days)
- Long format: one row per (date, ticker)
- Sorted by Date ascending, then Ticker ascending
- Pre-IPO days preserved as NaN rows (same as original, e.g. BJ in 2010-2018)

Usage:
    pip install yfinance pandas
    python fetch_new_stocks.py

Output: combined_new_stocks_cleaned.csv
"""

import time
import pandas as pd
import yfinance as yf

# ---------------- CONFIG ----------------
START_DATE = "2010-01-04"
END_DATE   = "2023-12-30"   # yfinance end is EXCLUSIVE, so +1 to include 2023-12-29
OUTPUT     = "combined_new_stocks_cleaned.csv"

# The 49 tickers already in your training set — these will be SKIPPED.
EXISTING_TICKERS = { None
}

# Candidate tickers to fetch — large-cap US names across sectors NOT in the
# original set. Edit this list freely; anything overlapping with
# EXISTING_TICKERS is automatically filtered out.
CANDIDATE_TICKERS = [
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG", "OXY", "PSX", "VLO", "MPC", "HAL",
    "PXD", "KMI", "WMB", "OKE", "BKR",
    # Industrials
    "BA", "CAT", "DE", "GE", "HON", "LMT", "RTX", "UPS", "FDX", "UNP",
    "MMM", "EMR", "ETN", "ITW", "PH", "ROK", "CSX", "NSC", "WM",
    # Materials & Chemicals
    "LIN", "APD", "DD", "DOW", "NEM", "FCX", "NUE", "SHW", "ECL", "PPG",
    # Utilities
    "NEE", "DUK", "SO", "AEP", "EXC", "SRE", "D", "XEL", "PEG", "ED",
    # Telecom / Media
    "CMCSA", "TMUS", "CHTR", "NFLX",
    # Healthcare
    "JNJ", "PFE", "MRK", "ABBV", "LLY", "BMY", "TMO", "ABT", "DHR", "MDT",
    "AMGN", "GILD", "CVS", "CI", "ELV", "HUM", "ISRG", "SYK", "BSX", "BDX",
    # REITs
    "AMT", "PLD", "EQIX", "CCI", "PSA", "O", "SPG", "WELL", "AVB", "EQR",
    # Tech (not in original)
    "ORCL", "IBM", "TXN", "QCOM", "AMD", "MU", "AMAT", "LRCX", "KLAC", "ADI",
    "NOW", "INTU", "PYPL", "SQ", "SHOP", "SNAP", "PINS", "UBER", "LYFT",
    # Consumer
    "SBUX", "MDLZ", "MO", "PM", "CL", "KMB", "GIS", "K", "HSY", "STZ",
    "EL", "CHD", "CLX", "TJX", "ROST", "BBY", "EBAY", "ETSY",
    # Financials (not in original)
    "BLK", "SCHW", "AXP", "V", "MA", "SPGI", "MCO", "ICE", "CME", "CB",
    "AON", "MMC", "AJG", "WTW",
"AVGO", "ADSK", "PANW", "FTNT", "CDNS", "SNPS", "CRWD", "SNOW", "TEAM", "MDB",
    "BX", "KKR", "APO", "MSCI", "NDAQ", "COF", "DFS", "SYF",
    "VRTX", "REGN", "HCA", "MCK", "COR", "CNC", "ZTS", "IDXX",
    "BKNG", "ABNB", "MAR", "HLT", "ORLY", "AZO", "LEN", "DHI", "TSCO",
    "ADM", "SYY", "MNST", "TSN", "STZ", "KVUE",
    "ACN", "RSG", "FAST", "GWW", "URI", "CPRT", "CTAS",
    "VICI", "CBRE", "DLR", "EXR",
    "PCG", "CEG", "FE", "MPLX", "FANG",   "AVGO", "ANET", "PLTR", "WDAY", "DDOG", "ZS", "PANW", "CRWD", "SNOW", "TEAM",
    "BX", "KKR", "APO", "NDAQ", "COF", "DFS", "TROW", "AMP",
    "VRTX", "REGN", "HCA", "MCK", "ZTS", "IDXX", "EW",
    "BKNG", "ABNB", "CMG", "LULU", "MAR", "YUM", "TSCO",
    "ACN", "RSG", "FAST", "GWW", "URI", "CPRT", "VMC",
    "CEG", "FANG", "PCG", "VICI", "DLR", "EXR", 'AAPL', 'ADBE', 'AEP', 'AFG', 'AIG', 'ALL', 'AMZN', 'APD', 'BA', 'BAC', 'BK', 'BKR',
 'C', 'CAT', 'CINF', 'COP', 'COST', 'CRM', 'CSCO', 'CSX', 'CVX', 'DD', 'DE', 'DG',
 'DIS', 'DLTR', 'DUK', 'ECL', 'EMR', 'EOG', 'ETN', 'EXC', 'FCX', 'FDX', 'GE', 'GOOGL',
 'GS', 'HAL', 'HD', 'HIG', 'HON', 'INTC', 'ITW', 'JPM', 'KO', 'KR', 'LIN', 'LMT',
 'LNC', 'LOW', 'MCD', 'MET', 'MMM', 'MS', 'MSFT', 'NEE', 'NEM', 'NKE', 'NSC', 'NUE',
 'NVDA', 'OKE', 'OXY', 'PEP', 'PG', 'PGR', 'PH', 'PNC', 'PPG', 'PRU', 'ROK', 'RTX',
 'SHW', 'SLB', 'SO', 'T', 'TD', 'TGT', 'TRV', 'UNH', 'UNP', 'UPS', 'USB', 'VLO', 'VZ',
 'WFC', 'WM', 'WMB', 'WMT', 'XOM', 'TSLA', 'KMI', 'MPC', 'PSX', 'META', 'BJ', 'DOW'
]

# Limit how many to fetch. Set to None for "all candidates not in existing".
MAX_TICKERS = "All"   # match original count; set to None for all

# Politeness delay between requests (seconds). yfinance is rate-limit friendly,
# but a small sleep avoids transient errors on big batches.
SLEEP_BETWEEN = 0.3
# ----------------------------------------


def fetch_one(ticker: str) -> pd.DataFrame:
    """Download one ticker over the date range and return a long-format DF
    with the exact 8 columns of the original file."""
    df = yf.download(
        ticker,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=False,   # keep raw OHLC + separate "Adj Close" — matches original
        progress=False,
        threads=False,
    )
    if df.empty:
        return pd.DataFrame()

    # yfinance can return a MultiIndex on columns even for a single ticker.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index().rename(columns={"Date": "Date"})
    df["Ticker"] = ticker
    return df[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume", "Ticker"]]


def main():
    # 1. Build the trading-day spine from SPY so pre-IPO tickers get NaN rows
    #    on dates the market was open — exactly mirroring the original file's
    #    handling of late-IPO tickers like BJ.
    print("Fetching SPY to establish the trading calendar...")
    spy = yf.download("SPY", start=START_DATE, end=END_DATE,
                      auto_adjust=False, progress=False, threads=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    trading_days = spy.index
    print(f"  {len(trading_days)} trading days "
          f"({trading_days.min().date()} to {trading_days.max().date()})\n")

    # 2. Build the final ticker list: candidates minus anything already in the
    #    original dataset, capped at MAX_TICKERS.
    to_fetch = [t for t in CANDIDATE_TICKERS if t not in EXISTING_TICKERS]
    # Deduplicate while preserving order
    seen = set()
    to_fetch = [t for t in to_fetch if not (t in seen or seen.add(t))]

    if MAX_TICKERS == "All":
        to_fetch = to_fetch

    elif MAX_TICKERS is not None:
        to_fetch = to_fetch[:MAX_TICKERS]


    print(f"Will fetch {len(to_fetch)} tickers (excluded "
          f"{len(set(CANDIDATE_TICKERS) & EXISTING_TICKERS)} overlapping with "
          f"the existing set)\n")

    # 3. Fetch each ticker, reindex to the full trading calendar so missing
    #    pre-IPO days become NaN rows.
    all_frames = []
    failed = []
    for i, ticker in enumerate(to_fetch, 1):
        print(f"[{i:2d}/{len(to_fetch)}] {ticker} ... ", end="", flush=True)
        try:
            df = fetch_one(ticker)
            if df.empty:
                raise ValueError("no data returned")
        except Exception as e:
            print(f"FAILED ({e})")
            failed.append(ticker)
            # Still append an all-NaN frame so the ticker shows up on every
            # date — matches how the original handled missing data.
            df = pd.DataFrame({
                "Date": trading_days,
                "Open": pd.NA, "High": pd.NA, "Low": pd.NA,
                "Close": pd.NA, "Adj Close": pd.NA, "Volume": pd.NA,
                "Ticker": ticker,
            })
            all_frames.append(df)
            continue

        # Reindex to the full trading calendar; pre-IPO days become NaN.
        df = df.set_index("Date").reindex(trading_days)
        df["Ticker"] = ticker
        df = df.reset_index().rename(columns={"index": "Date"})
        all_frames.append(df)
        n_real = df["Close"].notna().sum()
        print(f"OK ({n_real} real rows, {len(df) - n_real} NaN)")
        time.sleep(SLEEP_BETWEEN)

    # 4. Concat, format date string, sort, write.
    combined = pd.concat(all_frames, ignore_index=True)
    combined["Date"] = pd.to_datetime(combined["Date"]).dt.strftime("%Y-%m-%d")
    combined = combined.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    combined.to_csv(OUTPUT, index=False)

    # 5. Summary
    print(f"\n{'='*50}")
    print(f"Wrote {len(combined):,} rows to {OUTPUT}")
    print(f"Unique tickers: {combined['Ticker'].nunique()}")
    print(f"Date range: {combined['Date'].min()} to {combined['Date'].max()}")
    print(f"NaN OHLCV rows: {combined['Close'].isna().sum():,}")
    if failed:
        print(f"Failed tickers ({len(failed)}): {failed}")
    print(f"{'='*50}")

    # Sanity check vs original format
    print("\nFirst 10 rows:")
    print(combined.head(10).to_string())


if __name__ == "__main__":
    main()