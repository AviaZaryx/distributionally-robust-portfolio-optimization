import pandas as pd
import numpy as np


def clean_stock_data(input_file, output_file):
    print(f"--- Loading: {input_file} ---")
    df = pd.read_csv(input_file)

    # 1. Force conversion of common 'empty' strings to real NaNs
    # This catches "nan", "NaN", "null", and empty spaces
    df = df.replace(r'^\s*$', np.nan, regex=True)
    df = df.replace(['nan', 'NaN', 'null', 'None', 'None '], np.nan)

    # 2. Drop Duplicate Rows (Exact Ticker + Date overlaps)
    initial_count = len(df)
    df = df.drop_duplicates(subset=['Date', 'Ticker'], keep='first')
    if len(df) < initial_count:
        print(f"Dropped {initial_count - len(df):,} duplicate rows.")

    # 3. Identify the "Perfect" number of trading days
    # We find the ticker that has the most entries and assume that is the standard
    ticker_counts = df['Ticker'].value_counts()
    max_days = ticker_counts.max()
    print(f"Target number of trading days: {max_days}")

    # 4. Filter Strategy A: Remove any ticker with ANY remaining NaN values
    bad_tickers_nan = df[df.isna().any(axis=1)]['Ticker'].unique()

    # 5. Filter Strategy B: Remove any ticker that doesn't have the full day count
    # This catches tickers that just have 'missing rows' instead of 'nan rows'
    bad_tickers_incomplete = ticker_counts[ticker_counts < max_days].index.tolist()

    # Combine both lists of bad tickers
    all_bad_tickers = set(bad_tickers_nan) | set(bad_tickers_incomplete)

    # 6. Apply the filter
    df_cleaned = df[~df['Ticker'].isin(all_bad_tickers)]

    # Final cleanup: Sort properly
    df_cleaned = df_cleaned.sort_values(['Date', 'Ticker']).reset_index(drop=True)

    # 7. Detailed Report
    print(f"\n{'=' * 40}")
    print(f"FILTER REPORT")
    print(f"{'=' * 40}")
    print(f"Total Tickers in file:      {len(ticker_counts)}")
    print(f"Tickers with NaNs:          {len(bad_tickers_nan)}")
    print(f"Tickers with missing days:  {len(set(bad_tickers_incomplete) - set(bad_tickers_nan))}")
    print(f"Tickers REMOVED total:      {len(all_bad_tickers)}")
    print(f"Tickers REMAINING:         {len(df_cleaned['Ticker'].unique())}")
    print(f"{'=' * 40}")

    if len(all_bad_tickers) > 0:
        print("\nTop 10 removed tickers:")
        print(sorted(list(all_bad_tickers))[:10])

    # 8. Save
    df_cleaned.to_csv(output_file, index=False)
    print(f"\nCleaned file saved as: {output_file}")


if __name__ == "__main__":
    # Change these if your filenames are different
    INPUT = "merged_stock_data.csv"
    OUTPUT = "merged_stock_data.csv"

    clean_stock_data(INPUT, OUTPUT)