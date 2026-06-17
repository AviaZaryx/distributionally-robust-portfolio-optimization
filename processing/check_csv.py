import pandas as pd
import os


def inspect_csv(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    print(f"--- INSPECTING: {file_path} ---")

    # 1. Check Raw File Size
    file_size = os.path.getsize(file_path) / (1024 * 1024)
    print(f"File Size: {file_size:.2f} MB")

    # 2. Load Data
    # We load without parsing dates first to see the RAW string format
    df = pd.read_csv(file_path)

    print(f"Total Rows: {len(df):,}")
    print(f"Total Columns: {len(df.columns)}")

    # 3. Show Column Names and Data Types
    print("\n[ Column Types ]")
    print(df.dtypes)

    # 4. Show Raw Sample (First 5 rows)
    print("\n[ First 5 Rows Sample ]")
    print(df.head())

    # 5. Check for NaNs (Missing Data)
    nan_counts = df.isnull().sum()
    if nan_counts.sum() > 0:
        print("\n[ Missing Values Found ]")
        print(nan_counts[nan_counts > 0])
    else:
        print("\n[ No Missing Values Detected ]")

    # 6. CRITICAL: Check Date Format
    if 'Date' in df.columns:
        print("\n[ Date Format Analysis ]")
        raw_dates = df['Date'].head(5).tolist()
        print(f"Raw Date Strings from file: {raw_dates}")

        # Test if it's Day-First (like 13/01/2010)
        sample_date = str(df['Date'].iloc[0])
        if "/" in sample_date:
            parts = sample_date.split('/')
            if int(parts[0]) > 12:
                print("Detected Format: DD/MM/YYYY (Day-First)")
            else:
                print("Detected Format: Likely MM/DD/YYYY or DD/MM/YYYY (Ambiguous)")
        elif "-" in sample_date:
            print("Detected Format: YYYY-MM-DD (Standard ISO)")

    # 7. Ticker Check
    if 'Ticker' in df.columns:
        tickers = df['Ticker'].unique()
        print(f"\n[ Ticker Analysis ]")
        print(f"Unique Tickers: {len(tickers)}")
        print(f"Sample Tickers: {tickers[:10]}")

    # 8. Check for Duplicate Rows
    duplicates = df.duplicated(subset=['Date', 'Ticker']).sum()
    if duplicates > 0:
        print(f"\n[!] WARNING: Found {duplicates} duplicate Ticker/Date rows.")
    else:
        print("\n[ No Duplicates Found ]")


if __name__ == "__main__":
    # Change this to the filename you want to check
    FILENAME = "merged_stock_data.csv"
    inspect_csv(FILENAME)