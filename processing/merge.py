import pandas as pd

# File paths - update these to your actual filenames
file1 = 'combined_all_stocks_cleaned.csv'
file2 = 'combined_new_stocks_cleaned.csv'
output_file = 'merged_stock_data.csv'


def merge_stock_csvs(path1, path2, output_path):
    print("Reading files...")
    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)

    print(f"File 1 has {df1['Ticker'].nunique()} unique tickers.")
    print(f"File 2 has {df2['Ticker'].nunique()} unique tickers.")

    # 1. Stack the dataframes on top of each other
    # ignore_index=True creates a fresh ID for every row
    combined_df = pd.concat([df1, df2], ignore_index=True)

    # 2. Convert Date to datetime objects to ensure proper sorting
    combined_df['Date'] = pd.to_datetime(combined_df['Date'])

    # 3. Sort by Date first, then by Ticker name
    # This ensures that for '2023-01-01', you see all ~100 tickers in a row
    print("Sorting data by Date and Ticker...")
    combined_df = combined_df.sort_values(by=['Date', 'Ticker'])

    # 4. Optional: Clean up empty rows (like the 'BJ' row in your image)
    # This removes rows where price data is missing
    combined_df = combined_df.dropna(subset=['Adj Close'])

    # Save to new CSV
    combined_df.to_csv(output_path, index=False)

    print(f"Success! Merged file saved as: {output_path}")
    print(f"Total rows: {len(combined_df)}")
    print(f"Total unique tickers: {combined_df['Ticker'].nunique()}")


if __name__ == "__main__":
    merge_stock_csvs(file1, file2, output_file)