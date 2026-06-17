import pandas as pd

# 1. Define your file paths (using raw strings to avoid the backslash error)
input_file = r'D:\Downloads\pythonProject1\combined_all_stocks.csv'
output_file = r'D:\Downloads\pythonProject1\combined_all_stocks_cleaned.csv'


def reprocess_csv(file_path, save_path):
    print(f"Reading {file_path}...")
    df = pd.read_csv(file_path)

    # 2. Convert Date column to actual datetime objects
    df['Date'] = pd.to_datetime(df['Date'])

    # 3. Identify and Remove Duplicates (The "cannot reshape" fix)
    # This keeps the LAST entry for every Date/Ticker pair
    original_count = len(df)
    df = df.drop_duplicates(subset=['Date', 'Ticker'], keep='last')

    removed = original_count - len(df)
    if removed > 0:
        print(f"Removed {removed} duplicate rows.")

    # 4. Sort the data (Date first, then Ticker alphabetically)
    df = df.sort_values(by=['Date', 'Ticker'])

    # 5. Optional: Basic Data Cleaning
    # If a stock is missing a price for one day, fill it with the previous day's price
    df['Adj Close'] = df.groupby('Ticker')['Adj Close'].ffill()

    # 6. Save the cleaned file
    df.to_csv(save_path, index=False)
    print(f"Successfully saved cleaned file to: {save_path}")


# Run the process
reprocess_csv(input_file, output_file)