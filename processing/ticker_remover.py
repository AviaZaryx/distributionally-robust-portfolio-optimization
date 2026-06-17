import pandas as pd
import random


def remove_random_tickers(input_file, output_file, num_to_remove):
    print(f"Loading {input_file}...")
    df = pd.read_csv(input_file)

    # 1. Get list of unique tickers
    all_tickers = df['Ticker'].unique().tolist()
    total_initial = len(all_tickers)

    if num_to_remove >= total_initial:
        print(f"Error: You are trying to remove {num_to_remove} tickers, but only {total_initial} exist!")
        return

    # 2. Randomly select tickers to discard
    tickers_to_remove = random.sample(all_tickers, num_to_remove)

    # 3. Filter the dataframe
    # Keep only tickers NOT in the removal list
    df_subset = df[~df['Ticker'].isin(tickers_to_remove)]

    # 4. Final Stats
    remaining_tickers = df_subset['Ticker'].unique().tolist()

    print(f"\n--- Random Subset Report ---")
    print(f"Initial Tickers:      {total_initial}")
    print(f"Tickers Removed:      {num_to_remove}")
    print(f"Tickers Remaining:    {len(remaining_tickers)}")
    print(f"Rows Before:          {len(df):,}")
    print(f"Rows After:           {len(df_subset):,}")
    print("-" * 30)
    print(f"Removed: {sorted(tickers_to_remove)}")

    # 5. Save the new file
    df_subset.to_csv(output_file, index=False)
    print(f"\nSuccess: Subset saved to {output_file}")


if __name__ == "__main__":
    INPUT_FILE = r"D:\Downloads\pythonProject1\merged_stock_data.csv"
    OUTPUT_FILE = r"D:\Downloads\pythonProject1\merged_stock_data_half.csv"

    # --- CONFIGURATION ---
    # Change this number to however many tickers you want to delete
    NUMBER_OF_TICKERS_TO_REMOVE = 100
    # ---------------------

    remove_random_tickers(INPUT_FILE, OUTPUT_FILE, NUMBER_OF_TICKERS_TO_REMOVE)