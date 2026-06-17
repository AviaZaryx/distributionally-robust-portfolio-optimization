import pandas as pd

# Load the csv file
df = pd.read_csv(r"D:\Downloads\pythonProject1\merged_stock_data_half.csv")

# Count the number of unique values in the 'Ticker' column
unique_tickers_count = df['Ticker'].nunique()

# To see the actual list of unique tickers
unique_tickers_list = df['Ticker'].unique()

print(f"Total Unique Tickers: {unique_tickers_count}")
print(f"List of Tickers: {unique_tickers_list}")