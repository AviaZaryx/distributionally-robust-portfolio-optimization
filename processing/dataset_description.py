import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set a professional theme
sns.set_theme(style="whitegrid")


def describe_dataset(file_path):
    print(f"Loading data from: {file_path}...")

    # 1. LOAD DATA
    df = pd.read_csv(file_path)

    # FIX: dayfirst=True handles "13/01/2010" correctly.
    # We use errors='coerce' to handle any truly unparseable dates.
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')

    # Basic Stats
    tickers = sorted(df['Ticker'].unique())
    start_date = df['Date'].min().strftime('%Y-%m-%d')
    end_date = df['Date'].max().strftime('%Y-%m-%d')

    print("-" * 50)
    print("DATASET SUMMARY")
    print("-" * 50)
    print(f"Total Rows:         {len(df):,}")
    print(f"Unique Tickers:     {len(tickers)}")
    print(f"Date Range:         {start_date} to {end_date}")

    # Identify tickers with missing days
    # drop_duplicates prevents errors if the file has redundant rows
    pivot_check = df.drop_duplicates(subset=['Date', 'Ticker']).pivot(index='Date', columns='Ticker',
                                                                      values='Adj Close')
    missing_data_tickers = pivot_check.columns[pivot_check.isnull().any()].tolist()

    print(f"\nTickers with gaps or late IPOs ({len(missing_data_tickers)}):")
    print(missing_data_tickers if len(missing_data_tickers) < 10 else f"{missing_data_tickers[:10]}... and more")
    print("-" * 50)

    # 2. VISUALIZATION: STRETCHED COVERAGE MAP (Shades of Blue)
    print("Generating Stretched Coverage Map...")
    plt.figure(figsize=(26, 10))

    # cmap='Blues' - missing data will be white, existing data will be blue
    # If you want missing data to be the dark part, use 'Blues_r'
    ax = sns.heatmap(
        pivot_check.isnull(),
        cbar=False,
        cmap='Blues',
        xticklabels=True
    )

    plt.xticks(rotation=90, fontsize=8)
    plt.xlabel("Tickers", fontsize=12, fontweight='bold')

    y_labels = [d.strftime('%Y-%m-%d') for d in pivot_check.index]
    step = max(1, len(y_labels) // 20)
    ax.set_yticks(np.arange(0, len(y_labels), step))
    ax.set_yticklabels(y_labels[::step])
    plt.ylabel("Time (YYYY-MM-DD)", fontsize=12, fontweight='bold')

    plt.title(f"Data Coverage Map (Blue = Missing Data)\nFile: {file_path}", fontsize=16)
    plt.tight_layout()
    plt.savefig('1_coverage_map_stretched.png', dpi=300)
    plt.close()

    # 3. VISUALIZATION: PRICE DISTRIBUTION (Steel Blue Boxplot)
    print("Generating Price Distribution Plot...")
    plt.figure(figsize=(26, 8))
    # Using a solid professional blue (SteelBlue)
    sns.boxplot(data=df, x='Ticker', y='Adj Close', color="steelblue")
    plt.yscale('log')
    plt.xticks(rotation=90, fontsize=8)
    plt.title("Price Range per Ticker (Log Scale)", fontsize=16)
    plt.ylabel("Adjusted Close Price (Log Scale)")
    plt.tight_layout()
    plt.savefig('2_price_distribution.png', dpi=300)
    plt.close()

    # 4. VISUALIZATION: AVERAGE TRADING VOLUME (Midnight Blue Bars)
    print("Generating Volume Liquidity Plot...")
    plt.figure(figsize=(20, 8))
    avg_volume = df.groupby('Ticker')['Volume'].mean().sort_values(ascending=False)

    # Using a dark blue (MidnightBlue)
    avg_volume.head(50).plot(kind='bar', color='midnightblue', alpha=0.8)
    plt.title("Top 50 Tickers by Average Daily Volume", fontsize=16)
    plt.ylabel("Average Volume")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('3_volume_liquidity.head.png', dpi=300)
    plt.close()

    print("\nSUCCESS: 3 Blue-themed Report images saved.")


if __name__ == "__main__":
    FILENAME = 'merged_stock_data.csv'

    try:
        describe_dataset(FILENAME)
    except FileNotFoundError:
        print(f"Error: Could not find '{FILENAME}'.")
    except Exception as e:
        print(f"An error occurred: {e}")