import numpy as np
import pandas as pd
import warnings
import time

warnings.filterwarnings('ignore')

# ---------------- CONFIG ----------------
CSV_FILE = 'combined_all_stocks_cleaned.csv'
START_DATE = '2017-01-01'
END_DATE = '2021-12-31'
EST_WIN = 250
PRED_WIN = 21


def load_data(csv_path):
    try:
        df = pd.read_csv(csv_path, parse_dates=['Date'])
    except:
        return pd.DataFrame()
    mask = (df['Date'] >= START_DATE) & (df['Date'] <= END_DATE)
    df = df.loc[mask].sort_values('Date')
    df_adj = df.pivot(index='Date', columns='Ticker', values='Adj Close').sort_index()
    df_adj = df_adj.dropna(axis=1, how='any').dropna(how='any')
    if df_adj.empty:
        return pd.DataFrame()
    return df_adj / df_adj.iloc[0]


def get_stats(r):
    s = pd.Series(r)
    if s.empty:
        return 0, 0, 0, 0, 0
    mn = s.mean() * 252
    sd = s.std() * np.sqrt(252)
    sh = mn / sd if sd > 0 else 0
    ds = s[s < 0].std() * np.sqrt(252)
    so = mn / ds if ds > 0 else 0
    cum_ret = (1 + s).cumprod()
    md = (cum_ret / cum_ret.cummax() - 1).min()
    return mn, sd, sh, so, md


def run_1n_gamma(csv=CSV_FILE):
    data = load_data(csv)
    if data.empty:
        print("Data is empty.")
        return pd.DataFrame()

    # --- SETTINGS ---
    gamma_range = [i for i in range(11)]
    FIXED_DELTA = 0.005

    results = []

    # 1/N weights are constant
    n = data.shape[1]
    w = np.ones(n) / n

    for g in gamma_range:
        # START TIMER FOR THIS GAMMA
        t_start = time.time()

        rets_1n = []
        # Backtest loop executed for each gamma to measure individual runtime
        for i in range(EST_WIN, len(data) - PRED_WIN, PRED_WIN):
            pred_data = data.iloc[i:i + PRED_WIN].pct_change().dropna()
            if pred_data.empty:
                continue

            # Calculate returns for this period
            rets_1n.extend(pred_data.values @ w)

        # Calculate overall stats
        stats = get_stats(rets_1n)

        # END TIMER FOR THIS GAMMA
        t_end = time.time()
        runtime = t_end - t_start

        results.append({
            'gamma_risk': g,
            'delta': FIXED_DELTA,
            'Runtime (s)': runtime,  # Added Runtime
            'Mean_1N': stats[0],
            'Std_1N': stats[1],
            'Sharpe_1N': stats[2],
            'Sortino_1N': stats[3],
            'MDD_1N': stats[4]
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    df_res = run_1n_gamma()
    print("\n--- 1/N PORTFOLIO RESULTS (Gamma Sweep with Runtime) ---")
    # Showing the full range to demonstrate per-delta/gamma runtime consistency
    print(df_res[['gamma_risk', 'Runtime (s)', 'Sharpe_1N', 'Mean_1N']].to_string(index=False))