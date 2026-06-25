import numpy as np
import pandas as pd
import warnings
import time

warnings.filterwarnings('ignore')

# ---------------- CONFIG ----------------
SCALE_FACTOR = 1
CSV_FILE = r'D:\Downloads\pythonProject1\combined_all_stocks_cleaned.csv'
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
    if s.empty: return 0, 0, 0, 0, 0
    mn = s.mean() * 252
    sd = s.std() * np.sqrt(252)
    sh = mn / sd if sd > 0 else 0
    ds = s[s < 0].std() * np.sqrt(252)
    so = mn / ds if ds > 0 else 0
    md = ((1 + s).cumprod() / (1 + s).cumprod().cummax() - 1).min()
    return mn, sd, sh, so, md


# --- Added optional toggle for signature consistency ---
def run_1n(data = load_data(CSV_FILE), csv=CSV_FILE, use_custom_tol=False, custom_tol=1e-7, d = 0.003, FIXED_GAMMA = 5):
    if data.empty: return pd.DataFrame()

    results = []

    # 1/N weights are constant
    n = data.shape[1]
    w = np.ones(n) / n

    # START TIMER FOR THIS DELTA
    delta_start_time = time.time()

    rets_1n = []
    stats_summary = []

    # Backtest loop executed for each delta
    for i in range(EST_WIN, len(data) - PRED_WIN, PRED_WIN):
        pred_data = data.iloc[i:i + PRED_WIN].pct_change().dropna()
        if pred_data.empty: continue

        # Note: Because 1/N is analytical, 'custom_tol' is not used here
        # but is included to keep the code compatible with standardized callers.

        # Tracking "status" for consistency
        stats_summary.append("Analytical (Closed-form)")
        rets_1n.extend(pred_data.values @ w)

    # Calculate statistics
    stats = get_stats(rets_1n)

    # END TIMER FOR THIS DELTA
    delta_end_time = time.time()
    delta_runtime = delta_end_time - delta_start_time

    results.append({
        'Gamma': FIXED_GAMMA,
        'Delta': d,
        'Runtime': delta_runtime,  # Added runtime here
        'Mean Return': stats[0] / SCALE_FACTOR,
        'STD DEV': stats[1] / SCALE_FACTOR,
        'Sharpe Ratio': stats[2],
        'Sortino Ratio': stats[3],
        'Max Drawdown': stats[4]
    })

    # ONLY PRINT STATUS FOR THE FIRST DELTA (as it's the same for all)
    if __name__ == "__main__":
        unique_statuses = pd.Series(stats_summary).value_counts()
        print("\n--- Solver Status Summary (1/N Portfolio) ---")
        print(unique_statuses)
        if use_custom_tol:
            print(f"(Note: Tolerance requested at {custom_tol}, but 1/N is analytical)")
        else:
            print("Note: 1/N is non-iterative; it always converges analytically.")

    return pd.DataFrame(results)


if __name__ == "__main__":
    # use_custom_tol set to False as default
    df = run_1n(use_custom_tol=False, custom_tol=1e-7)

    print("\n--- 1/N PORTFOLIO RESULTS (Delta Sweep with Runtime) ---")
    # Including Runtime in the print output
    cols_to_print = ['delta', 'Runtime (s)', 'sharpe_ratio', 'mean_return', 'std_dev']
    print(df.to_string(index=False))