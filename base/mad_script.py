import numpy as np
import pandas as pd
import cvxpy as cp
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


def optimize_mad(returns, gamma, use_custom_tol=False, custom_tol=1e-7):
    n = returns.shape[1]
    T = returns.shape[0]
    mu = returns.mean().values
    w = cp.Variable(n)
    z = cp.Variable(T)

    mean_port = mu @ w
    constraints = [
        cp.sum(w) == 1, w >= 0,
        z >= (returns.values @ w) - mean_port,
        z >= -((returns.values @ w) - mean_port)
    ]

    mad_risk = cp.sum(z) / T
    obj = cp.Maximize(mu @ w - gamma * mad_risk)

    prob = cp.Problem(obj, constraints)

    # Prepare solver settings
    solver_opts = {'solver': cp.ECOS, 'verbose': False}
    if use_custom_tol:
        solver_opts.update({
            'abstol': custom_tol,
            'reltol': custom_tol,
            'feastol': custom_tol
        })

    try:
        prob.solve(**solver_opts)
        status = prob.status
    except Exception as e:
        status = f"Error: {str(e)}"

    if w.value is None or status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        return np.ones(n) / n, status

    res = np.maximum(w.value, 0)
    return res / res.sum(), status


def get_stats(r):
    s = pd.Series(r)
    mn = s.mean() * 252
    sd = s.std() * np.sqrt(252)
    sh = mn / sd if sd > 0 else 0
    ds = s[s < 0].std() * np.sqrt(252)
    so = mn / ds if ds > 0 else 0
    md = ((1 + s).cumprod() / (1 + s).cumprod().cummax() - 1).min()
    return mn, sd, sh, so, md


def run_mad(data = load_data(CSV_FILE), csv=CSV_FILE, use_custom_tol=False, custom_tol=1e-7, delta = 0.003, GAMMA_RISK = 0.5):
    if data.empty: return pd.DataFrame()

    results = []

    # Iterate through each delta and measure runtime
    d = delta * 1

    delta_start_time = time.time()

    rets_mad = []
    stats_summary = []

    # Run the backtest window logic for this delta
    for i in range(EST_WIN, len(data) - PRED_WIN, PRED_WIN):
        est = data.iloc[i - EST_WIN:i].pct_change().dropna()
        pred = data.iloc[i:i + PRED_WIN].pct_change().dropna()
        if est.empty: continue

        # Run Optimization with optional tolerance settings
        w, status = optimize_mad(est, GAMMA_RISK, use_custom_tol, custom_tol)
        stats_summary.append(status)
        rets_mad.extend(pred.values @ w)

        if not rets_mad: return pd.DataFrame()

    # Calculate statistics
    stats = get_stats(rets_mad)

    delta_end_time = time.time()
    delta_runtime = delta_end_time - delta_start_time

    results.append({
        'Delta': d,
        'Gamma': GAMMA_RISK,
        'Runtime': delta_runtime,
        'Mean Return': stats[0],
        'STD DEV': stats[1],
        'Sharpe Ratio': stats[2],
        'Sortino Ratio': stats[3],
        'Max Drawdown': stats[4]
    })

    # Optional: Print status for the first delta to confirm solver health
    if __name__ == "__main__":
        unique_statuses = pd.Series(stats_summary).value_counts()
        print(f"\n--- Solver Status Summary (Sample Delta {d}) ---")
        print(unique_statuses)

    return pd.DataFrame(results)


if __name__ == "__main__":
    # Toggle use_custom_tol=True to use the custom_tol value
    df_res = run_mad(use_custom_tol=False, custom_tol=1e-7)

    print("\n--- CLASSIC MAD RESULTS (Delta Sweep with Runtime) ---")
    # Including Runtime in the print output

    print(df_res.to_string(index=False))