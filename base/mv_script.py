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


def optimize_mv(returns, gamma, use_custom_tol=False, custom_tol=1e-7):
    n = returns.shape[1]
    mu = returns.mean().values
    Sigma = returns.cov().values

    # Numerical stability
    min_eig = np.min(np.linalg.eigvalsh(Sigma))
    if min_eig < 1e-8:
        Sigma += (1e-8 - min_eig) * np.eye(n)

    w = cp.Variable(n)
    risk = cp.quad_form(w, Sigma)
    obj = cp.Maximize(mu @ w - gamma * risk)
    constraints = [cp.sum(w) == 1, w >= 0]
    prob = cp.Problem(obj, constraints)

    # Prepare solver options
    solver_opts = {'solver': cp.OSQP}
    if use_custom_tol:
        solver_opts.update({
            'eps_abs': custom_tol,
            'eps_rel': custom_tol,
            'eps_prim_inf': custom_tol,
            'eps_dual_inf': custom_tol
        })

    status = "error"
    try:
        prob.solve(**solver_opts)
        status = prob.status
    except Exception as e:
        status = f"Error: {str(e)}"

    if w.value is None or status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        return np.ones(n) / n, status

    res = np.maximum(w.value, 0)
    return res / res.sum(), status


def run_mv(data = load_data(CSV_FILE), csv=CSV_FILE, use_custom_tol=False, custom_tol=1e-7, d = 0.003, FIXED_GAMMA = 5):
    if data.empty: return pd.DataFrame()

    results = []

    # Iterate through each delta and measure runtime for each
    start_time = time.time()  # START TIMER

    rets_mv = []
    stats_summary = []

    # Run the backtest for this delta
    for i in range(EST_WIN, len(data) - PRED_WIN, PRED_WIN):
        est = data.iloc[i - EST_WIN:i].pct_change().dropna()
        pred = data.iloc[i:i + PRED_WIN].pct_change().dropna()
        if est.empty: continue

        # Pass tolerance toggle through
        w, status = optimize_mv(est, FIXED_GAMMA, use_custom_tol, custom_tol)
        stats_summary.append(status)
        rets_mv.extend(pred.values @ w)

    end_time = time.time()  # END TIMER
    runtime = end_time - start_time

    # Calculate statistics
    stats = get_stats(rets_mv)

    results.append({
        'Gamma': FIXED_GAMMA,
        'Delta': d,
        'Runtime': runtime,
        'Mean Return': stats[0] / SCALE_FACTOR,
        'STD DEV': stats[1] / SCALE_FACTOR,
        'Sharpe Ratio': stats[2],
        'Sortino Ratio': stats[3],
        'Max Drawdown': stats[4]
    })

    # Print status summary for the first delta as a health check
    if __name__ == "__main__":
        unique_statuses = pd.Series(stats_summary).value_counts()
        print(f"\n--- Solver Status (Sample for Delta {d}) ---")
        print(unique_statuses)

    return pd.DataFrame(results)


if __name__ == "__main__":
    # Toggle use_custom_tol to True to use customizable tolerances
    df = run_mv(use_custom_tol=False, custom_tol=1e-7)

    print("\n--- CLASSIC MV RESULTS (Baseline with Per-Delta Runtime) ---")
    # Output the table showing individual runtimes
    cols_to_show = ['delta', 'Runtime (s)', 'sharpe_ratio', 'mean_return', 'std_dev']
    print(df.to_string(index=False))