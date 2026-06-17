import numpy as np
import pandas as pd
import cvxpy as cp
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


def optimize_mv(returns, gamma):
    n = returns.shape[1]
    mu = returns.mean().values
    Sigma = returns.cov().values

    # Numerical stability for Sigma
    min_eig = np.min(np.linalg.eigvalsh(Sigma))
    if min_eig < 1e-8:
        Sigma += (1e-8 - min_eig) * np.eye(n)

    w = cp.Variable(n)
    risk = cp.quad_form(w, Sigma)

    # Standard MV Objective: Maximize Return - Gamma * Risk
    obj = cp.Maximize(mu @ w - gamma * risk)
    constraints = [cp.sum(w) == 1, w >= 0]

    prob = cp.Problem(obj, constraints)
    try:
        prob.solve(solver=cp.SCS, verbose=False)
    except:
        pass

    if w.value is None:
        return np.ones(n) / n

    res = np.maximum(w.value, 0)
    return res / res.sum()


def run_mv_gamma(csv=CSV_FILE):
    data = load_data(csv)
    if data.empty:
        print("Data is empty.")
        return pd.DataFrame()

    # --- PARAMETERS ---
    FIXED_DELTA = 0.005  # Placeholder for tabular consistency
    gamma_range = [i for i in range(11)]

    results = []

    for g in gamma_range:
        # START TIMER FOR THIS GAMMA
        t_start = time.perf_counter()

        rets_mv = []

        # Backtest loop for the current Gamma
        for i in range(EST_WIN, len(data) - PRED_WIN, PRED_WIN):
            est = data.iloc[i - EST_WIN:i].pct_change().dropna()
            pred = data.iloc[i:i + PRED_WIN].pct_change().dropna()

            if est.empty or pred.empty:
                continue

            # Optimize for the specific Gamma
            w = optimize_mv(est, g)

            # Calculate and store returns
            rets_mv.extend(pred.values @ w)

        # Performance metrics
        stats = get_stats(rets_mv)

        # END TIMER AND CALCULATE RUNTIME
        t_end = time.perf_counter()
        runtime = t_end - t_start

        results.append({
            'gamma_risk': g,
            'delta': FIXED_DELTA,
            'Runtime (s)': runtime,  # Added Runtime here
            'mean_return': stats[0],
            'std_dev': stats[1],
            'sharpe_ratio': stats[2],
            'sortino_ratio': stats[3],
            'max_drawdown': stats[4]
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    df_res = run_mv_gamma()
    print("\n--- CLASSIC MEAN-VARIANCE RESULTS (Gamma Sweep with Runtime) ---")
    print(df_res[['gamma_risk', 'Runtime (s)', 'mean_return', 'sharpe_ratio', 'std_dev']].to_string(index=False))