import numpy as np
import pandas as pd
import cvxpy as cp
import warnings
import time

warnings.filterwarnings('ignore')

# ---------------- CONFIG ----------------
SCALE_FACTOR = 1
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


def optimize_mad(returns, gamma):
    """
    Classic MAD Optimization:
    Maximize (Expected Return - Gamma * MAD_Risk)
    """
    n = returns.shape[1]
    T = returns.shape[0]
    mu = returns.mean().values

    # Variables
    w = cp.Variable(n)
    z = cp.Variable(T)  # Auxiliary for absolute deviations

    mean_port = mu @ w

    # Constraints for MAD (L1 norm)
    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        z >= (returns.values @ w) - mean_port,
        z >= -((returns.values @ w) - mean_port)
    ]

    mad_risk = cp.sum(z) / T
    obj = cp.Maximize(mu @ w - gamma * mad_risk)

    prob = cp.Problem(obj, constraints)
    try:
        prob.solve(solver=cp.ECOS, verbose=False)
    except:
        pass

    if w.value is None:
        return np.ones(n) / n

    res = np.maximum(w.value, 0)
    return res / res.sum()


def run_mad_gamma(csv=CSV_FILE):
    data = load_data(csv)
    if data.empty: return pd.DataFrame()

    gamma_range = [i * 0.1 for i in range(11)]
    FIXED_DELTA = 0.005

    results = []
    for g in gamma_range:
        # START TIMER FOR THIS GAMMA
        gamma_start_time = time.time()

        rets_mad = []
        for i in range(EST_WIN, len(data) - PRED_WIN, PRED_WIN):
            est = data.iloc[i - EST_WIN:i].pct_change().dropna()
            pred = data.iloc[i:i + PRED_WIN].pct_change().dropna()
            if est.empty: continue

            w = optimize_mad(est, g)
            rets_mad.extend(pred.values @ w)

        stats = get_stats(rets_mad)

        # END TIMER FOR THIS GAMMA
        gamma_end_time = time.time()
        gamma_runtime = gamma_end_time - gamma_start_time

        results.append({
            'gamma_risk': g * 10,
            'delta': FIXED_DELTA,
            'Runtime (s)': gamma_runtime,  # Added Runtime
            'Mean_MAD': stats[0],
            'Std_MAD': stats[1],
            'Sharpe_MAD': stats[2],
            'Sortino_MAD': stats[3],
            'MDD_MAD': stats[4]
        })
    return pd.DataFrame(results)


if __name__ == "__main__":
    df_res = run_mad_gamma()
    print("\n--- CLASSIC MAD RESULTS (Gamma Sweep with Runtime) ---")
    print(df_res[['gamma_risk', 'Runtime (s)', 'Mean_MAD', 'Sharpe_MAD', 'Std_MAD']].to_string(index=False))