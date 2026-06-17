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

# Internal math rescale (Plot-Delta 0.1 -> Internal Eps 0.002)
DELTA_RESCALE_FACTOR = 1


# --- 1. Data Loading ---
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


# --- 2. Model: DRO-MAD (Gamma/Penalty Form - Remark 1) ---
def optimize_dromad_gamma(returns, epsilon, gamma):
    """
    Follows Remark 1 of Chen et al. (2022).
    Objective: Minimize - (Worst-case Mean) + Gamma * (Worst-case MAD)
    """
    n = returns.shape[1]
    T = returns.shape[0]
    mu = returns.mean().values
    xi = returns.values

    x = cp.Variable(n)
    y1 = cp.Variable(T)
    y2 = cp.Variable(T)
    mad_wc = cp.Variable()

    # Robust MAD Constraints (Theorem 1)
    constraints = [
        cp.sum(x) == 1,
        x >= 0,
        y1 >= (mu @ x - xi @ x) - epsilon,
        y1 >= -(mu @ x - xi @ x) + epsilon,
        y2 >= (mu @ x - xi @ x) + epsilon,
        y2 >= -(mu @ x - xi @ x) - epsilon,
        mad_wc >= cp.sum(y1) / T + epsilon,
        mad_wc >= cp.sum(y2) / T + epsilon
    ]

    # Robust Mean = Historical Mean - Uncertainty Radius
    robust_mean = mu @ x - epsilon

    # Objective: Minimize Negative Robust Utility
    obj = cp.Minimize(-robust_mean + gamma * mad_wc)

    prob = cp.Problem(obj, constraints)
    try:
        prob.solve(solver=cp.ECOS)
    except:
        return np.ones(n) / n

    if x.value is None: return np.ones(n) / n
    res = np.maximum(x.value, 0)
    return res / (res.sum() + 1e-8)


# --- 3. Backtest Engine (Gamma Sweep) ---
def run_dro_mad_gamma(csv=CSV_FILE):
    data = load_data(csv)
    if data.empty: return pd.DataFrame()

    # --- SETTINGS ---
    # We fix Delta at a standard level (0.03 -> internal 0.0006)
    FIXED_DELTA_PLOT = 0.005
    internal_epsilon = FIXED_DELTA_PLOT * DELTA_RESCALE_FACTOR

    # We sweep Gamma (Risk Aversion)
    gamma_range = [i * 0.1 for i in range(11)]

    results = []

    for g in gamma_range:
        # START TIMER FOR THIS GAMMA
        gamma_start_time = time.time()

        rets = []

        for i in range(EST_WIN, len(data) - PRED_WIN, PRED_WIN):
            est = data.iloc[i - EST_WIN:i].pct_change().dropna()
            pred = data.iloc[i:i + PRED_WIN].pct_change().dropna()
            if est.empty: continue

            # Optimize using the Gamma Penalty version
            w = optimize_dromad_gamma(est, internal_epsilon, g)
            rets.extend(pred.values @ w)

        if not rets: continue

        # Performance Metrics
        s = pd.Series(rets)
        mean_ann = s.mean() * 252
        std_ann = s.std() * np.sqrt(252)
        sharpe = mean_ann / std_ann if std_ann > 0 else 0
        mdd = ((1 + s).cumprod() / (1 + s).cumprod().cummax() - 1).min()

        # END TIMER FOR THIS GAMMA
        gamma_end_time = time.time()
        gamma_runtime = gamma_end_time - gamma_start_time

        results.append({
            'gamma_risk': g * 10,
            'Delta': FIXED_DELTA_PLOT,
            'Runtime (s)': gamma_runtime,  # Added Runtime here
            'Mean Return': mean_ann,
            'Std Dev': std_ann,
            'Sharpe Ratio': sharpe,
            'Max Drawdown': mdd
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    df = run_dro_mad_gamma()
    print("\n--- DRO-MAD GAMMA SENSITIVITY ---")
    print(df.to_string(index=False))