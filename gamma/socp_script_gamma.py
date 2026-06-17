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


# --- 1. Data Loading ---
def load_data(csv_path):
    try:
        df = pd.read_csv(csv_path, parse_dates=['Date'])
    except FileNotFoundError:
        print(f"Error: {csv_path} not found.")
        return pd.DataFrame()

    mask = (df['Date'] >= START_DATE) & (df['Date'] <= END_DATE)
    df = df.loc[mask].sort_values('Date')

    # Pivot and clean
    df_adj = df.pivot(index='Date', columns='Ticker', values='Adj Close').sort_index()
    df_adj = df_adj.dropna(axis=1, how='any')
    df_adj = df_adj.dropna(how='any')

    if df_adj.empty:
        return pd.DataFrame()

    return df_adj / df_adj.iloc[0]


# --- 2. Model: SOCP with Trading AND Holding Costs ---
def optimize_socp(returns, prev_w, delta, gamma_risk, gamma_trd, gamma_hld, rho):
    n = returns.shape[1]
    mu = returns.mean().values
    Sigma = returns.cov().values

    # Numerical Stability: Ensure Sigma is Positive Definite
    min_eig = np.min(np.linalg.eigvalsh(Sigma))
    if min_eig < 1e-8:
        Sigma += (1e-8 - min_eig) * np.eye(n)

    try:
        L = np.linalg.cholesky(Sigma).T
    except:
        return np.ones(n) / n

    # Variables
    w = cp.Variable(n)  # Portfolio weights
    t = cp.Variable()  # Wasserstein auxiliary
    q = cp.Variable()  # Risk auxiliary (Variance)
    turnover = cp.Variable(n)  # Trading cost auxiliary
    holding = cp.Variable(n)  # Holding cost auxiliary

    # Objective: Minimize -Expected_Return + Robust_Penalty + Risk_Penalty + Costs
    robust_penalty = rho * delta

    obj = cp.Minimize(
        -mu @ w
        + robust_penalty * t
        + gamma_risk * q
        + gamma_trd * cp.sum(turnover)
        + gamma_hld * cp.sum(holding)
    )

    constraints = [
        # 1. Budget & Long Only
        cp.sum(w) == 1,
        w >= 0,

        # 2. Wasserstein SOC Constraint: ||w||_2 <= t
        cp.SOC(t, w),

        # 3. Risk Constraint: w.T @ Sigma @ w <= q
        cp.sum_squares(L @ w) <= q,
        q >= 0,

        # 4. Trading Cost Linearization: u >= |w - w_prev|
        turnover >= w - prev_w,
        turnover >= -(w - prev_w),
        turnover >= 0,

        # 5. Holding Cost Linearization: v >= |w|
        holding >= w,
        holding >= -w,
        holding >= 0
    ]

    prob = cp.Problem(obj, constraints)
    try:
        prob.solve(solver=cp.ECOS, verbose=False)
    except:
        pass

    if w.value is None:
        return np.ones(n) / n

    res = np.maximum(w.value, 0)
    return res / res.sum()


# --- 3. Backtest Engine (Sweeping Gamma, Fixed Delta) ---
def run_socp_gamma(csv=CSV_FILE):
    data = load_data(csv)
    if data.empty:
        print("Data is empty. Check CSV path and dates.")
        return pd.DataFrame()

    # --- PARAMETERS ---
    FIXED_DELTA = 0.005

    # Sweep through Gamma Risk (Aversion)
    gamma_range = [i for i in range(11)]

    rho_param = 1.5
    G_TRD = 0.001
    G_HLD = 0.001

    results = []

    for g in gamma_range:
        # START TIMER FOR EACH GAMMA
        t_start = time.perf_counter()

        rets = []
        prev_w = np.zeros(data.shape[1])

        # Rolling Window Backtest
        for i in range(EST_WIN, len(data) - PRED_WIN, PRED_WIN):
            est_data = data.iloc[i - EST_WIN:i].pct_change().dropna()
            pred_data = data.iloc[i:i + PRED_WIN].pct_change().dropna()

            if est_data.empty or pred_data.empty:
                continue

            # Run Optimization
            w = optimize_socp(est_data, prev_w, FIXED_DELTA, g, G_TRD, G_HLD, rho_param)

            # Calculate returns for the prediction period
            raw_ret_array = pred_data.values @ w

            # Apply Transaction Cost to the first day of the period
            trd_cost = G_TRD * np.sum(np.abs(w - prev_w))
            hld_cost = G_HLD * np.sum(np.abs(w))

            raw_ret_array[0] -= (trd_cost + hld_cost)

            rets.extend(raw_ret_array)
            prev_w = w

        # --- Calculate Performance Metrics ---
        if not rets:
            continue

        s = pd.Series(rets)
        mean_ann = s.mean() * 252
        std_ann = s.std() * np.sqrt(252)
        sharpe = mean_ann / std_ann if std_ann > 0 else 0

        downside_std = s[s < 0].std() * np.sqrt(252)
        sortino = mean_ann / downside_std if downside_std > 0 else 0

        cum_ret = (1 + s).cumprod()
        mdd = (cum_ret / cum_ret.cummax() - 1).min()

        # END TIMER AND CALCULATE RUNTIME
        t_end = time.perf_counter()
        gamma_runtime = t_end - t_start

        results.append({
            'gamma_risk': g,
            'delta': FIXED_DELTA,
            'Runtime (s)': gamma_runtime,  # Added Runtime per Gamma
            'mean_return': mean_ann / SCALE_FACTOR,
            'std_dev': std_ann / SCALE_FACTOR,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': mdd
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    final_results = run_socp_gamma()
    print("\n" + "=" * 65)
    print("FINAL SUMMARY (Fixed Delta=0.005)")
    print("=" * 65)
    # Added Runtime (s) to the output display
    print(final_results[['gamma_risk', 'Runtime (s)', 'mean_return', 'sharpe_ratio', 'max_drawdown']].to_string(
        index=False))