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
    except:
        return pd.DataFrame()
    mask = (df['Date'] >= START_DATE) & (df['Date'] <= END_DATE)
    df = df.loc[mask].sort_values('Date')
    df_adj = df.pivot(index='Date', columns='Ticker', values='Adj Close').sort_index()
    df_adj = df_adj.dropna(axis=1, how='any')
    df_adj = df_adj.dropna(how='any')
    if df_adj.empty:
        return pd.DataFrame()
    return df_adj / df_adj.iloc[0]


# --- 2. Model: Non-Convex Mean-Variance (Moehle et al.) ---
def optimize_moehle_nonconvex(returns, prev_w, gamma_risk, fixed_trd_cost, lin_trd_cost, fixed_hld_cost):
    n = returns.shape[1]
    mu = returns.mean().values
    Sigma = returns.cov().values

    # Numerical stability
    min_eig = np.min(np.linalg.eigvalsh(Sigma))
    if min_eig < 1e-8:
        Sigma += (1e-8 - min_eig) * np.eye(n)

    # --- Variables ---
    w = cp.Variable(n)
    abs_trade = cp.Variable(n)
    z_trade = cp.Variable(n, boolean=True)
    z_hold = cp.Variable(n, boolean=True)

    risk_term = cp.quad_form(w, Sigma)
    cost_trade = fixed_trd_cost * cp.sum(z_trade) + lin_trd_cost * cp.sum(abs_trade)
    cost_hold = fixed_hld_cost * cp.sum(z_hold)

    obj = cp.Minimize(-mu @ w + gamma_risk * risk_term + cost_trade + cost_hold)

    M = 1.0  # Big-M bound
    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        abs_trade >= w - prev_w,
        abs_trade >= -(w - prev_w),
        abs_trade <= M * z_trade,
        w <= M * z_hold
    ]

    prob = cp.Problem(obj, constraints)

    status = "error"
    try:
        # MIQP requires a branch-and-bound solver like ECOS_BB
        prob.solve(solver=cp.ECOS_BB, verbose=False)
        status = prob.status
    except Exception as e:
        status = f"Error: {str(e)[:50]}..." # Truncate long error messages

    if w.value is None or status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        return np.ones(n) / n, status

    res = np.maximum(w.value, 0)
    return res / res.sum(), status


# --- 3. Backtest Engine ---
def run_moehle_gamma(csv=CSV_FILE):
    data = load_data(csv)
    if data.empty: return pd.DataFrame()

    FIXED_DELTA = 0.005
    gamma_range = [i for i in range(11)]

    FIXED_TRD_COST = 0.0001
    FIXED_HLD_COST = 0.0001
    LIN_TRD_COST = 0.0010

    results = []

    for g in gamma_range:
        rets = []
        prev_w = np.zeros(data.shape[1])
        stats_summary = []

        for i in range(EST_WIN, len(data) - PRED_WIN, PRED_WIN):
            est = data.iloc[i - EST_WIN:i].pct_change().dropna()
            pred = data.iloc[i:i + PRED_WIN].pct_change().dropna()

            if est.empty or pred.empty: continue

            # 1. Run Optimization and capture status
            w, status = optimize_moehle_nonconvex(est, prev_w, g, FIXED_TRD_COST, LIN_TRD_COST, FIXED_HLD_COST)
            stats_summary.append(status)

            # 2. Calculate Period Return
            raw_ret_array = pred.values @ w

            # 3. Deduct costs
            trade_mag = np.abs(w - prev_w)
            epsilon = 1e-5
            cost_lin = LIN_TRD_COST * np.sum(trade_mag)
            cost_fix_trd = FIXED_TRD_COST * np.sum(trade_mag > epsilon)
            cost_fix_hld = FIXED_HLD_COST * np.sum(w > epsilon)

            total_cost = cost_lin + cost_fix_trd + cost_fix_hld
            raw_ret_array[0] -= total_cost

            rets.extend(raw_ret_array)
            prev_w = w

        # ONLY PRINT STATUS IF SCRIPT IS RUN DIRECTLY
        if __name__ == "__main__":
            unique_statuses = pd.Series(stats_summary).value_counts()
            print(f"\n--- Solver Status Summary (Moehle, Gamma={g}) ---")
            print(unique_statuses)

        if not rets: continue

        s = pd.Series(rets)
        mean_ann = s.mean() * 252
        std_ann = s.std() * np.sqrt(252)
        sharpe = mean_ann / std_ann if std_ann > 0 else 0
        ds = s[s < 0].std() * np.sqrt(252)
        sortino = mean_ann / ds if ds > 0 else 0
        mdd = ((1 + s).cumprod() / (1 + s).cumprod().cummax() - 1).min()

        results.append({
            'gamma_risk': g,
            'delta': FIXED_DELTA,
            'mean_return': mean_ann / SCALE_FACTOR,
            'std_dev': std_ann / SCALE_FACTOR,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': mdd
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    df_res = run_moehle_gamma()
    print("\n--- MOEHLE RESULTS SUMMARY ---")
    print(df_res[['gamma_risk', 'mean_return', 'sharpe_ratio']].to_string(index=False))