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
def optimize_moehle_nonconvex(returns, prev_w, gamma_risk, fixed_trd_cost, lin_trd_cost, fixed_hld_cost,
                              use_custom_tol=False, custom_tol=1e-7):
    n = returns.shape[1]
    mu = returns.mean().values
    Sigma = returns.cov().values

    # Regularization
    min_eig = np.min(np.linalg.eigvalsh(Sigma))
    if min_eig < 1e-8:
        Sigma += (1e-8 - min_eig) * np.eye(n)

    # Variables
    w = cp.Variable(n)
    abs_trade = cp.Variable(n)
    z_trade = cp.Variable(n, boolean=True)
    z_hold = cp.Variable(n, boolean=True)

    risk_term = cp.quad_form(w, Sigma)
    cost_trade = fixed_trd_cost * cp.sum(z_trade) + lin_trd_cost * cp.sum(abs_trade)
    cost_hold = fixed_hld_cost * cp.sum(z_hold)

    obj = cp.Minimize(-mu @ w + gamma_risk * risk_term + cost_trade + cost_hold)

    M = 1.0
    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        abs_trade >= w - prev_w,
        abs_trade >= -(w - prev_w),
        abs_trade <= M * z_trade,
        w <= M * z_hold
    ]

    prob = cp.Problem(obj, constraints)

    # Prepare solver settings
    solver_kwargs = {"verbose": False}
    if use_custom_tol:
        solver_kwargs.update({
            'abstol': custom_tol,
            'reltol': custom_tol,
            'feastol': custom_tol
        })

    status = "not_solved"
    try:
        # Attempt solving with ECOS_BB for boolean variables
        prob.solve(solver=cp.ECOS_BB, **solver_kwargs)
        status = prob.status
    except cp.error.SolverError:
        try:
            # Fallback
            prob.solve(**solver_kwargs)
            status = prob.status
        except Exception as e:
            status = f"Solver_Error: {str(e)}"

    if w.value is None or status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        return np.ones(n) / n, status

    res = np.maximum(w.value, 0)
    return res / res.sum(), status


# --- 3. Backtest Engine ---
def run_moehle_paper(data = load_data(CSV_FILE), csv=CSV_FILE, use_custom_tol=False, custom_tol=1e-7, d = 0.003, g = 5,
FIXED_TRD_COST = 0.0001, FIXED_HLD_COST = 0.0001, LIN_TRD_COST = 0.0010):

    if data.empty: return pd.DataFrame()

    results = []

    # Start timer for each delta
    delta_start_time = time.time()

    rets_gamma = []
    prev_w = np.zeros(data.shape[1])
    stats_summary = []

    for i in range(EST_WIN, len(data) - PRED_WIN, PRED_WIN):
        est = data.iloc[i - EST_WIN:i].pct_change().dropna()
        pred = data.iloc[i:i + PRED_WIN].pct_change().dropna()

        if est.empty: continue

        # Optimization with potential tolerance override
        w, status = optimize_moehle_nonconvex(
            est, prev_w, g, FIXED_TRD_COST, LIN_TRD_COST, FIXED_HLD_COST,
            use_custom_tol=use_custom_tol, custom_tol=custom_tol
        )
        stats_summary.append(status)

        raw_ret = pred.values @ w

        trade_mag = np.abs(w - prev_w)
        cost_lin = LIN_TRD_COST * np.sum(trade_mag)

        epsilon = 1e-5
        has_traded = (trade_mag > epsilon).astype(float)
        cost_fix_trd = FIXED_TRD_COST * np.sum(has_traded)

        has_held = (w > epsilon).astype(float)
        cost_fix_hld = FIXED_HLD_COST * np.sum(has_held)

        total_cost = cost_lin + cost_fix_trd + cost_fix_hld
        raw_ret[0] -= total_cost

        rets_gamma.extend(raw_ret)
        prev_w = w

    # Calculate Runtime for this delta
    delta_end_time = time.time()
    runtime = delta_end_time - delta_start_time

    if __name__ == "__main__":
        unique_statuses = pd.Series(stats_summary).value_counts()
        print(f"\n--- Solver Status (Delta={d}, Gamma={g}) ---")
        print(unique_statuses)

    s = pd.Series(rets_gamma)
    mean = s.mean() * 252
    std = s.std() * np.sqrt(252)
    sharpe = mean / std if std > 0 else 0
    ds = s[s < 0].std() * np.sqrt(252)
    sortino = mean / ds if ds > 0 else 0
    mdd = ((1 + s).cumprod() / (1 + s).cumprod().cummax() - 1).min()

    results.append({
        'Delta': d,
        'Runtime': runtime,
        'Gamma': g,
        'Mean Return': mean / SCALE_FACTOR,
        'STD DEV': std / SCALE_FACTOR,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Max Drawdown': mdd
    })

    return pd.DataFrame(results)


if __name__ == "__main__":
    # To use customizable tolerance, set use_custom_tol=True
    df_res = run_moehle_paper(use_custom_tol=False, custom_tol=1e-7)
    print("\n--- MOEHLE RESULTS SUMMARY ---")
    print(df_res.to_string(index=False))