import numpy as np
import pandas as pd
import cvxpy as cp
import warnings
import time

warnings.filterwarnings('ignore')

# ---------------- CONFIG ----------------
SCALE_FACTOR = 1
CSV_FILE = 'combined_10_stocks.csv'
START_DATE = '2017-01-01'
END_DATE = '2021-12-31'
EST_WIN = 250
PRED_WIN = 21


# --- 1. Data Loading ---
def load_data(csv_path):
    try:
        df = pd.read_csv(csv_path, parse_dates=['Date'])
    except FileNotFoundError:
        return pd.DataFrame()

    mask = (df['Date'] >= START_DATE) & (df['Date'] <= END_DATE)
    df = df.loc[mask].sort_values('Date')

    # Pivot data
    if 'Ticker' not in df.columns or 'Adj Close' not in df.columns:
        return pd.DataFrame()

    df_adj = df.pivot(index='Date', columns='Ticker', values='Adj Close').sort_index()
    df_adj = df_adj.dropna(axis=1, how='any')
    df_adj = df_adj.dropna(how='any')

    if df_adj.empty:
        return pd.DataFrame()

    return df_adj / df_adj.iloc[0]


# --- 2. Model: SOCP with Trading AND Holding Costs ---
def optimize_socp(returns, prev_w, delta, gamma_risk, gamma_trd, gamma_hld, rho,
                  use_custom_tol=False, custom_tol=1e-7):
    n = returns.shape[1]
    mu = returns.mean().values
    Sigma = returns.cov().values

    # Regularization for numerical stability
    min_eig = np.min(np.linalg.eigvalsh(Sigma))
    if min_eig < 1e-8:
        Sigma += (1e-8 - min_eig) * np.eye(n)

    try:
        L = np.linalg.cholesky(Sigma).T
    except:
        return np.ones(n) / n, "Cholesky_Fail"

    # Variables
    w = cp.Variable(n)
    t = cp.Variable()
    q = cp.Variable()
    turnover = cp.Variable(n)
    holding = cp.Variable(n)

    robust_penalty = rho * delta

    obj = cp.Minimize(
        -mu @ w
        + robust_penalty * t
        + gamma_risk * q
        + gamma_trd * cp.sum(turnover)
        + gamma_hld * cp.sum(holding)
    )

    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        cp.SOC(t, w),
        cp.sum_squares(L @ w) <= q,
        q >= 0,
        turnover >= w - prev_w,
        turnover >= -(w - prev_w),
        holding >= w
    ]

    prob = cp.Problem(obj, constraints)

    # Solver options based on toggle
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


# --- 3. Backtest Engine ---
def run_socp(csv=CSV_FILE, use_custom_tol=False, custom_tol=1e-7):
    data = load_data(csv)
    if data.empty: return pd.DataFrame()

    delta_range = [0, 0.005, 0.01]
    gamma_range = [5]
    rho_param = 1

    G_TRD = 0.001
    G_HLD = 0.001

    results = []
    statuses = []

    for g in gamma_range:
        for d in delta_range:
            # START TIMER FOR THIS SPECIFIC DELTA
            delta_start_time = time.time()

            rets = []
            prev_w = np.zeros(data.shape[1])
            stats_summary = []

            for i in range(EST_WIN, len(data) - PRED_WIN, PRED_WIN):
                est_data = data.iloc[i - EST_WIN:i].pct_change().dropna()
                pred_data = data.iloc[i:i + PRED_WIN].pct_change().dropna()

                if est_data.empty or pred_data.empty: continue

                # 1. Run Optimization (Pass toggle and tolerance through)
                w, status = optimize_socp(est_data, prev_w, d, g, G_TRD, G_HLD, rho_param,
                                          use_custom_tol, custom_tol)
                stats_summary.append(status)

                # 2. Calculate raw returns
                raw_ret_array = pred_data.values @ w

                # 3. Calculate friction costs
                trd_cost = G_TRD * np.sum(np.abs(w - prev_w))
                hld_cost = G_HLD * np.sum(np.abs(w))

                # 4. Subtract costs
                raw_ret_array[0] -= (trd_cost + hld_cost)

                rets.extend(raw_ret_array)
                prev_w = w

            # END TIMER FOR THIS SPECIFIC DELTA
            delta_end_time = time.time()
            runtime_for_delta = delta_end_time - delta_start_time

            unique_statuses = pd.Series(stats_summary).value_counts()
            statuses.append(unique_statuses)

            s = pd.Series(rets)
            mean = s.mean() * 252
            std = s.std() * np.sqrt(252)
            sharpe = mean / std if std > 0 else 0
            downside = s[s < 0].std() * np.sqrt(252)
            sortino = mean / downside if downside > 0 else 0
            mdd = ((1 + s).cumprod() / (1 + s).cumprod().cummax() - 1).min()

            results.append({
                'gamma_risk': g,
                'delta': d,
                'Runtime (s)': runtime_for_delta,
                'mean_return': mean / SCALE_FACTOR,
                'std_dev': std / SCALE_FACTOR,
                'sharpe_ratio': sharpe,
                'sortino_ratio': sortino,
                'max_drawdown': mdd
            })

    if __name__ == "__main__":
        print("\n--- Solver Statuses per Delta ---")
        print(statuses)

    return pd.DataFrame(results)


if __name__ == "__main__":
    # To enable custom tolerance, set use_custom_tol=True
    df_res = run_socp(use_custom_tol=False, custom_tol=1e-7)

    print("\n--- SOCP RESULTS SUMMARY ---")
    # Show Delta and Runtime specifically in the output
    cols_to_print = ['delta', 'Runtime (s)', 'sharpe_ratio', 'mean_return']
    print(df_res[cols_to_print].to_string(index=False))