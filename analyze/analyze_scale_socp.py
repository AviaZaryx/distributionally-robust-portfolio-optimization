import numpy as np
import pandas as pd
import cvxpy as cp
import warnings
import time
import csv
import os

warnings.filterwarnings('ignore')

# ---------------- CONFIG ----------------
CSV_FILE = r'D:\Downloads\pythonProject1\merged_stock_data.csv'
CSV_OUTPUT = r'D:\Downloads\pythonProject1\time_scale_size.csv'
START_DATE = '2017-01-01'
END_DATE = '2021-12-31'
EST_WIN = 250
PRED_WIN = 21
N_PARTS = 20

# --- CONSOLIDATED TIMERS (All 4 components) ---
timers = {
    "Data Processing": 0,
    "Model Setup & Compilation": 0,
    "Solver (Math)": 0,
    "Backtest & Post-processing": 0
}


def reset_timers():
    for key in timers:
        timers[key] = 0


def load_data(csv_path):
    t_start = time.time()
    try:
        df = pd.read_csv(csv_path, parse_dates=['Date'])
        mask = (df['Date'] >= START_DATE) & (df['Date'] <= END_DATE)
        df = df.loc[mask].sort_values('Date')
        df_adj = df.pivot(index='Date', columns='Ticker', values='Adj Close').sort_index()
        df_adj = df_adj.dropna(axis=1, how='any').dropna(how='any')
        res = (df_adj / df_adj.iloc[0]) if not df_adj.empty else pd.DataFrame()
    except:
        res = pd.DataFrame()
    # Initial load counted as data processing
    timers["Data Processing"] += (time.time() - t_start)
    return res


def optimize_socp(returns, prev_w, delta, gamma_risk, gamma_trd, gamma_hld, rho):
    t_comp_start = time.time()
    n = returns.shape[1]
    mu = returns.mean().values
    Sigma = returns.cov().values

    # Numerical Stability
    min_eig = np.min(np.linalg.eigvalsh(Sigma))
    if min_eig < 1e-8: Sigma += (1e-8 - min_eig) * np.eye(n)

    try:
        L = np.linalg.cholesky(Sigma).T
    except:
        return np.ones(n) / n

    # CVXPY Variables
    w = cp.Variable(n)
    t, q = cp.Variable(), cp.Variable()
    turnover, holding = cp.Variable(n), cp.Variable(n)

    robust_penalty = rho * delta
    obj = cp.Minimize(-mu @ w + robust_penalty * t + gamma_risk * q +
                      gamma_trd * cp.sum(turnover) + gamma_hld * cp.sum(holding))

    constraints = [
        cp.sum(w) == 1, w >= 0, cp.SOC(t, w),
        cp.sum_squares(L @ w) <= q, q >= 0,
        turnover >= w - prev_w, turnover >= -(w - prev_w),
        holding >= w
    ]

    prob = cp.Problem(obj, constraints)
    # Record Setup Time before solve call
    timers["Model Setup & Compilation"] += (time.time() - t_comp_start)

    try:
        t_before_solve = time.time()
        # Using ECOS for SOCP problems
        prob.solve(solver=cp.ECOS)
        t_after_solve = time.time()

        math_time = prob.solver_stats.solve_time if prob.solver_stats else 0
        total_overhead = (t_after_solve - t_before_solve) - math_time

        timers["Solver (Math)"] += math_time
        timers["Model Setup & Compilation"] += total_overhead
        status = prob.status
    except:
        status = "Error"

    t_post = time.time()
    res_w = w.value if (w.value is not None and status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]) else np.ones(n) / n
    timers["Backtest & Post-processing"] += (time.time() - t_post)
    return res_w


def update_scale_csv(solver_name, scaling_values, size_labels=None, filename=CSV_OUTPUT):
    rows = []
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        with open(filename, mode='r', newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

    new_data = {"solver": solver_name}
    for i, val in enumerate(scaling_values):
        label = size_labels[i] if (size_labels and i < len(size_labels)) else i + 1
        new_data[f"N={label}"] = val

    found = False
    for row in rows:
        if row.get("solver") == solver_name:
            row.update(new_data)
            found = True
            break
    if not found:
        rows.append(new_data)

    all_keys = set().union(*(r.keys() for r in rows))

    def sort_key(k):
        if k == 'solver': return (0, 0)
        if 'N=' in k: return (1, int(k.split('=')[1]))
        return (2, k)

    sorted_fieldnames = sorted(list(all_keys), key=sort_key)
    with open(filename, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=sorted_fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_socp_scaling(data, solver_name="SOCP"):
    all_tickers = data.columns.tolist()
    total_stocks = len(all_tickers)
    step = total_stocks // N_PARTS

    scaling_results = []
    actual_labels = []

    print(f"Starting scaling analysis for {solver_name}...")

    for p in range(1, N_PARTS + 1):
        num_stocks = p * step if p < N_PARTS else total_stocks

        # Reset all 4 timers for this specific increment
        reset_timers()

        t_slice_start = time.time()
        subset_data = data.iloc[:, :num_stocks]
        timers["Data Processing"] += (time.time() - t_slice_start)

        print(f"Executing Run {p}/{N_PARTS} with {num_stocks} assets...")

        prev_w = np.zeros(num_stocks)
        # Core backtest loop
        for i in range(EST_WIN, len(subset_data) - PRED_WIN, PRED_WIN):
            t_est_start = time.time()
            est = subset_data.iloc[i - EST_WIN:i].pct_change().dropna()
            timers["Data Processing"] += (time.time() - t_est_start)

            if est.empty: continue

            # Using standard robust/cost parameters for test
            w = optimize_socp(est, prev_w, 0.05, 5, 0.001, 0.001, 1)
            prev_w = w

        # Construct the 4-component tuple: (Data, Setup, Math, Post)
        res_tuple = (
            round(timers["Data Processing"], 6),
            round(timers["Model Setup & Compilation"], 6),
            round(timers["Solver (Math)"], 6),
            round(timers["Backtest & Post-processing"], 6)
        )

        scaling_results.append(str(res_tuple))
        actual_labels.append(num_stocks)
        print(f"  Done. Total Time: {sum(res_tuple):.2f}s")

    update_scale_csv(solver_name, scaling_results, size_labels=actual_labels)
    print(f"\nScaling results for {solver_name} saved to {CSV_OUTPUT}")


if __name__ == "__main__":
    # 1. Load data
    data = load_data(CSV_FILE)

    if not data.empty:
        # 2. Run Scaling Analysis
        run_socp_scaling(data)
    else:
        print("Data could not be loaded. Check CSV_FILE path.")