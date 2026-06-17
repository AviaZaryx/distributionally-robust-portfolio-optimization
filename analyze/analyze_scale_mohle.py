import numpy as np
import pandas as pd
import cvxpy as cp
import warnings
import time
import csv
import os

warnings.filterwarnings('ignore')

# ---------------- CONFIG ----------------
CSV_INPUT = r'D:\Downloads\pythonProject1\merged_stock_data.csv'
CSV_OUTPUT = r'D:\Downloads\pythonProject1\time_scale_size.csv'

START_DATE = '2017-01-01'
END_DATE = '2021-12-31'
EST_WIN = 250
PRED_WIN = 21

# IMPORTANT: Moehle is NP-Hard. Capped at 40 to prevent infinite runs.
MAX_ASSETS_FOR_MOEHLE = 200
INCREMENTS = 20

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


# --- 1. DATA LOADING ---
def load_data(csv_path):
    t_start = time.time()
    try:
        df = pd.read_csv(csv_path, parse_dates=['Date'])
        mask = (df['Date'] >= START_DATE) & (df['Date'] <= END_DATE)
        df = df.loc[mask].sort_values('Date')
        df_adj = df.pivot(index='Date', columns='Ticker', values='Adj Close').sort_index()
        df_adj = df_adj.dropna(axis=1, how='any').dropna(how='any')
        res = (df_adj / df_adj.iloc[0]) if not df_adj.empty else pd.DataFrame()
    except Exception as e:
        print(f"Data Load Error: {e}")
        res = pd.DataFrame()

    timers["Data Processing"] += (time.time() - t_start)
    return res


# --- 2. MODEL FORMULATION ---
def optimize_moehle_nonconvex(returns, prev_w, gamma_risk, f_trd, l_trd, f_hld):
    t_setup_start = time.time()
    n = returns.shape[1]
    mu = returns.mean().values
    Sigma = returns.cov().values

    # Regularization
    min_eig = np.min(np.linalg.eigvalsh(Sigma))
    if min_eig < 1e-8: Sigma += (1e-8 - min_eig) * np.eye(n)

    # Variables
    w = cp.Variable(n)
    abs_trade = cp.Variable(n)
    z_trade = cp.Variable(n, boolean=True)
    z_hold = cp.Variable(n, boolean=True)

    risk_term = cp.quad_form(w, Sigma)
    obj = cp.Minimize(-mu @ w + gamma_risk * risk_term +
                      f_trd * cp.sum(z_trade) + l_trd * cp.sum(abs_trade) + f_hld * cp.sum(z_hold))

    M = 1.0
    constraints = [
        cp.sum(w) == 1, w >= 0,
        abs_trade >= w - prev_w, abs_trade >= -(w - prev_w),
        abs_trade <= M * z_trade, w <= M * z_hold
    ]
    prob = cp.Problem(obj, constraints)
    timers["Model Setup & Compilation"] += (time.time() - t_setup_start)

    try:
        t_before_solve = time.time()
        prob.solve(solver=cp.ECOS_BB)
        t_after_solve = time.time()

        solve_time = prob.solver_stats.solve_time if prob.solver_stats else (t_after_solve - t_before_solve)
        overhead = (t_after_solve - t_before_solve) - solve_time

        timers["Solver (Math)"] += solve_time
        timers["Model Setup & Compilation"] += overhead
        status = prob.status
    except:
        status = "Error"

    t_post_start = time.time()
    res_w = w.value if (w.value is not None and status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]) else np.ones(n) / n
    timers["Backtest & Post-processing"] += (time.time() - t_post_start)
    return res_w


# --- 3. CSV UPDATE ---
def update_scale_csv(solver_name, scaling_values, size_labels):
    rows = []
    if os.path.exists(CSV_OUTPUT) and os.path.getsize(CSV_OUTPUT) > 0:
        with open(CSV_OUTPUT, mode='r', newline='') as f:
            rows = list(csv.DictReader(f))

    new_data = {"solver": solver_name}
    for i, val in enumerate(scaling_values):
        new_data[f"N={size_labels[i]}"] = val

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

    sorted_fields = sorted(list(all_keys), key=sort_key)
    with open(CSV_OUTPUT, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=sorted_fields)
        writer.writeheader()
        writer.writerows(rows)


# --- 4. EXECUTION LOOP ---
def run_scaling():
    data = load_data(CSV_INPUT)
    if data.empty: return

    available_tickers = data.columns.tolist()
    step = MAX_ASSETS_FOR_MOEHLE // INCREMENTS
    test_sizes = [i * step for i in range(1, INCREMENTS + 1)]

    scaling_results = []
    actual_labels = []

    for n in test_sizes:
        if n > len(available_tickers): break

        print(f"Executing Moehle Scale: N={n}...")
        reset_timers()

        t_slice_start = time.time()
        subset = data.iloc[:, :n]
        timers["Data Processing"] += (time.time() - t_slice_start)

        prev_w = np.zeros(n)

        # Backtest Loop
        for i in range(EST_WIN, len(subset) - PRED_WIN, PRED_WIN):
            t_est_start = time.time()
            est = subset.iloc[i - EST_WIN:i].pct_change().dropna()
            timers["Data Processing"] += (time.time() - t_est_start)

            if est.empty: continue
            w = optimize_moehle_nonconvex(est, prev_w, 5, 0.0001, 0.0010, 0.0001)
            prev_w = w

        # Construct the 4-tuple: (Data, Setup, Math, Post)
        res_tuple = (
            round(timers["Data Processing"], 6),
            round(timers["Model Setup & Compilation"], 6),
            round(timers["Solver (Math)"], 6),
            round(timers["Backtest & Post-processing"], 6)
        )

        scaling_results.append(str(res_tuple))
        actual_labels.append(n)
        print(f"  Done. Total: {sum(res_tuple):.2f}s")

    update_scale_csv("Moehle", scaling_results, actual_labels)
    print(f"\nScaling results for 'Moehle' saved to {CSV_OUTPUT}")


if __name__ == "__main__":
    run_scaling()