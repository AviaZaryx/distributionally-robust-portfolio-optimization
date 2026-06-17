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
DELTA_RESCALE_FACTOR = 1
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


def optimize_dromad(returns, epsilon, rho_target):
    t_setup_start = time.time()
    n, T = returns.shape[1], returns.shape[0]
    mu, xi = returns.mean().values, returns.values
    x, y1, y2, t = cp.Variable(n), cp.Variable(T), cp.Variable(T), cp.Variable()

    base_con = [cp.sum(x) == 1, x >= 0]
    c1 = base_con + [
        y1 >= mu @ x - xi @ x - epsilon, y1 >= -mu @ x + xi @ x + epsilon,
        y2 >= mu @ x - xi @ x + epsilon, y2 >= -mu @ x + xi @ x - epsilon,
        mu @ x - epsilon >= rho_target, t >= cp.sum(y1) / T + epsilon, t >= cp.sum(y2) / T + epsilon
    ]
    c2 = base_con + [
        y1 >= xi @ x - rho_target, y1 >= -xi @ x + rho_target,
        y2 >= mu @ x - xi @ x + epsilon, y2 >= -mu @ x + xi @ x - epsilon,
        mu @ x - epsilon <= rho_target, mu @ x + epsilon >= rho_target,
        t >= cp.sum(y1) / T + epsilon, t >= cp.sum(y2) / T + epsilon
    ]
    prob1, prob2 = cp.Problem(cp.Minimize(t), c1), cp.Problem(cp.Minimize(t), c2)
    timers["Model Setup & Compilation"] += (time.time() - t_setup_start)

    best_x = np.ones(n) / n
    for p in [prob1, prob2]:
        try:
            t_before = time.time()
            p.solve(solver=cp.ECOS)
            t_after = time.time()

            math_time = p.solver_stats.solve_time if p.solver_stats else 0
            overhead = (t_after - t_before) - math_time

            timers["Solver (Math)"] += math_time
            timers["Model Setup & Compilation"] += overhead

            if p.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                best_x = x.value
        except:
            pass

    t_post = time.time()
    res = np.maximum(best_x, 0)
    final_w = res / (res.sum() + 1e-8)
    timers["Backtest & Post-processing"] += (time.time() - t_post)
    return final_w


def update_scale_csv(solver_name, scaling_values, size_labels=None, filename=CSV_OUTPUT):
    rows = []
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        with open(filename, mode='r', newline='') as f:
            rows = list(csv.DictReader(f))

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


def run_scaling_analysis():
    data = load_data(CSV_FILE)
    if data.empty: return

    all_tickers = data.columns.tolist()
    total_stocks = len(all_tickers)
    step = total_stocks // N_PARTS

    scaling_results = []
    actual_labels = []
    solver_key = "DRO-MAD"  # Standardized name

    print(f"Total stocks found: {total_stocks}. Running {N_PARTS} scale increments.")

    for p in range(1, N_PARTS + 1):
        num_stocks = p * step if p < N_PARTS else total_stocks
        subset_tickers = all_tickers[:num_stocks]

        # Reset all 4 timers for this increment
        reset_timers()

        t_slice_start = time.time()
        subset_data = data[subset_tickers]
        timers["Data Processing"] += (time.time() - t_slice_start)

        print(f"Executing Run {p}/{N_PARTS} with {num_stocks} assets...")

        # Backtest Loop
        for i in range(EST_WIN, len(subset_data) - PRED_WIN, PRED_WIN):
            t_est_start = time.time()
            est_data = subset_data.iloc[i - EST_WIN:i].pct_change().dropna()
            timers["Data Processing"] += (time.time() - t_est_start)

            if est_data.empty: continue
            _ = optimize_dromad(est_data, 0.05 * DELTA_RESCALE_FACTOR, 0.0012)

        # Construct the 4-component tuple
        res_tuple = (
            round(timers["Data Processing"], 6),
            round(timers["Model Setup & Compilation"], 6),
            round(timers["Solver (Math)"], 6),
            round(timers["Backtest & Post-processing"], 6)
        )

        scaling_results.append(str(res_tuple))
        actual_labels.append(num_stocks)
        print(f"  N={num_stocks} -> Total: {sum(res_tuple):.2f}s")

    update_scale_csv(solver_key, scaling_results, size_labels=actual_labels)
    print(f"\nScaling analysis for {solver_key} saved to {CSV_OUTPUT}")


if __name__ == "__main__":
    run_scaling_analysis()