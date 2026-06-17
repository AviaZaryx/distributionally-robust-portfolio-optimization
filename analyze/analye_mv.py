import numpy as np
import pandas as pd
import cvxpy as cp
import warnings
import time
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ---------------- CONFIG ----------------
SCALE_FACTOR = 1
CSV_FILE = r"D:\Downloads\pythonProject1\merged_stock_data_half.csv"
START_DATE = '2017-01-01'
END_DATE = '2021-12-31'
EST_WIN = 250
PRED_WIN = 21

# --- CONSOLIDATED TIMERS ---
timers = {
    "Data Processing": 0,
    "Model Setup & Compilation": 0,
    "Solver (Math)": 0,
    "Backtest & Post-processing": 0
}


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
    timers["Data Processing"] += (time.time() - t_start)
    return res


def optimize_mv(returns, gamma, use_custom_tol=False, custom_tol=1e-7):
    # --- SETUP & COMPILATION TIMER START ---
    t_comp_start = time.time()
    n = returns.shape[1]
    mu = returns.mean().values
    Sigma = returns.cov().values

    # Numerical stability
    min_eig = np.min(np.linalg.eigvalsh(Sigma))
    if min_eig < 1e-8:
        Sigma += (1e-8 - min_eig) * np.eye(n)

    w = cp.Variable(n)
    risk = cp.quad_form(w, Sigma)
    obj = cp.Maximize(mu @ w - gamma * risk)
    constraints = [cp.sum(w) == 1, w >= 0]
    prob = cp.Problem(obj, constraints)

    solver_opts = {'solver': cp.OSQP}
    if use_custom_tol:
        solver_opts.update({
            'eps_abs': custom_tol,
            'eps_rel': custom_tol,
            'eps_prim_inf': custom_tol,
            'eps_dual_inf': custom_tol
        })

    # --- SOLVE CALL ---
    try:
        t_before_solve_call = time.time()
        prob.solve(**solver_opts)
        t_after_solve_call = time.time()

        # Pure solver time (from OSQP stats)
        pure_math_time = prob.solver_stats.solve_time if prob.solver_stats else 0
        total_solve_overhead = (t_after_solve_call - t_before_solve_call) - pure_math_time

        # Add pure math to solver
        timers["Solver (Math)"] += pure_math_time

        # Add setup + CVXPY compilation overhead to Compilation
        timers["Model Setup & Compilation"] += (t_before_solve_call - t_comp_start) + total_solve_overhead
        status = prob.status
    except Exception:
        status = "Solver_Error"

    # --- POST-PROCESSING (Grouped with Backtest) ---
    t_post = time.time()
    if w.value is None or status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        res_weights = np.ones(n) / n
    else:
        res = np.maximum(w.value, 0)
        res_weights = res / res.sum()
    timers["Backtest & Post-processing"] += (time.time() - t_post)

    return res_weights, status


def run_mv():
    data = load_data(CSV_FILE)
    if data.empty: return pd.DataFrame()

    delta_range = [0.05]
    FIXED_GAMMA = 5
    results = []

    for d in delta_range:
        rets_mv = []
        for i in range(EST_WIN, len(data) - PRED_WIN, PRED_WIN):
            # Data Processing Timer (Slicing)
            t_slice = time.time()
            est = data.iloc[i - EST_WIN:i].pct_change().dropna()
            pred = data.iloc[i:i + PRED_WIN].pct_change().dropna()
            timers["Data Processing"] += (time.time() - t_slice)

            if est.empty: continue

            # Optimization (Timers handled inside function)
            w, status = optimize_mv(est, FIXED_GAMMA)

            # Backtest Timer (Realization)
            t_real = time.time()
            rets_mv.extend(pred.values @ w)
            timers["Backtest & Post-processing"] += (time.time() - t_real)

        # Backtest Timer (Stats Calculation)
        t_stats = time.time()
        s = pd.Series(rets_mv)
        if not s.empty:
            mn = s.mean() * 252
            sd = s.std() * np.sqrt(252)
            results.append({'delta': d, 'Mean': mn, 'Sharpe': mn / sd if sd > 0 else 0})
        timers["Backtest & Post-processing"] += (time.time() - t_stats)

    return pd.DataFrame(results)


def visualize_timing(timers_dict):
    filtered = {k: v for k, v in timers_dict.items() if v > 0}
    labels = list(filtered.keys())
    values = list(filtered.values())
    total_time = sum(values)

    legend_labels = [f"{l} ({(v / total_time) * 100:.2f}%)" for l, v in zip(labels, values)]

    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))

    wedges, texts, autotexts = ax.pie(
        values,
        labels=None,
        autopct='%1.1f%%',
        startangle=140,
        colors=colors,
        pctdistance=0.85
    )

    ax.legend(wedges, legend_labels,
              title=f"Time Distribution (Total: {total_time:.2f}s)",
              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1))

    plt.tight_layout()
    plt.show()

def comp_analyze_mv():
    run_mv()
    return timers


import csv
import os


def update_analyzed_csv(solver_name, timers_dict, filename=r"D:\Downloads\pythonProject1\analyzed.csv"):
    rows = []
    fieldnames = ["solver"]
    file_exists = os.path.isfile(filename)

    # 1. Read existing data if the file exists
    if file_exists:
        with open(filename, mode='r', newline='') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames if reader.fieldnames else fieldnames
            rows = list(reader)

    # 2. Ensure all keys in timers_dict are in fieldnames (add new columns if needed)
    for key in timers_dict.keys():
        if key not in fieldnames:
            fieldnames.append(key)

    # 3. Check if the solver already has a row
    found = False
    for row in rows:
        if row.get("solver") == solver_name:
            # Update existing row with new timer values
            row.update({k: str(v) for k, v in timers_dict.items()})
            found = True
            break

    # 4. If solver row not found, create a new row
    if not found:
        new_row = {"solver": solver_name}
        new_row.update({k: str(v) for k, v in timers_dict.items()})
        rows.append(new_row)

    # 5. Write everything back to the CSV
    with open(filename, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    # Your existing functions
    run_mv()
    visualize_timing(timers)

    # Logic to save/update analyzed.csv
    update_analyzed_csv("mv", timers)

    print("\n--- CONSOLIDATED TIMERS ---")
    for key, val in timers.items():
        print(f"{key}: {val:.4f}s")
    print(f"\nResults saved to analyzed.csv")

