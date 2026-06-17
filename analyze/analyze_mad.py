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


def optimize_mad(returns, gamma, use_custom_tol=False, custom_tol=1e-7):
    # --- SETUP & COMPILATION TIMER START ---
    t_comp_start = time.time()

    n = returns.shape[1]
    T = returns.shape[0]
    mu = returns.mean().values
    w = cp.Variable(n)
    z = cp.Variable(T)

    mean_port = mu @ w
    constraints = [
        cp.sum(w) == 1, w >= 0,
        z >= (returns.values @ w) - mean_port,
        z >= -((returns.values @ w) - mean_port)
    ]

    mad_risk = cp.sum(z) / T
    obj = cp.Maximize(mu @ w - gamma * mad_risk)
    prob = cp.Problem(obj, constraints)

    solver_opts = {'solver': cp.ECOS, 'verbose': False}
    if use_custom_tol:
        solver_opts.update({'abstol': custom_tol, 'reltol': custom_tol, 'feastol': custom_tol})

    # --- SOLVE CALL ---
    try:
        t_before_solve_call = time.time()
        prob.solve(**solver_opts)
        t_after_solve_call = time.time()

        # Pure solver time (from ECOS stats)
        pure_math_time = prob.solver_stats.solve_time if prob.solver_stats else 0

        # Compilation includes the Variable/Constraint setup AND the CVXPY overhead during the solve call
        total_solve_overhead = (t_after_solve_call - t_before_solve_call) - pure_math_time
        setup_time = (t_before_solve_call - t_comp_start)

        timers["Model Setup & Compilation"] += setup_time + total_solve_overhead
        timers["Solver (Math)"] += pure_math_time
        status = prob.status
    except Exception as e:
        status = f"Error: {str(e)}"

    # --- POST-PROCESSING (Grouped with Backtest) ---
    t_post = time.time()
    if w.value is None or status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        res_weights = np.ones(n) / n
    else:
        res = np.maximum(w.value, 0)
        res_weights = res / res.sum()

    timers["Backtest & Post-processing"] += (time.time() - t_post)
    return res_weights, status


def run_mad(csv=CSV_FILE, use_custom_tol=False, custom_tol=1e-7):
    data = load_data(csv)
    if data.empty: return pd.DataFrame()

    delta_range = [0.05]
    GAMMA_RISK = 0.5
    results = []

    for d in delta_range:
        rets_mad = []
        for i in range(EST_WIN, len(data) - PRED_WIN, PRED_WIN):
            # Data Processing (Slicing)
            t_s = time.time()
            est = data.iloc[i - EST_WIN:i].pct_change().dropna()
            pred = data.iloc[i:i + PRED_WIN].pct_change().dropna()
            timers["Data Processing"] += (time.time() - t_s)

            if est.empty: continue

            # Optimization (Setup, Compilation, and Math handled inside)
            w, status = optimize_mad(est, GAMMA_RISK, use_custom_tol, custom_tol)

            # Backtest (Realization)
            t_r = time.time()
            rets_mad.extend(pred.values @ w)
            timers["Backtest & Post-processing"] += (time.time() - t_r)

        if not rets_mad: continue

        # Backtest (Stats Calculation)
        t_m = time.time()
        s = pd.Series(rets_mad)
        mn = s.mean() * 252
        sd = s.std() * np.sqrt(252)
        results.append({'Delta': d, 'Mean': mn, 'Sharpe': mn / sd if sd > 0 else 0})
        timers["Backtest & Post-processing"] += (time.time() - t_m)

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
              title=f"MAD Model Execution Time (Total: {total_time:.2f}s)",
              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1))

    plt.tight_layout()
    plt.show()


import csv
import os


def update_analyzed_csv(solver_name, timers_dict, filename=r"D:\Downloads\pythonProject1\analyzed.csv"):
    rows = []
    fieldnames = ["solver"]

    # 1. Read existing data if the file exists
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        with open(filename, mode='r', newline='') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            rows = list(reader)

    # 2. Add any new keys from timers to the fieldnames (columns)
    for key in timers_dict.keys():
        if key not in fieldnames:
            fieldnames.append(key)

    # 3. Search for the solver row and update it, or prepare to add a new one
    found = False
    for row in rows:
        if row.get("solver") == solver_name:
            # Update the existing row with new timer values
            row.update({k: str(v) for k, v in timers_dict.items()})
            found = True
            break

    if not found:
        # Create a new row if solver doesn't exist
        new_row = {"solver": solver_name}
        new_row.update({k: str(v) for k, v in timers_dict.items()})
        rows.append(new_row)

    # 4. Write the updated data back to the CSV
    with open(filename, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    # Run the solver
    df_res = run_mad()
    visualize_timing(timers)

    # Save/Update results in analyzed.csv
    update_analyzed_csv("mad", timers)

    print("\n--- MAD CONSOLIDATED SUMMARY ---")
    for key, val in timers.items():
        print(f"{key}: {val:.4f}s")
    print(f"\nResults saved to analyzed.csv")

def comp_analyze_mad():
    run_mad()
    return timers
