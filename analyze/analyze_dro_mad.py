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
DELTA_RESCALE_FACTOR = 1

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


def optimize_dromad(returns, epsilon, rho_target, use_custom_tol=False, custom_tol=1e-7):
    # --- SETUP & COMPILATION TIMER START ---
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

    setup_time = time.time() - t_setup_start
    timers["Model Setup & Compilation"] += setup_time

    # --- SOLVE ---
    best_obj, best_x, final_status = np.inf, np.ones(n) / n, "infeasible"
    solver_kwargs = {'solver': cp.ECOS}
    if use_custom_tol:
        solver_kwargs.update({'abstol': custom_tol, 'reltol': custom_tol, 'feastol': custom_tol})

    for p in [prob1, prob2]:
        try:
            t_before_solve_call = time.time()
            p.solve(**solver_kwargs)
            t_after_solve_call = time.time()

            # Pure solver math time
            pure_math_time = p.solver_stats.solve_time if p.solver_stats else 0

            # Compilation/Overhead for this specific problem solve call
            solve_overhead = (t_after_solve_call - t_before_solve_call) - pure_math_time

            timers["Solver (Math)"] += pure_math_time
            timers["Model Setup & Compilation"] += solve_overhead

            # Weight logic (Post-processing)
            t_post = time.time()
            if p.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE] and p.value < best_obj:
                best_obj, best_x, final_status = p.value, x.value, p.status
            timers["Backtest & Post-processing"] += (time.time() - t_post)
        except:
            pass

    t_clean = time.time()
    res = np.maximum(best_x, 0)
    final_w = res / (res.sum() + 1e-8)
    timers["Backtest & Post-processing"] += (time.time() - t_clean)

    return final_w, final_status


def run_backtest():
    data = load_data(CSV_FILE)
    if data.empty: return

    delta_range = [0.05]
    for d in delta_range:
        rets = []
        for i in range(EST_WIN, len(data) - PRED_WIN, PRED_WIN):
            # Data Processing (Slicing)
            t_s = time.time()
            est_data = data.iloc[i - EST_WIN:i].pct_change().dropna()
            pred_data = data.iloc[i:i + PRED_WIN].pct_change().dropna()
            timers["Data Processing"] += (time.time() - t_s)

            if est_data.empty: continue

            # Optimization (Setup, Compilation, Math, and Post-processing handled inside)
            w, _ = optimize_dromad(est_data, d * DELTA_RESCALE_FACTOR, 0.0012)

            # Backtest (Realization)
            t_r = time.time()
            rets.extend(pred_data.values @ w)
            timers["Backtest & Post-processing"] += (time.time() - t_r)

        # Backtest (Stats Calc)
        t_m = time.time()
        _ = pd.Series(rets).mean()
        timers["Backtest & Post-processing"] += (time.time() - t_m)


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
              title=f"DRO-MAD Execution Time (Total: {total_time:.2f}s)",
              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1))

    plt.tight_layout()
    plt.show()


import csv
import os


def update_analyzed_csv(solver_name, timers_dict, filename=r"D:\Downloads\pythonProject1\analyzed.csv"):
    rows = []
    fieldnames = ["solver"]

    # 1. Read existing data if the file exists and is not empty
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        with open(filename, mode='r', newline='') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            rows = list(reader)

    # 2. Add any new timer keys to the fieldnames list
    for key in timers_dict.keys():
        if key not in fieldnames:
            fieldnames.append(key)

    # 3. Check for the existing solver row
    found = False
    for row in rows:
        if row.get("solver") == solver_name:
            # Update the existing row with new timer values
            row.update({k: str(v) for k, v in timers_dict.items()})
            found = True
            break

    # 4. If not found, append a new row (it will be directly below the previous solver)
    if not found:
        new_row = {"solver": solver_name}
        new_row.update({k: str(v) for k, v in timers_dict.items()})
        rows.append(new_row)

    # 5. Save everything back to the CSV
    with open(filename, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    # Execute DRO-MAD solver logic
    run_backtest()
    visualize_timing(timers)

    # Log results to CSV under the name "dro_mad"
    update_analyzed_csv("dro_mad", timers)

    print("\n--- DRO-MAD CONSOLIDATED SUMMARY ---")
    for key, val in timers.items():
        print(f"{key}: {val:.4f}s")
    print(f"\nResults saved to analyzed.csv")
def comp_analyze_dro_mad():
    run_backtest()
    return timers

