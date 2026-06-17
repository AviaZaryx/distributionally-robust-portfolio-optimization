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


# --- 1. Data Loading ---
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


# --- 2. Model: SOCP with Trading AND Holding Costs ---
def optimize_socp(returns, prev_w, delta, gamma_risk, gamma_trd, gamma_hld, rho,
                  use_custom_tol=False, custom_tol=1e-7):
    # --- SETUP & COMPILATION START ---
    t_comp_start = time.time()

    n = returns.shape[1]
    mu = returns.mean().values
    Sigma = returns.cov().values

    # Numerical stability & Cholesky
    min_eig = np.min(np.linalg.eigvalsh(Sigma))
    if min_eig < 1e-8:
        Sigma += (1e-8 - min_eig) * np.eye(n)

    try:
        L = np.linalg.cholesky(Sigma).T
    except:
        timers["Model Setup & Compilation"] += (time.time() - t_comp_start)
        return np.ones(n) / n, "Cholesky_Fail"

    # Variables
    w = cp.Variable(n)
    t = cp.Variable()
    q = cp.Variable()
    turnover = cp.Variable(n)
    holding = cp.Variable(n)

    robust_penalty = rho * delta
    obj = cp.Minimize(
        -mu @ w + robust_penalty * t + gamma_risk * q + gamma_trd * cp.sum(turnover) + gamma_hld * cp.sum(holding))

    constraints = [
        cp.sum(w) == 1, w >= 0,
        cp.SOC(t, w),
        cp.sum_squares(L @ w) <= q,
        q >= 0,
        turnover >= w - prev_w,
        turnover >= -(w - prev_w),
        holding >= w
    ]

    prob = cp.Problem(obj, constraints)
    solver_opts = {'solver': cp.ECOS, 'verbose': False}
    if use_custom_tol:
        solver_opts.update({'abstol': custom_tol, 'reltol': custom_tol, 'feastol': custom_tol})

    # --- SOLVE CALL ---
    try:
        t_before_solve_call = time.time()
        prob.solve(**solver_opts)
        t_after_solve_call = time.time()

        # Pure solver math time
        pure_math_time = prob.solver_stats.solve_time if prob.solver_stats else 0

        # Calculate overhead (total solve duration minus pure math)
        total_solve_overhead = (t_after_solve_call - t_before_solve_call) - pure_math_time
        setup_time = (t_before_solve_call - t_comp_start)

        timers["Model Setup & Compilation"] += setup_time + total_solve_overhead
        timers["Solver (Math)"] += pure_math_time
        status = prob.status
    except Exception:
        status = "Error"

    # --- POST-PROCESSING (Grouped with Backtest) ---
    t_post = time.time()
    if w.value is None or status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        res_w = np.ones(n) / n
    else:
        res_w = np.maximum(w.value, 0)
        res_w = res_w / res_w.sum()

    timers["Backtest & Post-processing"] += (time.time() - t_post)
    return res_w, status


# --- 3. Backtest Engine ---
def run_socp():
    data = load_data(CSV_FILE)
    if data.empty: return pd.DataFrame()

    delta_range = [0.05]
    rho_param, G_TRD, G_HLD = 1, 0.001, 0.001
    results = []

    for d in delta_range:
        rets = []
        prev_w = np.zeros(data.shape[1])

        for i in range(EST_WIN, len(data) - PRED_WIN, PRED_WIN):
            # Data Processing (Slicing)
            t_s = time.time()
            est_data = data.iloc[i - EST_WIN:i].pct_change().dropna()
            pred_data = data.iloc[i:i + PRED_WIN].pct_change().dropna()
            timers["Data Processing"] += (time.time() - t_s)

            if est_data.empty: continue

            w, status = optimize_socp(est_data, prev_w, d, 5, G_TRD, G_HLD, rho_param)

            # Backtest (Realization & Costs)
            t_r = time.time()
            raw_ret_array = pred_data.values @ w
            trd_cost = G_TRD * np.sum(np.abs(w - prev_w))
            hld_cost = G_HLD * np.sum(np.abs(w))
            raw_ret_array[0] -= (trd_cost + hld_cost)
            rets.extend(raw_ret_array)
            prev_w = w
            timers["Backtest & Post-processing"] += (time.time() - t_r)

        # Backtest (Stats Calculation)
        t_m = time.time()
        s = pd.Series(rets)
        if not s.empty:
            mn = s.mean() * 252
            sd = s.std() * np.sqrt(252)
            results.append({'delta': d, 'Mean': mn, 'Sharpe': mn / sd if sd > 0 else 0})
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
              title=f"SOCP Execution Time (Total: {total_time:.2f}s)",
              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1))

    plt.tight_layout()
    plt.show()


import csv
import os


def update_analyzed_csv(solver_name, timers_dict, filename=r"D:\Downloads\pythonProject1\analyzed.csv"):
    rows = []
    fieldnames = ["solver"]

    # 1. Read existing data
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        with open(filename, mode='r', newline='') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            rows = list(reader)

    # 2. Add any new timer keys as new columns (fieldnames)
    for key in timers_dict.keys():
        if key not in fieldnames:
            fieldnames.append(key)

    # 3. Update existing row or prepare to append a new one
    found = False
    for row in rows:
        if row.get("solver") == solver_name:
            # Update matching row
            row.update({k: str(v) for k, v in timers_dict.items()})
            found = True
            break

    if not found:
        # Create new row at the bottom if solver doesn't exist
        new_row = {"solver": solver_name}
        new_row.update({k: str(v) for k, v in timers_dict.items()})
        rows.append(new_row)

    # 4. Write back to the file
    with open(filename, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    # Execute SOCP solver
    df_res = run_socp()
    visualize_timing(timers)

    # Log results to CSV
    update_analyzed_csv("socp", timers)

    print("\n--- SOCP CONSOLIDATED SUMMARY ---")
    for key, val in timers.items():
        print(f"{key}: {val:.4f}s")
    print(f"\nResults saved to analyzed.csv")

def comp_analyze_socp():
    run_socp()
    return timers