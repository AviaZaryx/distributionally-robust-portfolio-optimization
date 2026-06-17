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


# --- 2. Model: Non-Convex Mean-Variance (Moehle et al.) ---
def optimize_moehle_nonconvex(returns, prev_w, gamma_risk, fixed_trd_cost, lin_trd_cost, fixed_hld_cost,
                              use_custom_tol=False, custom_tol=1e-7):
    # --- SETUP & COMPILATION TIMER START ---
    t_comp_start = time.time()

    n = returns.shape[1]
    mu = returns.mean().values
    Sigma = returns.cov().values

    # Regularization
    min_eig = np.min(np.linalg.eigvalsh(Sigma))
    if min_eig < 1e-8:
        Sigma += (1e-8 - min_eig) * np.eye(n)

    # Variables (Includes Boolean for non-convex costs)
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

    solver_kwargs = {"verbose": False}
    if use_custom_tol:
        solver_kwargs.update({'abstol': custom_tol, 'reltol': custom_tol, 'feastol': custom_tol})

    # --- SOLVE CALL ---
    status = "not_solved"
    try:
        t_before_solve_call = time.time()
        # ECOS_BB is required for the Boolean variables
        prob.solve(solver=cp.ECOS_BB, **solver_kwargs)
        t_after_solve_call = time.time()

        # Pure solver time (Branch and Bound math)
        pure_math_time = prob.solver_stats.solve_time if prob.solver_stats else 0

        # Compilation includes the Variable/Constraint setup AND the CVXPY overhead during the solve call
        total_solve_overhead = (t_after_solve_call - t_before_solve_call) - pure_math_time
        setup_time = (t_before_solve_call - t_comp_start)

        timers["Model Setup & Compilation"] += setup_time + total_solve_overhead
        timers["Solver (Math)"] += pure_math_time
        status = prob.status
    except Exception:
        status = "Solver_Error"

    # --- POST-PROCESSING (Grouped with Backtest) ---
    t_post = time.time()
    if w.value is None or status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        res_w = np.ones(n) / n
    else:
        res = np.maximum(w.value, 0)
        res_w = res / res.sum()

    timers["Backtest & Post-processing"] += (time.time() - t_post)
    return res_w, status


# --- 3. Backtest Engine ---
def run_moehle_paper(csv=CSV_FILE, use_custom_tol=False, custom_tol=1e-7):
    data = load_data(csv)
    if data.empty: return pd.DataFrame()

    delta_range = [0.05]
    gamma_range = [5]
    FIXED_TRD_COST, FIXED_HLD_COST, LIN_TRD_COST = 0.0001, 0.0001, 0.0010
    results = []

    for d in delta_range:
        for g in gamma_range:
            rets_gamma = []
            prev_w = np.zeros(data.shape[1])

            for i in range(EST_WIN, len(data) - PRED_WIN, PRED_WIN):
                # Data Processing (Slicing)
                t_s = time.time()
                est = data.iloc[i - EST_WIN:i].pct_change().dropna()
                pred = data.iloc[i:i + PRED_WIN].pct_change().dropna()
                timers["Data Processing"] += (time.time() - t_s)

                if est.empty: continue

                w, status = optimize_moehle_nonconvex(est, prev_w, g, FIXED_TRD_COST, LIN_TRD_COST, FIXED_HLD_COST,
                                                      use_custom_tol, custom_tol)

                # Backtest (Realization & Costs)
                t_r = time.time()
                raw_ret = pred.values @ w
                trade_mag = np.abs(w - prev_w)
                epsilon = 1e-5
                cost_lin = LIN_TRD_COST * np.sum(trade_mag)
                cost_fix_trd = FIXED_TRD_COST * np.sum((trade_mag > epsilon).astype(float))
                cost_fix_hld = FIXED_HLD_COST * np.sum((w > epsilon).astype(float))

                raw_ret[0] -= (cost_lin + cost_fix_trd + cost_fix_hld)
                rets_gamma.extend(raw_ret)
                prev_w = w
                timers["Backtest & Post-processing"] += (time.time() - t_r)

            # Backtest (Stats Calculation)
            t_m = time.time()
            s = pd.Series(rets_gamma)
            mean = s.mean() * 252
            std = s.std() * np.sqrt(252)
            results.append({'delta': d, 'Mean': mean, 'Sharpe': mean / std if std > 0 else 0})
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
              title=f"Moehle Model Time Distribution (Total: {total_time:.2f}s)",
              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1))

    plt.tight_layout()
    plt.show()


import csv
import os


def update_analyzed_csv(solver_name, timers_dict, filename=r"D:\Downloads\pythonProject1\analyzed.csv"):
    rows = []
    fieldnames = ["solver"]

    # 1. Read existing data if the file exists and isn't empty
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        with open(filename, mode='r', newline='') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            rows = list(reader)

    # 2. Add any new keys from timers to the list of columns
    for key in timers_dict.keys():
        if key not in fieldnames:
            fieldnames.append(key)

    # 3. Check if 'moehle' already exists; if so, update. If not, append.
    found = False
    for row in rows:
        if row.get("solver") == solver_name:
            # Update the existing row with new timer values
            row.update({k: str(v) for k, v in timers_dict.items()})
            found = True
            break

    if not found:
        # If no row for moehle, create a new one at the end
        new_row = {"solver": solver_name}
        new_row.update({k: str(v) for k, v in timers_dict.items()})
        rows.append(new_row)

    # 4. Write the updated list back to the CSV
    with open(filename, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    df_res = run_moehle_paper()
    visualize_timing(timers)

    # Save/Update results in analyzed.csv under the name "moehle"
    update_analyzed_csv("moehle", timers)

    print("\n--- MOEHLE CONSOLIDATED SUMMARY ---")
    for key, val in timers.items():
        print(f"{key}: {val:.4f}s")
    print(f"\nResults saved to analyzed.csv")

def comp_analyze_mohle():
    run_moehle_paper()
    return timers