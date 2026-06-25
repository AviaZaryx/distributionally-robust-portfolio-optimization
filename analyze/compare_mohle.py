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
CSV_RUNS_OUTPUT = r'D:\Downloads\pythonProject1\mohle_sampled_run.csv'

START_DATE = '2017-01-01'
END_DATE = '2021-12-31'
EST_WIN = 250
PRED_WIN = 21

POOL_SIZE = 200
NUM_TRIALS = 10
INCREMENTS = 20

timers = {
    "Data Processing": 0,
    "Model Setup & Compilation": 0,
    "Solver (Math)": 0,
    "Backtest & Post-processing": 0
}


def reset_timers():
    for key in timers:
        timers[key] = 0.0


# --- 1. DATA LOADING ---
def load_data(csv_path):
    t_start = time.perf_counter()
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

    timers["Data Processing"] += (time.perf_counter() - t_start)
    return res


# --- 2. MODEL FORMULATION ---
def optimize_moehle_nonconvex(returns, prev_w, gamma_risk, f_trd, l_trd, f_hld):
    t_setup_start = time.perf_counter()
    n = returns.shape[1]
    mu = returns.mean().values
    Sigma = returns.cov().values

    min_eig = np.min(np.linalg.eigvalsh(Sigma))
    if min_eig < 1e-8: Sigma += (1e-8 - min_eig) * np.eye(n)

    w = cp.Variable(n)
    abs_trade = cp.Variable(n)
    z_trade = cp.Variable(n, boolean=True)
    z_hold = cp.Variable(n, boolean=True)

    risk_term = cp.quad_form(w, Sigma)
    obj = cp.Minimize(-mu @ w + gamma_risk * risk_term +
                      f_trd * cp.sum(z_trade) + l_trd * cp.sum(abs_trade) + f_hld * cp.sum(z_hold))

    constraints = [
        cp.sum(w) == 1, w >= 0,
        abs_trade >= w - prev_w, abs_trade >= -(w - prev_w),
        abs_trade <= 1.0 * z_trade, w <= 1.0 * z_hold
    ]
    prob = cp.Problem(obj, constraints)
    timers["Model Setup & Compilation"] += (time.perf_counter() - t_setup_start)

    try:
        t_before_solve = time.perf_counter()
        prob.solve(solver=cp.ECOS_BB)
        t_after_solve = time.perf_counter()
        solve_time = prob.solver_stats.solve_time if prob.solver_stats else (t_after_solve - t_before_solve)
        overhead = (t_after_solve - t_before_solve) - solve_time
        timers["Solver (Math)"] += solve_time
        timers["Model Setup & Compilation"] += overhead
        status = prob.status
    except:
        status = "Error"

    t_post_start = time.perf_counter()
    res_w = w.value if (w.value is not None and status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]) else np.ones(n) / n
    timers["Backtest & Post-processing"] += (time.perf_counter() - t_post_start)
    return res_w


# --- 3. CSV UPDATES ---

def format_as_decimal(val):
    """Ensures value is a plain decimal string with 6 places, no scientific notation."""
    return f"{float(val):.6f}"


def update_scale_csv(solver_name, scaling_values, size_labels):
    """Updates averages in time_scale_size.csv using plain decimal strings."""
    rows = []
    if os.path.exists(CSV_OUTPUT) and os.path.getsize(CSV_OUTPUT) > 0:
        with open(CSV_OUTPUT, mode='r', newline='') as f:
            rows = list(csv.DictReader(f))

    new_data = {"solver": solver_name}
    for i, val_str in enumerate(scaling_values):
        new_data[f"N={size_labels[i]}"] = val_str

    found = False
    for row in rows:
        if row.get("solver") == solver_name:
            row.update(new_data)
            found = True
            break
    if not found: rows.append(new_data)

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


def save_raw_runs(raw_data_list):
    """Saves every individual trial to mohle_sampled_run.csv with forced decimal formatting."""
    fieldnames = ["N", "Trial", "Data Processing", "Model Setup & Compilation", "Solver (Math)",
                  "Backtest & Post-processing", "Total Time"]

    with open(CSV_RUNS_OUTPUT, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for entry in raw_data_list:
            # Format every timing field as a decimal string before writing
            formatted_row = {
                "N": entry["N"],
                "Trial": entry["Trial"],
                "Data Processing": format_as_decimal(entry["Data Processing"]),
                "Model Setup & Compilation": format_as_decimal(entry["Model Setup & Compilation"]),
                "Solver (Math)": format_as_decimal(entry["Solver (Math)"]),
                "Backtest & Post-processing": format_as_decimal(entry["Backtest & Post-processing"]),
                "Total Time": format_as_decimal(entry["Total Time"])
            }
            writer.writerow(formatted_row)


# --- 4. EXECUTION LOOP ---
def run_scaling():
    data = load_data(CSV_INPUT)
    if data.empty: return

    full_pool = data.columns.tolist()[:POOL_SIZE]
    step = POOL_SIZE // INCREMENTS
    test_sizes = [i * step for i in range(1, INCREMENTS + 1)]

    scaling_results = []
    actual_labels = []
    all_individual_runs = []

    for n in test_sizes:
        print(f"\n--- Scaling Moehle: N={n} (Averaging {NUM_TRIALS} runs) ---")
        trial_accumulator = np.zeros(4)

        for trial in range(1, NUM_TRIALS + 1):
            reset_timers()
            selected_tickers = np.random.choice(full_pool, n, replace=False)

            t_slice_start = time.perf_counter()
            subset = data[selected_tickers]
            timers["Data Processing"] += (time.perf_counter() - t_slice_start)

            prev_w = np.zeros(n)
            for i in range(EST_WIN, len(subset) - PRED_WIN, PRED_WIN):
                t_est_start = time.perf_counter()
                est = subset.iloc[i - EST_WIN:i].pct_change().dropna()
                timers["Data Processing"] += (time.perf_counter() - t_est_start)
                if est.empty: continue
                w = optimize_moehle_nonconvex(est, prev_w, 5, 0.0001, 0.0010, 0.0001)
                prev_w = w

            # Capture raw data
            total_t = sum(timers.values())
            all_individual_runs.append({
                "N": n, "Trial": trial,
                "Data Processing": timers["Data Processing"],
                "Model Setup & Compilation": timers["Model Setup & Compilation"],
                "Solver (Math)": timers["Solver (Math)"],
                "Backtest & Post-processing": timers["Backtest & Post-processing"],
                "Total Time": total_t
            })

            trial_accumulator[0] += timers["Data Processing"]
            trial_accumulator[1] += timers["Model Setup & Compilation"]
            trial_accumulator[2] += timers["Solver (Math)"]
            trial_accumulator[3] += timers["Backtest & Post-processing"]
            print(f"  Trial {trial}/{NUM_TRIALS} complete.")

        # Calculate averages and format them as decimal strings within the tuple string
        avg_vals = trial_accumulator / NUM_TRIALS
        formatted_tuple = "(" + ", ".join([format_as_decimal(v) for v in avg_vals]) + ")"

        scaling_results.append(formatted_tuple)
        actual_labels.append(n)
        print(f"Finished N={n}. Avg Total Time: {sum(avg_vals):.2f}s")

    update_scale_csv("Moehle_Sampled", scaling_results, actual_labels)
    save_raw_runs(all_individual_runs)

    print(f"\nAveraged results (decimals) updated in {CSV_OUTPUT}")
    print(f"Individual trial results (decimals) saved to {CSV_RUNS_OUTPUT}")


if __name__ == "__main__":
    run_scaling()