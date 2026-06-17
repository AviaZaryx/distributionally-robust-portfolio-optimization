import numpy as np
import pandas as pd
import warnings
import time
import csv
import os

warnings.filterwarnings('ignore')

# ---------------- CONFIG ----------------
CSV_FILE = r'D:\Downloads\pythonProject1\merged_stock_data.csv'
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
    timers["Data Processing"] += (time.time() - t_start)
    return res


def update_scale_csv(solver_name, scaling_values, size_labels=None,
                     filename=r"D:\Downloads\pythonProject1\time_scale_size.csv"):
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

    sorted_fields = sorted(list(all_keys), key=sort_key)
    with open(filename, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=sorted_fields)
        writer.writeheader()
        writer.writerows(rows)


def run_1n_scaling(data, solver_name="1/N"):
    all_tickers = data.columns.tolist()
    total_tickers = len(all_tickers)
    chunk_size = total_tickers // N_PARTS

    scaling_results = []
    actual_labels = []

    for p in range(1, N_PARTS + 1):
        num_to_use = p * chunk_size if p < N_PARTS else total_tickers

        # 1. Data Processing Time
        t_dp_start = time.time()
        subset_data = data.iloc[:, :num_to_use]
        reset_timers()
        timers["Data Processing"] += (time.time() - t_dp_start)

        # 2. Setup/Compilation Time
        t_setup_start = time.time()
        n = subset_data.shape[1]
        w = np.ones(n) / n
        timers["Model Setup & Compilation"] += (time.time() - t_setup_start)

        # Backtest Loop
        for i in range(EST_WIN, len(subset_data) - PRED_WIN, PRED_WIN):
            # 3. Solver (Math) Time - Simulated by weights multiplication
            t_math_start = time.time()
            pred_data = subset_data.iloc[i:i + PRED_WIN].pct_change().dropna()
            if not pred_data.empty:
                _ = pred_data.values @ w
            timers["Solver (Math)"] += (time.time() - t_math_start)

            # 4. Post-processing Time
            t_post_start = time.time()
            # (Any cleanup or metric calculation goes here)
            timers["Backtest & Post-processing"] += (time.time() - t_post_start)

        # Construct the 4-tuple
        result_tuple = (
            round(timers["Data Processing"], 6),
            round(timers["Model Setup & Compilation"], 6),
            round(timers["Solver (Math)"], 6),
            round(timers["Backtest & Post-processing"], 6)
        )

        scaling_results.append(str(result_tuple))
        actual_labels.append(num_to_use)
        print(f"N={num_to_use} -> Tuple: {result_tuple}")

    update_scale_csv(solver_name, scaling_results, size_labels=actual_labels)


if __name__ == "__main__":
    data = load_data(CSV_FILE)
    if not data.empty:
        run_1n_scaling(data)