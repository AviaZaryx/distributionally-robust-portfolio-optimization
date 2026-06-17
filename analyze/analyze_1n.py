import numpy as np
import pandas as pd
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


def run_1n():
    data = load_data(CSV_FILE)
    if data.empty: return pd.DataFrame()

    delta_range = [0.05]
    results = []

    for d in delta_range:
        rets_1n = []

        # 1/N Weight Generation (Grouped with Post-processing as it's analytical)
        t_calc_start = time.time()
        n = data.shape[1]
        w = np.ones(n) / n
        timers["Backtest & Post-processing"] += (time.time() - t_calc_start)

        for i in range(EST_WIN, len(data) - PRED_WIN, PRED_WIN):
            # Data Processing (Slicing)
            t_s = time.time()
            pred_data = data.iloc[i:i + PRED_WIN].pct_change().dropna()
            timers["Data Processing"] += (time.time() - t_s)

            if pred_data.empty: continue

            # Backtest (Realization)
            t_r = time.time()
            rets_1n.extend(pred_data.values @ w)
            timers["Backtest & Post-processing"] += (time.time() - t_r)

        # Backtest (Stats Calculation)
        t_m = time.time()
        s = pd.Series(rets_1n)
        if not s.empty:
            mn = s.mean() * 252
            sd = s.std() * np.sqrt(252)
            results.append({'delta': d, 'Mean': mn, 'Sharpe': mn / sd if sd > 0 else 0})
        timers["Backtest & Post-processing"] += (time.time() - t_m)

    return pd.DataFrame(results)


def visualize_timing(timers_dict):
    filtered = {k: v for k, v in timers_dict.items()}  # Keep 0s to show comparison
    labels = list(filtered.keys())
    values = list(filtered.values())
    total_time = sum(values)

    legend_labels = [f"{l} ({(v / total_time) * 100:.2f}%)" for l, v in zip(labels, values)]

    fig, ax = plt.subplots(figsize=(12, 7))
    # Use a set of distinct colors
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))

    wedges, texts, autotexts = ax.pie(
        values,
        labels=None,
        autopct=lambda p: '{:.1f}%'.format(p) if p > 0 else '',
        startangle=140,
        colors=colors,
        pctdistance=0.85
    )

    ax.legend(wedges, legend_labels,
              title=f"1/N Model Execution Time (Total: {total_time:.4f}s)",
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

    # 2. Add any new timer keys to the header list
    for key in timers_dict.keys():
        if key not in fieldnames:
            fieldnames.append(key)

    # 3. Find existing row to update, or prepare to append a new one
    found = False
    for row in rows:
        if row.get("solver") == solver_name:
            # Convert values to string to ensure CSV compatibility
            row.update({k: str(v) for k, v in timers_dict.items()})
            found = True
            break

    if not found:
        # Create a new row at the bottom (directly below the last solver)
        new_row = {"solver": solver_name}
        new_row.update({k: str(v) for k, v in timers_dict.items()})
        rows.append(new_row)

    # 4. Write all data back to the file
    with open(filename, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    # Execute 1/N solver
    df_res = run_1n()
    visualize_timing(timers)

    # Log results to CSV under the name "1n"
    update_analyzed_csv("1n", timers)

    print("\n--- 1/N CONSOLIDATED SUMMARY ---")
    for key, val in timers.items():
        print(f"{key}: {val:.4f}s")
    print(f"\nResults saved to analyzed.csv")
def comp_analyze_1n():
    run_1n()
    return timers

