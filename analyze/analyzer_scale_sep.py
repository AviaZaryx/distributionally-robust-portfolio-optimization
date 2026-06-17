import pandas as pd
import matplotlib.pyplot as plt
import ast
import os
import numpy as np

# --- CONFIGURATION ---
CSV_FILE = 'time_scale_size.csv'
STOCK_DATA_FILE = 'merged_stock_data.csv'

GLOBAL_COLORS = {
    'SOCP': '#000000',
    'Classic MV': '#d62728',
    'Classic MAD': '#ff7f0e',
    'Moehle': '#1f77b4',
    'DRO-MAD': '#1b9e77',
    '1/N': '#984ea3'
}

NAME_MAP = {
    'socp': 'SOCP',
    'mv': 'Classic MV',
    'mad': 'Classic MAD',
    'moehle': 'Moehle',
    'dro_mad': 'DRO-MAD',
    '1n': '1/N'
}


def get_total_stock_count():
    if os.path.exists(STOCK_DATA_FILE):
        df = pd.read_csv(STOCK_DATA_FILE)
        return len(df['Ticker'].unique())
    return 100


def load_and_plot_separate():
    if not os.path.exists(CSV_FILE):
        print(f"Error: {CSV_FILE} not found.")
        return

    df_scale = pd.read_csv(CSV_FILE)
    total_stocks = get_total_stock_count()
    run_cols = [c for c in df_scale.columns if c.startswith('run_')]
    n_parts = len(run_cols)

    # Calculate X-axis (Stocks)
    step = total_stocks // n_parts
    x_axis = [((i + 1) * step if (i + 1) < n_parts else total_stocks) for i in range(n_parts)]

    # Create two subplots (Top for Compilation, Bottom for Math)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    for _, row in df_scale.iterrows():
        solver_id = row['solver']
        model_name = NAME_MAP.get(solver_id, solver_id)
        color = GLOBAL_COLORS.get(model_name, '#95a5a6')

        comp_times = []
        math_times = []

        for col in run_cols:
            val = row[col]
            if pd.isna(val):
                comp_times.append(0)
                math_times.append(0)
                continue

            try:
                comp, math = ast.literal_eval(val)
                comp_times.append(comp)
                math_times.append(math)
            except:
                comp_times.append(0)
                math_times.append(0)

        # Plot Compilation Time (Top Plot)
        ax1.plot(x_axis, comp_times, label=model_name, color=color,
                 marker='o', linestyle='-', linewidth=2)

        # Plot Solver Math Time (Bottom Plot)
        ax2.plot(x_axis, math_times, label=model_name, color=color,
                 marker='s', linestyle='--', linewidth=2)

    # --- Styling Top Plot ---
    ax1.set_title('Model Setup & Compilation Time Scaling', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Time (s)', fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.7)
    ax1.legend(loc='upper left', fontsize=9)

    # --- Styling Bottom Plot ---
    ax2.set_title('Pure Solver (Math) Time Scaling', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Number of Stocks', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Time (s)', fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.7)

    # Optional: If Moehle is 100x slower than 1/N, use log scale for Math
    # ax2.set_yscale('log')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    load_and_plot_separate()