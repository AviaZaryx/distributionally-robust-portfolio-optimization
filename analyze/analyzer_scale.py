import pandas as pd
import matplotlib.pyplot as plt
import ast
import os
import numpy as np

# --- CONFIGURATION ---
CSV_FILE = r'D:\Downloads\pythonProject1\time_scale_size.csv'

# Standard colors for your paper
GLOBAL_COLORS = {
    'SOCP': '#000000',  # Black
    'Classic MV': '#d62728',  # Red
    'Classic MAD': '#ff7f0e',  # Orange
    'Moehle': '#1f77b4',  # Blue
    'DRO-MAD': '#1b9e77',  # Green
    '1/N': '#984ea3'  # Purple
}

# Standardize solver names from CSV to Legend
NAME_MAP = {
    'socp': 'SOCP',
    'mv': 'Classic MV',
    'mad': 'Classic MAD',
    'dro_mad': 'DRO-MAD',
    'moehle': 'Moehle',
    '1n': '1/N',
    'SOCP': 'SOCP',
    'DRO-MAD': 'DRO-MAD',
    'Classic MV': 'Classic MV',
    'Classic MAD': 'Classic MAD',
    'Moehle': 'Moehle',
    '1/N': '1/N'
}


def load_and_plot_scaling_linear():
    if not os.path.exists(CSV_FILE):
        print(f"Error: {CSV_FILE} not found.")
        return

    # 1. Load the data
    df_scale = pd.read_csv(CSV_FILE)

    # 2. Identify and Sort N= columns numerically (N=1, N=10, N=100, N=200)
    # This prevents the "zig-zag" caused by alphabetical sorting
    data_cols = [c for c in df_scale.columns if c.startswith('N=')]
    data_cols.sort(key=lambda x: int(x.split('=')[1]))

    plt.figure(figsize=(13, 8))

    # 3. Process each solver row
    for _, row in df_scale.iterrows():
        solver_id = row['solver']
        model_name = NAME_MAP.get(solver_id, solver_id)
        color = GLOBAL_COLORS.get(model_name, '#95a5a6')

        x_vals = []
        y_vals = []

        for col in data_cols:
            cell_value = row[col]

            # Skip empty cells
            if pd.isna(cell_value) or str(cell_value).strip() == "":
                continue

            try:
                # Parse the 4-tuple: (Data, Setup, Math, Post)
                parsed_val = ast.literal_eval(str(cell_value))

                # Sum all 4 components for TOTAL TIME
                if isinstance(parsed_val, (tuple, list)):
                    total_time = sum(parsed_val)
                else:
                    total_time = float(parsed_val)

                x_vals.append(int(col.split('=')[1]))
                y_vals.append(total_time)

            except (ValueError, SyntaxError):
                continue

        # 4. Plot the line if data exists
        if x_vals:
            # Re-sort points by X to ensure smooth lines
            sorted_idx = np.argsort(x_vals)
            plt.plot(
                np.array(x_vals)[sorted_idx],
                np.array(y_vals)[sorted_idx],
                label=model_name,
                color=color,
                marker='o',
                linewidth=2.5,
                markersize=7,
                # SOCP is zorder 10 to stay on top of other convex models
                zorder=10 if model_name == 'SOCP' else 5
            )

    # --- STYLING ---
    plt.title('Total Computational Time vs. Asset Universe Size ($N$)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Number of Assets ($N$)', fontsize=14, fontweight='bold')
    plt.ylabel('Total Execution Time [Seconds]', fontsize=14, fontweight='bold')

    # Explicitly linear scale
    plt.yscale('linear')

    # Grid and Legend
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title="Optimization Models", title_fontsize='12', fontsize=11, loc='upper left')

    # Adjust layout to prevent clipping
    plt.tight_layout()

    # Save the plot for your paper
    # plt.savefig('computational_scaling_linear.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    load_and_plot_scaling_linear()