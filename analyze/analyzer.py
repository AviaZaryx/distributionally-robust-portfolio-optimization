import csv
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

# --- MAPPING & CONFIGURATION ---
GLOBAL_COLORS = {
    'socp': '#000000',  # Black
    'mv': '#d62728',  # Red
    'mad': '#ff7f0e',  # Orange
    'moehle': '#1f77b4',  # Blue
    'dro_mad': '#1b9e77',  # Teal
    '1n': '#984ea3'  # Purple
}

CATEGORIES = ["Data Processing", "Model Setup & Compilation", "Solver (Math)", "Backtest & Post-processing"]
# High-density hatches for better visibility on small bars
HATCHES = ['', '////', '....', 'xxxx']


def get_high_contrast_gradient(hex_color, step):
    """
    Uses an aggressive non-linear blend to ensure dark colors
    (like SOCP/Black) become clearly visible gray/white segments.
    """
    rgb = np.array(mcolors.to_rgb(hex_color))
    white = np.array([1.0, 1.0, 1.0])

    # Step-based white-mix fractions:
    # 0% white (base), 45% white, 75% white, 92% white
    fractions = [0.0, 0.45, 0.75, 0.92]
    f = fractions[step]

    return rgb + (white - rgb) * f


def load_and_sort_data(filename=r"D:\Downloads\pythonProject1\analyzed.csv"):
    if not os.path.exists(filename):
        print(f"Error: {filename} not found.")
        return []

    data = []
    with open(filename, mode='r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            processed_row = {"solver": row["solver"]}
            total_time = 0.0
            for cat in CATEGORIES:
                val = float(row.get(cat, 0.0))
                processed_row[cat] = val
                total_time += val
            processed_row["total"] = total_time
            data.append(processed_row)

    return sorted(data, key=lambda x: x["total"])


def visualize_timing_gradient():
    sorted_data = load_and_sort_data(r"D:\Downloads\pythonProject1\analyzed.csv")
    if not sorted_data: return

    n_solvers = len(sorted_data)
    indices = np.arange(n_solvers)
    solver_labels = [d["solver"] for d in sorted_data]

    plt.figure(figsize=(12, 7))
    bottoms = np.zeros(n_solvers)

    # Plot categories one by one
    for i, cat in enumerate(CATEGORIES):
        cat_values = [d[cat] for d in sorted_data]

        for s_idx, d in enumerate(sorted_data):
            base_color = GLOBAL_COLORS.get(d["solver"].lower(), "#95a5a6")
            face_color = get_high_contrast_gradient(base_color, i)

            # Determine if the background is dark. If so, make the hatch lines white.
            # This is crucial for SOCP (Black) and 1n (Purple)
            luminance = 0.299 * face_color[0] + 0.587 * face_color[1] + 0.114 * face_color[2]
            hatch_color = 'white' if luminance < 0.45 else 'black'

            # Plot individual bar segments to allow per-segment hatch colors
            plt.bar(
                indices[s_idx], cat_values[s_idx], bottom=bottoms[s_idx],
                color=face_color,
                edgecolor=hatch_color,  # Match hatch color to edge for visibility
                linewidth=0.5,
                hatch=HATCHES[i],
                width=0.7
            )

        bottoms += np.array(cat_values)

    # --- LEGEND ---
    # Legend shows the patterns clearly with high contrast
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor='#cccccc', edgecolor='black',
                      hatch=HATCHES[i], label=CATEGORIES[i])
        for i in range(len(CATEGORIES))
    ]

    plt.legend(
        handles=legend_elements[::-1],
        title="Timing Breakdown"
              "",
        loc='upper left'
    )

    # Styling
    plt.ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    plt.xlabel('Solvers (Fastest to Slowest)', fontsize=12, fontweight='bold')
    plt.title('Solver Performance Analysis', fontsize=14, pad=15)
    plt.xticks(indices, solver_labels)
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize_timing_gradient()