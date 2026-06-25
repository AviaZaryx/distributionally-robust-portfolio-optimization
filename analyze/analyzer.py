import csv, os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

GLOBAL_COLORS = {'socp': '#000000', 'mv': '#d62728', 'mad': '#ff7f0e', 'moehle': '#1f77b4', 'dro_mad': '#1b9e77',
                 '1n': '#984ea3'}
NAME_MAP = {'socp': 'SOCP', 'mv': 'Classic MV', 'mad': 'Classic MAD', 'dro_mad': 'DRO-MAD', 'moehle': 'Moehle',
            '1n': '1/N'}
CATEGORIES = ["Data Processing", "Setup & Compilation", "Solver (Math)", "Post-processing"]
HATCHES = ['', '////', '....', 'xxxx']


def get_high_contrast_gradient(hex_color, step):
    rgb = np.array(mcolors.to_rgb(hex_color))
    fractions = [0.0, 0.45, 0.75, 0.92]
    return rgb + (np.array([1.0, 1.0, 1.0]) - rgb) * fractions[step]


def visualize_timing_gradient():
    filename = r"D:\Downloads\pythonProject1\analyzed.csv"
    data = []
    with open(filename, mode='r') as f:
        for row in csv.DictReader(f):
            row_data = {"solver": row["solver"], "total": 0.0}
            for cat, short_cat in zip(
                    ["Data Processing", "Model Setup & Compilation", "Solver (Math)", "Backtest & Post-processing"],
                    CATEGORIES):
                val = float(row.get(cat, 0.0))
                row_data[short_cat] = val
                row_data["total"] += val
            data.append(row_data)

    sorted_data = sorted(data, key=lambda x: x["total"])

    # LARGE SYNCHRONIZED CANVAS
    plt.figure(figsize=(16, 12))
    bottoms = np.zeros(len(sorted_data))

    for i, cat in enumerate(CATEGORIES):
        cat_values = [d[cat] for d in sorted_data]
        for s_idx, d in enumerate(sorted_data):
            face_color = get_high_contrast_gradient(GLOBAL_COLORS.get(d["solver"].lower(), "#95a5a6"), i)
            lum = 0.299 * face_color[0] + 0.587 * face_color[1] + 0.114 * face_color[2]
            plt.bar(s_idx, cat_values[s_idx], bottom=bottoms[s_idx], color=face_color,
                    edgecolor='white' if lum < 0.45 else 'black', linewidth=2, hatch=HATCHES[i], width=0.7)
        bottoms += np.array(cat_values)

    # --- DOUBLED FONTS & UNITS ---
    plt.title('Time Breakdown by Component', fontsize=40, fontweight='bold', pad=40)
    plt.ylabel('Execution Time [Seconds]', fontsize=32, fontweight='bold', labelpad=20)
    plt.xlabel('Optimization Models', fontsize=32, fontweight='bold', labelpad=20)
    plt.xticks(range(len(sorted_data)), [NAME_MAP.get(d["solver"].lower(), d["solver"]) for d in sorted_data],
               fontsize=22)
    plt.yticks(fontsize=26)

    # --- DOUBLED LEGEND ---
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor='#cccccc', edgecolor='black', hatch=HATCHES[i], label=CATEGORIES[i]) for i
        in range(len(CATEGORIES))]
    plt.legend(handles=legend_elements[::-1], title="Components", title_fontsize='30', fontsize=26,
               loc='upper left', frameon=True, shadow=True, borderpad=1.5, labelspacing=1.2)

    plt.grid(axis='y', linestyle='--', alpha=0.4, linewidth=2)
    plt.tight_layout()
    plt.savefig('time.png', dpi=300, bbox_inches='tight')
    plt.show()


visualize_timing_gradient()