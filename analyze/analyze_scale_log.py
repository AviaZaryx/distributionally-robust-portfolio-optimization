import pandas as pd
import matplotlib.pyplot as plt
import os
import re

CSV_FILE = r'D:\Downloads\pythonProject1\time_scale_size.csv'
GLOBAL_COLORS = {'SOCP': '#000000', 'Classic MV': '#d62728', 'Classic MAD': '#ff7f0e', 'Moehle_Sampled': '#1f77b4',
                 'DRO-MAD': '#1b9e77', '1/N': '#984ea3'}
NAME_MAP = {'socp': 'SOCP', 'mv': 'Classic MV', 'mad': 'Classic MAD', 'dro_mad': 'DRO-MAD', 'moehle': 'Moehle_Sampled',
            '1n': '1/N'}


def load_and_plot_scaling_log():
    if not os.path.exists(CSV_FILE):
        print(f"File not found: {CSV_FILE}")
        return

    df_scale = pd.read_csv(CSV_FILE)
    data_cols = sorted([c for c in df_scale.columns if c.startswith('N=')], key=lambda x: int(x.split('=')[1]))

    # --- INCREASED HEIGHT: (Width=16, Height=20) ---
    plt.figure(figsize=(16, 20))

    plot_data_list = []

    for _, row in df_scale.iterrows():
        raw_name = str(row['solver']).strip().lower()
        model_name = NAME_MAP.get(raw_name, row['solver'])

        x_vals, y_vals = [], []
        for col in data_cols:
            val = str(row[col])
            if pd.isna(row[col]) or val.strip() == "" or val.lower() == 'nan':
                continue

            numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", val)
            if numbers:
                total_time = sum(float(n) for n in numbers)
                # Avoid plotting 0 in log scale (will result in error or missing point)
                if total_time > 0:
                    x_vals.append(int(col.split('=')[1]))
                    y_vals.append(total_time)

        if x_vals:
            # Determine the runtime specifically at N=100 for sorting
            val_at_100 = 0
            for x, y in zip(x_vals, y_vals):
                if x == 100:
                    val_at_100 = y
                    break

            plot_data_list.append({
                'name': model_name,
                'x': x_vals,
                'y': y_vals,
                'sort_val': val_at_100,
                'color': GLOBAL_COLORS.get(model_name, '#95a5a6'),
                'zorder': 10 if model_name == 'SOCP' else 5
            })

    # Sort legend by runtime at N=100
    plot_data_list.sort(key=lambda item: item['sort_val'], reverse=True)

    # Now plot the sorted data
    for item in plot_data_list:
        plt.plot(item['x'], item['y'], label=item['name'], color=item['color'],
                 marker='o', linewidth=6, markersize=16, zorder=item['zorder'])

    # --- LOG SCALE TRANSFORMATION ---
    plt.xscale('log')
    plt.yscale('log')

    # --- STYLING ---
    plt.title('Computational Scalability (Log-Log)', fontsize=40, fontweight='bold', pad=40)
    plt.xlabel('Number of Assets ($N$)', fontsize=32, fontweight='bold', labelpad=20)
    plt.ylabel('Total Time [Seconds]', fontsize=32, fontweight='bold', labelpad=20)

    # Adjusting tick parameters for log scale visibility
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)

    #plt.legend(title="Models", title_fontsize='20', fontsize=16, loc='upper left', frameon=True, shadow=True, borderpad=1.5, labelspacing=1.2)

    # Use which='both' to show grid lines for minor log ticks
    plt.grid(False)

    plt.tight_layout()
    plt.savefig('time_scale_log.png', dpi=300, bbox_inches='tight')
    print("Plot saved successfully with Log-Log scale.")
    plt.show()


if __name__ == "__main__":
    load_and_plot_scaling_log()