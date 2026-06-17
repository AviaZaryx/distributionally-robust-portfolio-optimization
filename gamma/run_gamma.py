import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd

# --- 1. IMPORT GAMMA-SPECIFIC MODULES ---
import socp_script_gamma
import mv_script_gamma
import mad_script_gamma
import one_n_script_gamma
import dro_mad_gamma
import mean_var_w_trd_hld_gamma

# --- 2. GLOBAL CONFIGURATION ---
GLOBAL_COLORS = {
    'SOCP': '#000000',
    'Classic MV': '#d62728',
    'Classic MAD': '#ff7f0e',
    'Moehle': '#1f77b4',
    'DRO-MAD': '#1b9e77',
    '1/N': '#984ea3'
}

csv_file = 'combined_all_stocks_cleaned.csv'

# --- 3. EXECUTE MODELS ---
print("--- STARTING GAMMA SENSITIVITY ANALYSIS ---")

result_socp = socp_script_gamma.run_socp_gamma(csv_file)
result_mv = mv_script_gamma.run_mv_gamma(csv_file)
result_mad = mad_script_gamma.run_mad_gamma(csv_file)
result_1n = one_n_script_gamma.run_1n_gamma(csv_file)
result_dro_mad = dro_mad_gamma.run_dro_mad_gamma(csv_file)
result_moehle = mean_var_w_trd_hld_gamma.run_moehle_gamma(csv_file)


# --- 4. ROBUST COLUMN STANDARDIZATION ---
def standardize_columns(df):
    # Strip spaces and lowercase
    df.columns = [str(c).strip().lower().replace(' ', '_') for c in df.columns]
    mapping = {
        'mean_return': ['mean_return', 'mean_mad', 'mean_1n', 'annualized_return', 'mean'],
        'std_dev': ['std_dev', 'std_mad', 'std_1n', 'volatility', 'std'],
        'sharpe_ratio': ['sharpe_ratio', 'sharpe_mad', 'sharpe_1n', 'sharpe'],
        'sortino_ratio': ['sortino_ratio', 'sortino_mad', 'sortino_1n', 'sortino'],
        'max_drawdown': ['max_drawdown', 'mdd_mad', 'mdd_1n', 'mdd', 'drawdown'],
        'gamma_risk': ['gamma_risk', 'gamma'],
        'runtime_s': ['runtime_(s)', 'runtime_s', 'runtime']
    }
    new_cols = {}
    for target, variations in mapping.items():
        for var in variations:
            if var in df.columns:
                new_cols[var] = target
                break
    return df.rename(columns=new_cols)


# Standardize all
result_socp = standardize_columns(result_socp)
result_mv = standardize_columns(result_mv)
result_mad = standardize_columns(result_mad)
result_1n = standardize_columns(result_1n)
result_dro_mad = standardize_columns(result_dro_mad)
result_moehle = standardize_columns(result_moehle)

# --- 5. PLOTTING: LINE CHARTS ---
plt.style.use('seaborn-v0_8-darkgrid')


def plot_gamma_metric(metric_key, ylabel, title):
    plt.figure(figsize=(12, 7))

    better_map = {
        'mean_return': '(Higher is Better)',
        'std_dev': '(Lower is Better)',
        'sharpe_ratio': '(Higher is Better)',
        'sortino_ratio': '(Higher is Better)',
        'max_drawdown': '(Less Negative is Better)',
        'runtime_s': '(Lower is Better)'
    }

    directional_info = better_map.get(metric_key, "")

    def plot_line(df, label, color_key, style, marker):
        if 'gamma_risk' in df.columns and metric_key in df.columns:
            lw = 3.5 if color_key == 'SOCP' else 2.5
            zo = 15 if color_key == 'SOCP' else 5
            plt.plot(df['gamma_risk'], df[metric_key], linestyle=style, marker=marker,
                     color=GLOBAL_COLORS[color_key], label=label, linewidth=lw, zorder=zo)

    plot_line(result_socp, 'SOCP', 'SOCP', '-', 'o')
    plot_line(result_dro_mad, 'Wasserstein DRO-MAD', 'DRO-MAD', '-', '^')
    plot_line(result_mv, 'Classic MV', 'Classic MV', '--', 'x')
    plot_line(result_mad, 'Classic MAD', 'Classic MAD', '-.', 's')
    plot_line(result_moehle, 'Moehle MIQP', 'Moehle', '--', 'd')
    plot_line(result_1n, '1/N Portfolio', '1/N', ':', None)

    plt.xlabel('Gamma (Risk Aversion)')
    plt.ylabel(ylabel)
    plt.title(f"{title} {directional_info}", fontweight='bold')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


# --- 6. PLOTTING: RUNTIMES (PER-GAMMA) ---
def plot_runtimes_per_gamma():
    plt.figure(figsize=(12, 7))

    models = [
        (result_socp, 'SOCP', 'SOCP', '-'),
        (result_dro_mad, 'Wasserstein DRO-MAD', 'DRO-MAD', '-'),
        (result_mv, 'Classic MV', 'Classic MV', '--'),
        (result_mad, 'Classic MAD', 'Classic MAD', '-.'),
        (result_moehle, 'Moehle MIQP', 'Moehle', '--'),
        (result_1n, '1/N Portfolio', '1/N', ':')
    ]

    for df, label, color_key, style in models:
        if 'runtime_s' in df.columns:
            lw = 3.5 if color_key == 'SOCP' else 2.5
            plt.plot(df['gamma_risk'], df['runtime_s'], linestyle=style, marker='o',
                     markersize=4, color=GLOBAL_COLORS[color_key], label=label, linewidth=lw)

    plt.xlabel('Gamma (Risk Aversion)')
    plt.ylabel('Execution Time per Backtest (Seconds)')
    plt.title('Computational Performance per Gamma (Lower is Better)', fontweight='bold')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


# --- 7. PLOTTING: SPIDER CHARTS ---
def generate_spider_plots_gamma():
    # 1. Removed Runtime from display labels
    metrics_display = ['Return (Higher)', 'Vol (Lower)', 'Sharpe (Higher)',
                       'Sortino (Higher)', 'MDD (Lower)']
    num_vars = len(metrics_display)

    # Standardizing gammas based on SOCP result
    unique_gammas = sorted(result_socp['gamma_risk'].unique())
    target_gammas = [unique_gammas[0], unique_gammas[len(unique_gammas) // 2], unique_gammas[-1]]

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    for g_val in target_gammas:
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        labels = ['SOCP', 'DRO-MAD', 'Classic MV', 'Classic MAD', 'Moehle', '1/N']
        dfs = [result_socp, result_dro_mad, result_mv, result_mad, result_moehle, result_1n]
        color_keys = ['SOCP', 'DRO-MAD', 'Classic MV', 'Classic MAD', 'Moehle', '1/N']

        rows = []
        for df in dfs:
            # Find closest gamma row
            subset = df[np.isclose(df['gamma_risk'], g_val)]
            if subset.empty:
                rows.append([0] * num_vars)
                continue
            r = subset.iloc[0]
            # 2. Removed runtime_s from the data collection list
            rows.append([r.get('mean_return', 0),
                         r.get('std_dev', 0),
                         r.get('sharpe_ratio', 0),
                         r.get('sortino_ratio', 0),
                         abs(r.get('max_drawdown', 0))])

        data = np.array(rows)
        norm_data = np.zeros_like(data)

        for i in range(num_vars):
            col = data[:, i]
            # 3. Updated indices: Index 1 is Vol, Index 4 is MDD (Lower is better)
            if i in [1, 4]:
                best_val = col.min()
                norm_data[:, i] = [best_val / v if v != 0 else 1.0 for v in col]
            else:  # Return/Ratios: Higher is better
                best_val = col.max()
                norm_data[:, i] = [v / best_val if best_val != 0 else 0.0 for v in col]

        for idx, label in enumerate(labels):
            values = norm_data[idx].tolist()
            values += values[:1]
            lw = 4 if color_keys[idx] == 'SOCP' else 2.5
            ax.plot(angles, values, color=GLOBAL_COLORS[color_keys[idx]], linewidth=lw, label=label, zorder=10)
            ax.fill(angles, values, color=GLOBAL_COLORS[color_keys[idx]], alpha=0.05)

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles[:-1]), metrics_display)
        ax.set_ylim(0, 1.1)
        plt.title(f"Performance Snapshot (Gamma = {g_val})\nScale: 1.0 = Best Metric Found", weight='bold', pad=30)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.show()


# --- 8. NUMERICAL REPORTS ---
def print_numerical_reports_gamma():
    print("\n" + "=" * 100 + "\nDETAILED NUMERICAL COMPARISON (GAMMA SWEEP)\n" + "=" * 100)

    summary_dfs = []
    for df, name in [(result_socp, 'SOCP'), (result_dro_mad, 'DRO-MAD'), (result_mv, 'Classic MV')]:
        print(f"\n>> {name} Stats:")
        cols = ['gamma_risk', 'mean_return', 'sharpe_ratio', 'runtime_s']
        print(df[cols].round(4).to_string(index=False))


# --- EXECUTE ALL ---
plot_gamma_metric('mean_return', 'Annualized Return', 'Mean Return vs Gamma')
plot_gamma_metric('std_dev', 'Annualized Std Dev', 'Risk vs Gamma')
plot_gamma_metric('sharpe_ratio', 'Sharpe Ratio', 'Sharpe Ratio vs Gamma')
plot_gamma_metric('sortino_ratio', 'Sortino Ratio', 'Sortino Ratio vs Gamma')
plot_gamma_metric('max_drawdown', 'Max Drawdown', 'Max Drawdown vs Gamma')

# Per-Gamma Runtime Line Chart
plot_runtimes_per_gamma()

# Radar charts and numerical tables
generate_spider_plots_gamma()
print_numerical_reports_gamma()