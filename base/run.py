import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import time
import pandas as pd

# --- 1. IMPORT MODULES ---
import socp_script
import mv_script
import mad_script
import one_n_script
import dro_mad
import mean_var_w_trd_hld

# --- 2. GLOBAL COLOR CONFIGURATION ---
GLOBAL_COLORS = {
    'SOCP': '#000000',
    'Classic MV': '#d62728',
    'Classic MAD': '#ff7f0e',
    'Moehle': '#1f77b4',
    'DRO-MAD': '#1b9e77',
    '1/N': '#984ea3'
}

# --- 3. EXECUTE MODELS ---
csv_file = r'D:\Downloads\pythonProject1\merged_stock_data_half.csv'

print("1. Running SOCP...")
result_socp = socp_script.run_socp(csv_file)
result_socp.columns = result_socp.columns.str.strip()

print("2. Running Classic MV...")
result_mv = mv_script.run_mv(csv_file)
result_mv.columns = result_mv.columns.str.strip()

print("3. Running Classic MAD...")
result_mad = mad_script.run_mad(csv_file)
result_mad.columns = result_mad.columns.str.strip()

print("4. Running 1/N Benchmark...")
result_1n = one_n_script.run_1n(csv_file)
result_1n.columns = result_1n.columns.str.strip()

print("5. Running Wasserstein DRO-MAD...")
result_dro_mad = dro_mad.run_dro_mad(csv_file)
result_dro_mad.columns = result_dro_mad.columns.str.strip()

print("6. Running Moehle (Non-Convex)...")
result_moehle = mean_var_w_trd_hld.run_moehle_paper(csv_file)
result_moehle.columns = result_moehle.columns.str.strip()

# --- 4. DATA PRE-PROCESSING ---
# Normalize delta column names to lowercase 'delta' for internal plotting logic
for df in [result_mad, result_dro_mad]:
    if 'Delta' in df.columns:
        df.rename(columns={'Delta': 'delta'}, inplace=True)

df_mv = result_mv.sort_values('delta')
df_mad = result_mad.sort_values('delta')
df_dro = result_dro_mad.sort_values('delta')
df_moehle = result_moehle.sort_values('delta')
df_socp = result_socp.sort_values('delta')
df_1n = result_1n.sort_values('delta')

delta_bench = df_mv['delta'].values

# --- 5. PLOTTING: LINE CHARTS ---
plt.style.use('seaborn-v0_8-darkgrid')


def plot_metric(metric_name, ylabel, title):
    plt.figure(figsize=(12, 7))

    # Define directional indicators
    better_map = {
        'mean': '(Higher is Better)',
        'std': '(Lower is Better)',
        'sharpe': '(Higher is Better)',
        'sortino': '(Higher is Better)',
        'mdd': '(Less Negative is Better)'
    }

    col_map = {'mean': 'mean_return', 'std': 'std_dev', 'sharpe': 'sharpe_ratio', 'sortino': 'sortino_ratio',
               'mdd': 'max_drawdown'}

    # Adjust for DRO-MAD column naming differences if they exist
    dro_col_map = {'mean': 'Mean Return', 'std': 'Std Dev', 'sharpe': 'Sharpe Ratio', 'sortino': 'Sortino Ratio',
                   'mdd': 'Max Drawdown'}
    mad_col_map = {'mean': 'Mean_MAD', 'std': 'Std_MAD', 'sharpe': 'Sharpe_MAD', 'sortino': 'Sortino_MAD',
                   'mdd': 'MDD_MAD'}

    socp_gammas = sorted(df_socp['gamma_risk'].unique())
    styles = ['-', '--', ':', '-.']

    for i, g in enumerate(socp_gammas):
        subset = df_socp[df_socp['gamma_risk'] == g]
        plt.plot(subset['delta'], subset[col_map[metric_name]],
                 marker='o', markersize=5, linewidth=2.5,
                 linestyle=styles[i % len(styles)],
                 color=GLOBAL_COLORS['SOCP'],
                 label=f'SOCP', zorder=10)

    plt.plot(df_mv['delta'], df_mv[col_map[metric_name]], '--x', color=GLOBAL_COLORS['Classic MV'], label='Classic MV')
    plt.plot(df_mad['delta'], df_mad[mad_col_map[metric_name]], '-.s', color=GLOBAL_COLORS['Classic MAD'],
             label='Classic MAD')
    plt.plot(df_moehle['delta'], df_moehle[col_map[metric_name]], '--d', color=GLOBAL_COLORS['Moehle'], label='Moehle')
    plt.plot(df_dro['delta'], df_dro[dro_col_map[metric_name]], '^-', color=GLOBAL_COLORS['DRO-MAD'], linewidth=2,
             label='Wasserstein DRO-MAD')
    plt.plot(df_1n['delta'], df_1n[col_map[metric_name]], color=GLOBAL_COLORS['1/N'], linestyle=':', linewidth=3,
             label='1/N Portfolio')

    plt.xlabel('Delta')
    plt.ylabel(ylabel)
    plt.title(f"{title}", fontweight='bold')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


# --- 6. PLOTTING: RUNTIMES (PER-DELTA) ---
def plot_runtimes_per_delta(delta_axis):
    plt.figure(figsize=(12, 7))

    models = [
        (df_socp[df_socp['gamma_risk'] == 3], 'SOCP', 'SOCP', '-'),
        (df_mv, 'Classic MV', 'Classic MV', '--'),
        (df_mad, 'Classic MAD', 'Classic MAD', '-.'),
        (df_dro, 'Wasserstein DRO-MAD', 'DRO-MAD', '-'),
        (df_moehle, 'Moehle (Non-Convex)', 'Moehle', '--'),
        (df_1n, '1/N Portfolio', '1/N', ':')
    ]

    for df_res, label, color_key, style in models:
        # Check if runtime column exists
        runtime_col = 'Runtime (s)'
        if runtime_col in df_res.columns:
            plt.plot(df_res['delta'], df_res[runtime_col],
                     linestyle=style, marker='o', markersize=4,
                     color=GLOBAL_COLORS[color_key], linewidth=2.5,
                     label=f"{label}")

    plt.xlabel('Delta')
    plt.ylabel('Execution Time per Backtest (Seconds)')
    plt.title('Computational Performance per Delta (Lower is Better)', fontweight='bold')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


# --- 7. PLOTTING: SPIDER CHARTS (WITH LOCAL RUNTIME) ---
def generate_individual_spider_plots():
    # Labels for the Spider Plot
    metrics = ['Return', 'Vol', 'Sharpe',
               'Sortino', 'MDD']
    num_vars = len(metrics)

    unique_deltas = sorted(df_dro['delta'].unique())
    target_deltas = [unique_deltas[0], unique_deltas[len(unique_deltas) // 2], unique_deltas[-1]]

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    for d_val in target_deltas:
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        # --- STYLE UPDATES START HERE ---
        fig.patch.set_facecolor('white')  # Background outside the circle
        ax.set_facecolor('white')  # Background inside the circle
        ax.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)  # Gray grid
        ax.spines['polar'].set_color('gray')  # Make the outer circular border gray
        # --- STYLE UPDATES END HERE ---

        # Gathering stats for the specific delta point
        model_data = []
        labels = ['SOCP', 'DRO-MAD', 'Classic MV', 'Classic MAD', 'Moehle', '1/N']
        color_keys = ['SOCP', 'DRO-MAD', 'Classic MV', 'Classic MAD', 'Moehle', '1/N']

        def get_row_vals(df, d, cols, is_mad=False, is_dro=False):
            subset = df[np.isclose(df['delta'], d)]
            if subset.empty: return [0] * 5  # Adjusted to 5 to match metrics length
            r = subset.iloc[0]
            if is_dro:
                return [r['Mean Return'], r['Std Dev'], r['Sharpe Ratio'], r['Sortino Ratio'], abs(r['Max Drawdown'])]
            if is_mad:
                return [r['Mean_MAD'], r['Std_MAD'], r['Sharpe_MAD'], r['Sortino_MAD'], abs(r['MDD_MAD'])]
            return [r['mean_return'], r['std_dev'], r['sharpe_ratio'], r['sortino_ratio'], abs(r['max_drawdown'])]

        model_data.append(get_row_vals(df_socp, d_val, None))
        model_data.append(get_row_vals(df_dro, d_val, None, is_dro=True))
        model_data.append(get_row_vals(df_mv, d_val, None))
        model_data.append(get_row_vals(df_mad, d_val, None, is_mad=True))
        model_data.append(get_row_vals(df_moehle, d_val, None))
        model_data.append(get_row_vals(df_1n, d_val, None))

        data = np.array(model_data)
        norm_data = np.zeros_like(data)
        for i in range(num_vars):
            col = data[:, i]
            if i in [1, 4]:  # Vol and MDD: Lower is better (Invert)
                best_val = col.min()
                norm_data[:, i] = [best_val / v if v != 0 else 1.0 for v in col]
            else:  # Return, Sharpe, Sortino: Higher is better
                best_val = col.max()
                norm_data[:, i] = [v / best_val if best_val != 0 else 0.0 for v in col]

        for idx, name in enumerate(labels):
            values = norm_data[idx].tolist()
            values += values[:1]
            lw = 4 if color_keys[idx] == 'SOCP' else 2.5
            ax.plot(angles, values, color=GLOBAL_COLORS[color_keys[idx]], linewidth=lw, label=name, zorder=10)
            # ax.fill(angles, values, color=GLOBAL_COLORS[color_keys[idx]], alpha=0.05)

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_yticklabels([])
        ax.set_thetagrids(np.degrees(angles[:-1]), metrics, fontsize=30)  # Slightly reduced font for clarity
        ax.set_ylim(0, 1.1)
        plt.show()


def export_full_results_to_csv(filename="backtest_results_master.csv"):
    """
    Standardizes and combines all model results into a single CSV.
    """
    all_dfs = []

    # Mapping dictionary to standardize column names across different scripts
    # Key = Original Column Name : Value = Standardized Name
    column_mapper = {
        # Standard names (SOCP, Moehle, MV, 1/N)
        'mean_return': 'Mean_Return',
        'std_dev': 'Volatility',
        'sharpe_ratio': 'Sharpe_Ratio',
        'sortino_ratio': 'Sortino_Ratio',
        'max_drawdown': 'Max_Drawdown',
        'Runtime (s)': 'Runtime_Sec',
        'gamma_risk': 'Gamma',
        'delta': 'Delta',

        # DRO-MAD specific names
        'Mean Return': 'Mean_Return',
        'Std Dev': 'Volatility',
        'Sharpe Ratio': 'Sharpe_Ratio',
        'Sortino Ratio': 'Sortino_Ratio',
        'Max Drawdown': 'Max_Drawdown',

        # Classic MAD specific names
        'Mean_MAD': 'Mean_Return',
        'Std_MAD': 'Volatility',
        'Sharpe_MAD': 'Sharpe_Ratio',
        'Sortino_MAD': 'Sortino_Ratio',
        'MDD_MAD': 'Max_Drawdown'
    }

    # List of tuples: (DataFrame, Model Name)
    models_to_process = [
        (df_socp, 'SOCP'),
        (df_mv, 'Classic MV'),
        (df_mad, 'Classic MAD'),
        (df_dro, 'DRO-MAD'),
        (df_moehle, 'Moehle'),
        (df_1n, '1/N Portfolio')
    ]

    for original_df, model_name in models_to_process:
        # Create a copy to avoid modifying the original dataframes used for plotting
        temp_df = original_df.copy()

        # Add Model identifier
        temp_df['Model'] = model_name

        # Rename columns based on the mapper
        temp_df = temp_df.rename(columns=column_mapper)

        # Ensure 'Gamma' column exists (fill with 0 or N/A for models that don't use it)
        if 'Gamma' not in temp_df.columns:
            temp_df['Gamma'] = 0

        # Select only the columns we want in the final CSV
        cols_to_keep = ['Model', 'Delta', 'Gamma', 'Mean_Return', 'Volatility',
                        'Sharpe_Ratio', 'Sortino_Ratio', 'Max_Drawdown', 'Runtime_Sec']

        # Only keep columns that actually exist in the temp_df
        existing_cols = [c for c in cols_to_keep if c in temp_df.columns]
        all_dfs.append(temp_df[existing_cols])

    # Combine everything
    master_df = pd.concat(all_dfs, ignore_index=True)

    # Sort for readability
    master_df = master_df.sort_values(by=['Model', 'Gamma', 'Delta'])

    # Export
    master_df.to_csv(filename, index=False)
    print(f"\n[SUCCESS] Master results exported to: {filename}")
    return master_df

# --- 8. NUMERICAL REPORTS ---
def print_numerical_reports():
    print("\n" + "=" * 100 + "\nDETAILED NUMERICAL COMPARISON\n" + "=" * 100)

    # Calculate total runtimes for the summary table
    summary_runtimes = {
        'SOCP': df_socp[df_socp['gamma_risk'] == 3]['Runtime (s)'].sum(),
        'Classic MV': df_mv['Runtime (s)'].sum(),
        'Classic MAD': df_mad['Runtime (s)'].sum(),
        'DRO-MAD': df_dro['Runtime (s)'].sum(),
        'Moehle': df_moehle['Runtime (s)'].sum(),
        '1/N': df_1n['Runtime (s)'].sum()
    }

    runtime_df = pd.DataFrame(list(summary_runtimes.items()), columns=['Model', 'Total Sweep Time (s)'])
    print(runtime_df.sort_values('Total Sweep Time (s)').to_string(index=False))
    print("=" * 100)


# --- EXECUTE ALL ---
plot_metric('mean', 'Annualized Return', 'Mean Return vs Delta')
plot_metric('std', 'Annualized Std Dev', 'Risk vs Delta')
plot_metric('sharpe', 'Sharpe Ratio', 'Sharpe Ratio vs Delta')
plot_metric('sortino', 'Sortino Ratio', 'Sortino Ratio vs Delta')
plot_metric('mdd', 'Max Drawdown', 'Max Drawdown vs Delta')

# New Runtime Line Graph featuring per-delta values
plot_runtimes_per_delta(delta_bench)

# Spider Charts using local runtime data
generate_individual_spider_plots()

# Summary table
print_numerical_reports()

master_results = export_full_results_to_csv("Portfolio_Optimization_Results.csv")