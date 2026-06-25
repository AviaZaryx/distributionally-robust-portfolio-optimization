import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

# --- 1. IMPORT MODULES ---
# Ensure these .py files are in the same folder as this script
import socp_script
import mv_script
import mad_script
import one_n_script
import dro_mad
import mean_var_w_trd_hld

# --- 2. GLOBAL CONFIGURATION ---
GLOBAL_COLORS = {
    'SOCP': '#000000',
    'Classic MV': '#d62728',
    'Classic MAD': '#ff7f0e',
    'Moehle': '#1f77b4',
    'DRO-MAD': '#1b9e77',
    '1/N': '#984ea3'
}

START_DATE = '2017-01-01'
END_DATE = '2021-12-31'
EST_WIN = 250
PRED_WIN = 21
SCALE_FACTOR = 1.0  # Adjust if your returns are scaled (e.g., 100 for percent)


def load_data(csv_path):
    try:
        df = pd.read_csv(csv_path, parse_dates=['Date'])
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return pd.DataFrame()

    mask = (df['Date'] >= START_DATE) & (df['Date'] <= END_DATE)
    df = df.loc[mask].sort_values('Date')
    df_adj = df.pivot(index='Date', columns='Ticker', values='Adj Close').sort_index()
    df_adj = df_adj.dropna(axis=1, how='any')
    df_adj = df_adj.dropna(how='any')

    if df_adj.empty:
        return pd.DataFrame()
    return df_adj


# --- 3. EXECUTE MODELS & BUILD BIG MATRIX ---
csv_file = r'D:\Downloads\pythonProject1\merged_stock_data_half.csv'
data = load_data(csv_file)

all_results_list = []

# Define your parameter ranges here
deltas = [0, 0.05, 0.1]
gammas = [0, 5, 10]

for delta in deltas:
    for gamma in gammas:
        print(f">>> Running Models for Delta: {delta}, Gamma: {gamma}...")

        # Execute each script
        # Note: Ensure these functions return a DataFrame with at least one row
        res_socp = socp_script.run_socp(data, d=delta, g=gamma)
        res_mv = mv_script.run_mv(data, d=delta, FIXED_GAMMA=gamma)
        res_mad = mad_script.run_mad(data, delta=delta, GAMMA_RISK=gamma)
        res_1n = one_n_script.run_1n(data, d=delta, FIXED_GAMMA=gamma)
        res_dro_mad = dro_mad.run_dro_mad(data, delta=delta, FIXED_GAMMA_FOR_CONSISTENCY=gamma)
        res_moehle = mean_var_w_trd_hld.run_moehle_paper(data, d=delta, g=gamma)

        temp_list = [res_socp, res_mv, res_mad, res_1n, res_dro_mad, res_moehle]
        labels = ["SOCP", "Classic MV", "Classic MAD", "1/N", "DRO-MAD", "Moehle"]

        for df, name in zip(temp_list, labels):
            if df is not None and not df.empty:
                df = df.copy()
                df.columns = df.columns.str.strip()

                # --- UNIFY COLUMN NAMES ---
                # This mapping converts various script outputs to your target naming convention
                rename_map = {
                    'mean_return': 'Mean Return', 'Mean_MAD': 'Mean Return',
                    'std_dev': 'STD DEV', 'Std_MAD': 'STD DEV', 'Std Dev': 'STD DEV',
                    'sharpe_ratio': 'Sharpe Ratio', 'Sharpe_MAD': 'Sharpe Ratio',
                    'sortino_ratio': 'Sortino Ratio', 'Sortino_MAD': 'Sortino Ratio',
                    'max_drawdown': 'Max Drawdown', 'MDD_MAD': 'Max Drawdown'
                }
                df = df.rename(columns=rename_map)

                # Assign loop parameters to the row
                df['Model_Type'] = name
                df['Delta'] = delta
                df['Gamma'] = gamma

                # Keep only the unified columns we care about
                required_cols = ['Model_Type', 'Delta', 'Gamma', 'Mean Return',
                                 'STD DEV', 'Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown']
                # Only keep columns that actually exist
                existing_cols = [c for c in required_cols if c in df.columns]
                all_results_list.append(df[existing_cols])

# Create the Final Big Matrix
big_matrix = pd.concat(all_results_list, ignore_index=True)
print("\n--- FINAL COMBINED RESULTS ---")
print(big_matrix.head())
big_matrix.to_csv("portfolio_optimization_results.csv", index=False)


# --- 4. SENSITIVITY ANALYSIS FOR LINE STYLES ---
def calculate_model_line_styles(df_all):
    """
    Determines line styles based on parameter sensitivity:
    - Dash ('--'): Not change by delta
    - Dotted (':'): No change by gamma
    - Solid ('-'): Change in all (or doesn't fit the specific invariant cases)
    """
    styles = {}
    metrics = ['Mean Return', 'STD DEV', 'Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown']
    models = df_all['Model_Type'].unique()

    for model in models:
        subset = df_all[df_all['Model_Type'] == model]

        # Check if it changes with Delta (group by Gamma and see if metrics vary across Deltas)
        # We sum the standard deviations; if it's 0, it's invariant.
        delta_variance = subset.groupby('Gamma')[metrics].std().sum().sum()
        # Check if it changes with Gamma (group by Delta and see if metrics vary across Gammas)
        gamma_variance = subset.groupby('Delta')[metrics].std().sum().sum()

        is_invariant_delta = delta_variance < 1e-9
        is_invariant_gamma = gamma_variance < 1e-9

        if is_invariant_delta and not is_invariant_gamma:
            styles[model] = '--'  # Dash: not change by delta
        elif is_invariant_gamma and not is_invariant_delta:
            styles[model] = ':'  # Dotted: no change by gamma
        else:
            styles[model] = '-'  # Solid: change in all (or invariant to both, e.g., 1/N)

    return styles


# --- 5. UPDATED GLOBAL SPIDER PLOT FUNCTION ---
def generate_global_spider_plots(df_all):
    # Pre-calculate line styles based on the full dataset sensitivity
    model_line_styles = calculate_model_line_styles(df_all)

    metrics = ['Return', 'Vol', 'Sharpe', 'Sortino', 'MDD']
    num_vars = len(metrics)

    # Calculate Global Best values for normalization
    g_max_ret = df_all['Mean Return'].max()
    g_max_sharpe = df_all['Sharpe Ratio'].max()
    g_max_sortino = df_all['Sortino Ratio'].max()
    g_min_vol = df_all['STD DEV'].min()
    g_min_mdd = df_all['Max Drawdown'].abs().min()

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    unique_deltas = sorted(df_all['Delta'].unique())
    unique_gammas = sorted(df_all['Gamma'].unique())

    for d_val in unique_deltas:
        for g_val in unique_gammas:
            subset = df_all[(df_all['Delta'] == d_val) & (df_all['Gamma'] == g_val)]
            if subset.empty: continue

            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
            fig.patch.set_facecolor('white')
            ax.set_facecolor('white')
            ax.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

            for _, row in subset.iterrows():
                m_name = row['Model_Type']

                norm_vals = [
                    row['Mean Return'] / g_max_ret if g_max_ret != 0 else 0,
                    g_min_vol / row['STD DEV'] if row['STD DEV'] != 0 else 0,
                    row['Sharpe Ratio'] / g_max_sharpe if g_max_sharpe != 0 else 0,
                    row['Sortino Ratio'] / g_max_sortino if g_max_sortino != 0 else 0,
                    g_min_mdd / abs(row['Max Drawdown']) if row['Max Drawdown'] != 0 else 0
                ]
                values = norm_vals + [norm_vals[0]]

                # Get visual properties
                lw = 4 if m_name == 'SOCP' else 2.5
                color = GLOBAL_COLORS.get(m_name, '#CCCCCC')

                # --- APPLY NEW LINESTYLE LOGIC ---
                lstyle = model_line_styles.get(m_name, '-')

                ax.plot(angles, values, color=color, linewidth=lw,
                        linestyle=lstyle, label=m_name, zorder=10)

            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            ax.set_yticklabels([])
            ax.set_thetagrids(np.degrees(angles[:-1]), metrics, fontsize=20)  # Adjusted size for readability
            ax.set_ylim(0, 1.1)

            # plt.title(f"Radar Chart: Delta={d_val}, Gamma={g_val}", size=15, y=1.1)
            # Optional: Add a legend to verify styles
            # plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
            plt.tight_layout()
            plt.show()


# --- 6. RUN ---
# Assuming big_matrix is already generated from your loop
generate_global_spider_plots(big_matrix)