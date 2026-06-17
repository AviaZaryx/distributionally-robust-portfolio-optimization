import numpy as np
import pandas as pd
import cvxpy as cp
import warnings
import time

warnings.filterwarnings('ignore')

# ---------------- CONFIG ----------------
SCALE_FACTOR = 1
CSV_FILE = 'combined_all_stocks_cleaned.csv'
START_DATE = '2017-01-01'
END_DATE = '2021-12-31'
EST_WIN = 250
PRED_WIN = 21

# --- RESCALE LOGIC ---
DELTA_RESCALE_FACTOR = 1


# --- 1. Data Loading ---
def load_data(csv_path):
    try:
        df = pd.read_csv(csv_path, parse_dates=['Date'])
    except:
        return pd.DataFrame()
    mask = (df['Date'] >= START_DATE) & (df['Date'] <= END_DATE)
    df = df.loc[mask].sort_values('Date')
    df_adj = df.pivot(index='Date', columns='Ticker', values='Adj Close').sort_index()
    df_adj = df_adj.dropna(axis=1, how='any')
    df_adj = df_adj.dropna(how='any')
    if df_adj.empty:
        return pd.DataFrame()
    return df_adj / df_adj.iloc[0]


# --- 2. Model: DRO-MAD (Theorem 1 Math) ---
def optimize_dromad(returns, epsilon, rho_target, use_custom_tol=False, custom_tol=1e-7):
    n = returns.shape[1]
    T = returns.shape[0]
    mu = returns.mean().values
    xi = returns.values

    # Variables
    x = cp.Variable(n)
    y1 = cp.Variable(T)
    y2 = cp.Variable(T)
    t = cp.Variable()

    base_con = [cp.sum(x) == 1, x >= 0]

    # LP 1: Case where robust mean is above target
    c1 = base_con + [
        y1 >= mu @ x - xi @ x - epsilon,
        y1 >= -mu @ x + xi @ x + epsilon,
        y2 >= mu @ x - xi @ x + epsilon,
        y2 >= -mu @ x + xi @ x - epsilon,
        mu @ x - epsilon >= rho_target,
        t >= cp.sum(y1) / T + epsilon,
        t >= cp.sum(y2) / T + epsilon
    ]
    prob1 = cp.Problem(cp.Minimize(t), c1)

    # LP 2: Case where target return is within uncertainty bounds
    c2 = base_con + [
        y1 >= xi @ x - rho_target,
        y1 >= -xi @ x + rho_target,
        y2 >= mu @ x - xi @ x + epsilon,
        y2 >= -mu @ x + xi @ x - epsilon,
        mu @ x - epsilon <= rho_target,
        mu @ x + epsilon >= rho_target,
        t >= cp.sum(y1) / T + epsilon,
        t >= cp.sum(y2) / T + epsilon
    ]
    prob2 = cp.Problem(cp.Minimize(t), c2)

    best_obj = np.inf
    best_x = np.ones(n) / n
    final_status = "infeasible"

    # Solver kwargs based on toggle
    solver_kwargs = {'solver': cp.ECOS}
    if use_custom_tol:
        solver_kwargs.update({
            'abstol': custom_tol,
            'reltol': custom_tol,
            'feastol': custom_tol
        })

    # Solve Both and find the best
    for p in [prob1, prob2]:
        try:
            p.solve(**solver_kwargs)
            if p.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                if p.value < best_obj:
                    best_obj = p.value
                    best_x = x.value
                    final_status = p.status
            elif final_status == "infeasible":
                final_status = p.status
        except:
            pass

    res = np.maximum(best_x, 0)
    return res / (res.sum() + 1e-8), final_status


# --- 3. Backtest Engine ---
def run_dro_mad(csv=CSV_FILE, use_custom_tol=False, custom_tol=1e-7):
    data = load_data(csv)
    if data.empty:
        return pd.DataFrame()

    delta_range = [0, 0.005, 0.01]
    rho_target = 0.0012
    FIXED_GAMMA_FOR_CONSISTENCY = 0.5

    results = []
    statuses = []

    for d in delta_range:
        # START TIMER FOR THIS DELTA
        delta_start_time = time.time()

        rets = []
        stats_summary = []
        internal_epsilon = d * DELTA_RESCALE_FACTOR

        for i in range(EST_WIN, len(data) - PRED_WIN, PRED_WIN):
            est_data = data.iloc[i - EST_WIN:i].pct_change().dropna()
            pred_data = data.iloc[i:i + PRED_WIN].pct_change().dropna()

            if est_data.empty: continue

            # 1. Run Optimization and get status (Pass through tolerance settings)
            w, status = optimize_dromad(est_data, internal_epsilon, rho_target, use_custom_tol, custom_tol)
            stats_summary.append(status)

            # 2. Portfolio realization
            rets.extend(pred_data.values @ w)

        if not rets: continue

        # END TIMER FOR THIS DELTA
        delta_end_time = time.time()
        delta_runtime = delta_end_time - delta_start_time

        # Print solver status summary for this delta
        unique_statuses = pd.Series(stats_summary).value_counts()
        statuses.append(unique_statuses)

        # Stats Calculation
        s = pd.Series(rets)
        mean_ann = s.mean() * 252
        std_ann = s.std() * np.sqrt(252)
        sharpe = mean_ann / std_ann if std_ann > 0 else 0
        sortino = mean_ann / (s[s < 0].std() * np.sqrt(252)) if (s[s < 0].std() > 0) else 0
        mdd = ((1 + s).cumprod() / (1 + s).cumprod().cummax() - 1).min()

        results.append({
            'Delta': d,
            'Runtime (s)': delta_runtime,  # Added runtime here
            'gamma_risk': FIXED_GAMMA_FOR_CONSISTENCY,
            'Mean Return': mean_ann / SCALE_FACTOR,
            'Std Dev': std_ann / SCALE_FACTOR,
            'Sharpe Ratio': sharpe,
            'Sortino Ratio': sortino,
            'Max Drawdown': mdd
        })

    if __name__ == "__main__":
        print("\n--- ALL DELTA STATUSES ---")
        print(statuses)

    return pd.DataFrame(results)


if __name__ == "__main__":
    # To use custom tolerance, set use_custom_tol=True and specify custom_tol
    df = run_dro_mad(use_custom_tol=False, custom_tol=1e-7)
    print("\n--- FINAL NUMERICAL RESULTS ---")
    print(df.to_string(index=False))