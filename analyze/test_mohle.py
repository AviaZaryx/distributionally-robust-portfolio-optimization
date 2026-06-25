import numpy as np
import pandas as pd
import cvxpy as cp
import time
import os
import csv
import warnings

warnings.filterwarnings('ignore')

# ---------------- CONFIG ----------------
CSV_INPUT = r'D:\Downloads\pythonProject1\merged_stock_data.csv'
CSV_OUTPUT = r'D:\Downloads\pythonProject1\mohle_test_result.csv'

START_DATE = '2017-01-01'
END_DATE = '2021-12-31'
EST_WIN = 250
PRED_WIN = 21
TARGET_N = 140
NUM_SAMPLES = 10

# Financial Parameters
GAMMA_RISK = 10
FIXED_TRD_COST = 0.0001
FIXED_HLD_COST = 0.0001
LIN_TRD_COST = 0.0010

# --- TIMING CATEGORIES ---
# 1: Data Processing, 2: Setup/Compilation, 3: Solver (Math), 4: Post-processing
timers = {"Data": 0, "Setup": 0, "Math": 0, "Post": 0}


def reset_timers():
    for key in timers: timers[key] = 0


# --- OPTIMIZATION ENGINE ---
def optimize_moehle(returns, prev_w):
    t_setup_start = time.perf_counter()

    n = returns.shape[1]
    mu = returns.mean().values
    Sigma = returns.cov().values

    # Regularization
    min_eig = np.min(np.linalg.eigvalsh(Sigma))
    if min_eig < 1e-8: Sigma += (1e-8 - min_eig) * np.eye(n)

    w = cp.Variable(n)
    abs_trade = cp.Variable(n)
    z_trade = cp.Variable(n, boolean=True)
    z_hold = cp.Variable(n, boolean=True)

    risk_term = cp.quad_form(w, Sigma)
    obj = cp.Minimize(-mu @ w + GAMMA_RISK * risk_term +
                      FIXED_TRD_COST * cp.sum(z_trade) +
                      LIN_TRD_COST * cp.sum(abs_trade) +
                      FIXED_HLD_COST * cp.sum(z_hold))

    constraints = [
        cp.sum(w) == 1, w >= 0,
        abs_trade >= w - prev_w, abs_trade >= -(w - prev_w),
        abs_trade <= 1.0 * z_trade, w <= 1.0 * z_hold
    ]
    prob = cp.Problem(obj, constraints)

    # Record Setup Time (Initial part)
    timers["Setup"] += (time.perf_counter() - t_setup_start)

    try:
        t_before_solve = time.perf_counter()
        prob.solve(solver=cp.ECOS_BB)
        t_after_solve = time.perf_counter()

        # Extract pure Math time vs CVXPY Overhead
        solve_math_time = prob.solver_stats.solve_time if prob.solver_stats else (t_after_solve - t_before_solve)
        overhead = (t_after_solve - t_before_solve) - solve_math_time

        timers["Math"] += solve_math_time
        timers["Setup"] += overhead
        status = prob.status
    except:
        status = "Error"

    t_post_start = time.perf_counter()
    res_w = w.value if (w.value is not None and status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]) else np.ones(n) / n
    timers["Post"] += (time.perf_counter() - t_post_start)

    return res_w, status


# --- PROGRESS BAR HELPER ---
def print_progress(iteration, total, prefix='', suffix='', length=30):
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
    if iteration == total: print()


# --- MAIN RUNNER ---
def run_test():
    print(f"Loading data from {CSV_INPUT}...")
    t_data_start = time.perf_counter()
    df = pd.read_csv(CSV_INPUT, parse_dates=['Date'])
    mask = (df['Date'] >= START_DATE) & (df['Date'] <= END_DATE)
    df = df.loc[mask].sort_values('Date')
    df_adj = df.pivot(index='Date', columns='Ticker', values='Adj Close').dropna(axis=1)
    all_tickers = df_adj.columns.tolist()
    initial_data_time = time.perf_counter() - t_data_start

    print(f"Found {len(all_tickers)} tickers. Starting {NUM_SAMPLES} samples of N={TARGET_N}...")

    fieldnames = ['Run', 'Sharpe', 'Ann_Return', 'Max_Drawdown', 'Time_Data', 'Time_Setup', 'Time_Math', 'Time_Post',
                  'Total_Wait']

    with open(CSV_OUTPUT, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for run_idx in range(1, NUM_SAMPLES + 1):
            reset_timers()
            timers["Data"] += initial_data_time  # Account for initial load in first categories

            # Random Selection
            selected = np.random.choice(all_tickers, TARGET_N, replace=False)
            subset = df_adj[selected]

            prev_w = np.zeros(TARGET_N)
            portfolio_returns = []

            # Backtest Loop
            intervals = list(range(EST_WIN, len(subset) - PRED_WIN, PRED_WIN))
            for i, j in enumerate(intervals):
                # Data Phase
                t_d = time.perf_counter()
                est = subset.iloc[j - EST_WIN:j].pct_change().dropna()
                pred = subset.iloc[j:j + PRED_WIN].pct_change().dropna()
                timers["Data"] += (time.perf_counter() - t_d)

                if est.empty: continue

                # Optimization Phase (Setup + Math + Post internal)
                w, status = optimize_moehle(est, prev_w)

                # Final Post-Processing (Returns/Costs)
                t_p = time.perf_counter()
                raw_ret = pred.values @ w
                trade_mag = np.abs(w - prev_w)
                cost = (LIN_TRD_COST * np.sum(trade_mag) +
                        FIXED_TRD_COST * np.sum(trade_mag > 1e-5) +
                        FIXED_HLD_COST * np.sum(w > 1e-5))

                net_ret = raw_ret.copy()
                net_ret[0] -= cost
                portfolio_returns.extend(net_ret)
                prev_w = w
                timers["Post"] += (time.perf_counter() - t_p)

                # Progress indicator for inner loop
                print_progress(i + 1, len(intervals), prefix=f'Run {run_idx}/{NUM_SAMPLES}', suffix=f'Status: {status}')

            # Calculate Performance
            s = pd.Series(portfolio_returns)
            ann_ret = s.mean() * 252
            ann_vol = s.std() * np.sqrt(252)
            sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
            mdd = ((1 + s).cumprod() / (1 + s).cumprod().cummax() - 1).min()

            total_wait = sum(timers.values())

            # Log to CSV
            writer.writerow({
                'Run': run_idx,
                'Sharpe': round(sharpe, 4),
                'Ann_Return': round(ann_ret, 4),
                'Max_Drawdown': round(mdd, 4),
                'Time_Data': round(timers["Data"], 4),
                'Time_Setup': round(timers["Setup"], 4),
                'Time_Math': round(timers["Math"], 4),
                'Time_Post': round(timers["Post"], 4),
                'Total_Wait': round(total_wait, 4)
            })
            f.flush()  # Ensure it writes to disk immediately

    print(f"\nTest Complete. Detailed results saved to: {CSV_OUTPUT}")


if __name__ == "__main__":
    run_test()