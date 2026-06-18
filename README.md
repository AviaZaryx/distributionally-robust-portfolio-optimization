# Portfolio Optimization and Efficiency Suite

This repository provides a framework for backtesting and comparing various portfolio optimization models. It evaluates traditional strategies against modern robust methods across a range of turnover constraints (Delta).

## Directory Structure

*   **base/**: Contains the core optimization models including Mean-Variance, MAD, SOCP, and Wasserstein DRO. These scripts are configured to sweep through various Delta constraints.
*   **analyze/**: Contains scripts focused on computational efficiency and performance analysis of the optimization algorithms.
*   **processing/**: Utility folder for data management, including data generation, cleaning, ticker removal, and dataset formatting.
*   **gamma/**: Model variations that specifically explore the risk-aversion parameter (Gamma).
*   **run.py**: The central execution script. It imports the modules, runs the backtests for all models, generates performance visualizations, and exports results to CSV.

## High-Level Workflow

1.  **Data Preparation**: Data is handled through scripts in the processing directory to ensure clean, consistent input for the solvers.
2.  **Execution**: run.py calls the model functions located in the base directory. Each model processes a range of Delta values to simulate different turnover levels.
3.  **Analysis**: The suite measures standard financial metrics (Mean Return, Volatility, Sharpe, etc.) alongside computational metrics (Execution Time).
4.  **Visualization**: The system generates line charts for metric trends, execution time comparisons, and spider charts for multi-dimensional performance snapshots.

## Usage

To execute the entire pipeline—from running the models to generating the final reports and plots—run the main script:

```bash
python run.py
