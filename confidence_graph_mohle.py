import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- CONFIG ---
CSV_RUNS_INPUT = r'D:\Downloads\pythonProject1\mohle_sampled_run.csv'


def plot_confidence_intervals():
    if not os.path.exists(CSV_RUNS_INPUT):
        print(f"Error: {CSV_RUNS_INPUT} not found. Run the simulation first.")
        return

    # 1. Load the data
    df = pd.read_csv(CSV_RUNS_INPUT)

    # 2. Set the visual style
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))

    # 3. Create the line plot
    # ci=95 tells seaborn to calculate and shade the 95% confidence interval
    # markers=True adds dots at each N value
    ax = sns.lineplot(
        data=df,
        x='N',
        y='Total Time',
        marker='o',
        color='blue',
        errorbar=('ci', 95),
        label='Mean Total Time (95% CI Shaded)'
    )

    # 4. Customizing the labels
    plt.title('Scaling Analysis: Moehle Optimization Time vs Portfolio Size (N)', fontsize=14)
    plt.xlabel('Number of Assets (N)', fontsize=12)
    plt.ylabel('Total Execution Time (Seconds)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Force X-axis to show the specific N values we tested
    plt.xticks(df['N'].unique())

    # 5. Optional: Plot individual components (e.g., Solver Time) to see what causes variance
    # sns.lineplot(data=df, x='N', y='Solver (Math)', marker='s', color='red', errorbar=('ci', 95), label='Solver Math Time')

    plt.legend()

    # Save the plot
    output_plot = CSV_RUNS_INPUT.replace('.csv', '.png')
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    print(f"Graph saved to: {output_plot}")

    plt.show()


if __name__ == "__main__":
    plot_confidence_intervals()