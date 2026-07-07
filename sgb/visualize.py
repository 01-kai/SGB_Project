# sgb/visualize.py

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "outputs" / "raw"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            "Run the related experiment first."
        )


def mean_ci_95(series):
    values = series.dropna().astype(float)

    if len(values) == 0:
        return pd.Series({
            "mean": np.nan,
            "ci95": np.nan,
            "count": 0,
        })

    mean = values.mean()

    if len(values) < 2:
        ci95 = 0.0
    else:
        ci95 = 1.96 * values.std(ddof=1) / np.sqrt(len(values))

    return pd.Series({
        "mean": mean,
        "ci95": ci95,
        "count": len(values),
    })


def save_figure(fig, filename):
    output_path = FIGURE_DIR / filename

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved figure: {output_path}")


def plot_figure_2_gmi_time_series(shock_step=100):
    input_path = RAW_DIR / "exp2_results.csv"
    require_file(input_path)

    df = pd.read_csv(input_path)

    required_columns = {"step", "framework", "seed", "E_GMI"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns in exp2_results.csv: {missing}")

    grouped = (
        df
        .groupby(["framework", "step"])["E_GMI"]
        .apply(mean_ci_95)
        .reset_index()
    )

    grouped = grouped.pivot_table(
        index=["framework", "step"],
        columns="level_2",
        values="E_GMI",
    ).reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))

    for framework, group in grouped.groupby("framework"):
        group = group.sort_values("step")

        x = group["step"].astype(float).to_numpy()
        y = group["mean"].astype(float).to_numpy()
        ci = group["ci95"].astype(float).to_numpy()

        ax.plot(
            x,
            y,
            label=framework,
            linewidth=2,
        )

        ax.fill_between(
            x,
            y - ci,
            y + ci,
            alpha=0.15,
        )

    ax.axvline(
        shock_step,
        linestyle="--",
        linewidth=1.5,
        label="Security Incident at t=100",
    )

    ax.set_title("Figure 2. GMI Time Series Before and After Security Incident")
    ax.set_xlabel("Simulation Step")
    ax.set_ylabel("Ecosystem GMI")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_figure(fig, "figure2_gmi_time_series.png")


def plot_figure_3_threshold_tradeoff():
    input_path = RAW_DIR / "exp3_results.csv"
    require_file(input_path)

    df = pd.read_csv(input_path)

    required_columns = {
        "threshold",
        "framework",
        "admitted_fraction",
        "recovery_time",
        "steady_E_GMI",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns in exp3_results.csv: {missing}")

    grouped = (
        df
        .groupby(["framework", "threshold"])
        .agg({
            "admitted_fraction": "mean",
            "recovery_time": "mean",
            "steady_E_GMI": "mean",
        })
        .reset_index()
    )

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    for framework, group in grouped.groupby("framework"):
        group = group.sort_values("threshold")

        x = group["threshold"].astype(float)

        ax1.plot(
            x,
            group["admitted_fraction"].astype(float),
            marker="o",
            linewidth=2,
            label=f"{framework} coverage",
        )

        ax2.plot(
            x,
            group["recovery_time"].astype(float),
            marker="s",
            linestyle="--",
            linewidth=2,
            label=f"{framework} recovery",
        )

    ax1.set_title("Figure 3. Threshold–Resilience–Coverage Trade-off")
    ax1.set_xlabel("NDEL Onboarding GMI Threshold")
    ax1.set_ylabel("Admitted Fraction")
    ax2.set_ylabel("Recovery Time")
    ax1.set_ylim(0, 1.05)

    ax1.grid(True, alpha=0.3)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()

    ax1.legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        loc="best",
        fontsize=8,
    )

    save_figure(fig, "figure3_threshold_tradeoff.png")


def compute_pareto_front(df):
    valid = df.dropna(subset=["throughput", "recovery_time"]).copy()

    if valid.empty:
        return valid

    valid = valid.sort_values(
        by=["throughput", "recovery_time"],
        ascending=[False, True],
    )

    pareto_rows = []
    best_recovery_so_far = float("inf")

    for _, row in valid.iterrows():
        recovery_time = row["recovery_time"]

        if recovery_time < best_recovery_so_far:
            pareto_rows.append(row)
            best_recovery_so_far = recovery_time

    return pd.DataFrame(pareto_rows)


def plot_figure_4_pareto_front():
    input_path = RAW_DIR / "exp4_results.csv"
    require_file(input_path)

    df = pd.read_csv(input_path)

    required_columns = {
        "profile",
        "throughput",
        "recovery_time",
        "steady_E_GMI",
        "success_rate",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns in exp4_results.csv: {missing}")

    grouped = (
        df
        .groupby("profile")
        .agg({
            "throughput": "mean",
            "recovery_time": "mean",
            "steady_E_GMI": "mean",
            "success_rate": "mean",
        })
        .reset_index()
    )

    pareto_df = compute_pareto_front(grouped)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(
        grouped["throughput"],
        grouped["recovery_time"],
        s=90,
        label="Maturity profile",
    )

    for _, row in grouped.iterrows():
        ax.annotate(
            row["profile"],
            (
                row["throughput"],
                row["recovery_time"],
            ),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=9,
        )

    if not pareto_df.empty:
        pareto_df = pareto_df.sort_values("throughput")

        ax.plot(
            pareto_df["throughput"],
            pareto_df["recovery_time"],
            marker="o",
            linewidth=2,
            label="Pareto front",
        )

    ax.set_title("Figure 4. Pareto Front: Exchange Throughput vs. Recovery Time")
    ax.set_xlabel("Exchange Throughput: Successful Events per Step")
    ax.set_ylabel("Recovery Time After DQ Shock")

    ax.grid(True, alpha=0.3)
    ax.legend()

    save_figure(fig, "figure4_pareto_front.png")


def create_exp4_summary_table():
    input_path = RAW_DIR / "exp4_results.csv"
    require_file(input_path)

    df = pd.read_csv(input_path)

    grouped = (
        df
        .groupby("profile")
        .agg({
            "throughput": "mean",
            "recovery_time": "mean",
            "steady_E_GMI": "mean",
            "success_rate": "mean",
        })
        .reset_index()
    )

    grouped = grouped.sort_values("throughput")

    output_path = TABLE_DIR / "table5_resilience_efficiency_summary.csv"
    grouped.to_csv(output_path, index=False)

    print(f"Saved table: {output_path}")


def main():
    print("Generating SGB figures...")

    plot_figure_2_gmi_time_series(shock_step=100)
    plot_figure_3_threshold_tradeoff()
    plot_figure_4_pareto_front()
    create_exp4_summary_table()

    print("All figures generated.")


if __name__ == "__main__":
    main()