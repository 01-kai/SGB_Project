# sgb/experiments.py

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from sgb.model import SGBModel
from sgb.metrics import (
    compute_gmi,
    compute_egmi,
    confidence_interval_95,
    compute_recovery_time,
)
from sgb.perturbations import (
    apply_security_incident,
    apply_regulatory_change,
    apply_dq_shock,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "sgb" / "config.yaml"

RAW_DIR = PROJECT_ROOT / "outputs" / "raw"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

FRAMEWORKS = ["DBL", "DMM", "ISF"]


def ensure_output_dirs():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_agents_by_uid(model):
    return {agent.uid: agent for agent in model.agents}


def get_agent_gmis(model, active_uids=None):
    values = {}

    for agent in model.agents:
        if active_uids is not None and agent.uid not in active_uids:
            continue

        values[agent.uid] = compute_gmi(
            agent.dimensions,
            model.framework_name,
        )

    return values


def custom_collect_metrics(model, active_uids=None):
    active_agents = []

    for agent in model.agents:
        if active_uids is None or agent.uid in active_uids:
            active_agents.append(agent)

    agent_gmis = [
        compute_gmi(agent.dimensions, model.framework_name)
        for agent in active_agents
    ]

    egmi = compute_egmi(agent_gmis)

    recent_events = [
        event for event in model.event_records
        if event["step"] == model.current_step
    ]

    if recent_events:
        successful_events = sum(event["success"] for event in recent_events)
        total_events = len(recent_events)
        success_rate = successful_events / total_events
    else:
        successful_events = 0
        total_events = 0
        success_rate = 0.0

    row = {
        "step": model.current_step,
        "framework": model.framework_name,
        "E_GMI": egmi,
        "success_rate": success_rate,
        "successful_events": successful_events,
        "total_events": total_events,
    }

    for dimension in ["D1", "D2", "D3", "D4", "D5", "SVC"]:
        values = [
            agent.dimensions[dimension]
            for agent in active_agents
            if dimension in agent.dimensions
        ]

        row[dimension] = float(np.mean(values)) if values else np.nan

    model.records.append(row)


def run_active_step(model, active_uids):
    agents_by_uid = get_agents_by_uid(model)

    active_agents = [
        agent for agent in model.agents
        if agent.uid in active_uids
    ]

    model.rng.shuffle(active_agents)

    for agent in active_agents:
        active_neighbors = [
            uid for uid in model.graph.neighbors(agent.uid)
            if uid in active_uids
        ]

        if active_neighbors:
            number_of_events = max(1, int(agent.exchange_volume / 500))
            sample_size = min(number_of_events, len(active_neighbors))

            selected_uids = model.rng.choice(
                active_neighbors,
                size=sample_size,
                replace=False,
            )

            for partner_uid in selected_uids:
                partner = agents_by_uid[int(partner_uid)]
                event = model.exchange(agent, partner)
                agent.update_dimensions(event)

        agent.recover()
        agent.record_gmi()

    custom_collect_metrics(model, active_uids=active_uids)
    model.current_step += 1


def format_mean_ci(values, decimals=3):
    mean, ci = confidence_interval_95(values)
    return f"{mean:.{decimals}f} ± {ci:.{decimals}f}"


def anova_pvalue(df, metric, group_col="framework"):
    groups = []

    for _, group in df.groupby(group_col):
        values = group[metric].dropna().values

        if len(values) > 1:
            groups.append(values)

    if len(groups) < 2:
        return np.nan

    _, p_value = stats.f_oneway(*groups)
    return p_value


def format_pvalue(p_value):
    if pd.isna(p_value):
        return "NA"

    if p_value < 0.001:
        return "p < 0.001"

    return f"p = {p_value:.3f}"


def run_experiment_1(config):
    """
    Experiment 1: Framework Comparison.

    No perturbation is applied here.
    This experiment compares DBL, DMM, and ISF under normal conditions.
    """

    ensure_output_dirs()

    seeds = range(config["simulation"]["seeds"])
    total_steps = config["simulation"]["T"]

    all_results = []
    summary_rows = []

    total_runs = len(list(seeds)) * len(FRAMEWORKS)
    run_counter = 0

    for seed in seeds:
        for framework in FRAMEWORKS:
            run_counter += 1

            print(
                f"[Experiment 1] Run {run_counter}/{total_runs} | seed={seed} | framework={framework}",
                flush=True,
            )

            model = SGBModel(
                config=config,
                framework_name=framework,
                seed=seed,
            )

            for _ in range(total_steps):
                model.step()

            result_df = model.get_results()
            result_df["seed"] = seed
            result_df["framework"] = framework

            all_results.append(result_df)

            steady_state_df = result_df[result_df["step"] >= total_steps - 50]

            summary_rows.append({
                "seed": seed,
                "framework": framework,
                "steady_E_GMI": steady_state_df["E_GMI"].mean(),
                "success_rate": steady_state_df["success_rate"].mean(),
                "D1": steady_state_df["D1"].mean(),
                "D4": steady_state_df["D4"].mean(),
            })

    raw_df = pd.concat(all_results, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)

    raw_path = RAW_DIR / "exp1_results.csv"
    summary_path = RAW_DIR / "exp1_summary_by_seed.csv"
    table_path = TABLE_DIR / "table2_framework_comparison.csv"

    raw_df.to_csv(raw_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    table_rows = []

    metrics = [
        ("Steady-State E-GMI", "steady_E_GMI", 3),
        ("Exchange Success Rate", "success_rate", 3),
        ("D1 Data Quality mean score", "D1", 3),
        ("D4 Regulatory Compliance mean score", "D4", 3),
    ]

    for label, metric, decimals in metrics:
        row = {"Metric": label}

        for framework in FRAMEWORKS:
            values = summary_df[summary_df["framework"] == framework][metric]
            row[framework] = format_mean_ci(values, decimals=decimals)

        p_value = anova_pvalue(summary_df, metric)
        row["Statistical Sig."] = format_pvalue(p_value)

        table_rows.append(row)

    table_df = pd.DataFrame(table_rows)
    table_df.to_csv(table_path, index=False)

    print("Experiment 1 complete.")
    print(f"Raw results saved to: {raw_path}")
    print(f"Summary saved to: {summary_path}")
    print(f"Table 2 saved to: {table_path}")


def run_experiment_2(config):
    """
    Experiment 2: Failure Propagation.

    A Security Incident is injected at t = 100 into the highest-degree node.
    """

    ensure_output_dirs()

    seeds = range(config["simulation"]["seeds"])
    total_steps = config["simulation"]["T"]

    shock_config = config["perturbations"]["security_incident"]
    shock_step = shock_config["step"]
    min_impact = shock_config["min_impact"]
    max_impact = shock_config["max_impact"]
    observation_window = shock_config["observation_window"]

    all_results = []
    summary_rows = []

    total_runs = len(list(seeds)) * len(FRAMEWORKS)
    run_counter = 0

    for seed in seeds:
        for framework in FRAMEWORKS:
            run_counter += 1

            print(
                f"[Experiment 2] Run {run_counter}/{total_runs} | seed={seed} | framework={framework}",
                flush=True,
            )

            model = SGBModel(
                config=config,
                framework_name=framework,
                seed=seed,
            )

            target_uid = max(model.graph.degree, key=lambda item: item[1])[0]
            baseline_agent_gmis = None

            for step in range(total_steps):
                if step == shock_step:
                    baseline_agent_gmis = get_agent_gmis(model)

                    impact = float(model.rng.uniform(min_impact, max_impact))

                    apply_security_incident(
                        model=model,
                        target_uid=target_uid,
                        impact=impact,
                    )

                model.step()

            result_df = model.get_results()
            result_df["seed"] = seed
            result_df["framework"] = framework
            result_df["target_uid"] = target_uid

            all_results.append(result_df)

            if baseline_agent_gmis is None:
                baseline_agent_gmis = get_agent_gmis(model)

            agents_by_uid = get_agents_by_uid(model)
            affected_uids = []

            for uid, baseline_gmi in baseline_agent_gmis.items():
                agent = agents_by_uid[uid]

                start = shock_step
                end = min(
                    shock_step + observation_window,
                    len(agent.gmi_history),
                )

                post_values = agent.gmi_history[start:end]

                if not post_values:
                    continue

                min_post_gmi = min(post_values)

                if min_post_gmi < baseline_gmi - 0.02:
                    affected_uids.append(uid)

            if affected_uids:
                distances = []

                for uid in affected_uids:
                    try:
                        distance = get_shortest_path_distance(
                            model.graph,
                            target_uid,
                            uid,
                        )
                        distances.append(distance)
                    except Exception:
                        pass

                max_cascade_depth = max(distances) if distances else 0
            else:
                max_cascade_depth = 0

            fraction_affected = len(affected_uids) / model.N

            recovery_time = compute_recovery_time(
                series=result_df["E_GMI"],
                shock_step=shock_step,
                baseline_window=20,
                tolerance=0.02,
            )

            summary_rows.append({
                "seed": seed,
                "framework": framework,
                "target_uid": target_uid,
                "max_cascade_depth": max_cascade_depth,
                "fraction_agents_affected": fraction_affected,
                "recovery_time": recovery_time,
            })

    raw_df = pd.concat(all_results, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)

    raw_path = RAW_DIR / "exp2_results.csv"
    summary_path = RAW_DIR / "exp2_summary_by_seed.csv"
    table_path = TABLE_DIR / "table3_propagation_metrics.csv"

    raw_df.to_csv(raw_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    table_rows = []

    table_metrics = [
        ("Max Cascade Depth", "max_cascade_depth"),
        ("Fraction of Agents Affected", "fraction_agents_affected"),
        ("Recovery Time", "recovery_time"),
    ]

    for label, metric in table_metrics:
        row = {"Metric": label}

        for framework in FRAMEWORKS:
            values = summary_df[summary_df["framework"] == framework][metric].dropna()
            row[framework] = format_mean_ci(values, decimals=3)

        p_value = anova_pvalue(summary_df, metric)
        row["Statistical Sig."] = format_pvalue(p_value)

        table_rows.append(row)

    table_df = pd.DataFrame(table_rows)
    table_df.to_csv(table_path, index=False)

    print("Experiment 2 complete.")
    print(f"Raw results saved to: {raw_path}")
    print(f"Summary saved to: {summary_path}")
    print(f"Table 3 saved to: {table_path}")


def get_shortest_path_distance(graph, source, target):
    import networkx as nx

    return nx.shortest_path_length(
        graph,
        source=source,
        target=target,
    )


def run_experiment_3(config):
    """
    Experiment 3: NDEL Onboarding Threshold.

    Thresholds from 0.3 to 0.8 are tested.
    Only agents with initial GMI >= threshold are admitted.
    Regulatory Change is injected at t = 100.
    """

    ensure_output_dirs()

    seeds = range(config["simulation"]["seeds"])
    total_steps = config["simulation"]["T"]

    thresholds = config["experiment_3"]["thresholds"]

    shock_config = config["perturbations"]["regulatory_change"]
    shock_step = shock_config["step"]
    min_impact = shock_config["min_impact"]
    max_impact = shock_config["max_impact"]

    summary_rows = []

    total_runs = len(thresholds) * len(list(seeds)) * len(FRAMEWORKS)
    run_counter = 0

    for threshold in thresholds:
        for seed in seeds:
            for framework in FRAMEWORKS:
                run_counter += 1

                print(
                    f"[Experiment 3] Run {run_counter}/{total_runs} | "
                    f"threshold={threshold} | seed={seed} | framework={framework}",
                    flush=True,
                )

                model = SGBModel(
                    config=config,
                    framework_name=framework,
                    seed=seed,
                )

                initial_gmis = get_agent_gmis(model)

                active_uids = {
                    uid for uid, gmi in initial_gmis.items()
                    if gmi >= threshold
                }

                admitted_fraction = len(active_uids) / model.N

                if len(active_uids) == 0:
                    summary_rows.append({
                        "threshold": threshold,
                        "seed": seed,
                        "framework": framework,
                        "admitted_fraction": admitted_fraction,
                        "steady_E_GMI": np.nan,
                        "recovery_time": np.nan,
                        "success_rate": np.nan,
                    })
                    continue

                for step in range(total_steps):
                    if step == shock_step:
                        impact = float(model.rng.uniform(min_impact, max_impact))

                        apply_regulatory_change(
                            model=model,
                            impact=impact,
                        )

                    run_active_step(
                        model=model,
                        active_uids=active_uids,
                    )

                result_df = model.get_results()
                steady_state_df = result_df[result_df["step"] >= total_steps - 50]

                recovery_time = compute_recovery_time(
                    series=result_df["E_GMI"],
                    shock_step=shock_step,
                    baseline_window=20,
                    tolerance=0.02,
                )

                summary_rows.append({
                    "threshold": threshold,
                    "seed": seed,
                    "framework": framework,
                    "admitted_fraction": admitted_fraction,
                    "steady_E_GMI": steady_state_df["E_GMI"].mean(),
                    "recovery_time": recovery_time,
                    "success_rate": steady_state_df["success_rate"].mean(),
                })

    summary_df = pd.DataFrame(summary_rows)

    raw_path = RAW_DIR / "exp3_results.csv"
    table_path = TABLE_DIR / "table4_recommended_threshold.csv"

    summary_df.to_csv(raw_path, index=False)

    table_df = build_threshold_recommendation_table(summary_df)
    table_df.to_csv(table_path, index=False)

    print("Experiment 3 complete.")
    print(f"Raw results saved to: {raw_path}")
    print(f"Table 4 saved to: {table_path}")


def build_threshold_recommendation_table(summary_df):
    rows = []

    for framework, group in summary_df.groupby("framework"):
        grouped = group.groupby("threshold").agg({
            "admitted_fraction": "mean",
            "steady_E_GMI": "mean",
            "recovery_time": "mean",
            "success_rate": "mean",
        }).reset_index()

        valid = grouped.dropna().copy()

        if valid.empty:
            continue

        valid["coverage_score"] = normalize_positive(valid["admitted_fraction"])
        valid["egmi_score"] = normalize_positive(valid["steady_E_GMI"])
        valid["recovery_score"] = normalize_negative(valid["recovery_time"])

        valid["policy_score"] = (
            0.40 * valid["coverage_score"]
            + 0.35 * valid["egmi_score"]
            + 0.25 * valid["recovery_score"]
        )

        best = valid.sort_values("policy_score", ascending=False).iloc[0]

        rows.append({
            "Framework": framework,
            "Recommended Threshold": best["threshold"],
            "Admitted Fraction": best["admitted_fraction"],
            "Steady-State E-GMI": best["steady_E_GMI"],
            "Recovery Time": best["recovery_time"],
            "Success Rate": best["success_rate"],
            "Policy Score": best["policy_score"],
        })

    return pd.DataFrame(rows)


def normalize_positive(series):
    min_value = series.min()
    max_value = series.max()

    if max_value == min_value:
        return pd.Series([1.0] * len(series), index=series.index)

    return (series - min_value) / (max_value - min_value)


def normalize_negative(series):
    min_value = series.min()
    max_value = series.max()

    if max_value == min_value:
        return pd.Series([1.0] * len(series), index=series.index)

    return 1.0 - ((series - min_value) / (max_value - min_value))


def run_experiment_4(config):
    """
    Experiment 4: Resilience-Efficiency Trade-off.

    Maturity distributions are swept from low maturity to high maturity.
    DQ Shock is injected at t = 100.
    """

    ensure_output_dirs()

    seeds = range(config["simulation"]["seeds"])
    total_steps = config["simulation"]["T"]

    beta_profiles = config["experiment_4"]["beta_profiles"]

    shock_config = config["perturbations"]["dq_shock"]
    shock_step = shock_config["step"]
    min_impact = shock_config["min_impact"]
    max_impact = shock_config["max_impact"]

    summary_rows = []

    experiment_frameworks = ["ISF"]

    total_runs = len(beta_profiles) * len(list(seeds)) * len(experiment_frameworks)
    run_counter = 0

    for profile_name, beta_params in beta_profiles.items():
        beta_override = beta_override_all_dimensions(beta_params)

        for seed in seeds:
            for framework in experiment_frameworks:
                run_counter += 1

                print(
                    f"[Experiment 4] Run {run_counter}/{total_runs} | "
                    f"profile={profile_name} | seed={seed} | framework={framework}",
                    flush=True,
                )

                model = SGBModel(
                    config=config,
                    framework_name=framework,
                    seed=seed,
                    beta_override=beta_override,
                )

                for step in range(total_steps):
                    if step == shock_step:
                        impact = float(model.rng.uniform(min_impact, max_impact))

                        apply_dq_shock(
                            model=model,
                            impact=impact,
                        )

                    model.step()

                result_df = model.get_results()
                steady_state_df = result_df[result_df["step"] >= total_steps - 50]

                recovery_time = compute_recovery_time(
                    series=result_df["E_GMI"],
                    shock_step=shock_step,
                    baseline_window=20,
                    tolerance=0.02,
                )

                throughput = result_df["successful_events"].mean()

                summary_rows.append({
                    "profile": profile_name,
                    "beta_alpha": beta_params[0],
                    "beta_beta": beta_params[1],
                    "seed": seed,
                    "framework": framework,
                    "throughput": throughput,
                    "recovery_time": recovery_time,
                    "steady_E_GMI": steady_state_df["E_GMI"].mean(),
                    "success_rate": steady_state_df["success_rate"].mean(),
                })

    summary_df = pd.DataFrame(summary_rows)

    raw_path = RAW_DIR / "exp4_results.csv"
    summary_df.to_csv(raw_path, index=False)

    print("Experiment 4 complete.")
    print(f"Raw results saved to: {raw_path}")


def beta_override_all_dimensions(beta_params):
    return {
        "D1": beta_params,
        "D2": beta_params,
        "D3": beta_params,
        "D4": beta_params,
        "D5": beta_params,
        "SVC": beta_params,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run SGB simulation experiments."
    )

    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        choices=["1", "2", "3", "4", "all"],
        help="Experiment number to run: 1, 2, 3, 4, or all.",
    )

    args = parser.parse_args()
    config = load_config()

    if args.experiment == "1":
        run_experiment_1(config)

    elif args.experiment == "2":
        run_experiment_2(config)

    elif args.experiment == "3":
        run_experiment_3(config)

    elif args.experiment == "4":
        run_experiment_4(config)

    elif args.experiment == "all":
        run_experiment_1(config)
        run_experiment_2(config)
        run_experiment_3(config)
        run_experiment_4(config)


if __name__ == "__main__":
    main()