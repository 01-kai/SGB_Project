from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from sgb.config import load_config
from sgb.convergence_utils import (
    FRAMEWORK_METRICS,
    ConvergenceConfigurationError,
    ConvergenceError,
    ConvergenceStudyResult,
    detect_block_convergence,
    framework_ranking,
    load_convergence_config,
    run_single_trajectory,
    save_convergence_result,
    summarize_values,
    validate_convergence_config,
)
from sgb.dynamics import load_dynamics_config


def run_convergence_study(
    base_config: Mapping[str, Any],
    dynamics_config: Mapping[str, Any],
    study_config: Mapping[str, Any],
) -> ConvergenceStudyResult:
    """Run the configured convergence-study type."""

    validate_convergence_config(
        study_config
    )

    study_type = str(
        study_config[
            "study_type"
        ]
    )

    if study_type == "time_convergence":
        return run_time_convergence(
            base_config,
            dynamics_config,
            study_config,
        )

    if study_type == "seed_convergence":
        return run_seed_convergence(
            base_config,
            dynamics_config,
            study_config,
        )

    return run_population_convergence(
        base_config,
        dynamics_config,
        study_config,
    )


def run_time_convergence(
    base_config: Mapping[str, Any],
    dynamics_config: Mapping[str, Any],
    study_config: Mapping[str, Any],
) -> ConvergenceStudyResult:
    """Determine whether candidate simulation horizons are sufficient."""

    validate_convergence_config(
        study_config
    )

    metrics = list(
        study_config[
            "metrics"
        ]
    )

    maximum_horizon = int(
        study_config[
            "maximum_horizon"
        ]
    )

    population_size = int(
        study_config[
            "simulation"
        ]["population_size"]
    )

    seed_plan = study_config[
        "seed_plan"
    ]

    seeds = range(
        int(
            seed_plan[
                "start_seed"
            ]
        ),
        int(
            seed_plan[
                "start_seed"
            ]
        )
        + int(
            seed_plan[
                "seed_count"
            ]
        ),
    )

    trajectories: list[
        pd.DataFrame
    ] = []

    for seed in seeds:
        trajectory = (
            run_single_trajectory(
                base_config=base_config,
                dynamics_config=(
                    dynamics_config
                ),
                population_size=(
                    population_size
                ),
                horizon=(
                    maximum_horizon
                ),
                seed=seed,
            )
        )

        selected = trajectory[
            [
                "step",
                *metrics,
            ]
        ].copy()

        selected.insert(
            0,
            "seed",
            seed,
        )

        trajectories.append(
            selected
        )

    replicate_trajectories = (
        pd.concat(
            trajectories,
            ignore_index=True,
        )
    )

    confidence = float(
        study_config[
            "analysis"
        ]["confidence_level"]
    )

    summary_rows: list[
        dict[str, Any]
    ] = []

    for step in sorted(
        replicate_trajectories[
            "step"
        ].unique()
    ):
        step_rows = (
            replicate_trajectories.loc[
                replicate_trajectories[
                    "step"
                ]
                == step
            ]
        )

        for metric in metrics:
            summary_rows.append(
                {
                    "step": int(
                        step
                    ),
                    "metric": metric,
                    **summarize_values(
                        step_rows[
                            metric
                        ],
                        confidence_level=(
                            confidence
                        ),
                    ),
                }
            )

    trajectory_summary = (
        pd.DataFrame(
            summary_rows
        )
    )

    detection = study_config[
        "detection"
    ]

    convergence_rows: list[
        dict[str, Any]
    ] = []

    for metric in metrics:
        values = (
            trajectory_summary.loc[
                trajectory_summary[
                    "metric"
                ]
                == metric
            ]
            .sort_values(
                "step"
            )["mean"]
        )

        convergence_rows.append(
            {
                "metric": metric,
                **detect_block_convergence(
                    values,
                    rolling_window=int(
                        detection[
                            "rolling_window"
                        ]
                    ),
                    required_stable_windows=int(
                        detection[
                            "required_stable_windows"
                        ]
                    ),
                    minimum_step=int(
                        detection[
                            "minimum_step"
                        ]
                    ),
                    slope_tolerance=float(
                        detection[
                            "slope_tolerance"
                        ]
                    ),
                    mean_difference_tolerance=float(
                        detection[
                            "mean_difference_tolerance"
                        ]
                    ),
                ),
            }
        )

    metric_convergence = (
        pd.DataFrame(
            convergence_rows
        )
    )

    all_converged = bool(
        metric_convergence[
            "converged"
        ].all()
    )

    convergence_step = (
        int(
            metric_convergence[
                "convergence_step"
            ].max()
        )
        if all_converged
        else None
    )

    observation_steps = int(
        detection[
            "required_observation_steps"
        ]
    )

    horizon_rows: list[
        dict[str, Any]
    ] = []

    for horizon in study_config[
        "candidate_horizons"
    ]:
        accepted = bool(
            convergence_step
            is not None
            and int(
                horizon
            )
            >= (
                convergence_step
                + observation_steps
            )
        )

        horizon_rows.append(
            {
                "horizon": int(
                    horizon
                ),
                "overall_convergence_step": (
                    convergence_step
                ),
                "required_observation_steps": (
                    observation_steps
                ),
                "observation_steps_after_convergence": (
                    int(horizon)
                    - convergence_step
                    if convergence_step
                    is not None
                    else None
                ),
                "accepted": accepted,
            }
        )

    horizon_summary = (
        pd.DataFrame(
            horizon_rows
        )
    )

    accepted_horizons = (
        horizon_summary.loc[
            horizon_summary[
                "accepted"
            ],
            "horizon",
        ]
    )

    recommended = (
        int(
            accepted_horizons.min()
        )
        if not accepted_horizons.empty
        else None
    )

    return ConvergenceStudyResult(
        study_type=(
            "time_convergence"
        ),
        tables={
            "replicate_trajectories": (
                replicate_trajectories
            ),
            "trajectory_summary": (
                trajectory_summary
            ),
            "metric_convergence": (
                metric_convergence
            ),
            "horizon_summary": (
                horizon_summary
            ),
        },
        metadata={
            "study_type": (
                "time_convergence"
            ),
            "seed_count": int(
                seed_plan[
                    "seed_count"
                ]
            ),
            "population_size": (
                population_size
            ),
            "maximum_horizon": (
                maximum_horizon
            ),
            "all_metrics_converged": (
                all_converged
            ),
            "overall_convergence_step": (
                convergence_step
            ),
            "recommended_horizon": (
                recommended
            ),
            "passed": (
                recommended
                is not None
            ),
        },
    )


def run_seed_convergence(
    base_config: Mapping[str, Any],
    dynamics_config: Mapping[str, Any],
    study_config: Mapping[str, Any],
) -> ConvergenceStudyResult:
    """Measure confidence-interval width as seed count increases."""

    validate_convergence_config(
        study_config
    )

    metrics = list(
        study_config[
            "metrics"
        ]
    )

    counts = [
        int(value)
        for value
        in study_config[
            "candidate_seed_counts"
        ]
    ]

    start_seed = int(
        study_config[
            "seed_plan"
        ]["start_seed"]
    )

    population_size = int(
        study_config[
            "simulation"
        ]["population_size"]
    )

    horizon = int(
        study_config[
            "simulation"
        ]["horizon"]
    )

    terminal_window = int(
        study_config[
            "simulation"
        ]["terminal_window"]
    )

    seed_rows: list[
        dict[str, Any]
    ] = []

    for seed in range(
        start_seed,
        start_seed
        + max(counts),
    ):
        trajectory = (
            run_single_trajectory(
                base_config=base_config,
                dynamics_config=(
                    dynamics_config
                ),
                population_size=(
                    population_size
                ),
                horizon=horizon,
                seed=seed,
            )
        )

        terminal = trajectory.tail(
            terminal_window
        )

        seed_rows.append(
            {
                "seed": seed,
                **{
                    metric: float(
                        terminal[
                            metric
                        ].mean()
                    )
                    for metric
                    in metrics
                },
            }
        )

    seed_metric_values = (
        pd.DataFrame(
            seed_rows
        )
    )

    analysis = study_config[
        "analysis"
    ]

    confidence = float(
        analysis[
            "confidence_level"
        ]
    )

    maximum_relative_width = float(
        analysis[
            "maximum_relative_ci_width"
        ]
    )

    denominator_floor = float(
        analysis[
            "relative_ci_denominator_floor"
        ]
    )

    ranking_tolerance = float(
        analysis[
            "ranking_tolerance"
        ]
    )

    reference_subset = (
        seed_metric_values.iloc[
            : max(counts)
        ]
    )

    reference_ranking = (
        framework_ranking(
            {
                metric: float(
                    reference_subset[
                        metric
                    ].mean()
                )
                for metric
                in FRAMEWORK_METRICS
            },
            tolerance=(
                ranking_tolerance
            ),
        )
    )

    ci_rows: list[
        dict[str, Any]
    ] = []

    ranking_rows: list[
        dict[str, Any]
    ] = []

    for count in counts:
        subset = (
            seed_metric_values.iloc[
                :count
            ]
        )

        metric_acceptance: list[
            bool
        ] = []

        for metric in metrics:
            statistics = (
                summarize_values(
                    subset[
                        metric
                    ],
                    confidence_level=(
                        confidence
                    ),
                )
            )

            denominator = max(
                abs(
                    float(
                        statistics[
                            "mean"
                        ]
                    )
                ),
                denominator_floor,
            )

            relative_width = (
                float(
                    statistics[
                        "ci_width"
                    ]
                )
                / denominator
            )

            accepted_metric = (
                relative_width
                <= maximum_relative_width
            )

            metric_acceptance.append(
                accepted_metric
            )

            ci_rows.append(
                {
                    "candidate_seed_count": (
                        count
                    ),
                    "metric": metric,
                    **statistics,
                    "relative_ci_width": (
                        relative_width
                    ),
                    "maximum_relative_ci_width": (
                        maximum_relative_width
                    ),
                    "accepted_metric": (
                        accepted_metric
                    ),
                }
            )

        ranking = (
            framework_ranking(
                {
                    metric: float(
                        subset[
                            metric
                        ].mean()
                    )
                    for metric
                    in FRAMEWORK_METRICS
                },
                tolerance=(
                    ranking_tolerance
                ),
            )
        )

        ranking_matches = (
            ranking
            == reference_ranking
        )

        ranking_rows.append(
            {
                "candidate_seed_count": (
                    count
                ),
                "ranking": ranking,
                "reference_ranking": (
                    reference_ranking
                ),
                "matches_reference": (
                    ranking_matches
                ),
                "all_metric_ci_widths_acceptable": (
                    all(
                        metric_acceptance
                    )
                ),
                "accepted": bool(
                    all(
                        metric_acceptance
                    )
                    and ranking_matches
                ),
            }
        )

    ci_summary = pd.DataFrame(
        ci_rows
    )

    ranking_summary = (
        pd.DataFrame(
            ranking_rows
        )
    )

    accepted_counts = (
        ranking_summary.loc[
            ranking_summary[
                "accepted"
            ],
            "candidate_seed_count",
        ]
    )

    recommended = (
        int(
            accepted_counts.min()
        )
        if not accepted_counts.empty
        else None
    )

    return ConvergenceStudyResult(
        study_type=(
            "seed_convergence"
        ),
        tables={
            "seed_metric_values": (
                seed_metric_values
            ),
            "ci_summary": (
                ci_summary
            ),
            "ranking_summary": (
                ranking_summary
            ),
        },
        metadata={
            "study_type": (
                "seed_convergence"
            ),
            "start_seed": (
                start_seed
            ),
            "maximum_seed_count": (
                max(counts)
            ),
            "population_size": (
                population_size
            ),
            "horizon": horizon,
            "terminal_window": (
                terminal_window
            ),
            "reference_ranking": (
                reference_ranking
            ),
            "recommended_seed_count": (
                recommended
            ),
            "passed": (
                recommended
                is not None
            ),
        },
    )


def run_population_convergence(
    base_config: Mapping[str, Any],
    dynamics_config: Mapping[str, Any],
    study_config: Mapping[str, Any],
) -> ConvergenceStudyResult:
    """Compare candidate population sizes with the largest population."""

    validate_convergence_config(
        study_config
    )

    metrics = list(
        study_config[
            "metrics"
        ]
    )

    sizes = [
        int(value)
        for value
        in study_config[
            "candidate_population_sizes"
        ]
    ]

    seed_plan = study_config[
        "seed_plan"
    ]

    seeds = range(
        int(
            seed_plan[
                "start_seed"
            ]
        ),
        int(
            seed_plan[
                "start_seed"
            ]
        )
        + int(
            seed_plan[
                "seed_count"
            ]
        ),
    )

    horizon = int(
        study_config[
            "simulation"
        ]["horizon"]
    )

    terminal_window = int(
        study_config[
            "simulation"
        ]["terminal_window"]
    )

    confidence = float(
        study_config[
            "analysis"
        ]["confidence_level"]
    )

    difference_tolerance = float(
        study_config[
            "analysis"
        ][
            "material_difference_tolerance"
        ]
    )

    ranking_tolerance = float(
        study_config[
            "analysis"
        ]["ranking_tolerance"]
    )

    value_rows: list[
        dict[str, Any]
    ] = []

    for population_size in sizes:
        for seed in seeds:
            trajectory = (
                run_single_trajectory(
                    base_config=base_config,
                    dynamics_config=(
                        dynamics_config
                    ),
                    population_size=(
                        population_size
                    ),
                    horizon=horizon,
                    seed=seed,
                )
            )

            terminal = trajectory.tail(
                terminal_window
            )

            value_rows.append(
                {
                    "population_size": (
                        population_size
                    ),
                    "seed": seed,
                    **{
                        metric: float(
                            terminal[
                                metric
                            ].mean()
                        )
                        for metric
                        in metrics
                    },
                }
            )

    population_metric_values = (
        pd.DataFrame(
            value_rows
        )
    )

    summary_rows: list[
        dict[str, Any]
    ] = []

    ranking_rows: list[
        dict[str, Any]
    ] = []

    for population_size in sizes:
        subset = (
            population_metric_values.loc[
                population_metric_values[
                    "population_size"
                ]
                == population_size
            ]
        )

        for metric in metrics:
            summary_rows.append(
                {
                    "population_size": (
                        population_size
                    ),
                    "metric": metric,
                    **summarize_values(
                        subset[
                            metric
                        ],
                        confidence_level=(
                            confidence
                        ),
                    ),
                }
            )

        ranking_rows.append(
            {
                "population_size": (
                    population_size
                ),
                "ranking": (
                    framework_ranking(
                        {
                            metric: float(
                                subset[
                                    metric
                                ].mean()
                            )
                            for metric
                            in FRAMEWORK_METRICS
                        },
                        tolerance=(
                            ranking_tolerance
                        ),
                    )
                ),
            }
        )

    population_summary = (
        pd.DataFrame(
            summary_rows
        )
    )

    ranking_summary = (
        pd.DataFrame(
            ranking_rows
        )
    )

    reference_size = max(
        sizes
    )

    reference_ranking = str(
        ranking_summary.loc[
            ranking_summary[
                "population_size"
            ]
            == reference_size,
            "ranking",
        ].iloc[0]
    )

    reference_means = {
        metric: float(
            population_summary.loc[
                (
                    population_summary[
                        "population_size"
                    ]
                    == reference_size
                )
                & (
                    population_summary[
                        "metric"
                    ]
                    == metric
                ),
                "mean",
            ].iloc[0]
        )
        for metric in metrics
    }

    comparison_rows: list[
        dict[str, Any]
    ] = []

    acceptance_rows: list[
        dict[str, Any]
    ] = []

    for population_size in sizes:
        metric_acceptance: list[
            bool
        ] = []

        for metric in metrics:
            candidate_mean = float(
                population_summary.loc[
                    (
                        population_summary[
                            "population_size"
                        ]
                        == population_size
                    )
                    & (
                        population_summary[
                            "metric"
                        ]
                        == metric
                    ),
                    "mean",
                ].iloc[0]
            )

            difference = abs(
                candidate_mean
                - reference_means[
                    metric
                ]
            )

            accepted_metric = (
                difference
                <= difference_tolerance
            )

            metric_acceptance.append(
                accepted_metric
            )

            comparison_rows.append(
                {
                    "population_size": (
                        population_size
                    ),
                    "reference_population_size": (
                        reference_size
                    ),
                    "metric": metric,
                    "candidate_mean": (
                        candidate_mean
                    ),
                    "reference_mean": (
                        reference_means[
                            metric
                        ]
                    ),
                    "absolute_difference": (
                        difference
                    ),
                    "material_difference_tolerance": (
                        difference_tolerance
                    ),
                    "accepted_metric": (
                        accepted_metric
                    ),
                }
            )

        ranking = str(
            ranking_summary.loc[
                ranking_summary[
                    "population_size"
                ]
                == population_size,
                "ranking",
            ].iloc[0]
        )

        ranking_matches = (
            ranking
            == reference_ranking
        )

        acceptance_rows.append(
            {
                "population_size": (
                    population_size
                ),
                "ranking": ranking,
                "reference_ranking": (
                    reference_ranking
                ),
                "matches_reference_ranking": (
                    ranking_matches
                ),
                "all_metric_differences_acceptable": (
                    all(
                        metric_acceptance
                    )
                ),
                "accepted": bool(
                    all(
                        metric_acceptance
                    )
                    and ranking_matches
                ),
            }
        )

    population_comparison = (
        pd.DataFrame(
            comparison_rows
        )
    )

    population_acceptance = (
        pd.DataFrame(
            acceptance_rows
        )
    )

    accepted_sizes = (
        population_acceptance.loc[
            population_acceptance[
                "accepted"
            ],
            "population_size",
        ]
    )

    recommended = (
        int(
            accepted_sizes.min()
        )
        if not accepted_sizes.empty
        else None
    )

    return ConvergenceStudyResult(
        study_type=(
            "population_convergence"
        ),
        tables={
            "population_metric_values": (
                population_metric_values
            ),
            "population_summary": (
                population_summary
            ),
            "population_comparison": (
                population_comparison
            ),
            "ranking_summary": (
                ranking_summary
            ),
            "population_acceptance": (
                population_acceptance
            ),
        },
        metadata={
            "study_type": (
                "population_convergence"
            ),
            "seed_count": int(
                seed_plan[
                    "seed_count"
                ]
            ),
            "horizon": horizon,
            "terminal_window": (
                terminal_window
            ),
            "reference_population_size": (
                reference_size
            ),
            "reference_ranking": (
                reference_ranking
            ),
            "recommended_population_size": (
                recommended
            ),
            "passed": (
                recommended
                is not None
            ),
        },
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the convergence command-line parser."""

    parser = argparse.ArgumentParser(
        prog="python -m sgb.convergence",
        description=(
            "Run an SGB convergence study."
        ),
    )

    parser.add_argument(
        "--study",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--fail-on-rejection",
        action="store_true",
    )

    return parser


def main(
    argv: Sequence[str] | None = None,
) -> int:
    """Run one convergence study from the command line."""

    arguments = (
        build_parser().parse_args(
            argv
        )
    )

    try:
        study_config = (
            load_convergence_config(
                arguments.study
            )
        )

        base_config = load_config(
            study_config[
                "reference_config"
            ]
        )

        dynamics_config = (
            load_dynamics_config(
                study_config[
                    "dynamics_config"
                ]
            )
        )

        result = (
            run_convergence_study(
                base_config,
                dynamics_config,
                study_config,
            )
        )

        output_directory = (
            arguments.output
            if arguments.output
            is not None
            else study_config[
                "outputs"
            ]["directory"]
        )

        paths = (
            save_convergence_result(
                result,
                output_directory,
            )
        )

        print(
            json.dumps(
                result.metadata,
                indent=2,
                sort_keys=True,
            )
        )

        print(
            "Output directory: "
            f"{paths['metadata'].parent}"
        )

        if (
            arguments.fail_on_rejection
            and not result.passed
        ):
            return 1

        return 0

    except (
        ConvergenceConfigurationError,
        ConvergenceError,
    ) as error:
        print(
            f"Convergence error: {error}"
        )

        return 2


if __name__ == "__main__":
    raise SystemExit(
        main()
    )