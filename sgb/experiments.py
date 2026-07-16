from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from sgb.config import PROJECT_ROOT, load_config
from sgb.convergence_utils import (
    detect_block_convergence,
    framework_ranking,
    summarize_values,
)
from sgb.dimensions import ALL_DIMENSIONS
from sgb.dynamics import (
    NeutralOperationalDynamics,
    load_dynamics_config,
)
from sgb.frameworks import (
    aggregate_agent_scores,
    build_framework_definitions,
    evaluate_agents,
    evaluate_trajectory,
)
from sgb.model import SGBModel
from sgb.perturbations import apply_security_incident


EXPERIMENT_DIRECTORY = (
    PROJECT_ROOT
    / "configs"
    / "experiments"
)

FRAMEWORKS = (
    "DBL",
    "DMM",
    "ISF",
)

FRAMEWORK_METRICS = tuple(
    f"E_GMI_{framework}"
    for framework in FRAMEWORKS
)

INVERSE_SUBMETRICS = {
    "duplicate_rate",
    "normalized_breach_latency",
    "policy_violation_rate",
}


class ExperimentConfigurationError(ValueError):
    """Raised when an experiment configuration is invalid."""


class ExperimentError(RuntimeError):
    """Raised when an experiment cannot be completed."""


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """Complete result produced by one experiment."""

    experiment_id: int
    experiment_name: str
    tables: dict[str, pd.DataFrame]
    metadata: dict[str, Any]

    @property
    def passed(self) -> bool:
        return bool(
            self.metadata.get(
                "passed",
                False,
            )
        )


def load_experiment_config(
    path: str | Path,
) -> dict[str, Any]:
    """Load and validate one experiment configuration."""

    config_path = Path(path)

    if not config_path.is_absolute():
        config_path = (
            PROJECT_ROOT
            / config_path
        )

    config_path = config_path.resolve()

    if not config_path.is_file():
        raise ExperimentConfigurationError(
            "Experiment configuration does not exist: "
            f"{config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ExperimentConfigurationError(
            "Experiment configuration root must be a mapping."
        )

    validate_experiment_config(
        config
    )

    config["_metadata"] = {
        "source_path": str(
            config_path
        ),
    }

    return config


def validate_experiment_config(
    config: Mapping[str, Any],
) -> None:
    """Validate one of the four experiment configurations."""

    if config.get("schema_version") != 1:
        raise ExperimentConfigurationError(
            "schema_version must be 1."
        )

    if config.get("config_kind") != "experiment":
        raise ExperimentConfigurationError(
            "config_kind must be 'experiment'."
        )

    experiment_id = config.get(
        "experiment_id"
    )

    if (
        isinstance(experiment_id, bool)
        or not isinstance(experiment_id, int)
        or experiment_id not in {1, 2, 3, 4}
    ):
        raise ExperimentConfigurationError(
            "experiment_id must be one of 1, 2, 3, or 4."
        )

    _non_empty_string(
        config,
        "experiment_name",
    )

    configuration = _mapping(
        config,
        "configuration",
    )

    if configuration.get("status") not in {
        "candidate",
        "final",
    }:
        raise ExperimentConfigurationError(
            "configuration.status must be "
            "'candidate' or 'final'."
        )

    _non_empty_string(
        configuration,
        "description",
    )

    _non_empty_string(
        config,
        "reference_config",
    )

    _non_empty_string(
        config,
        "dynamics_config",
    )

    seed_plan = _mapping(
        config,
        "seed_plan",
    )

    _non_negative_integer(
        seed_plan,
        "start_seed",
    )

    _positive_integer(
        seed_plan,
        "seed_count",
    )

    simulation = _mapping(
        config,
        "simulation",
    )

    _positive_integer(
        simulation,
        "population_size",
    )

    analysis = _mapping(
        config,
        "analysis",
    )

    confidence = _finite_number(
        analysis,
        "confidence_level",
    )

    if not 0.0 < confidence < 1.0:
        raise ExperimentConfigurationError(
            "analysis.confidence_level must be in (0, 1)."
        )

    _non_empty_string(
        _mapping(
            config,
            "outputs",
        ),
        "directory",
    )

    if experiment_id in {1, 2, 3}:
        _validate_steady_state_config(
            _mapping(
                config,
                "steady_state",
            )
        )

    if experiment_id == 1:
        _positive_integer(
            simulation,
            "terminal_window",
        )

        maturity = _mapping(
            config,
            "maturity_levels",
        )

        cutoffs = maturity.get(
            "cutoffs"
        )
        labels = maturity.get(
            "labels"
        )

        if (
            not isinstance(cutoffs, list)
            or len(cutoffs) != 4
            or cutoffs != sorted(cutoffs)
        ):
            raise ExperimentConfigurationError(
                "maturity_levels.cutoffs must contain "
                "four sorted values."
            )

        if (
            not isinstance(labels, list)
            or len(labels) != 5
            or any(
                not isinstance(label, str)
                or not label.strip()
                for label in labels
            )
        ):
            raise ExperimentConfigurationError(
                "maturity_levels.labels must contain five labels."
            )

        for cutoff in cutoffs:
            _unit_value(
                cutoff,
                "maturity cutoff",
            )

        _non_empty_string(
            maturity,
            "rationale",
        )

    elif experiment_id == 2:
        shock = _mapping(
            config,
            "shock",
        )

        if shock.get("type") != "security_incident":
            raise ExperimentConfigurationError(
                "Experiment 2 shock.type must be "
                "'security_incident'."
            )

        if (
            shock.get("target_selection")
            != "highest_degree"
        ):
            raise ExperimentConfigurationError(
                "Experiment 2 target_selection must be "
                "'highest_degree'."
            )

        _unit_number(
            shock,
            "impact",
        )

        _non_negative_integer(
            shock,
            "maximum_network_distance",
        )

        decay = _finite_number(
            shock,
            "distance_decay",
        )

        if not 0.0 < decay <= 1.0:
            raise ExperimentConfigurationError(
                "shock.distance_decay must be in (0, 1]."
            )

        recovery = _mapping(
            config,
            "recovery",
        )

        _positive_integer(
            recovery,
            "maximum_observation_steps",
        )

        _unit_number(
            recovery,
            "recovery_tolerance",
        )

        _positive_integer(
            recovery,
            "required_consecutive_steps",
        )

    elif experiment_id == 3:
        _positive_integer(
            simulation,
            "observation_steps",
        )

        scan = _mapping(
            config,
            "threshold_scan",
        )

        coarse = _mapping(
            scan,
            "coarse",
        )
        refinement = _mapping(
            scan,
            "refinement",
        )

        minimum = _unit_number(
            coarse,
            "minimum",
        )
        maximum = _unit_number(
            coarse,
            "maximum",
        )
        step = _finite_number(
            coarse,
            "step",
        )

        if maximum <= minimum:
            raise ExperimentConfigurationError(
                "Threshold maximum must exceed minimum."
            )

        if step <= 0.0:
            raise ExperimentConfigurationError(
                "Threshold coarse step must be positive."
            )

        radius = _finite_number(
            refinement,
            "radius",
        )
        refined_step = _finite_number(
            refinement,
            "step",
        )

        if radius <= 0.0 or refined_step <= 0.0:
            raise ExperimentConfigurationError(
                "Threshold refinement values must be positive."
            )

        if scan.get("endpoint_score_rule") != "minimum":
            raise ExperimentConfigurationError(
                "endpoint_score_rule must be 'minimum'."
            )

        if scan.get("optimization_metric") != "f1_score":
            raise ExperimentConfigurationError(
                "optimization_metric must be 'f1_score'."
            )

        if scan.get("tie_breaker") != "lower_threshold":
            raise ExperimentConfigurationError(
                "tie_breaker must be 'lower_threshold'."
            )

    else:
        horizon = _positive_integer(
            simulation,
            "horizon",
        )

        terminal_window = _positive_integer(
            simulation,
            "terminal_window",
        )

        if terminal_window > horizon:
            raise ExperimentConfigurationError(
                "terminal_window cannot exceed horizon."
            )

        levels = config.get(
            "maturity_levels"
        )

        if (
            not isinstance(levels, list)
            or len(levels) < 2
        ):
            raise ExperimentConfigurationError(
                "Experiment 4 requires at least two maturity levels."
            )

        names: set[str] = set()

        for level in levels:
            if not isinstance(level, Mapping):
                raise ExperimentConfigurationError(
                    "Every maturity level must be a mapping."
                )

            name = _non_empty_string(
                level,
                "name",
            )

            if name in names:
                raise ExperimentConfigurationError(
                    f"Duplicate maturity level: {name}."
                )

            names.add(name)

            alpha = _positive_number(
                level,
                "alpha",
            )
            beta = _positive_number(
                level,
                "beta",
            )

            if not math.isfinite(
                alpha + beta
            ):
                raise ExperimentConfigurationError(
                    "Maturity Beta parameters must be finite."
                )

        endpoint = _mapping(
            config,
            "endpoint_acceptance",
        )

        low_maximum = _unit_number(
            endpoint,
            "maximum_low_effective_mean",
        )

        high_minimum = _unit_number(
            endpoint,
            "minimum_high_effective_mean",
        )

        if high_minimum <= low_maximum:
            raise ExperimentConfigurationError(
                "High endpoint requirement must exceed low requirement."
            )


def run_experiment(
    base_config: Mapping[str, Any],
    dynamics_config: Mapping[str, Any],
    experiment_config: Mapping[str, Any],
) -> ExperimentResult:
    """Run one configured experiment."""

    validate_experiment_config(
        experiment_config
    )

    experiment_id = int(
        experiment_config[
            "experiment_id"
        ]
    )

    if experiment_id == 1:
        return run_experiment_1(
            base_config,
            dynamics_config,
            experiment_config,
        )

    if experiment_id == 2:
        return run_experiment_2(
            base_config,
            dynamics_config,
            experiment_config,
        )

    if experiment_id == 3:
        return run_experiment_3(
            base_config,
            dynamics_config,
            experiment_config,
        )

    return run_experiment_4(
        base_config,
        dynamics_config,
        experiment_config,
    )


def run_experiment_1(
    base_config: Mapping[str, Any],
    dynamics_config: Mapping[str, Any],
    experiment_config: Mapping[str, Any],
) -> ExperimentResult:
    """Compare all frameworks on shared steady-state trajectories."""

    validate_experiment_config(
        experiment_config
    )

    definitions = build_framework_definitions(
        base_config
    )

    confidence = float(
        experiment_config[
            "analysis"
        ]["confidence_level"]
    )

    ranking_tolerance = float(
        experiment_config[
            "analysis"
        ]["ranking_tolerance"]
    )

    terminal_window = int(
        experiment_config[
            "simulation"
        ]["terminal_window"]
    )

    seed_framework_rows: list[
        dict[str, Any]
    ] = []

    maturity_rows: list[
        dict[str, Any]
    ] = []

    convergence_rows: list[
        dict[str, Any]
    ] = []

    cutoffs = [
        float(value)
        for value in experiment_config[
            "maturity_levels"
        ]["cutoffs"]
    ]

    labels = list(
        experiment_config[
            "maturity_levels"
        ]["labels"]
    )

    for seed in _seeds(
        experiment_config
    ):
        model, steady_state_step = (
            _create_and_stabilize_model(
                base_config=base_config,
                dynamics_config=dynamics_config,
                experiment_config=experiment_config,
                seed=seed,
                additional_steps=terminal_window,
            )
        )

        for _ in range(
            terminal_window
        ):
            model.step()
            model.event_records.clear()

        scored_trajectory = evaluate_trajectory(
            model.get_results(),
            definitions,
        )

        terminal = scored_trajectory.tail(
            terminal_window
        )

        agent_scores = evaluate_agents(
            model.agents.to_list(),
            definitions,
        )

        seed_scores: dict[
            str,
            float,
        ] = {}

        for framework in FRAMEWORKS:
            metric = (
                f"E_GMI_{framework}"
            )

            mean_score = float(
                terminal[
                    metric
                ].mean()
            )

            seed_scores[
                metric
            ] = mean_score

            seed_framework_rows.append(
                {
                    "seed": seed,
                    "steady_state_step": (
                        steady_state_step
                    ),
                    "framework": framework,
                    "E_GMI": mean_score,
                    "success_rate": float(
                        terminal[
                            "success_rate"
                        ].mean()
                    ),
                    "volume_success_rate": float(
                        terminal[
                            "volume_success_rate"
                        ].mean()
                    ),
                }
            )

            score_column = (
                f"GMI_{framework}"
            )

            for score in agent_scores[
                score_column
            ]:
                maturity_rows.append(
                    {
                        "seed": seed,
                        "framework": framework,
                        "maturity_level": (
                            classify_maturity(
                                float(score),
                                cutoffs,
                                labels,
                            )
                        ),
                    }
                )

        convergence_rows.append(
            {
                "seed": seed,
                "steady_state_step": (
                    steady_state_step
                ),
                "ranking": framework_ranking(
                    seed_scores,
                    tolerance=(
                        ranking_tolerance
                    ),
                ),
            }
        )

    seed_framework_scores = pd.DataFrame(
        seed_framework_rows
    )

    summary_rows: list[
        dict[str, Any]
    ] = []

    framework_means: dict[
        str,
        float,
    ] = {}

    for framework in FRAMEWORKS:
        values = seed_framework_scores.loc[
            seed_framework_scores[
                "framework"
            ]
            == framework,
            "E_GMI",
        ]

        statistics = summarize_values(
            values,
            confidence_level=confidence,
        )

        framework_means[
            f"E_GMI_{framework}"
        ] = float(
            statistics[
                "mean"
            ]
        )

        summary_rows.append(
            {
                "framework": framework,
                **statistics,
            }
        )

    framework_summary = pd.DataFrame(
        summary_rows
    )

    overall_ranking = framework_ranking(
        framework_means,
        tolerance=ranking_tolerance,
    )

    maturity_distribution = (
        pd.DataFrame(
            maturity_rows
        )
        .groupby(
            [
                "framework",
                "maturity_level",
            ],
            as_index=False,
        )
        .size()
        .rename(
            columns={
                "size": "organization_seed_count",
            }
        )
    )

    return ExperimentResult(
        experiment_id=1,
        experiment_name=(
            "framework_comparison"
        ),
        tables={
            "seed_framework_scores": (
                seed_framework_scores
            ),
            "framework_summary": (
                framework_summary
            ),
            "maturity_distribution": (
                maturity_distribution
            ),
            "seed_convergence": (
                pd.DataFrame(
                    convergence_rows
                )
            ),
        },
        metadata={
            "experiment_id": 1,
            "experiment_name": (
                "framework_comparison"
            ),
            "seed_count": len(
                list(
                    _seeds(
                        experiment_config
                    )
                )
            ),
            "overall_ranking": (
                overall_ranking
            ),
            "shared_trajectory": True,
            "passed": True,
        },
    )


def run_experiment_2(
    base_config: Mapping[str, Any],
    dynamics_config: Mapping[str, Any],
    experiment_config: Mapping[str, Any],
) -> ExperimentResult:
    """Measure framework-neutral security propagation and recovery."""

    validate_experiment_config(
        experiment_config
    )

    definitions = build_framework_definitions(
        base_config
    )

    shock_config = experiment_config[
        "shock"
    ]
    recovery_config = experiment_config[
        "recovery"
    ]

    maximum_recovery_steps = int(
        recovery_config[
            "maximum_observation_steps"
        ]
    )

    recovery_tolerance = float(
        recovery_config[
            "recovery_tolerance"
        ]
    )

    required_consecutive = int(
        recovery_config[
            "required_consecutive_steps"
        ]
    )

    recovery_rows: list[
        dict[str, Any]
    ] = []

    propagation_rows: list[
        dict[str, Any]
    ] = []

    trajectory_rows: list[
        dict[str, Any]
    ] = []

    for seed in _seeds(
        experiment_config
    ):
        model, steady_state_step = (
            _create_and_stabilize_model(
                base_config=base_config,
                dynamics_config=dynamics_config,
                experiment_config=experiment_config,
                seed=seed,
                additional_steps=(
                    maximum_recovery_steps
                ),
            )
        )

        pre_scores = _mean_framework_scores(
            model,
            definitions,
        )

        target_uid = max(
            model.graph.degree,
            key=lambda item: (
                item[1],
                -item[0],
            ),
        )[0]

        target_degree = int(
            model.graph.degree[
                target_uid
            ]
        )

        shock_records = apply_security_incident(
            model=model,
            target_uid=target_uid,
            impact=float(
                shock_config[
                    "impact"
                ]
            ),
            max_distance=int(
                shock_config[
                    "maximum_network_distance"
                ]
            ),
            decay=float(
                shock_config[
                    "distance_decay"
                ]
            ),
        )

        post_scores = _mean_framework_scores(
            model,
            definitions,
        )

        recovery_times: dict[
            str,
            int | None,
        ] = {
            framework: None
            for framework in FRAMEWORKS
        }

        stable_counts = {
            framework: 0
            for framework in FRAMEWORKS
        }

        for framework in FRAMEWORKS:
            trajectory_rows.append(
                {
                    "seed": seed,
                    "recovery_step": 0,
                    "framework": framework,
                    "E_GMI": post_scores[
                        framework
                    ],
                    "pre_shock_E_GMI": (
                        pre_scores[
                            framework
                        ]
                    ),
                }
            )

        model.event_records.clear()

        for recovery_step in range(
            1,
            maximum_recovery_steps + 1,
        ):
            model.step()
            model.event_records.clear()

            current_scores = (
                _mean_framework_scores(
                    model,
                    definitions,
                )
            )

            for framework in FRAMEWORKS:
                current = current_scores[
                    framework
                ]

                trajectory_rows.append(
                    {
                        "seed": seed,
                        "recovery_step": (
                            recovery_step
                        ),
                        "framework": framework,
                        "E_GMI": current,
                        "pre_shock_E_GMI": (
                            pre_scores[
                                framework
                            ]
                        ),
                    }
                )

                recovered_now = (
                    current
                    >= (
                        pre_scores[
                            framework
                        ]
                        - recovery_tolerance
                    )
                )

                stable_counts[
                    framework
                ] = (
                    stable_counts[
                        framework
                    ]
                    + 1
                    if recovered_now
                    else 0
                )

                if (
                    recovery_times[
                        framework
                    ]
                    is None
                    and stable_counts[
                        framework
                    ]
                    >= required_consecutive
                ):
                    recovery_times[
                        framework
                    ] = (
                        recovery_step
                        - required_consecutive
                        + 1
                    )

            if all(
                recovery_times[
                    framework
                ]
                is not None
                for framework in FRAMEWORKS
            ):
                break

        for framework in FRAMEWORKS:
            recovery_rows.append(
                {
                    "seed": seed,
                    "framework": framework,
                    "steady_state_step": (
                        steady_state_step
                    ),
                    "pre_shock_E_GMI": (
                        pre_scores[
                            framework
                        ]
                    ),
                    "post_shock_E_GMI": (
                        post_scores[
                            framework
                        ]
                    ),
                    "immediate_loss": (
                        pre_scores[
                            framework
                        ]
                        - post_scores[
                            framework
                        ]
                    ),
                    "recovery_time": (
                        recovery_times[
                            framework
                        ]
                    ),
                    "recovered": (
                        recovery_times[
                            framework
                        ]
                        is not None
                    ),
                }
            )

        d2_losses = [
            float(
                record[
                    "D2_before"
                ]
                - record[
                    "D2_after"
                ]
            )
            for record in shock_records
        ]

        propagation_rows.append(
            {
                "seed": seed,
                "steady_state_step": (
                    steady_state_step
                ),
                "target_uid": (
                    target_uid
                ),
                "target_degree": (
                    target_degree
                ),
                "affected_organizations": (
                    len(
                        shock_records
                    )
                ),
                "maximum_network_distance": (
                    max(
                        record[
                            "network_distance"
                        ]
                        for record
                        in shock_records
                    )
                ),
                "mean_D2_loss": float(
                    np.mean(
                        d2_losses
                    )
                ),
                "maximum_D2_loss": float(
                    np.max(
                        d2_losses
                    )
                ),
            }
        )

    recovery_results = pd.DataFrame(
        recovery_rows
    )

    propagation_results = pd.DataFrame(
        propagation_rows
    )

    recovery_trajectory = pd.DataFrame(
        trajectory_rows
    )

    confidence = float(
        experiment_config[
            "analysis"
        ]["confidence_level"]
    )

    summary_rows: list[
        dict[str, Any]
    ] = []

    for framework in FRAMEWORKS:
        subset = recovery_results.loc[
            recovery_results[
                "framework"
            ]
            == framework
        ]

        recovered_values = subset.loc[
            subset[
                "recovered"
            ],
            "recovery_time",
        ]

        if recovered_values.empty:
            recovery_statistics = {
                "sample_count": 0,
                "mean": math.nan,
                "standard_deviation": math.nan,
                "ci_lower": math.nan,
                "ci_upper": math.nan,
                "ci_width": math.nan,
            }
        else:
            recovery_statistics = (
                summarize_values(
                    recovered_values,
                    confidence_level=(
                        confidence
                    ),
                )
            )

        summary_rows.append(
            {
                "framework": framework,
                "mean_immediate_loss": float(
                    subset[
                        "immediate_loss"
                    ].mean()
                ),
                "recovered_fraction": float(
                    subset[
                        "recovered"
                    ].mean()
                ),
                "mean_recovery_time": (
                    recovery_statistics[
                        "mean"
                    ]
                ),
                "recovery_time_ci_lower": (
                    recovery_statistics[
                        "ci_lower"
                    ]
                ),
                "recovery_time_ci_upper": (
                    recovery_statistics[
                        "ci_upper"
                    ]
                ),
            }
        )

    all_recovered = bool(
        recovery_results[
            "recovered"
        ].all()
    )

    return ExperimentResult(
        experiment_id=2,
        experiment_name=(
            "security_propagation_and_recovery"
        ),
        tables={
            "recovery_results": (
                recovery_results
            ),
            "propagation_results": (
                propagation_results
            ),
            "recovery_trajectory": (
                recovery_trajectory
            ),
            "framework_recovery_summary": (
                pd.DataFrame(
                    summary_rows
                )
            ),
        },
        metadata={
            "experiment_id": 2,
            "experiment_name": (
                "security_propagation_and_recovery"
            ),
            "shared_shock": True,
            "framework_independent_dynamics": (
                True
            ),
            "all_framework_runs_recovered": (
                all_recovered
            ),
            "passed": True,
        },
    )


def run_experiment_3(
    base_config: Mapping[str, Any],
    dynamics_config: Mapping[str, Any],
    experiment_config: Mapping[str, Any],
) -> ExperimentResult:
    """Scan and refine GMI thresholds using shared exchange events."""

    validate_experiment_config(
        experiment_config
    )

    definitions = build_framework_definitions(
        base_config
    )

    observation_steps = int(
        experiment_config[
            "simulation"
        ]["observation_steps"]
    )

    scan = experiment_config[
        "threshold_scan"
    ]

    coarse = scan[
        "coarse"
    ]
    refinement = scan[
        "refinement"
    ]

    coarse_thresholds = _float_range(
        float(
            coarse[
                "minimum"
            ]
        ),
        float(
            coarse[
                "maximum"
            ]
        ),
        float(
            coarse[
                "step"
            ]
        ),
    )

    seed_threshold_rows: list[
        dict[str, Any]
    ] = []

    event_rows: list[
        dict[str, Any]
    ] = []

    for seed in _seeds(
        experiment_config
    ):
        model, steady_state_step = (
            _create_and_stabilize_model(
                base_config=base_config,
                dynamics_config=dynamics_config,
                experiment_config=experiment_config,
                seed=seed,
                additional_steps=(
                    observation_steps
                ),
            )
        )

        agent_scores = evaluate_agents(
            model.agents.to_list(),
            definitions,
        )

        score_maps = {
            framework: dict(
                zip(
                    agent_scores[
                        "uid"
                    ].astype(int),
                    agent_scores[
                        f"GMI_{framework}"
                    ].astype(float),
                    strict=True,
                )
            )
            for framework in FRAMEWORKS
        }

        model.event_records.clear()

        for _ in range(
            observation_steps
        ):
            model.step()

        events = model.get_events()

        if events.empty:
            raise ExperimentError(
                "Experiment 3 produced no exchange events."
            )

        events = events.copy(
            deep=True
        )

        events.insert(
            0,
            "seed",
            seed,
        )

        events.insert(
            1,
            "steady_state_step",
            steady_state_step,
        )

        event_rows.extend(
            events.to_dict(
                orient="records"
            )
        )

        for framework in FRAMEWORKS:
            endpoint_scores = np.minimum(
                events[
                    "sender_uid"
                ].map(
                    score_maps[
                        framework
                    ]
                ).to_numpy(
                    dtype=float
                ),
                events[
                    "receiver_uid"
                ].map(
                    score_maps[
                        framework
                    ]
                ).to_numpy(
                    dtype=float
                ),
            )

            outcomes = events[
                "success"
            ].to_numpy(
                dtype=bool
            )

            volumes = events[
                "volume"
            ].to_numpy(
                dtype=float
            )

            coarse_results = [
                {
                    "threshold": threshold,
                    **_classification_metrics(
                        endpoint_scores=(
                            endpoint_scores
                        ),
                        outcomes=outcomes,
                        volumes=volumes,
                        threshold=threshold,
                    ),
                }
                for threshold
                in coarse_thresholds
            ]

            coarse_best = _select_threshold(
                coarse_results
            )

            refined_minimum = max(
                0.0,
                coarse_best
                - float(
                    refinement[
                        "radius"
                    ]
                ),
            )

            refined_maximum = min(
                1.0,
                coarse_best
                + float(
                    refinement[
                        "radius"
                    ]
                ),
            )

            refined_thresholds = _float_range(
                refined_minimum,
                refined_maximum,
                float(
                    refinement[
                        "step"
                    ]
                ),
            )

            all_thresholds = sorted(
                set(
                    coarse_thresholds
                )
                | set(
                    refined_thresholds
                )
            )

            coarse_set = set(
                coarse_thresholds
            )
            refined_set = set(
                refined_thresholds
            )

            for threshold in all_thresholds:
                if (
                    threshold in coarse_set
                    and threshold
                    in refined_set
                ):
                    phase = (
                        "coarse_and_refined"
                    )
                elif threshold in coarse_set:
                    phase = "coarse"
                else:
                    phase = "refined"

                seed_threshold_rows.append(
                    {
                        "seed": seed,
                        "steady_state_step": (
                            steady_state_step
                        ),
                        "framework": framework,
                        "phase": phase,
                        "threshold": threshold,
                        **_classification_metrics(
                            endpoint_scores=(
                                endpoint_scores
                            ),
                            outcomes=outcomes,
                            volumes=volumes,
                            threshold=threshold,
                        ),
                    }
                )

    seed_threshold_results = pd.DataFrame(
        seed_threshold_rows
    )

    confidence = float(
        experiment_config[
            "analysis"
        ]["confidence_level"]
    )

    summary_rows: list[
        dict[str, Any]
    ] = []

    metric_columns = [
        "precision",
        "recall",
        "specificity",
        "f1_score",
        "balanced_accuracy",
        "event_coverage",
        "volume_coverage",
        "eligible_success_rate",
    ]

    for (
        framework,
        threshold,
    ), subset in seed_threshold_results.groupby(
        [
            "framework",
            "threshold",
        ],
        sort=True,
    ):
        row: dict[str, Any] = {
            "framework": framework,
            "threshold": float(
                threshold
            ),
            "seed_count": int(
                subset[
                    "seed"
                ].nunique()
            ),
        }

        for metric in metric_columns:
            statistics = summarize_values(
                subset[
                    metric
                ],
                confidence_level=confidence,
            )

            row[
                f"{metric}_mean"
            ] = statistics[
                "mean"
            ]

            row[
                f"{metric}_ci_lower"
            ] = statistics[
                "ci_lower"
            ]

            row[
                f"{metric}_ci_upper"
            ] = statistics[
                "ci_upper"
            ]

        summary_rows.append(
            row
        )

    threshold_summary = pd.DataFrame(
        summary_rows
    )

    recommendation_rows: list[
        dict[str, Any]
    ] = []

    for framework in FRAMEWORKS:
        subset = threshold_summary.loc[
            threshold_summary[
                "framework"
            ]
            == framework
        ].copy()

        subset = subset.sort_values(
            [
                "f1_score_mean",
                "threshold",
            ],
            ascending=[
                False,
                True,
            ],
        )

        selected = subset.iloc[
            0
        ]

        recommendation_rows.append(
            {
                "framework": framework,
                "recommended_threshold": float(
                    selected[
                        "threshold"
                    ]
                ),
                "mean_f1_score": float(
                    selected[
                        "f1_score_mean"
                    ]
                ),
                "mean_balanced_accuracy": float(
                    selected[
                        "balanced_accuracy_mean"
                    ]
                ),
                "mean_event_coverage": float(
                    selected[
                        "event_coverage_mean"
                    ]
                ),
                "mean_eligible_success_rate": float(
                    selected[
                        "eligible_success_rate_mean"
                    ]
                ),
            }
        )

    recommendations = pd.DataFrame(
        recommendation_rows
    )

    return ExperimentResult(
        experiment_id=3,
        experiment_name=(
            "maturity_threshold_selection"
        ),
        tables={
            "seed_threshold_results": (
                seed_threshold_results
            ),
            "threshold_summary": (
                threshold_summary
            ),
            "threshold_recommendations": (
                recommendations
            ),
            "shared_exchange_events": (
                pd.DataFrame(
                    event_rows
                )
            ),
        },
        metadata={
            "experiment_id": 3,
            "experiment_name": (
                "maturity_threshold_selection"
            ),
            "coarse_full_range_scanned": (
                min(
                    coarse_thresholds
                )
                == 0.0
                and max(
                    coarse_thresholds
                )
                == 1.0
            ),
            "shared_events": True,
            "optimization_metric": (
                "f1_score"
            ),
            "passed": True,
        },
    )


def run_experiment_4(
    base_config: Mapping[str, Any],
    dynamics_config: Mapping[str, Any],
    experiment_config: Mapping[str, Any],
) -> ExperimentResult:
    """Sweep verified maturity distributions and score all frameworks."""

    validate_experiment_config(
        experiment_config
    )

    horizon = int(
        experiment_config[
            "simulation"
        ]["horizon"]
    )

    terminal_window = int(
        experiment_config[
            "simulation"
        ]["terminal_window"]
    )

    levels = list(
        experiment_config[
            "maturity_levels"
        ]
    )

    endpoint = experiment_config[
        "endpoint_acceptance"
    ]

    level_means = {
        str(
            level[
                "name"
            ]
        ): (
            float(
                level[
                    "alpha"
                ]
            )
            / (
                float(
                    level[
                        "alpha"
                    ]
                )
                + float(
                    level[
                        "beta"
                    ]
                )
            )
        )
        for level in levels
    }

    first_level_name = str(
        levels[
            0
        ]["name"]
    )

    last_level_name = str(
        levels[
            -1
        ]["name"]
    )

    low_endpoint_passed = bool(
        level_means[
            first_level_name
        ]
        < float(
            endpoint[
                "maximum_low_effective_mean"
            ]
        )
    )

    high_endpoint_passed = bool(
        level_means[
            last_level_name
        ]
        > float(
            endpoint[
                "minimum_high_effective_mean"
            ]
        )
    )

    if not (
        low_endpoint_passed
        and high_endpoint_passed
    ):
        raise ExperimentConfigurationError(
            "Experiment 4 Beta endpoints do not satisfy "
            "the configured low/high requirements."
        )

    level_seed_rows: list[
        dict[str, Any]
    ] = []

    dimension_rows: list[
        dict[str, Any]
    ] = []

    for level_index, level in enumerate(
        levels
    ):
        level_name = str(
            level[
                "name"
            ]
        )

        alpha = float(
            level[
                "alpha"
            ]
        )

        beta = float(
            level[
                "beta"
            ]
        )

        level_config = _config_with_maturity_distribution(
            base_config=base_config,
            positive_alpha=alpha,
            positive_beta=beta,
        )

        definitions = build_framework_definitions(
            level_config
        )

        for seed in _seeds(
            experiment_config
        ):
            model = _create_model(
                base_config=level_config,
                dynamics_config=dynamics_config,
                population_size=int(
                    experiment_config[
                        "simulation"
                    ]["population_size"]
                ),
                max_steps=horizon,
                seed=seed,
            )

            initial_results = model.agent_snapshots()

            for dimension in ALL_DIMENSIONS:
                dimension_rows.append(
                    {
                        "level_index": (
                            level_index
                        ),
                        "level": level_name,
                        "seed": seed,
                        "dimension": dimension,
                        "initial_mean": float(
                            np.mean(
                                [
                                    snapshot[
                                        "dimensions"
                                    ][dimension]
                                    for snapshot
                                    in initial_results
                                ]
                            )
                        ),
                    }
                )

            for _ in range(
                horizon
            ):
                model.step()
                model.event_records.clear()

            trajectory = evaluate_trajectory(
                model.get_results(),
                definitions,
            )

            terminal = trajectory.tail(
                terminal_window
            )

            for framework in FRAMEWORKS:
                level_seed_rows.append(
                    {
                        "level_index": (
                            level_index
                        ),
                        "level": level_name,
                        "seed": seed,
                        "theoretical_effective_mean": (
                            level_means[
                                level_name
                            ]
                        ),
                        "framework": framework,
                        "terminal_E_GMI": float(
                            terminal[
                                f"E_GMI_{framework}"
                            ].mean()
                        ),
                        "terminal_success_rate": float(
                            terminal[
                                "success_rate"
                            ].mean()
                        ),
                    }
                )

    level_seed_results = pd.DataFrame(
        level_seed_rows
    )

    confidence = float(
        experiment_config[
            "analysis"
        ]["confidence_level"]
    )

    summary_rows: list[
        dict[str, Any]
    ] = []

    for (
        level_index,
        level,
        framework,
    ), subset in level_seed_results.groupby(
        [
            "level_index",
            "level",
            "framework",
        ],
        sort=True,
    ):
        gmi_statistics = summarize_values(
            subset[
                "terminal_E_GMI"
            ],
            confidence_level=confidence,
        )

        success_statistics = summarize_values(
            subset[
                "terminal_success_rate"
            ],
            confidence_level=confidence,
        )

        summary_rows.append(
            {
                "level_index": int(
                    level_index
                ),
                "level": level,
                "framework": framework,
                "theoretical_effective_mean": float(
                    subset[
                        "theoretical_effective_mean"
                    ].iloc[0]
                ),
                "E_GMI_mean": (
                    gmi_statistics[
                        "mean"
                    ]
                ),
                "E_GMI_ci_lower": (
                    gmi_statistics[
                        "ci_lower"
                    ]
                ),
                "E_GMI_ci_upper": (
                    gmi_statistics[
                        "ci_upper"
                    ]
                ),
                "success_rate_mean": (
                    success_statistics[
                        "mean"
                    ]
                ),
                "success_rate_ci_lower": (
                    success_statistics[
                        "ci_lower"
                    ]
                ),
                "success_rate_ci_upper": (
                    success_statistics[
                        "ci_upper"
                    ]
                ),
            }
        )

    level_summary = pd.DataFrame(
        summary_rows
    ).sort_values(
        [
            "level_index",
            "framework",
        ]
    )

    initial_dimension_summary = (
        pd.DataFrame(
            dimension_rows
        )
        .groupby(
            [
                "level_index",
                "level",
                "dimension",
            ],
            as_index=False,
        )[
            "initial_mean"
        ]
        .mean()
    )

    return ExperimentResult(
        experiment_id=4,
        experiment_name=(
            "maturity_distribution_sweep"
        ),
        tables={
            "level_seed_results": (
                level_seed_results
            ),
            "level_summary": (
                level_summary
            ),
            "initial_dimension_summary": (
                initial_dimension_summary
            ),
        },
        metadata={
            "experiment_id": 4,
            "experiment_name": (
                "maturity_distribution_sweep"
            ),
            "low_endpoint_mean": (
                level_means[
                    first_level_name
                ]
            ),
            "high_endpoint_mean": (
                level_means[
                    last_level_name
                ]
            ),
            "low_endpoint_passed": (
                low_endpoint_passed
            ),
            "high_endpoint_passed": (
                high_endpoint_passed
            ),
            "all_frameworks_evaluated": (
                True
            ),
            "passed": bool(
                low_endpoint_passed
                and high_endpoint_passed
            ),
        },
    )


def classify_maturity(
    score: float,
    cutoffs: Sequence[float],
    labels: Sequence[str],
) -> str:
    """Classify one score using equal-width maturity cutoffs."""

    numeric_score = _unit_value(
        score,
        "score",
    )

    if len(labels) != len(cutoffs) + 1:
        raise ExperimentConfigurationError(
            "Maturity labels must contain one more "
            "entry than cutoffs."
        )

    for index, cutoff in enumerate(
        cutoffs
    ):
        if numeric_score < float(
            cutoff
        ):
            return str(
                labels[
                    index
                ]
            )

    return str(
        labels[
            -1
        ]
    )


def save_experiment_result(
    result: ExperimentResult,
    output_directory: str | Path,
) -> dict[str, Path]:
    """Write experiment tables, metadata, and a figure."""

    output_path = Path(
        output_directory
    )

    if not output_path.is_absolute():
        output_path = (
            PROJECT_ROOT
            / output_path
        )

    output_path = output_path.resolve()

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths: dict[
        str,
        Path,
    ] = {}

    for name, table in result.tables.items():
        path = (
            output_path
            / f"{name}.csv"
        )

        table.to_csv(
            path,
            index=False,
        )

        paths[
            name
        ] = path

    metadata_path = (
        output_path
        / "metadata.json"
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result.metadata,
            file,
            indent=2,
            sort_keys=True,
            default=_json_default,
        )

    paths[
        "metadata"
    ] = metadata_path

    figure_path = (
        output_path
        / (
            f"experiment_"
            f"{result.experiment_id}.png"
        )
    )

    if result.experiment_id == 1:
        _plot_experiment_1(
            result,
            figure_path,
        )
    elif result.experiment_id == 2:
        _plot_experiment_2(
            result,
            figure_path,
        )
    elif result.experiment_id == 3:
        _plot_experiment_3(
            result,
            figure_path,
        )
    else:
        _plot_experiment_4(
            result,
            figure_path,
        )

    paths[
        "figure"
    ] = figure_path

    return paths


def _create_and_stabilize_model(
    *,
    base_config: Mapping[str, Any],
    dynamics_config: Mapping[str, Any],
    experiment_config: Mapping[str, Any],
    seed: int,
    additional_steps: int,
) -> tuple[
    SGBModel,
    int,
]:
    steady = experiment_config[
        "steady_state"
    ]

    maximum_steps = int(
        steady[
            "maximum_steps"
        ]
    )

    stabilization_buffer = int(
        steady[
            "stabilization_buffer"
        ]
    )

    model = _create_model(
        base_config=base_config,
        dynamics_config=dynamics_config,
        population_size=int(
            experiment_config[
                "simulation"
            ]["population_size"]
        ),
        max_steps=(
            maximum_steps
            + stabilization_buffer
            + additional_steps
        ),
        seed=seed,
    )

    definitions = build_framework_definitions(
        base_config
    )

    check_interval = int(
        steady[
            "rolling_window"
        ]
    )

    for _ in range(
        maximum_steps
    ):
        model.step()
        model.event_records.clear()

        if (
            model.current_step
            < int(
                steady[
                    "minimum_step"
                ]
            )
            or model.current_step
            % check_interval
            != 0
        ):
            continue

        scored = evaluate_trajectory(
            model.get_results(),
            definitions,
        )

        results = []

        for metric in steady[
            "metrics"
        ]:
            results.append(
                detect_block_convergence(
                    scored[
                        metric
                    ],
                    rolling_window=int(
                        steady[
                            "rolling_window"
                        ]
                    ),
                    required_stable_windows=int(
                        steady[
                            "required_stable_windows"
                        ]
                    ),
                    minimum_step=int(
                        steady[
                            "minimum_step"
                        ]
                    ),
                    slope_tolerance=float(
                        steady[
                            "slope_tolerance"
                        ]
                    ),
                    mean_difference_tolerance=float(
                        steady[
                            "mean_difference_tolerance"
                        ]
                    ),
                )
            )

        if all(
            result[
                "converged"
            ]
            for result in results
        ):
            detected_step = max(
                int(
                    result[
                        "convergence_step"
                    ]
                )
                for result in results
            )

            for _ in range(
                stabilization_buffer
            ):
                model.step()
                model.event_records.clear()

            return (
                model,
                detected_step,
            )

    raise ExperimentError(
        f"Steady state was not detected for seed {seed} "
        f"within {maximum_steps} steps."
    )


def _create_model(
    *,
    base_config: Mapping[str, Any],
    dynamics_config: Mapping[str, Any],
    population_size: int,
    max_steps: int,
    seed: int,
) -> SGBModel:
    model_config = deepcopy(
        dict(
            base_config
        )
    )

    model_config.pop(
        "_metadata",
        None,
    )

    model_config[
        "simulation"
    ]["population_size"] = (
        population_size
    )

    model_config[
        "simulation"
    ]["max_steps"] = (
        max_steps
    )

    if (
        int(
            model_config[
                "network"
            ]["attachment_m"]
        )
        >= population_size
    ):
        raise ExperimentConfigurationError(
            "population_size must exceed network.attachment_m."
        )

    dynamics = NeutralOperationalDynamics(
        base_config=model_config,
        dynamics_config=dynamics_config,
    )

    return SGBModel(
        config=model_config,
        seed=seed,
        dynamics=dynamics,
    )


def _mean_framework_scores(
    model: SGBModel,
    definitions: Mapping[str, Any],
) -> dict[str, float]:
    scored_agents = evaluate_agents(
        model.agents.to_list(),
        definitions,
    )

    return {
        framework: float(
            scored_agents[
                f"GMI_{framework}"
            ].mean()
        )
        for framework in FRAMEWORKS
    }


def _classification_metrics(
    *,
    endpoint_scores: np.ndarray,
    outcomes: np.ndarray,
    volumes: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    predicted = (
        endpoint_scores
        >= threshold
    )

    true_positive = int(
        np.sum(
            predicted
            & outcomes
        )
    )

    false_positive = int(
        np.sum(
            predicted
            & ~outcomes
        )
    )

    false_negative = int(
        np.sum(
            ~predicted
            & outcomes
        )
    )

    true_negative = int(
        np.sum(
            ~predicted
            & ~outcomes
        )
    )

    precision = _safe_ratio(
        true_positive,
        true_positive
        + false_positive,
    )

    recall = _safe_ratio(
        true_positive,
        true_positive
        + false_negative,
    )

    specificity = _safe_ratio(
        true_negative,
        true_negative
        + false_positive,
    )

    f1_score = _safe_ratio(
        2.0
        * precision
        * recall,
        precision
        + recall,
    )

    balanced_accuracy = (
        recall
        + specificity
    ) / 2.0

    event_coverage = float(
        np.mean(
            predicted
        )
    )

    total_volume = float(
        np.sum(
            volumes
        )
    )

    eligible_volume = float(
        np.sum(
            volumes[
                predicted
            ]
        )
    )

    volume_coverage = _safe_ratio(
        eligible_volume,
        total_volume,
    )

    eligible_success_rate = _safe_ratio(
        true_positive,
        true_positive
        + false_positive,
    )

    return {
        "true_positive": (
            true_positive
        ),
        "false_positive": (
            false_positive
        ),
        "false_negative": (
            false_negative
        ),
        "true_negative": (
            true_negative
        ),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1_score": f1_score,
        "balanced_accuracy": (
            balanced_accuracy
        ),
        "event_coverage": (
            event_coverage
        ),
        "volume_coverage": (
            volume_coverage
        ),
        "eligible_success_rate": (
            eligible_success_rate
        ),
    }


def _select_threshold(
    rows: Sequence[
        Mapping[str, Any]
    ],
) -> float:
    if not rows:
        raise ExperimentError(
            "Threshold result collection is empty."
        )

    selected = sorted(
        rows,
        key=lambda row: (
            -float(
                row[
                    "f1_score"
                ]
            ),
            -float(
                row[
                    "balanced_accuracy"
                ]
            ),
            float(
                row[
                    "threshold"
                ]
            ),
        ),
    )[0]

    return float(
        selected[
            "threshold"
        ]
    )


def _float_range(
    minimum: float,
    maximum: float,
    step: float,
) -> list[float]:
    if step <= 0.0:
        raise ExperimentConfigurationError(
            "Range step must be positive."
        )

    values: list[
        float
    ] = []

    current = minimum

    while current <= (
        maximum
        + step
        * 1e-9
    ):
        values.append(
            round(
                min(
                    current,
                    maximum,
                ),
                10,
            )
        )

        current += step

    if values[
        -1
    ] != round(
        maximum,
        10,
    ):
        values.append(
            round(
                maximum,
                10,
            )
        )

    return sorted(
        set(
            values
        )
    )


def _config_with_maturity_distribution(
    *,
    base_config: Mapping[str, Any],
    positive_alpha: float,
    positive_beta: float,
) -> dict[str, Any]:
    config = deepcopy(
        dict(
            base_config
        )
    )

    config.pop(
        "_metadata",
        None,
    )

    distributions = config[
        "baseline"
    ]["initialization"][
        "submetric_distributions"
    ]

    for submetric in distributions:
        if submetric in INVERSE_SUBMETRICS:
            alpha = positive_beta
            beta = positive_alpha
        else:
            alpha = positive_alpha
            beta = positive_beta

        distributions[
            submetric
        ] = {
            "alpha": float(
                alpha
            ),
            "beta": float(
                beta
            ),
        }

    return config


def _plot_experiment_1(
    result: ExperimentResult,
    path: Path,
) -> None:
    summary = result.tables[
        "framework_summary"
    ]

    figure, axis = plt.subplots(
        figsize=(
            8,
            5,
        )
    )

    errors = (
        summary[
            "ci_upper"
        ]
        - summary[
            "ci_lower"
        ]
    ) / 2.0

    axis.bar(
        summary[
            "framework"
        ],
        summary[
            "mean"
        ],
        yerr=errors,
        capsize=4,
    )

    axis.set(
        xlabel="Framework",
        ylabel="E-GMI",
        title="Experiment 1: framework comparison",
    )

    figure.tight_layout()
    figure.savefig(
        path,
        dpi=200,
    )
    plt.close(
        figure
    )


def _plot_experiment_2(
    result: ExperimentResult,
    path: Path,
) -> None:
    trajectory = result.tables[
        "recovery_trajectory"
    ]

    summary = (
        trajectory
        .groupby(
            [
                "framework",
                "recovery_step",
            ],
            as_index=False,
        )[
            "E_GMI"
        ]
        .mean()
    )

    figure, axis = plt.subplots(
        figsize=(
            9,
            5,
        )
    )

    for framework in FRAMEWORKS:
        subset = summary.loc[
            summary[
                "framework"
            ]
            == framework
        ]

        axis.plot(
            subset[
                "recovery_step"
            ],
            subset[
                "E_GMI"
            ],
            label=framework,
        )

    axis.set(
        xlabel="Steps after shock",
        ylabel="Mean E-GMI",
        title="Experiment 2: security recovery",
    )

    axis.legend()
    figure.tight_layout()
    figure.savefig(
        path,
        dpi=200,
    )
    plt.close(
        figure
    )


def _plot_experiment_3(
    result: ExperimentResult,
    path: Path,
) -> None:
    summary = result.tables[
        "threshold_summary"
    ]

    figure, axis = plt.subplots(
        figsize=(
            9,
            5,
        )
    )

    for framework in FRAMEWORKS:
        subset = summary.loc[
            summary[
                "framework"
            ]
            == framework
        ].sort_values(
            "threshold"
        )

        axis.plot(
            subset[
                "threshold"
            ],
            subset[
                "f1_score_mean"
            ],
            label=framework,
        )

    axis.set(
        xlabel="GMI threshold",
        ylabel="Mean F1 score",
        title="Experiment 3: threshold selection",
    )

    axis.legend()
    figure.tight_layout()
    figure.savefig(
        path,
        dpi=200,
    )
    plt.close(
        figure
    )


def _plot_experiment_4(
    result: ExperimentResult,
    path: Path,
) -> None:
    summary = result.tables[
        "level_summary"
    ]

    figure, axis = plt.subplots(
        figsize=(
            9,
            5,
        )
    )

    for framework in FRAMEWORKS:
        subset = summary.loc[
            summary[
                "framework"
            ]
            == framework
        ].sort_values(
            "level_index"
        )

        axis.plot(
            subset[
                "theoretical_effective_mean"
            ],
            subset[
                "E_GMI_mean"
            ],
            marker="o",
            label=framework,
        )

    axis.set(
        xlabel="Theoretical effective maturity mean",
        ylabel="Terminal E-GMI",
        title="Experiment 4: maturity sweep",
    )

    axis.legend()
    figure.tight_layout()
    figure.savefig(
        path,
        dpi=200,
    )
    plt.close(
        figure
    )


def _safe_ratio(
    numerator: float,
    denominator: float,
) -> float:
    if math.isclose(
        denominator,
        0.0,
        abs_tol=1e-15,
    ):
        return 0.0

    return float(
        numerator
        / denominator
    )


def _seeds(
    config: Mapping[str, Any],
) -> range:
    seed_plan = config[
        "seed_plan"
    ]

    start = int(
        seed_plan[
            "start_seed"
        ]
    )

    count = int(
        seed_plan[
            "seed_count"
        ]
    )

    return range(
        start,
        start + count,
    )


def _validate_steady_state_config(
    steady: Mapping[str, Any],
) -> None:
    maximum_steps = _positive_integer(
        steady,
        "maximum_steps",
    )

    rolling_window = _positive_integer(
        steady,
        "rolling_window",
    )

    required_windows = _positive_integer(
        steady,
        "required_stable_windows",
    )

    minimum_step = _positive_integer(
        steady,
        "minimum_step",
    )

    _non_negative_number(
        steady,
        "slope_tolerance",
    )

    _non_negative_number(
        steady,
        "mean_difference_tolerance",
    )

    _non_negative_integer(
        steady,
        "stabilization_buffer",
    )

    metrics = steady.get(
        "metrics"
    )

    if (
        not isinstance(metrics, list)
        or not metrics
        or any(
            not isinstance(metric, str)
            or not metric.strip()
            for metric in metrics
        )
    ):
        raise ExperimentConfigurationError(
            "steady_state.metrics must be a non-empty list."
        )

    if (
        rolling_window
        * (
            required_windows
            + 1
        )
        > maximum_steps
    ):
        raise ExperimentConfigurationError(
            "steady_state.maximum_steps is too small."
        )

    if minimum_step > maximum_steps:
        raise ExperimentConfigurationError(
            "steady_state.minimum_step cannot exceed maximum_steps."
        )


def _mapping(
    container: Mapping[str, Any],
    key: str,
) -> Mapping[str, Any]:
    value = container.get(
        key
    )

    if not isinstance(
        value,
        Mapping,
    ):
        raise ExperimentConfigurationError(
            f"{key} must be a mapping."
        )

    return value


def _non_empty_string(
    container: Mapping[str, Any],
    key: str,
) -> str:
    value = container.get(
        key
    )

    if (
        not isinstance(value, str)
        or not value.strip()
    ):
        raise ExperimentConfigurationError(
            f"{key} must be a non-empty string."
        )

    return value.strip()


def _positive_integer(
    container: Mapping[str, Any],
    key: str,
) -> int:
    value = container.get(
        key
    )

    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ExperimentConfigurationError(
            f"{key} must be a positive integer."
        )

    return value


def _non_negative_integer(
    container: Mapping[str, Any],
    key: str,
) -> int:
    value = container.get(
        key
    )

    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ExperimentConfigurationError(
            f"{key} must be a non-negative integer."
        )

    return value


def _finite_number(
    container: Mapping[str, Any],
    key: str,
) -> float:
    return _number_value(
        container.get(
            key
        ),
        key,
    )


def _positive_number(
    container: Mapping[str, Any],
    key: str,
) -> float:
    value = _finite_number(
        container,
        key,
    )

    if value <= 0.0:
        raise ExperimentConfigurationError(
            f"{key} must be greater than zero."
        )

    return value


def _non_negative_number(
    container: Mapping[str, Any],
    key: str,
) -> float:
    value = _finite_number(
        container,
        key,
    )

    if value < 0.0:
        raise ExperimentConfigurationError(
            f"{key} must be non-negative."
        )

    return value


def _unit_number(
    container: Mapping[str, Any],
    key: str,
) -> float:
    return _unit_value(
        container.get(
            key
        ),
        key,
    )


def _unit_value(
    value: Any,
    name: str,
) -> float:
    numeric = _number_value(
        value,
        name,
    )

    if not 0.0 <= numeric <= 1.0:
        raise ExperimentConfigurationError(
            f"{name} must be in [0, 1]."
        )

    return numeric


def _number_value(
    value: Any,
    name: str,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(
            value,
            (int, float),
        )
    ):
        raise ExperimentConfigurationError(
            f"{name} must be numeric."
        )

    numeric = float(
        value
    )

    if not math.isfinite(
        numeric
    ):
        raise ExperimentConfigurationError(
            f"{name} must be finite."
        )

    return numeric


def _json_default(
    value: Any,
) -> Any:
    if isinstance(
        value,
        np.integer,
    ):
        return int(
            value
        )

    if isinstance(
        value,
        np.floating,
    ):
        return float(
            value
        )

    if isinstance(
        value,
        Path,
    ):
        return str(
            value
        )

    raise TypeError(
        f"Cannot serialize {type(value).__name__}."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m sgb.experiments",
        description="Run one SGB experiment.",
    )

    parser.add_argument(
        "--config",
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
    arguments = build_parser().parse_args(
        argv
    )

    try:
        experiment_config = (
            load_experiment_config(
                arguments.config
            )
        )

        base_config = load_config(
            experiment_config[
                "reference_config"
            ]
        )

        dynamics_config = (
            load_dynamics_config(
                experiment_config[
                    "dynamics_config"
                ]
            )
        )

        result = run_experiment(
            base_config,
            dynamics_config,
            experiment_config,
        )

        output_directory = (
            arguments.output
            if arguments.output
            is not None
            else experiment_config[
                "outputs"
            ]["directory"]
        )

        paths = save_experiment_result(
            result,
            output_directory,
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
        ExperimentConfigurationError,
        ExperimentError,
    ) as error:
        print(
            f"Experiment error: {error}"
        )

        return 2


if __name__ == "__main__":
    raise SystemExit(
        main()
    )