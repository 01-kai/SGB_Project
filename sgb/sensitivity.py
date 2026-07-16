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
from scipy.stats import spearmanr

from sgb.config import (
    EXPECTED_DIMENSION_SUBMETRICS,
    PROJECT_ROOT,
    load_config,
)
from sgb.convergence_utils import (
    framework_ranking,
    summarize_values,
)
from sgb.dimensions import (
    ALL_DIMENSIONS,
    compute_dimension_scores,
)
from sgb.dynamics import (
    NeutralOperationalDynamics,
    load_dynamics_config,
)
from sgb.frameworks import (
    build_framework_definitions,
    evaluate_agents,
    evaluate_snapshots,
)
from sgb.model import SGBModel
from sgb.perturbations import (
    apply_data_quality_shock,
    apply_regulatory_change,
    apply_security_incident,
)


SENSITIVITY_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "studies"
    / "sensitivity.yaml"
)

FRAMEWORKS = (
    "DBL",
    "DMM",
    "ISF",
)

DIRECT_CODES = {
    *(
        f"P{number}"
        for number in range(1, 15)
    ),
    "P21",
}

LINKED_CODES = {
    "P15",
    "P16",
    "P17",
    "P18",
    "P19",
    "P20",
}

DIMENSION_WEIGHT_CODES = {
    "P2": "D1",
    "P3": "D2",
    "P4": "D3",
    "P5": "D4",
    "P6": "D5",
    "P7": "SVC",
}

EXPECTED_OPERATIONS = {
    "P1": "framework_weights",
    "P2": "dimension_weights",
    "P3": "dimension_weights",
    "P4": "dimension_weights",
    "P5": "dimension_weights",
    "P6": "dimension_weights",
    "P7": "dimension_weights",
    "P8": "framework_weights",
    "P9": "organization_mix",
    "P10": "beta_concentration",
    "P11": "exchange_distribution",
    "P12": "network_attachment",
    "P13": "maturity_cutoffs",
    "P14": "shock_magnitude",
    "P21": "freshness_window",
}


class SensitivityConfigurationError(ValueError):
    """Raised when the sensitivity configuration is invalid."""


class SensitivityError(RuntimeError):
    """Raised when a sensitivity study cannot be completed."""


@dataclass(frozen=True, slots=True)
class SimulationProfile:
    """Execution profile for smoke or full sensitivity runs."""

    name: str
    start_seed: int
    seed_count: int
    population_size: int
    burn_in_steps: int
    observation_steps: int
    terminal_window: int

    @property
    def seeds(self) -> range:
        return range(
            self.start_seed,
            self.start_seed
            + self.seed_count,
        )


@dataclass(frozen=True, slots=True)
class RunArtifact:
    """Framework-independent result of one simulated trajectory."""

    seed: int
    operational_results: pd.DataFrame
    events: pd.DataFrame
    snapshots: list[dict[str, Any]]
    network_summary: dict[str, Any]


@dataclass(frozen=True, slots=True)
class SensitivityResult:
    """Complete outputs of one direct sensitivity run."""

    profile: str
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


def load_sensitivity_config(
    path: str | Path = SENSITIVITY_CONFIG_PATH,
) -> dict[str, Any]:
    """Load and validate the global sensitivity configuration."""

    config_path = Path(path)

    if not config_path.is_absolute():
        config_path = (
            PROJECT_ROOT
            / config_path
        )

    config_path = config_path.resolve()

    if not config_path.is_file():
        raise SensitivityConfigurationError(
            "Sensitivity configuration does not exist: "
            f"{config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise SensitivityConfigurationError(
            "Sensitivity configuration root must be a mapping."
        )

    validate_sensitivity_config(
        config
    )

    config["_metadata"] = {
        "source_path": str(
            config_path
        ),
    }

    return config


def validate_sensitivity_config(
    config: Mapping[str, Any],
) -> None:
    """Validate the complete P1-P21 sensitivity-study definition."""

    if config.get("schema_version") != 2:
        raise SensitivityConfigurationError(
            "schema_version must be 2."
        )

    if config.get("config_kind") != "study":
        raise SensitivityConfigurationError(
            "config_kind must be 'study'."
        )

    if config.get("study_type") != "sensitivity":
        raise SensitivityConfigurationError(
            "study_type must be 'sensitivity'."
        )

    configuration = _mapping(
        config,
        "configuration",
    )

    if configuration.get("status") not in {
        "candidate",
        "final",
    }:
        raise SensitivityConfigurationError(
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

    profiles = _mapping(
        config,
        "profiles",
    )

    for required_profile in {
        "smoke",
        "full",
    }:
        if required_profile not in profiles:
            raise SensitivityConfigurationError(
                f"Missing profile: {required_profile}."
            )

        _validate_profile(
            required_profile,
            profiles[
                required_profile
            ],
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
        raise SensitivityConfigurationError(
            "analysis.confidence_level must be in (0, 1)."
        )

    for key in {
        "framework_ranking_tolerance",
        "maximum_absolute_egmi_difference",
        "maximum_threshold_difference",
        "minimum_ranking_match_fraction",
        "minimum_dimension_rank_correlation",
        "minimum_maturity_classification_agreement",
    }:
        value = _finite_number(
            analysis,
            key,
        )

        if not 0.0 <= value <= 1.0:
            raise SensitivityConfigurationError(
                f"analysis.{key} must be in [0, 1]."
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

    coarse_minimum = _unit_number(
        coarse,
        "minimum",
    )
    coarse_maximum = _unit_number(
        coarse,
        "maximum",
    )
    coarse_step = _positive_number(
        coarse,
        "step",
    )

    if coarse_maximum <= coarse_minimum:
        raise SensitivityConfigurationError(
            "threshold_scan.coarse.maximum must exceed minimum."
        )

    if not math.isclose(
        coarse_minimum,
        0.0,
        abs_tol=1e-12,
    ) or not math.isclose(
        coarse_maximum,
        1.0,
        abs_tol=1e-12,
    ):
        raise SensitivityConfigurationError(
            "The coarse threshold scan must cover [0, 1]."
        )

    if coarse_step > 1.0:
        raise SensitivityConfigurationError(
            "threshold_scan.coarse.step cannot exceed 1."
        )

    _positive_number(
        refinement,
        "radius",
    )
    _positive_number(
        refinement,
        "step",
    )

    freshness = _mapping(
        config,
        "freshness_model",
    )

    _positive_integer(
        freshness,
        "baseline_window",
    )
    _non_empty_string(
        freshness,
        "rationale",
    )

    scenarios = config.get(
        "scenarios"
    )

    if (
        not isinstance(scenarios, list)
        or not scenarios
    ):
        raise SensitivityConfigurationError(
            "scenarios must be a non-empty list."
        )

    observed_codes: set[str] = set()
    observed_pairs: set[
        tuple[str, str]
    ] = set()

    for scenario in scenarios:
        if not isinstance(
            scenario,
            Mapping,
        ):
            raise SensitivityConfigurationError(
                "Every scenario must be a mapping."
            )

        code = _non_empty_string(
            scenario,
            "code",
        )
        name = _non_empty_string(
            scenario,
            "name",
        )
        operation = _non_empty_string(
            scenario,
            "operation",
        )

        if code not in DIRECT_CODES:
            raise SensitivityConfigurationError(
                f"Unsupported direct sensitivity code: {code}."
            )

        expected_operation = EXPECTED_OPERATIONS[
            code
        ]

        if operation != expected_operation:
            raise SensitivityConfigurationError(
                f"{code} must use operation "
                f"{expected_operation!r}."
            )

        if "value" not in scenario:
            raise SensitivityConfigurationError(
                f"{code}/{name} is missing value."
            )

        pair = (
            code,
            name,
        )

        if pair in observed_pairs:
            raise SensitivityConfigurationError(
                f"Duplicate scenario: {code}/{name}."
            )

        observed_pairs.add(
            pair
        )
        observed_codes.add(
            code
        )

    missing_direct_codes = (
        DIRECT_CODES
        - observed_codes
    )

    if missing_direct_codes:
        raise SensitivityConfigurationError(
            "Missing direct sensitivity scenarios: "
            + ", ".join(
                sorted(
                    missing_direct_codes
                )
            )
        )

    linked = _mapping(
        config,
        "linked_studies",
    )

    if set(linked) != LINKED_CODES:
        raise SensitivityConfigurationError(
            "linked_studies must contain P15-P20."
        )

    for code in sorted(
        LINKED_CODES
    ):
        linked_study = _mapping(
            linked,
            code,
        )

        _non_empty_string(
            linked_study,
            "type",
        )
        _non_empty_string(
            linked_study,
            "config",
        )

    _non_empty_string(
        _mapping(
            config,
            "outputs",
        ),
        "directory",
    )


def get_profile(
    study_config: Mapping[str, Any],
    profile_name: str,
) -> SimulationProfile:
    """Create a validated execution-profile object."""

    profiles = _mapping(
        study_config,
        "profiles",
    )

    if profile_name not in profiles:
        raise SensitivityConfigurationError(
            f"Unknown sensitivity profile: {profile_name!r}."
        )

    profile = profiles[
        profile_name
    ]

    _validate_profile(
        profile_name,
        profile,
    )

    terminal_window = int(
        profile[
            "terminal_window"
        ]
    )
    observation_steps = int(
        profile[
            "observation_steps"
        ]
    )

    if terminal_window > observation_steps:
        raise SensitivityConfigurationError(
            f"Profile {profile_name} terminal_window cannot "
            "exceed observation_steps."
        )

    return SimulationProfile(
        name=profile_name,
        start_seed=int(
            profile[
                "start_seed"
            ]
        ),
        seed_count=int(
            profile[
                "seed_count"
            ]
        ),
        population_size=int(
            profile[
                "population_size"
            ]
        ),
        burn_in_steps=int(
            profile[
                "burn_in_steps"
            ]
        ),
        observation_steps=(
            observation_steps
        ),
        terminal_window=(
            terminal_window
        ),
    )


def apply_scenario(
    base_config: Mapping[str, Any],
    scenario: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Return a modified configuration without changing the source mapping.

    Post-hoc operations such as maturity cutoffs and freshness windows do
    not alter the simulation configuration.
    """

    if not isinstance(base_config, Mapping):
        raise SensitivityConfigurationError(
            "base_config must be a mapping."
        )

    if not isinstance(scenario, Mapping):
        raise SensitivityConfigurationError(
            "scenario must be a mapping."
        )

    modified = deepcopy(
        dict(
            base_config
        )
    )

    modified.pop(
        "_metadata",
        None,
    )

    operation = str(
        scenario[
            "operation"
        ]
    )
    value = scenario[
        "value"
    ]

    if operation == "framework_weights":
        if not isinstance(value, Mapping):
            raise SensitivityConfigurationError(
                "framework_weights value must be a mapping."
            )

        for framework, weights in value.items():
            if framework not in modified[
                "frameworks"
            ]:
                raise SensitivityConfigurationError(
                    f"Unknown framework: {framework}."
                )

            if not isinstance(weights, Mapping):
                raise SensitivityConfigurationError(
                    f"Weights for {framework} must be a mapping."
                )

            modified[
                "frameworks"
            ][framework]["weights"] = {
                str(dimension): float(weight)
                for dimension, weight
                in weights.items()
            }

    elif operation == "dimension_weights":
        if not isinstance(value, Mapping):
            raise SensitivityConfigurationError(
                "dimension_weights value must be a mapping."
            )

        dimension = str(
            value[
                "dimension"
            ]
        )
        weights = value[
            "weights"
        ]

        if not isinstance(weights, Mapping):
            raise SensitivityConfigurationError(
                "dimension weights must be a mapping."
            )

        _set_dimension_weights(
            modified,
            dimension,
            weights,
        )

    elif operation == "organization_mix":
        if not isinstance(value, Mapping):
            raise SensitivityConfigurationError(
                "organization_mix value must be a mapping."
            )

        supplied_probabilities = value.get(
            "probabilities"
        )

        if not isinstance(
            supplied_probabilities,
            list,
        ):
            raise SensitivityConfigurationError(
                "organization_mix probabilities must be a list."
            )

        probability_mapping = modified[
            "population"
        ][
            "organization_type_distribution"
        ]["probabilities"]

        labels = list(
            probability_mapping
        )

        if len(labels) != len(
            supplied_probabilities
        ):
            raise SensitivityConfigurationError(
                "organization_mix probability count does not "
                "match the base organization types."
            )

        probabilities = [
            _unit_value(
                probability,
                "organization probability",
            )
            for probability
            in supplied_probabilities
        ]

        if not math.isclose(
            sum(probabilities),
            1.0,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise SensitivityConfigurationError(
                "organization probabilities must sum to 1.0."
            )

        modified[
            "population"
        ][
            "organization_type_distribution"
        ]["probabilities"] = dict(
            zip(
                labels,
                probabilities,
                strict=True,
            )
        )

    elif operation == "beta_concentration":
        if not isinstance(value, Mapping):
            raise SensitivityConfigurationError(
                "beta_concentration value must be a mapping."
            )

        concentration = _positive_value(
            value.get(
                "concentration"
            ),
            "concentration",
        )

        distributions = modified[
            "baseline"
        ]["initialization"][
            "submetric_distributions"
        ]

        for parameters in distributions.values():
            original_alpha = float(
                parameters[
                    "alpha"
                ]
            )
            original_beta = float(
                parameters[
                    "beta"
                ]
            )

            original_mean = (
                original_alpha
                / (
                    original_alpha
                    + original_beta
                )
            )

            parameters[
                "alpha"
            ] = (
                original_mean
                * concentration
            )
            parameters[
                "beta"
            ] = (
                (1.0 - original_mean)
                * concentration
            )

    elif operation == "exchange_distribution":
        if not isinstance(value, Mapping):
            raise SensitivityConfigurationError(
                "exchange_distribution value must be a mapping."
            )

        for key in {
            "minimum_volume",
            "maximum_volume",
            "pareto_alpha",
        }:
            modified[
                "exchange"
            ][key] = value[
                key
            ]

    elif operation == "network_attachment":
        if not isinstance(value, Mapping):
            raise SensitivityConfigurationError(
                "network_attachment value must be a mapping."
            )

        modified[
            "network"
        ]["attachment_m"] = int(
            value[
                "attachment_m"
            ]
        )

    elif operation in {
        "maturity_cutoffs",
        "shock_magnitude",
        "freshness_window",
    }:
        return modified

    else:
        raise SensitivityConfigurationError(
            f"Unsupported sensitivity operation: {operation!r}."
        )

    return modified


def transform_catalog_freshness(
    baseline_score: float,
    *,
    baseline_window: int,
    target_window: int,
) -> float:
    """
    Transform catalog freshness under an exponential age assumption.

    F(K) = 1 - (1 - F(K0)) ** (K / K0)
    """

    score = _unit_value(
        baseline_score,
        "baseline_score",
    )

    if (
        isinstance(baseline_window, bool)
        or not isinstance(
            baseline_window,
            int,
        )
        or baseline_window <= 0
    ):
        raise SensitivityConfigurationError(
            "baseline_window must be a positive integer."
        )

    if (
        isinstance(target_window, bool)
        or not isinstance(
            target_window,
            int,
        )
        or target_window <= 0
    ):
        raise SensitivityConfigurationError(
            "target_window must be a positive integer."
        )

    transformed = (
        1.0
        - (
            1.0
            - score
        ) ** (
            target_window
            / baseline_window
        )
    )

    return min(
        1.0,
        max(
            0.0,
            float(
                transformed
            ),
        ),
    )


def run_sensitivity_study(
    base_config: Mapping[str, Any],
    dynamics_config: Mapping[str, Any],
    study_config: Mapping[str, Any],
    *,
    profile_name: str,
    codes: Sequence[str] | None = None,
) -> SensitivityResult:
    """Run selected direct sensitivity studies using common random seeds."""

    validate_sensitivity_config(
        study_config
    )

    profile = get_profile(
        study_config,
        profile_name,
    )

    selected_codes = _resolve_codes(
        codes
    )

    selected_scenarios = [
        scenario
        for scenario in study_config[
            "scenarios"
        ]
        if scenario[
            "code"
        ] in selected_codes
    ]

    baseline_artifacts = {
        seed: _run_artifact(
            base_config=base_config,
            dynamics_config=(
                dynamics_config
            ),
            profile=profile,
            seed=seed,
        )
        for seed in profile.seeds
    }

    metric_rows: list[
        dict[str, Any]
    ] = []

    dimension_rank_rows: list[
        dict[str, Any]
    ] = []

    maturity_rows: list[
        dict[str, Any]
    ] = []

    freshness_rows: list[
        dict[str, Any]
    ] = []

    for code in sorted(
        selected_codes
    ):
        if code == "P14":
            continue

        for seed, artifact in (
            baseline_artifacts.items()
        ):
            metric_rows.extend(
                _score_artifact(
                    artifact=artifact,
                    scoring_config=(
                        base_config
                    ),
                    study_config=(
                        study_config
                    ),
                    profile=profile,
                    code=code,
                    scenario_name=(
                        "baseline"
                    ),
                    evaluation_basis=(
                        "shared_simulation"
                    ),
                )
            )

    for scenario in selected_scenarios:
        code = str(
            scenario[
                "code"
            ]
        )
        name = str(
            scenario[
                "name"
            ]
        )
        operation = str(
            scenario[
                "operation"
            ]
        )

        if operation == "framework_weights":
            scoring_config = apply_scenario(
                base_config,
                scenario,
            )

            for artifact in baseline_artifacts.values():
                metric_rows.extend(
                    _score_artifact(
                        artifact=artifact,
                        scoring_config=(
                            scoring_config
                        ),
                        study_config=(
                            study_config
                        ),
                        profile=profile,
                        code=code,
                        scenario_name=name,
                        evaluation_basis=(
                            "posthoc_framework_scoring"
                        ),
                    )
                )

        elif operation in {
            "dimension_weights",
            "organization_mix",
            "beta_concentration",
            "exchange_distribution",
            "network_attachment",
        }:
            scenario_config = apply_scenario(
                base_config,
                scenario,
            )

            for seed in profile.seeds:
                scenario_artifact = (
                    _run_artifact(
                        base_config=(
                            scenario_config
                        ),
                        dynamics_config=(
                            dynamics_config
                        ),
                        profile=profile,
                        seed=seed,
                    )
                )

                metric_rows.extend(
                    _score_artifact(
                        artifact=(
                            scenario_artifact
                        ),
                        scoring_config=(
                            scenario_config
                        ),
                        study_config=(
                            study_config
                        ),
                        profile=profile,
                        code=code,
                        scenario_name=name,
                        evaluation_basis=(
                            "rerun_simulation"
                        ),
                    )
                )

                if code in DIMENSION_WEIGHT_CODES:
                    dimension = (
                        DIMENSION_WEIGHT_CODES[
                            code
                        ]
                    )

                    correlation = (
                        _dimension_rank_correlation(
                            baseline_artifacts[
                                seed
                            ].snapshots,
                            scenario_artifact.snapshots,
                            dimension,
                        )
                    )

                    dimension_rank_rows.append(
                        {
                            "code": code,
                            "scenario": name,
                            "seed": seed,
                            "dimension": dimension,
                            "spearman_rank_correlation": (
                                correlation
                            ),
                            "accepted": bool(
                                correlation
                                >= float(
                                    study_config[
                                        "analysis"
                                    ][
                                        "minimum_dimension_rank_correlation"
                                    ]
                                )
                            ),
                        }
                    )

        elif operation == "maturity_cutoffs":
            maturity_rows.extend(
                _evaluate_maturity_cutoffs(
                    base_config=base_config,
                    baseline_artifacts=(
                        baseline_artifacts
                    ),
                    study_config=(
                        study_config
                    ),
                    scenario=scenario,
                )
            )

        elif operation == "freshness_window":
            window = int(
                scenario[
                    "value"
                ]["window"]
            )

            for artifact in baseline_artifacts.values():
                adjusted_snapshots = (
                    _snapshots_with_freshness_window(
                        snapshots=(
                            artifact.snapshots
                        ),
                        base_config=(
                            base_config
                        ),
                        baseline_window=int(
                            study_config[
                                "freshness_model"
                            ][
                                "baseline_window"
                            ]
                        ),
                        target_window=window,
                    )
                )

                metric_rows.extend(
                    _score_artifact(
                        artifact=RunArtifact(
                            seed=artifact.seed,
                            operational_results=(
                                artifact.operational_results
                            ),
                            events=artifact.events,
                            snapshots=(
                                adjusted_snapshots
                            ),
                            network_summary=(
                                artifact.network_summary
                            ),
                        ),
                        scoring_config=(
                            base_config
                        ),
                        study_config=(
                            study_config
                        ),
                        profile=profile,
                        code=code,
                        scenario_name=name,
                        evaluation_basis=(
                            "posthoc_freshness_mapping"
                        ),
                    )
                )

                freshness_rows.append(
                    {
                        "code": code,
                        "scenario": name,
                        "seed": artifact.seed,
                        "window": window,
                        "mean_D3": float(
                            np.mean(
                                [
                                    snapshot[
                                        "dimensions"
                                    ]["D3"]
                                    for snapshot
                                    in adjusted_snapshots
                                ]
                            )
                        ),
                    }
                )

        elif operation == "shock_magnitude":
            continue

        else:
            raise SensitivityError(
                f"Unsupported operation during run: {operation}."
            )

    shock_rows = _run_shock_sensitivity(
        base_config=base_config,
        dynamics_config=dynamics_config,
        study_config=study_config,
        profile=profile,
        scenarios=selected_scenarios,
        selected_codes=selected_codes,
    )

    scenario_metrics = pd.DataFrame(
        metric_rows
    )

    scenario_summary = (
        _build_scenario_summary(
            scenario_metrics,
            study_config,
        )
        if not scenario_metrics.empty
        else pd.DataFrame()
    )

    stability_summary = (
        _build_stability_summary(
            scenario_metrics,
            dimension_rank_rows,
            study_config,
        )
        if not scenario_metrics.empty
        else pd.DataFrame()
    )

    dimension_rank_stability = (
        pd.DataFrame(
            dimension_rank_rows
        )
    )

    maturity_cutoff_stability = (
        pd.DataFrame(
            maturity_rows
        )
    )

    freshness_window_stability = (
        pd.DataFrame(
            freshness_rows
        )
    )

    shock_sensitivity = pd.DataFrame(
        shock_rows
    )

    shock_summary = (
        _build_shock_summary(
            shock_sensitivity,
            study_config,
        )
        if not shock_sensitivity.empty
        else pd.DataFrame()
    )

    coverage_matrix = build_coverage_matrix(
        study_config
    )

    direct_stability_passed = bool(
        stability_summary.empty
        or stability_summary[
            "accepted"
        ].all()
    )

    dimension_rank_passed = bool(
        dimension_rank_stability.empty
        or dimension_rank_stability[
            "accepted"
        ].all()
    )

    maturity_passed = bool(
        maturity_cutoff_stability.empty
        or maturity_cutoff_stability[
            "accepted"
        ].all()
    )

    public_calibration_complete = bool(
        study_config[
            "public_calibration"
        ]["P9_complete"]
    )

    simulation_passed = bool(
        direct_stability_passed
        and dimension_rank_passed
        and maturity_passed
    )

    overall_passed = bool(
        simulation_passed
        and (
            public_calibration_complete
            or "P9"
            not in selected_codes
        )
    )

    return SensitivityResult(
        profile=profile.name,
        tables={
            "scenario_metrics": (
                scenario_metrics
            ),
            "scenario_summary": (
                scenario_summary
            ),
            "stability_summary": (
                stability_summary
            ),
            "dimension_rank_stability": (
                dimension_rank_stability
            ),
            "maturity_cutoff_stability": (
                maturity_cutoff_stability
            ),
            "freshness_window_stability": (
                freshness_window_stability
            ),
            "shock_sensitivity": (
                shock_sensitivity
            ),
            "shock_summary": (
                shock_summary
            ),
            "coverage_matrix": (
                coverage_matrix
            ),
        },
        metadata={
            "study_type": (
                "sensitivity"
            ),
            "profile": profile.name,
            "selected_codes": sorted(
                selected_codes
            ),
            "start_seed": (
                profile.start_seed
            ),
            "seed_count": (
                profile.seed_count
            ),
            "population_size": (
                profile.population_size
            ),
            "burn_in_steps": (
                profile.burn_in_steps
            ),
            "observation_steps": (
                profile.observation_steps
            ),
            "common_random_numbers": True,
            "simulation_stability_passed": (
                simulation_passed
            ),
            "P9_public_calibration_complete": (
                public_calibration_complete
            ),
            "linked_studies_required": sorted(
                LINKED_CODES
            ),
            "passed": overall_passed,
        },
    )


def build_coverage_matrix(
    study_config: Mapping[str, Any],
) -> pd.DataFrame:
    """Return the implementation and resolution status of P1-P21."""

    linked = study_config[
        "linked_studies"
    ]

    categories = {
        "P1": "III",
        "P2": "III",
        "P3": "III",
        "P4": "III",
        "P5": "III",
        "P6": "III",
        "P7": "III",
        "P8": "III",
        "P9": "I",
        "P10": "II",
        "P11": "II",
        "P12": "II",
        "P13": "III",
        "P14": "II",
        "P15": "II",
        "P16": "II",
        "P17": "II",
        "P18": "II",
        "P19": "II",
        "P20": "II",
        "P21": "II",
    }

    rows: list[
        dict[str, Any]
    ] = []

    for number in range(
        1,
        22,
    ):
        code = f"P{number}"

        if code in DIRECT_CODES:
            implementation = (
                "sgb.sensitivity"
            )

            if code == "P9":
                resolution_status = (
                    "simulation_envelope_complete_"
                    "public_calibration_pending"
                )
            else:
                resolution_status = (
                    "direct_sensitivity_implemented"
                )

            config_path = (
                "configs/studies/sensitivity.yaml"
            )
        else:
            linked_config = linked[
                code
            ]

            implementation = str(
                linked_config[
                    "type"
                ]
            )
            config_path = str(
                linked_config[
                    "config"
                ]
            )
            resolution_status = (
                "linked_study_implemented"
            )

        rows.append(
            {
                "code": code,
                "category": categories[
                    code
                ],
                "implementation": (
                    implementation
                ),
                "config": config_path,
                "resolution_status": (
                    resolution_status
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def save_sensitivity_result(
    result: SensitivityResult,
    output_directory: str | Path,
) -> dict[str, Path]:
    """Save every sensitivity table, metadata, and robustness figure."""

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
        / "sensitivity_robustness.png"
    )

    _plot_sensitivity(
        result.tables[
            "stability_summary"
        ],
        figure_path,
    )

    paths[
        "figure"
    ] = figure_path

    return paths


def _run_artifact(
    *,
    base_config: Mapping[str, Any],
    dynamics_config: Mapping[str, Any],
    profile: SimulationProfile,
    seed: int,
) -> RunArtifact:
    model = _create_model(
        base_config=base_config,
        dynamics_config=dynamics_config,
        profile=profile,
        seed=seed,
    )

    for _ in range(
        profile.burn_in_steps
    ):
        model.step()
        model.event_records.clear()

    model.event_records.clear()

    for _ in range(
        profile.observation_steps
    ):
        model.step()

    return RunArtifact(
        seed=seed,
        operational_results=(
            model.get_results()
        ),
        events=model.get_events(),
        snapshots=(
            model.agent_snapshots()
        ),
        network_summary=(
            model.network_summary()
        ),
    )


def _create_model(
    *,
    base_config: Mapping[str, Any],
    dynamics_config: Mapping[str, Any],
    profile: SimulationProfile,
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
        profile.population_size
    )

    model_config[
        "simulation"
    ]["max_steps"] = (
        profile.burn_in_steps
        + profile.observation_steps
    )

    attachment_m = int(
        model_config[
            "network"
        ]["attachment_m"]
    )

    if attachment_m >= profile.population_size:
        raise SensitivityConfigurationError(
            "network.attachment_m must be smaller than "
            "the profile population_size."
        )

    dynamics = NeutralOperationalDynamics(
        base_config=model_config,
        dynamics_config=(
            dynamics_config
        ),
    )

    return SGBModel(
        config=model_config,
        seed=seed,
        dynamics=dynamics,
    )


def _score_artifact(
    *,
    artifact: RunArtifact,
    scoring_config: Mapping[str, Any],
    study_config: Mapping[str, Any],
    profile: SimulationProfile,
    code: str,
    scenario_name: str,
    evaluation_basis: str,
) -> list[dict[str, Any]]:
    definitions = build_framework_definitions(
        scoring_config
    )

    scored_agents = evaluate_snapshots(
        snapshots=artifact.snapshots,
        definitions=definitions,
    )

    framework_scores = {
        framework: float(
            scored_agents[
                f"GMI_{framework}"
            ].mean()
        )
        for framework in FRAMEWORKS
    }

    ranking = framework_ranking(
        {
            f"E_GMI_{framework}": score
            for framework, score
            in framework_scores.items()
        },
        tolerance=float(
            study_config[
                "analysis"
            ][
                "framework_ranking_tolerance"
            ]
        ),
    )

    thresholds = _recommend_thresholds(
        scored_agents=(
            scored_agents
        ),
        events=artifact.events,
        study_config=study_config,
    )

    terminal = (
        artifact.operational_results.tail(
            profile.terminal_window
        )
    )

    operational_means = {
        column: float(
            terminal[
                column
            ].mean()
        )
        for column in [
            "success_rate",
            "volume_success_rate",
            "mean_operational_readiness",
            *ALL_DIMENSIONS,
        ]
    }

    rows: list[
        dict[str, Any]
    ] = []

    for framework in FRAMEWORKS:
        rows.append(
            {
                "code": code,
                "scenario": (
                    scenario_name
                ),
                "seed": artifact.seed,
                "framework": framework,
                "E_GMI": (
                    framework_scores[
                        framework
                    ]
                ),
                "recommended_threshold": (
                    thresholds[
                        framework
                    ]
                ),
                "framework_ranking": (
                    ranking
                ),
                "evaluation_basis": (
                    evaluation_basis
                ),
                "network_edge_count": (
                    artifact.network_summary[
                        "edge_count"
                    ]
                ),
                "network_density": (
                    artifact.network_summary[
                        "density"
                    ]
                ),
                **operational_means,
            }
        )

    return rows


def _recommend_thresholds(
    *,
    scored_agents: pd.DataFrame,
    events: pd.DataFrame,
    study_config: Mapping[str, Any],
) -> dict[str, float]:
    if events.empty:
        raise SensitivityError(
            "Threshold sensitivity requires exchange events."
        )

    scan = study_config[
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

    outcomes = events[
        "success"
    ].to_numpy(
        dtype=bool
    )

    recommendations: dict[
        str,
        float,
    ] = {}

    for framework in FRAMEWORKS:
        score_map = dict(
            zip(
                scored_agents[
                    "uid"
                ].astype(int),
                scored_agents[
                    f"GMI_{framework}"
                ].astype(float),
                strict=True,
            )
        )

        endpoint_scores = np.minimum(
            events[
                "sender_uid"
            ].map(
                score_map
            ).to_numpy(
                dtype=float
            ),
            events[
                "receiver_uid"
            ].map(
                score_map
            ).to_numpy(
                dtype=float
            ),
        )

        coarse_rows = [
            {
                "threshold": threshold,
                **_threshold_metrics(
                    endpoint_scores,
                    outcomes,
                    threshold,
                ),
            }
            for threshold
            in coarse_thresholds
        ]

        coarse_best = _select_best_threshold(
            coarse_rows
        )

        refined_thresholds = _float_range(
            max(
                0.0,
                coarse_best
                - float(
                    refinement[
                        "radius"
                    ]
                ),
            ),
            min(
                1.0,
                coarse_best
                + float(
                    refinement[
                        "radius"
                    ]
                ),
            ),
            float(
                refinement[
                    "step"
                ]
            ),
        )

        refined_rows = [
            {
                "threshold": threshold,
                **_threshold_metrics(
                    endpoint_scores,
                    outcomes,
                    threshold,
                ),
            }
            for threshold
            in refined_thresholds
        ]

        recommendations[
            framework
        ] = _select_best_threshold(
            [
                *coarse_rows,
                *refined_rows,
            ]
        )

    return recommendations


def _threshold_metrics(
    endpoint_scores: np.ndarray,
    outcomes: np.ndarray,
    threshold: float,
) -> dict[str, float]:
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

    return {
        "f1_score": f1_score,
        "balanced_accuracy": (
            (
                recall
                + specificity
            )
            / 2.0
        ),
    }


def _select_best_threshold(
    rows: Sequence[
        Mapping[str, Any]
    ],
) -> float:
    if not rows:
        raise SensitivityError(
            "Threshold rows cannot be empty."
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


def _dimension_rank_correlation(
    baseline_snapshots: Sequence[
        Mapping[str, Any]
    ],
    scenario_snapshots: Sequence[
        Mapping[str, Any]
    ],
    dimension: str,
) -> float:
    baseline_map = {
        int(
            snapshot[
                "uid"
            ]
        ): float(
            snapshot[
                "dimensions"
            ][dimension]
        )
        for snapshot
        in baseline_snapshots
    }

    scenario_map = {
        int(
            snapshot[
                "uid"
            ]
        ): float(
            snapshot[
                "dimensions"
            ][dimension]
        )
        for snapshot
        in scenario_snapshots
    }

    if set(
        baseline_map
    ) != set(
        scenario_map
    ):
        raise SensitivityError(
            "Rank-correlation populations do not match."
        )

    ordered_uids = sorted(
        baseline_map
    )

    baseline_values = [
        baseline_map[
            uid
        ]
        for uid in ordered_uids
    ]

    scenario_values = [
        scenario_map[
            uid
        ]
        for uid in ordered_uids
    ]

    result = spearmanr(
        baseline_values,
        scenario_values,
    )

    correlation = float(
        result.statistic
    )

    if math.isnan(
        correlation
    ):
        return (
            1.0
            if np.allclose(
                baseline_values,
                scenario_values,
            )
            else 0.0
        )

    return correlation


def _evaluate_maturity_cutoffs(
    *,
    base_config: Mapping[str, Any],
    baseline_artifacts: Mapping[
        int,
        RunArtifact,
    ],
    study_config: Mapping[str, Any],
    scenario: Mapping[str, Any],
) -> list[dict[str, Any]]:
    p13_scenarios = [
        item
        for item in study_config[
            "scenarios"
        ]
        if item[
            "code"
        ] == "P13"
    ]

    baseline_scenario = next(
        (
            item
            for item in p13_scenarios
            if item[
                "name"
            ] == "equal_width"
        ),
        None,
    )

    if baseline_scenario is None:
        raise SensitivityConfigurationError(
            "P13 requires an equal_width scenario."
        )

    reference_cutoffs = list(
        baseline_scenario[
            "value"
        ]["cutoffs"]
    )

    scenario_cutoffs = list(
        scenario[
            "value"
        ]["cutoffs"]
    )

    definitions = build_framework_definitions(
        base_config
    )

    rows: list[
        dict[str, Any]
    ] = []

    for seed, artifact in (
        baseline_artifacts.items()
    ):
        scored = evaluate_snapshots(
            artifact.snapshots,
            definitions,
        )

        for framework in FRAMEWORKS:
            scores = scored[
                f"GMI_{framework}"
            ].to_numpy(
                dtype=float
            )

            reference_labels = [
                _maturity_index(
                    score,
                    reference_cutoffs,
                )
                for score in scores
            ]

            scenario_labels = [
                _maturity_index(
                    score,
                    scenario_cutoffs,
                )
                for score in scores
            ]

            agreement = float(
                np.mean(
                    np.asarray(
                        reference_labels
                    )
                    == np.asarray(
                        scenario_labels
                    )
                )
            )

            minimum_agreement = float(
                study_config[
                    "analysis"
                ][
                    "minimum_maturity_classification_agreement"
                ]
            )

            rows.append(
                {
                    "code": "P13",
                    "scenario": (
                        scenario[
                            "name"
                        ]
                    ),
                    "seed": seed,
                    "framework": framework,
                    "classification_agreement": (
                        agreement
                    ),
                    "minimum_required_agreement": (
                        minimum_agreement
                    ),
                    "accepted": bool(
                        agreement
                        >= minimum_agreement
                    ),
                }
            )

    return rows


def _snapshots_with_freshness_window(
    *,
    snapshots: Sequence[
        Mapping[str, Any]
    ],
    base_config: Mapping[str, Any],
    baseline_window: int,
    target_window: int,
) -> list[dict[str, Any]]:
    adjusted: list[
        dict[str, Any]
    ] = []

    for source_snapshot in snapshots:
        snapshot = deepcopy(
            dict(
                source_snapshot
            )
        )

        submetrics = deepcopy(
            snapshot[
                "submetrics"
            ]
        )

        baseline_freshness = float(
            submetrics[
                "D3"
            ][
                "catalog_freshness"
            ]
        )

        submetrics[
            "D3"
        ][
            "catalog_freshness"
        ] = transform_catalog_freshness(
            baseline_freshness,
            baseline_window=(
                baseline_window
            ),
            target_window=(
                target_window
            ),
        )

        dimensions = compute_dimension_scores(
            submetrics_by_dimension=(
                submetrics
            ),
            formula_config=(
                base_config[
                    "dimension_formulas"
                ]
            ),
        )

        snapshot[
            "submetrics"
        ] = submetrics
        snapshot[
            "dimensions"
        ] = dimensions

        adjusted.append(
            snapshot
        )

    return adjusted


def _run_shock_sensitivity(
    *,
    base_config: Mapping[str, Any],
    dynamics_config: Mapping[str, Any],
    study_config: Mapping[str, Any],
    profile: SimulationProfile,
    scenarios: Sequence[
        Mapping[str, Any]
    ],
    selected_codes: set[str],
) -> list[dict[str, Any]]:
    shock_scenarios: list[
        Mapping[str, Any]
    ] = []

    if "P12" in selected_codes:
        shock_scenarios.extend(
            scenario
            for scenario in scenarios
            if scenario[
                "code"
            ] == "P12"
        )

    if "P14" in selected_codes:
        shock_scenarios.extend(
            scenario
            for scenario in scenarios
            if scenario[
                "code"
            ] == "P14"
        )

    rows: list[
        dict[str, Any]
    ] = []

    for scenario in shock_scenarios:
        code = str(
            scenario[
                "code"
            ]
        )

        if code == "P12":
            scenario_config = apply_scenario(
                base_config,
                scenario,
            )
            shock_type = "security"
            impact = float(
                study_config[
                    "shock_defaults"
                ][
                    "security_impact"
                ]
            )
        else:
            scenario_config = deepcopy(
                dict(
                    base_config
                )
            )
            shock_type = str(
                scenario[
                    "value"
                ][
                    "shock_type"
                ]
            )
            impact = float(
                scenario[
                    "value"
                ]["impact"]
            )

        for seed in profile.seeds:
            model = _create_model(
                base_config=(
                    scenario_config
                ),
                dynamics_config=(
                    dynamics_config
                ),
                profile=profile,
                seed=seed,
            )

            for _ in range(
                profile.burn_in_steps
                + profile.observation_steps
            ):
                model.step()
                model.event_records.clear()

            definitions = (
                build_framework_definitions(
                    scenario_config
                )
            )

            pre_scores = evaluate_agents(
                model.agents.to_list(),
                definitions,
            )

            pre_means = {
                framework: float(
                    pre_scores[
                        f"GMI_{framework}"
                    ].mean()
                )
                for framework in FRAMEWORKS
            }

            if shock_type == "security":
                target_uid = max(
                    model.graph.degree,
                    key=lambda item: (
                        item[1],
                        -item[0],
                    ),
                )[0]

                records = (
                    apply_security_incident(
                        model=model,
                        target_uid=(
                            target_uid
                        ),
                        impact=impact,
                        max_distance=int(
                            study_config[
                                "shock_defaults"
                            ][
                                "maximum_network_distance"
                            ]
                        ),
                        decay=float(
                            study_config[
                                "shock_defaults"
                            ][
                                "distance_decay"
                            ]
                        ),
                    )
                )

                dimension = "D2"

            elif shock_type == "data_quality":
                records = (
                    apply_data_quality_shock(
                        model=model,
                        impact=impact,
                    )
                )
                dimension = "D1"

            elif shock_type == "regulatory":
                records = (
                    apply_regulatory_change(
                        model=model,
                        impact=impact,
                    )
                )
                dimension = "D4"

            else:
                raise SensitivityConfigurationError(
                    f"Unknown shock type: {shock_type}."
                )

            post_scores = evaluate_agents(
                model.agents.to_list(),
                definitions,
            )

            post_means = {
                framework: float(
                    post_scores[
                        f"GMI_{framework}"
                    ].mean()
                )
                for framework in FRAMEWORKS
            }

            dimension_losses = [
                float(
                    record[
                        f"{dimension}_before"
                    ]
                    - record[
                        f"{dimension}_after"
                    ]
                )
                for record in records
            ]

            pre_ranking = framework_ranking(
                {
                    f"E_GMI_{framework}": (
                        pre_means[
                            framework
                        ]
                    )
                    for framework
                    in FRAMEWORKS
                },
                tolerance=float(
                    study_config[
                        "analysis"
                    ][
                        "framework_ranking_tolerance"
                    ]
                ),
            )

            post_ranking = framework_ranking(
                {
                    f"E_GMI_{framework}": (
                        post_means[
                            framework
                        ]
                    )
                    for framework
                    in FRAMEWORKS
                },
                tolerance=float(
                    study_config[
                        "analysis"
                    ][
                        "framework_ranking_tolerance"
                    ]
                ),
            )

            for framework in FRAMEWORKS:
                rows.append(
                    {
                        "code": code,
                        "scenario": (
                            scenario[
                                "name"
                            ]
                        ),
                        "seed": seed,
                        "framework": framework,
                        "shock_type": (
                            shock_type
                        ),
                        "impact": impact,
                        "affected_organizations": (
                            len(
                                records
                            )
                        ),
                        "affected_fraction": (
                            len(
                                records
                            )
                            / profile.population_size
                        ),
                        "mean_dimension_loss": float(
                            np.mean(
                                dimension_losses
                            )
                        ),
                        "maximum_dimension_loss": float(
                            np.max(
                                dimension_losses
                            )
                        ),
                        "pre_shock_E_GMI": (
                            pre_means[
                                framework
                            ]
                        ),
                        "post_shock_E_GMI": (
                            post_means[
                                framework
                            ]
                        ),
                        "immediate_E_GMI_loss": (
                            pre_means[
                                framework
                            ]
                            - post_means[
                                framework
                            ]
                        ),
                        "pre_shock_ranking": (
                            pre_ranking
                        ),
                        "post_shock_ranking": (
                            post_ranking
                        ),
                        "ranking_preserved": (
                            pre_ranking
                            == post_ranking
                        ),
                    }
                )

    return rows


def _build_scenario_summary(
    scenario_metrics: pd.DataFrame,
    study_config: Mapping[str, Any],
) -> pd.DataFrame:
    confidence = float(
        study_config[
            "analysis"
        ]["confidence_level"]
    )

    rows: list[
        dict[str, Any]
    ] = []

    for (
        code,
        scenario,
        framework,
    ), subset in scenario_metrics.groupby(
        [
            "code",
            "scenario",
            "framework",
        ],
        sort=True,
    ):
        gmi_statistics = summarize_values(
            subset[
                "E_GMI"
            ],
            confidence_level=confidence,
        )

        threshold_statistics = (
            summarize_values(
                subset[
                    "recommended_threshold"
                ],
                confidence_level=confidence,
            )
        )

        rows.append(
            {
                "code": code,
                "scenario": scenario,
                "framework": framework,
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
                "threshold_mean": (
                    threshold_statistics[
                        "mean"
                    ]
                ),
                "threshold_ci_lower": (
                    threshold_statistics[
                        "ci_lower"
                    ]
                ),
                "threshold_ci_upper": (
                    threshold_statistics[
                        "ci_upper"
                    ]
                ),
                "success_rate_mean": float(
                    subset[
                        "success_rate"
                    ].mean()
                ),
                "ranking_match_fraction_within_scenario": float(
                    subset[
                        "framework_ranking"
                    ].value_counts(
                        normalize=True
                    ).iloc[0]
                ),
                "seed_count": int(
                    subset[
                        "seed"
                    ].nunique()
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def _build_stability_summary(
    scenario_metrics: pd.DataFrame,
    dimension_rank_rows: Sequence[
        Mapping[str, Any]
    ],
    study_config: Mapping[str, Any],
) -> pd.DataFrame:
    analysis = study_config[
        "analysis"
    ]

    maximum_egmi_difference = float(
        analysis[
            "maximum_absolute_egmi_difference"
        ]
    )
    maximum_threshold_difference = float(
        analysis[
            "maximum_threshold_difference"
        ]
    )
    minimum_ranking_fraction = float(
        analysis[
            "minimum_ranking_match_fraction"
        ]
    )
    minimum_rank_correlation = float(
        analysis[
            "minimum_dimension_rank_correlation"
        ]
    )

    rank_table = pd.DataFrame(
        dimension_rank_rows
    )

    rows: list[
        dict[str, Any]
    ] = []

    for code in sorted(
        set(
            scenario_metrics[
                "code"
            ]
        )
    ):
        code_rows = scenario_metrics.loc[
            scenario_metrics[
                "code"
            ]
            == code
        ]

        baseline = code_rows.loc[
            code_rows[
                "scenario"
            ]
            == "baseline"
        ]

        if baseline.empty:
            continue

        for scenario in sorted(
            set(
                code_rows[
                    "scenario"
                ]
            )
            - {
                "baseline",
            }
        ):
            scenario_rows = code_rows.loc[
                code_rows[
                    "scenario"
                ]
                == scenario
            ]

            baseline_rankings = (
                baseline[
                    [
                        "seed",
                        "framework_ranking",
                    ]
                ]
                .drop_duplicates(
                    subset=[
                        "seed",
                    ]
                )
                .set_index(
                    "seed"
                )[
                    "framework_ranking"
                ]
            )

            scenario_rankings = (
                scenario_rows[
                    [
                        "seed",
                        "framework_ranking",
                    ]
                ]
                .drop_duplicates(
                    subset=[
                        "seed",
                    ]
                )
                .set_index(
                    "seed"
                )[
                    "framework_ranking"
                ]
            )

            common_seeds = (
                baseline_rankings.index.intersection(
                    scenario_rankings.index
                )
            )

            ranking_match_fraction = float(
                np.mean(
                    baseline_rankings.loc[
                        common_seeds
                    ].to_numpy()
                    == scenario_rankings.loc[
                        common_seeds
                    ].to_numpy()
                )
            )

            for framework in FRAMEWORKS:
                baseline_framework = baseline.loc[
                    baseline[
                        "framework"
                    ]
                    == framework
                ]

                scenario_framework = (
                    scenario_rows.loc[
                        scenario_rows[
                            "framework"
                        ]
                        == framework
                    ]
                )

                egmi_difference = abs(
                    float(
                        scenario_framework[
                            "E_GMI"
                        ].mean()
                    )
                    - float(
                        baseline_framework[
                            "E_GMI"
                        ].mean()
                    )
                )

                threshold_difference = abs(
                    float(
                        scenario_framework[
                            "recommended_threshold"
                        ].mean()
                    )
                    - float(
                        baseline_framework[
                            "recommended_threshold"
                        ].mean()
                    )
                )

                if (
                    not rank_table.empty
                    and code
                    in DIMENSION_WEIGHT_CODES
                ):
                    correlation_values = (
                        rank_table.loc[
                            (
                                rank_table[
                                    "code"
                                ]
                                == code
                            )
                            & (
                                rank_table[
                                    "scenario"
                                ]
                                == scenario
                            ),
                            "spearman_rank_correlation",
                        ]
                    )

                    mean_rank_correlation = float(
                        correlation_values.mean()
                    )
                else:
                    mean_rank_correlation = (
                        math.nan
                    )

                rank_correlation_accepted = bool(
                    math.isnan(
                        mean_rank_correlation
                    )
                    or mean_rank_correlation
                    >= minimum_rank_correlation
                )

                accepted = bool(
                    egmi_difference
                    <= maximum_egmi_difference
                    and threshold_difference
                    <= maximum_threshold_difference
                    and ranking_match_fraction
                    >= minimum_ranking_fraction
                    and rank_correlation_accepted
                )

                rows.append(
                    {
                        "code": code,
                        "scenario": scenario,
                        "framework": framework,
                        "absolute_E_GMI_difference": (
                            egmi_difference
                        ),
                        "maximum_allowed_E_GMI_difference": (
                            maximum_egmi_difference
                        ),
                        "absolute_threshold_difference": (
                            threshold_difference
                        ),
                        "maximum_allowed_threshold_difference": (
                            maximum_threshold_difference
                        ),
                        "ranking_match_fraction": (
                            ranking_match_fraction
                        ),
                        "minimum_required_ranking_match_fraction": (
                            minimum_ranking_fraction
                        ),
                        "mean_dimension_rank_correlation": (
                            mean_rank_correlation
                        ),
                        "minimum_required_dimension_rank_correlation": (
                            minimum_rank_correlation
                        ),
                        "accepted": accepted,
                    }
                )

    return pd.DataFrame(
        rows
    )


def _build_shock_summary(
    shock_sensitivity: pd.DataFrame,
    study_config: Mapping[str, Any],
) -> pd.DataFrame:
    confidence = float(
        study_config[
            "analysis"
        ]["confidence_level"]
    )

    rows: list[
        dict[str, Any]
    ] = []

    for (
        code,
        scenario,
        framework,
    ), subset in shock_sensitivity.groupby(
        [
            "code",
            "scenario",
            "framework",
        ],
        sort=True,
    ):
        loss_statistics = summarize_values(
            subset[
                "immediate_E_GMI_loss"
            ],
            confidence_level=confidence,
        )

        rows.append(
            {
                "code": code,
                "scenario": scenario,
                "framework": framework,
                "shock_type": str(
                    subset[
                        "shock_type"
                    ].iloc[0]
                ),
                "impact": float(
                    subset[
                        "impact"
                    ].iloc[0]
                ),
                "mean_affected_fraction": float(
                    subset[
                        "affected_fraction"
                    ].mean()
                ),
                "mean_dimension_loss": float(
                    subset[
                        "mean_dimension_loss"
                    ].mean()
                ),
                "mean_E_GMI_loss": (
                    loss_statistics[
                        "mean"
                    ]
                ),
                "E_GMI_loss_ci_lower": (
                    loss_statistics[
                        "ci_lower"
                    ]
                ),
                "E_GMI_loss_ci_upper": (
                    loss_statistics[
                        "ci_upper"
                    ]
                ),
                "ranking_preserved_fraction": float(
                    subset[
                        "ranking_preserved"
                    ].mean()
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def _set_dimension_weights(
    config: dict[str, Any],
    dimension: str,
    weights: Mapping[str, Any],
) -> None:
    if dimension not in EXPECTED_DIMENSION_SUBMETRICS:
        raise SensitivityConfigurationError(
            f"Unknown dimension: {dimension}."
        )

    expected_submetrics = (
        EXPECTED_DIMENSION_SUBMETRICS[
            dimension
        ]
    )

    supplied_submetrics = set(
        weights
    )

    if supplied_submetrics != expected_submetrics:
        raise SensitivityConfigurationError(
            f"Invalid submetrics for {dimension}."
        )

    numeric_weights = {
        submetric: _unit_value(
            weight,
            f"{dimension}.{submetric}",
        )
        for submetric, weight
        in weights.items()
    }

    if not math.isclose(
        sum(
            numeric_weights.values()
        ),
        1.0,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise SensitivityConfigurationError(
            f"{dimension} weights must sum to 1.0."
        )

    formula = config[
        "dimension_formulas"
    ][dimension]

    if (
        isinstance(formula, dict)
        and isinstance(
            formula.get(
                "weights"
            ),
            Mapping,
        )
    ):
        formula[
            "weights"
        ] = numeric_weights
        return

    if (
        isinstance(formula, dict)
        and expected_submetrics.issubset(
            set(
                formula
            )
        )
    ):
        for submetric, weight in (
            numeric_weights.items()
        ):
            formula[
                submetric
            ] = weight

        return

    raise SensitivityConfigurationError(
        f"Unsupported dimension formula structure for {dimension}."
    )


def _maturity_index(
    score: float,
    cutoffs: Sequence[float],
) -> int:
    numeric_score = _unit_value(
        score,
        "maturity score",
    )

    for index, cutoff in enumerate(
        cutoffs
    ):
        if numeric_score < float(
            cutoff
        ):
            return index

    return len(
        cutoffs
    )


def _resolve_codes(
    codes: Sequence[str] | None,
) -> set[str]:
    if codes is None or not codes:
        return set(
            DIRECT_CODES
        )

    normalized = {
        str(code).upper()
        for code in codes
    }

    if "ALL" in normalized:
        return set(
            DIRECT_CODES
        )

    invalid = (
        normalized
        - DIRECT_CODES
    )

    if invalid:
        raise SensitivityConfigurationError(
            "Unsupported direct sensitivity codes: "
            + ", ".join(
                sorted(
                    invalid
                )
            )
        )

    return normalized


def _float_range(
    minimum: float,
    maximum: float,
    step: float,
) -> list[float]:
    if step <= 0.0:
        raise SensitivityConfigurationError(
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


def _plot_sensitivity(
    stability_summary: pd.DataFrame,
    path: Path,
) -> None:
    figure, axis = plt.subplots(
        figsize=(
            12,
            6,
        )
    )

    if stability_summary.empty:
        axis.text(
            0.5,
            0.5,
            "No comparative stability rows were generated.",
            horizontalalignment="center",
            verticalalignment="center",
        )
        axis.set_axis_off()
    else:
        plot_data = (
            stability_summary.groupby(
                [
                    "code",
                    "scenario",
                ],
                as_index=False,
            )[
                "absolute_E_GMI_difference"
            ]
            .max()
        )

        labels = (
            plot_data[
                "code"
            ]
            + "\n"
            + plot_data[
                "scenario"
            ]
        )

        axis.bar(
            labels,
            plot_data[
                "absolute_E_GMI_difference"
            ],
        )

        axis.set(
            xlabel="Sensitivity scenario",
            ylabel="Maximum absolute E-GMI difference",
            title="Global sensitivity robustness",
        )

        axis.tick_params(
            axis="x",
            rotation=90,
        )

    figure.tight_layout()

    figure.savefig(
        path,
        dpi=200,
    )

    plt.close(
        figure
    )


def _validate_profile(
    name: str,
    profile: Any,
) -> None:
    if not isinstance(
        profile,
        Mapping,
    ):
        raise SensitivityConfigurationError(
            f"Profile {name} must be a mapping."
        )

    _non_negative_integer(
        profile,
        "start_seed",
    )

    for key in {
        "seed_count",
        "population_size",
        "burn_in_steps",
        "observation_steps",
        "terminal_window",
    }:
        _positive_integer(
            profile,
            key,
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
        raise SensitivityConfigurationError(
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
        not isinstance(
            value,
            str,
        )
        or not value.strip()
    ):
        raise SensitivityConfigurationError(
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
        or not isinstance(
            value,
            int,
        )
        or value <= 0
    ):
        raise SensitivityConfigurationError(
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
        or not isinstance(
            value,
            int,
        )
        or value < 0
    ):
        raise SensitivityConfigurationError(
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
    return _positive_value(
        container.get(
            key
        ),
        key,
    )


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


def _positive_value(
    value: Any,
    name: str,
) -> float:
    numeric = _number_value(
        value,
        name,
    )

    if numeric <= 0.0:
        raise SensitivityConfigurationError(
            f"{name} must be greater than zero."
        )

    return numeric


def _unit_value(
    value: Any,
    name: str,
) -> float:
    numeric = _number_value(
        value,
        name,
    )

    if not 0.0 <= numeric <= 1.0:
        raise SensitivityConfigurationError(
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
            (
                int,
                float,
                np.integer,
                np.floating,
            ),
        )
    ):
        raise SensitivityConfigurationError(
            f"{name} must be numeric."
        )

    numeric = float(
        value
    )

    if not math.isfinite(
        numeric
    ):
        raise SensitivityConfigurationError(
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
    """Build the sensitivity command-line parser."""

    parser = argparse.ArgumentParser(
        prog="python -m sgb.sensitivity",
        description=(
            "Run direct P1-P14 and P21 sensitivity studies."
        ),
    )

    parser.add_argument(
        "--study",
        type=Path,
        default=(
            SENSITIVITY_CONFIG_PATH
        ),
    )

    parser.add_argument(
        "--profile",
        choices=[
            "smoke",
            "full",
        ],
        default="smoke",
    )

    parser.add_argument(
        "--codes",
        nargs="*",
        default=None,
        help=(
            "Direct parameter codes such as P1 P2 P14. "
            "Use ALL or omit this argument for every direct code."
        ),
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
    """Run global sensitivity analysis from the command line."""

    arguments = build_parser().parse_args(
        argv
    )

    try:
        study_config = load_sensitivity_config(
            arguments.study
        )

        base_config = load_config(
            study_config[
                "reference_config"
            ]
        )

        dynamics_config = load_dynamics_config(
            study_config[
                "dynamics_config"
            ]
        )

        result = run_sensitivity_study(
            base_config=base_config,
            dynamics_config=(
                dynamics_config
            ),
            study_config=(
                study_config
            ),
            profile_name=(
                arguments.profile
            ),
            codes=arguments.codes,
        )

        output_directory = (
            arguments.output
            if arguments.output
            is not None
            else study_config[
                "outputs"
            ]["directory"]
        )

        paths = save_sensitivity_result(
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
        SensitivityConfigurationError,
        SensitivityError,
    ) as error:
        print(
            f"Sensitivity error: {error}"
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(
        main()
    )