from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from scipy.stats import t as student_t

from sgb.config import (
    EXPECTED_DIMENSION_SUBMETRICS,
    PROJECT_ROOT,
    load_config,
)
from sgb.dimensions import (
    ALL_DIMENSIONS,
    STANDARD_DIMENSIONS,
    compute_dimension_scores,
)
from sgb.model import SGBModel


CALIBRATION_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "studies"
    / "baseline_calibration.yaml"
)

INVERSE_SUBMETRICS = frozenset(
    {
        "duplicate_rate",
        "normalized_breach_latency",
        "policy_violation_rate",
    }
)

SUBMETRIC_DIMENSIONS = {
    submetric: dimension
    for dimension, submetrics
    in EXPECTED_DIMENSION_SUBMETRICS.items()
    for submetric in submetrics
}


class CalibrationConfigurationError(ValueError):
    """Raised when baseline-calibration configuration is invalid."""


class CalibrationError(RuntimeError):
    """Raised when the baseline-calibration study cannot run."""


@dataclass(frozen=True, slots=True)
class BaselineCalibrationResult:
    """Complete outputs of one baseline-calibration study."""

    seed_dimension_means: pd.DataFrame
    seed_submetric_means: pd.DataFrame
    dimension_summary: pd.DataFrame
    submetric_summary: pd.DataFrame
    table_1: pd.DataFrame
    metadata: dict[str, Any]

    @property
    def passed(self) -> bool:
        """Return the overall calibration decision."""

        return bool(
            self.metadata["passed"]
        )


def load_calibration_config(
    path: str | Path = CALIBRATION_CONFIG_PATH,
) -> dict[str, Any]:
    """Load and validate the baseline-calibration configuration."""

    config_path = Path(path)

    if not config_path.is_absolute():
        config_path = (
            PROJECT_ROOT
            / config_path
        )

    config_path = config_path.resolve()

    if not config_path.exists():
        raise CalibrationConfigurationError(
            "Calibration configuration does not exist: "
            f"{config_path}"
        )

    if not config_path.is_file():
        raise CalibrationConfigurationError(
            "Calibration configuration path is not a file: "
            f"{config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise CalibrationConfigurationError(
            "Calibration configuration root must be a mapping."
        )

    validate_calibration_config(
        config
    )

    config["_metadata"] = {
        "source_path": str(
            config_path
        ),
    }

    return config


def validate_calibration_config(
    config: Mapping[str, Any],
) -> None:
    """Validate the complete calibration-study configuration."""

    if config.get(
        "schema_version"
    ) != 1:
        raise CalibrationConfigurationError(
            "schema_version must be 1."
        )

    if config.get(
        "config_kind"
    ) != "baseline_calibration":
        raise CalibrationConfigurationError(
            "config_kind must be "
            "'baseline_calibration'."
        )

    configuration = _require_mapping(
        config,
        "configuration",
    )
    seed_plan = _require_mapping(
        config,
        "seed_plan",
    )
    analysis = _require_mapping(
        config,
        "analysis",
    )
    acceptance = _require_mapping(
        config,
        "acceptance",
    )
    outputs = _require_mapping(
        config,
        "outputs",
    )

    status = configuration.get(
        "status"
    )

    if status not in {
        "candidate",
        "final",
    }:
        raise CalibrationConfigurationError(
            "configuration.status must be "
            "'candidate' or 'final'."
        )

    _require_non_empty_string(
        configuration,
        "description",
    )

    reference_config = config.get(
        "reference_config"
    )

    if (
        not isinstance(
            reference_config,
            str,
        )
        or not reference_config.strip()
    ):
        raise CalibrationConfigurationError(
            "reference_config must be a non-empty string."
        )

    _require_non_negative_integer(
        seed_plan,
        "start_seed",
    )

    _require_positive_integer(
        seed_plan,
        "seed_count",
    )

    _require_positive_integer(
        config,
        "population_size",
    )

    dimensions = config.get(
        "dimensions"
    )

    if dimensions != list(
        ALL_DIMENSIONS
    ):
        raise CalibrationConfigurationError(
            "dimensions must contain D1-D5 and SVC "
            "in the standard order."
        )

    table_1_dimensions = config.get(
        "table_1_dimensions"
    )

    if table_1_dimensions != list(
        STANDARD_DIMENSIONS
    ):
        raise CalibrationConfigurationError(
            "table_1_dimensions must contain D1-D5 "
            "in the standard order."
        )

    confidence_level = (
        _require_finite_number(
            analysis,
            "confidence_level",
        )
    )

    if not (
        0.0
        < confidence_level
        < 1.0
    ):
        raise CalibrationConfigurationError(
            "analysis.confidence_level must be in (0, 1)."
        )

    target_minimum = (
        _require_unit_interval(
            acceptance,
            "target_minimum",
        )
    )

    target_maximum = (
        _require_unit_interval(
            acceptance,
            "target_maximum",
        )
    )

    if (
        target_maximum
        <= target_minimum
    ):
        raise CalibrationConfigurationError(
            "acceptance.target_maximum must be greater "
            "than target_minimum."
        )

    maximum_dimension_error = (
        _require_unit_interval(
            acceptance,
            "maximum_dimension_mean_error",
        )
    )

    maximum_submetric_error = (
        _require_unit_interval(
            acceptance,
            "maximum_submetric_mean_error",
        )
    )

    if maximum_dimension_error <= 0.0:
        raise CalibrationConfigurationError(
            "maximum_dimension_mean_error "
            "must be greater than zero."
        )

    if maximum_submetric_error <= 0.0:
        raise CalibrationConfigurationError(
            "maximum_submetric_mean_error "
            "must be greater than zero."
        )

    require_ci = acceptance.get(
        "require_dimension_ci_within_target"
    )

    if not isinstance(
        require_ci,
        bool,
    ):
        raise CalibrationConfigurationError(
            "require_dimension_ci_within_target "
            "must be Boolean."
        )

    output_directory = outputs.get(
        "directory"
    )

    if (
        not isinstance(
            output_directory,
            str,
        )
        or not output_directory.strip()
    ):
        raise CalibrationConfigurationError(
            "outputs.directory must be a non-empty string."
        )


def beta_mean(
    alpha: float,
    beta: float,
) -> float:
    """Calculate the theoretical mean of Beta(alpha, beta)."""

    alpha_value = (
        _require_positive_number_value(
            alpha,
            "alpha",
        )
    )
    beta_value = (
        _require_positive_number_value(
            beta,
            "beta",
        )
    )

    return (
        alpha_value
        / (
            alpha_value
            + beta_value
        )
    )


def calculate_theoretical_submetric_means(
    base_config: Mapping[str, Any],
) -> dict[str, float]:
    """Calculate theoretical raw means for all initialized submetrics."""

    distributions = _get_submetric_distributions(
        base_config
    )

    expected_submetrics = set(
        SUBMETRIC_DIMENSIONS
    )
    actual_submetrics = set(
        distributions
    )

    missing = (
        expected_submetrics
        - actual_submetrics
    )
    unexpected = (
        actual_submetrics
        - expected_submetrics
    )

    if missing or unexpected:
        details: list[str] = []

        if missing:
            details.append(
                "missing="
                + ",".join(
                    sorted(missing)
                )
            )

        if unexpected:
            details.append(
                "unexpected="
                + ",".join(
                    sorted(unexpected)
                )
            )

        raise CalibrationConfigurationError(
            "Invalid baseline submetric distributions: "
            + "; ".join(details)
        )

    means: dict[str, float] = {}

    for submetric, parameters in (
        distributions.items()
    ):
        if not isinstance(
            parameters,
            Mapping,
        ):
            raise CalibrationConfigurationError(
                f"Distribution for {submetric} "
                "must be a mapping."
            )

        means[submetric] = beta_mean(
            alpha=parameters.get(
                "alpha"
            ),
            beta=parameters.get(
                "beta"
            ),
        )

    return means


def calculate_theoretical_dimension_means(
    base_config: Mapping[str, Any],
) -> dict[str, float]:
    """
    Calculate dimension means implied by the Beta distributions.

    Inverse metrics are transformed by the dimension formulas already
    implemented in ``sgb.dimensions``.
    """

    submetric_means = (
        calculate_theoretical_submetric_means(
            base_config
        )
    )

    submetrics_by_dimension = {
        dimension: {
            submetric: (
                submetric_means[
                    submetric
                ]
            )
            for submetric in sorted(
                EXPECTED_DIMENSION_SUBMETRICS[
                    dimension
                ]
            )
        }
        for dimension in ALL_DIMENSIONS
    }

    formula_config = base_config.get(
        "dimension_formulas"
    )

    if not isinstance(
        formula_config,
        Mapping,
    ):
        raise CalibrationConfigurationError(
            "base_config.dimension_formulas "
            "must be a mapping."
        )

    return compute_dimension_scores(
        submetrics_by_dimension=(
            submetrics_by_dimension
        ),
        formula_config=formula_config,
    )


def run_baseline_calibration(
    base_config: Mapping[str, Any],
    calibration_config: Mapping[str, Any],
) -> BaselineCalibrationResult:
    """
    Run the baseline study using the actual SGBModel initializer.

    No simulation steps are executed. Measurements represent the initial
    synthetic population before operational dynamics begin.
    """

    if not isinstance(
        base_config,
        Mapping,
    ):
        raise CalibrationConfigurationError(
            "base_config must be a mapping."
        )

    if not isinstance(
        calibration_config,
        Mapping,
    ):
        raise CalibrationConfigurationError(
            "calibration_config must be a mapping."
        )

    validate_calibration_config(
        calibration_config
    )

    _validate_target_alignment(
        base_config=base_config,
        calibration_config=(
            calibration_config
        ),
    )

    model_config = deepcopy(
        dict(base_config)
    )

    population_size = int(
        calibration_config[
            "population_size"
        ]
    )

    attachment_m = int(
        model_config[
            "network"
        ]["attachment_m"]
    )

    if attachment_m >= population_size:
        raise CalibrationConfigurationError(
            "Calibration population_size must be greater "
            "than network.attachment_m."
        )

    model_config[
        "simulation"
    ]["population_size"] = (
        population_size
    )

    seed_plan = calibration_config[
        "seed_plan"
    ]

    start_seed = int(
        seed_plan["start_seed"]
    )
    seed_count = int(
        seed_plan["seed_count"]
    )

    seeds = list(
        range(
            start_seed,
            start_seed
            + seed_count,
        )
    )

    theoretical_submetric_means = (
        calculate_theoretical_submetric_means(
            model_config
        )
    )

    theoretical_dimension_means = (
        calculate_theoretical_dimension_means(
            model_config
        )
    )

    seed_dimension_rows: list[
        dict[str, Any]
    ] = []

    seed_submetric_rows: list[
        dict[str, Any]
    ] = []

    for seed in seeds:
        model = SGBModel(
            config=model_config,
            seed=seed,
        )

        snapshots = (
            model.agent_snapshots()
        )

        if len(
            snapshots
        ) != population_size:
            raise CalibrationError(
                "Model generated an unexpected "
                "population size."
            )

        dimension_row: dict[
            str,
            Any,
        ] = {
            "seed": seed,
            "population_size": (
                population_size
            ),
        }

        for dimension in ALL_DIMENSIONS:
            dimension_row[
                dimension
            ] = float(
                sum(
                    snapshot[
                        "dimensions"
                    ][dimension]
                    for snapshot
                    in snapshots
                )
                / population_size
            )

        seed_dimension_rows.append(
            dimension_row
        )

        for dimension in ALL_DIMENSIONS:
            for submetric in sorted(
                EXPECTED_DIMENSION_SUBMETRICS[
                    dimension
                ]
            ):
                mean_value = float(
                    sum(
                        snapshot[
                            "submetrics"
                        ][dimension][
                            submetric
                        ]
                        for snapshot
                        in snapshots
                    )
                    / population_size
                )

                seed_submetric_rows.append(
                    {
                        "seed": seed,
                        "population_size": (
                            population_size
                        ),
                        "dimension": (
                            dimension
                        ),
                        "submetric": (
                            submetric
                        ),
                        "mean": mean_value,
                    }
                )

    seed_dimension_means = (
        pd.DataFrame(
            seed_dimension_rows
        )
    )

    seed_submetric_means = (
        pd.DataFrame(
            seed_submetric_rows
        )
    )

    dimension_summary = (
        _build_dimension_summary(
            seed_dimension_means=(
                seed_dimension_means
            ),
            theoretical_means=(
                theoretical_dimension_means
            ),
            calibration_config=(
                calibration_config
            ),
        )
    )

    submetric_summary = (
        _build_submetric_summary(
            seed_submetric_means=(
                seed_submetric_means
            ),
            theoretical_means=(
                theoretical_submetric_means
            ),
            base_config=model_config,
            calibration_config=(
                calibration_config
            ),
        )
    )

    table_1_dimensions = list(
        calibration_config[
            "table_1_dimensions"
        ]
    )

    table_1 = (
        dimension_summary
        .set_index(
            "dimension"
        )
        .loc[
            table_1_dimensions
        ]
        .reset_index()
        [
            [
                "dimension",
                "theoretical_mean",
                "empirical_mean",
                "ci_lower",
                "ci_upper",
                "absolute_mean_error",
                "target_minimum",
                "target_maximum",
                "accepted",
            ]
        ]
    )

    dimension_passed = bool(
        dimension_summary[
            "accepted"
        ].all()
    )

    submetric_passed = bool(
        submetric_summary[
            "mean_error_acceptable"
        ].all()
    )

    overall_passed = bool(
        dimension_passed
        and submetric_passed
    )

    analysis = calibration_config[
        "analysis"
    ]
    acceptance = calibration_config[
        "acceptance"
    ]

    metadata = {
        "study": "baseline_calibration",
        "status": (
            calibration_config[
                "configuration"
            ]["status"]
        ),
        "start_seed": start_seed,
        "seed_count": seed_count,
        "population_size": (
            population_size
        ),
        "confidence_level": float(
            analysis[
                "confidence_level"
            ]
        ),
        "target_minimum": float(
            acceptance[
                "target_minimum"
            ]
        ),
        "target_maximum": float(
            acceptance[
                "target_maximum"
            ]
        ),
        "maximum_dimension_mean_error": float(
            acceptance[
                "maximum_dimension_mean_error"
            ]
        ),
        "maximum_submetric_mean_error": float(
            acceptance[
                "maximum_submetric_mean_error"
            ]
        ),
        "dimension_acceptance_passed": (
            dimension_passed
        ),
        "submetric_acceptance_passed": (
            submetric_passed
        ),
        "passed": overall_passed,
    }

    return BaselineCalibrationResult(
        seed_dimension_means=(
            seed_dimension_means
        ),
        seed_submetric_means=(
            seed_submetric_means
        ),
        dimension_summary=(
            dimension_summary
        ),
        submetric_summary=(
            submetric_summary
        ),
        table_1=table_1,
        metadata=metadata,
    )


def save_calibration_result(
    result: BaselineCalibrationResult,
    output_directory: str | Path,
) -> dict[str, Path]:
    """Write all calibration outputs to disk."""

    if not isinstance(
        result,
        BaselineCalibrationResult,
    ):
        raise CalibrationError(
            "result must be a BaselineCalibrationResult."
        )

    output_path = Path(
        output_directory
    )

    if not output_path.is_absolute():
        output_path = (
            PROJECT_ROOT
            / output_path
        )

    output_path = (
        output_path.resolve()
    )

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = {
        "seed_dimension_means": (
            output_path
            / "baseline_seed_dimension_means.csv"
        ),
        "seed_submetric_means": (
            output_path
            / "baseline_seed_submetric_means.csv"
        ),
        "dimension_summary": (
            output_path
            / "baseline_dimension_summary.csv"
        ),
        "submetric_summary": (
            output_path
            / "baseline_submetric_summary.csv"
        ),
        "table_1": (
            output_path
            / "table_1_baseline_dimensions.csv"
        ),
        "metadata": (
            output_path
            / "baseline_calibration_metadata.json"
        ),
    }

    result.seed_dimension_means.to_csv(
        paths["seed_dimension_means"],
        index=False,
    )

    result.seed_submetric_means.to_csv(
        paths["seed_submetric_means"],
        index=False,
    )

    result.dimension_summary.to_csv(
        paths["dimension_summary"],
        index=False,
    )

    result.submetric_summary.to_csv(
        paths["submetric_summary"],
        index=False,
    )

    result.table_1.to_csv(
        paths["table_1"],
        index=False,
    )

    with paths["metadata"].open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result.metadata,
            file,
            indent=2,
            sort_keys=True,
        )

    return paths


def _build_dimension_summary(
    *,
    seed_dimension_means: pd.DataFrame,
    theoretical_means: Mapping[str, float],
    calibration_config: Mapping[str, Any],
) -> pd.DataFrame:
    acceptance = calibration_config[
        "acceptance"
    ]
    analysis = calibration_config[
        "analysis"
    ]

    target_minimum = float(
        acceptance[
            "target_minimum"
        ]
    )
    target_maximum = float(
        acceptance[
            "target_maximum"
        ]
    )
    maximum_error = float(
        acceptance[
            "maximum_dimension_mean_error"
        ]
    )
    confidence_level = float(
        analysis[
            "confidence_level"
        ]
    )
    require_ci = bool(
        acceptance[
            "require_dimension_ci_within_target"
        ]
    )

    records: list[
        dict[str, Any]
    ] = []

    for dimension in calibration_config[
        "dimensions"
    ]:
        statistics = (
            _summarize_replicates(
                seed_dimension_means[
                    dimension
                ],
                confidence_level=(
                    confidence_level
                ),
            )
        )

        theoretical_mean = float(
            theoretical_means[
                dimension
            ]
        )

        absolute_error = abs(
            statistics[
                "mean"
            ]
            - theoretical_mean
        )

        theoretical_within_target = bool(
            target_minimum
            <= theoretical_mean
            <= target_maximum
        )

        empirical_within_target = bool(
            target_minimum
            <= statistics["mean"]
            <= target_maximum
        )

        ci_within_target = bool(
            statistics[
                "ci_lower"
            ]
            >= target_minimum
            and statistics[
                "ci_upper"
            ]
            <= target_maximum
        )

        error_acceptable = bool(
            absolute_error
            <= maximum_error
        )

        accepted = bool(
            theoretical_within_target
            and empirical_within_target
            and error_acceptable
            and (
                ci_within_target
                if require_ci
                else True
            )
        )

        records.append(
            {
                "dimension": dimension,
                "theoretical_mean": (
                    theoretical_mean
                ),
                "empirical_mean": (
                    statistics["mean"]
                ),
                "standard_deviation_across_seeds": (
                    statistics[
                        "standard_deviation"
                    ]
                ),
                "ci_lower": (
                    statistics[
                        "ci_lower"
                    ]
                ),
                "ci_upper": (
                    statistics[
                        "ci_upper"
                    ]
                ),
                "ci_width": (
                    statistics[
                        "ci_upper"
                    ]
                    - statistics[
                        "ci_lower"
                    ]
                ),
                "absolute_mean_error": (
                    absolute_error
                ),
                "target_minimum": (
                    target_minimum
                ),
                "target_maximum": (
                    target_maximum
                ),
                "theoretical_within_target": (
                    theoretical_within_target
                ),
                "empirical_within_target": (
                    empirical_within_target
                ),
                "ci_within_target": (
                    ci_within_target
                ),
                "mean_error_acceptable": (
                    error_acceptable
                ),
                "accepted": accepted,
                "seed_count": int(
                    statistics[
                        "sample_count"
                    ]
                ),
            }
        )

    return pd.DataFrame(
        records
    )


def _build_submetric_summary(
    *,
    seed_submetric_means: pd.DataFrame,
    theoretical_means: Mapping[str, float],
    base_config: Mapping[str, Any],
    calibration_config: Mapping[str, Any],
) -> pd.DataFrame:
    confidence_level = float(
        calibration_config[
            "analysis"
        ]["confidence_level"]
    )

    maximum_error = float(
        calibration_config[
            "acceptance"
        ][
            "maximum_submetric_mean_error"
        ]
    )

    distributions = (
        _get_submetric_distributions(
            base_config
        )
    )

    records: list[
        dict[str, Any]
    ] = []

    for dimension in ALL_DIMENSIONS:
        for submetric in sorted(
            EXPECTED_DIMENSION_SUBMETRICS[
                dimension
            ]
        ):
            values = (
                seed_submetric_means.loc[
                    (
                        seed_submetric_means[
                            "submetric"
                        ]
                        == submetric
                    ),
                    "mean",
                ]
            )

            statistics = (
                _summarize_replicates(
                    values,
                    confidence_level=(
                        confidence_level
                    ),
                )
            )

            theoretical_mean = float(
                theoretical_means[
                    submetric
                ]
            )

            absolute_error = abs(
                statistics[
                    "mean"
                ]
                - theoretical_mean
            )

            parameters = distributions[
                submetric
            ]

            records.append(
                {
                    "dimension": dimension,
                    "submetric": submetric,
                    "orientation": (
                        "inverse"
                        if submetric
                        in INVERSE_SUBMETRICS
                        else "positive"
                    ),
                    "alpha": float(
                        parameters[
                            "alpha"
                        ]
                    ),
                    "beta": float(
                        parameters[
                            "beta"
                        ]
                    ),
                    "theoretical_raw_mean": (
                        theoretical_mean
                    ),
                    "empirical_raw_mean": (
                        statistics[
                            "mean"
                        ]
                    ),
                    "standard_deviation_across_seeds": (
                        statistics[
                            "standard_deviation"
                        ]
                    ),
                    "ci_lower": (
                        statistics[
                            "ci_lower"
                        ]
                    ),
                    "ci_upper": (
                        statistics[
                            "ci_upper"
                        ]
                    ),
                    "absolute_mean_error": (
                        absolute_error
                    ),
                    "maximum_allowed_error": (
                        maximum_error
                    ),
                    "mean_error_acceptable": bool(
                        absolute_error
                        <= maximum_error
                    ),
                    "seed_count": int(
                        statistics[
                            "sample_count"
                        ]
                    ),
                }
            )

    return pd.DataFrame(
        records
    )


def _summarize_replicates(
    values: pd.Series,
    *,
    confidence_level: float,
) -> dict[str, float | int]:
    try:
        numeric_values = pd.to_numeric(
            values,
            errors="raise",
        ).astype(float)
    except (
        TypeError,
        ValueError,
    ) as error:
        raise CalibrationError(
            "Replicate values must be numeric."
        ) from error

    if numeric_values.empty:
        raise CalibrationError(
            "Cannot summarize empty replicate values."
        )

    if numeric_values.isna().any():
        raise CalibrationError(
            "Replicate values cannot contain missing values."
        )

    sample_count = int(
        len(numeric_values)
    )

    mean_value = float(
        numeric_values.mean()
    )

    if sample_count == 1:
        standard_deviation = 0.0
        ci_lower = mean_value
        ci_upper = mean_value
    else:
        standard_deviation = float(
            numeric_values.std(
                ddof=1
            )
        )

        critical_value = float(
            student_t.ppf(
                (
                    1.0
                    + confidence_level
                )
                / 2.0,
                df=(
                    sample_count
                    - 1
                ),
            )
        )

        standard_error = (
            standard_deviation
            / math.sqrt(
                sample_count
            )
        )

        half_width = (
            critical_value
            * standard_error
        )

        ci_lower = (
            mean_value
            - half_width
        )
        ci_upper = (
            mean_value
            + half_width
        )

    return {
        "sample_count": sample_count,
        "mean": mean_value,
        "standard_deviation": (
            standard_deviation
        ),
        "ci_lower": float(
            ci_lower
        ),
        "ci_upper": float(
            ci_upper
        ),
    }


def _validate_target_alignment(
    *,
    base_config: Mapping[str, Any],
    calibration_config: Mapping[str, Any],
) -> None:
    try:
        base_target = base_config[
            "baseline"
        ]["target_dimension_range"]
    except (
        KeyError,
        TypeError,
    ) as error:
        raise CalibrationConfigurationError(
            "Base configuration is missing "
            "baseline.target_dimension_range."
        ) from error

    study_target = calibration_config[
        "acceptance"
    ]

    base_minimum = float(
        base_target["minimum"]
    )
    base_maximum = float(
        base_target["maximum"]
    )

    study_minimum = float(
        study_target[
            "target_minimum"
        ]
    )
    study_maximum = float(
        study_target[
            "target_maximum"
        ]
    )

    if not (
        math.isclose(
            base_minimum,
            study_minimum,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            base_maximum,
            study_maximum,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise CalibrationConfigurationError(
            "Calibration target range must match "
            "baseline.target_dimension_range."
        )


def _get_submetric_distributions(
    base_config: Mapping[str, Any],
) -> Mapping[str, Any]:
    try:
        distributions = base_config[
            "baseline"
        ]["initialization"][
            "submetric_distributions"
        ]
    except (
        KeyError,
        TypeError,
    ) as error:
        raise CalibrationConfigurationError(
            "Base configuration is missing "
            "baseline submetric distributions."
        ) from error

    if not isinstance(
        distributions,
        Mapping,
    ):
        raise CalibrationConfigurationError(
            "Baseline submetric distributions "
            "must be a mapping."
        )

    return distributions


def _require_mapping(
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
        raise CalibrationConfigurationError(
            f"{key} must be a mapping."
        )

    return value


def _require_non_empty_string(
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
        raise CalibrationConfigurationError(
            f"{key} must be a non-empty string."
        )

    return value.strip()


def _require_non_negative_integer(
    container: Mapping[str, Any],
    key: str,
) -> int:
    value = container.get(
        key
    )

    if (
        isinstance(
            value,
            bool,
        )
        or not isinstance(
            value,
            int,
        )
    ):
        raise CalibrationConfigurationError(
            f"{key} must be an integer."
        )

    if value < 0:
        raise CalibrationConfigurationError(
            f"{key} must be zero or greater."
        )

    return value


def _require_positive_integer(
    container: Mapping[str, Any],
    key: str,
) -> int:
    value = container.get(
        key
    )

    if (
        isinstance(
            value,
            bool,
        )
        or not isinstance(
            value,
            int,
        )
    ):
        raise CalibrationConfigurationError(
            f"{key} must be an integer."
        )

    if value <= 0:
        raise CalibrationConfigurationError(
            f"{key} must be greater than zero."
        )

    return value


def _require_finite_number(
    container: Mapping[str, Any],
    key: str,
) -> float:
    return _require_finite_number_value(
        container.get(
            key
        ),
        key,
    )


def _require_unit_interval(
    container: Mapping[str, Any],
    key: str,
) -> float:
    value = _require_finite_number(
        container,
        key,
    )

    if not (
        0.0
        <= value
        <= 1.0
    ):
        raise CalibrationConfigurationError(
            f"{key} must be in [0, 1]."
        )

    return value


def _require_positive_number_value(
    value: Any,
    name: str,
) -> float:
    numeric_value = (
        _require_finite_number_value(
            value,
            name,
        )
    )

    if numeric_value <= 0.0:
        raise CalibrationConfigurationError(
            f"{name} must be greater than zero."
        )

    return numeric_value


def _require_finite_number_value(
    value: Any,
    name: str,
) -> float:
    if (
        isinstance(
            value,
            bool,
        )
        or not isinstance(
            value,
            (int, float),
        )
    ):
        raise CalibrationConfigurationError(
            f"{name} must be numeric."
        )

    numeric_value = float(
        value
    )

    if not math.isfinite(
        numeric_value
    ):
        raise CalibrationConfigurationError(
            f"{name} must be finite."
        )

    return numeric_value


def build_parser() -> argparse.ArgumentParser:
    """Build the calibration command-line parser."""

    parser = argparse.ArgumentParser(
        prog="python -m sgb.calibration",
        description=(
            "Run the SGB baseline-calibration study."
        ),
    )

    parser.add_argument(
        "--study",
        type=Path,
        default=(
            CALIBRATION_CONFIG_PATH
        ),
        help=(
            "Path to the baseline-calibration YAML file."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional output-directory override."
        ),
    )

    parser.add_argument(
        "--fail-on-rejection",
        action="store_true",
        help=(
            "Return a non-zero exit code when calibration "
            "acceptance criteria are not satisfied."
        ),
    )

    return parser


def main(
    argv: Sequence[str] | None = None,
) -> int:
    """Run baseline calibration from the command line."""

    parser = build_parser()
    arguments = parser.parse_args(
        argv
    )

    try:
        calibration_config = (
            load_calibration_config(
                arguments.study
            )
        )

        base_config = load_config(
            calibration_config[
                "reference_config"
            ]
        )

        result = (
            run_baseline_calibration(
                base_config=base_config,
                calibration_config=(
                    calibration_config
                ),
            )
        )

        output_directory = (
            arguments.output
            if arguments.output
            is not None
            else calibration_config[
                "outputs"
            ]["directory"]
        )

        paths = save_calibration_result(
            result=result,
            output_directory=(
                output_directory
            ),
        )

        print(
            "\nBaseline dimension calibration:"
        )

        print(
            result.dimension_summary[
                [
                    "dimension",
                    "theoretical_mean",
                    "empirical_mean",
                    "ci_lower",
                    "ci_upper",
                    "absolute_mean_error",
                    "accepted",
                ]
            ].to_string(
                index=False
            )
        )

        print(
            "\nCalibration decision:",
            (
                "PASSED"
                if result.passed
                else "REJECTED"
            ),
        )

        print(
            "\nOutput directory:",
            paths[
                "metadata"
            ].parent,
        )

        if (
            arguments.fail_on_rejection
            and not result.passed
        ):
            return 1

        return 0

    except (
        CalibrationConfigurationError,
        CalibrationError,
    ) as error:
        print(
            f"Calibration error: {error}"
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(
        main()
    )