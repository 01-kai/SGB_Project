from __future__ import annotations

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
from scipy.stats import t as student_t

from sgb.config import PROJECT_ROOT
from sgb.dimensions import ALL_DIMENSIONS
from sgb.dynamics import NeutralOperationalDynamics
from sgb.frameworks import (
    build_framework_definitions,
    evaluate_trajectory,
)
from sgb.model import SGBModel


KNOWN_METRICS = {
    "success_rate",
    "volume_success_rate",
    "mean_operational_readiness",
    *ALL_DIMENSIONS,
    "E_GMI_DBL",
    "E_GMI_DMM",
    "E_GMI_ISF",
}

FRAMEWORK_METRICS = (
    "E_GMI_DBL",
    "E_GMI_DMM",
    "E_GMI_ISF",
)

SUPPORTED_STUDIES = {
    "seed_convergence",
    "time_convergence",
    "population_convergence",
}


class ConvergenceConfigurationError(ValueError):
    """Raised when a convergence-study configuration is invalid."""


class ConvergenceError(RuntimeError):
    """Raised when a convergence study cannot be completed."""


@dataclass(frozen=True, slots=True)
class ConvergenceStudyResult:
    """Tables and metadata produced by one convergence study."""

    study_type: str
    tables: dict[str, pd.DataFrame]
    metadata: dict[str, Any]

    @property
    def passed(self) -> bool:
        """Return the study acceptance decision."""

        return bool(
            self.metadata.get(
                "passed",
                False,
            )
        )


def load_convergence_config(
    path: str | Path,
) -> dict[str, Any]:
    """Load and validate one convergence-study file."""

    config_path = Path(path)

    if not config_path.is_absolute():
        config_path = (
            PROJECT_ROOT
            / config_path
        )

    config_path = (
        config_path.resolve()
    )

    if not config_path.is_file():
        raise ConvergenceConfigurationError(
            "Convergence configuration does not exist: "
            f"{config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(
            file
        )

    if not isinstance(
        config,
        dict,
    ):
        raise ConvergenceConfigurationError(
            "Convergence configuration root "
            "must be a mapping."
        )

    validate_convergence_config(
        config
    )

    config["_metadata"] = {
        "source_path": str(
            config_path
        ),
    }

    return config


def validate_convergence_config(
    config: Mapping[str, Any],
) -> None:
    """Validate seed, time, or population convergence configuration."""

    if config.get(
        "schema_version"
    ) != 2:
        raise ConvergenceConfigurationError(
            "schema_version must be 2."
        )

    if config.get(
        "config_kind"
    ) != "study":
        raise ConvergenceConfigurationError(
            "config_kind must be 'study'."
        )

    study_type = config.get(
        "study_type"
    )

    if study_type not in SUPPORTED_STUDIES:
        raise ConvergenceConfigurationError(
            "Unsupported convergence study_type: "
            f"{study_type!r}."
        )

    _non_empty_string(
        config,
        "reference_config",
    )

    _non_empty_string(
        config,
        "dynamics_config",
    )

    metrics = _metric_list(
        config
    )

    unknown = (
        set(metrics)
        - KNOWN_METRICS
    )

    if unknown:
        raise ConvergenceConfigurationError(
            "Unknown convergence metrics: "
            + ", ".join(
                sorted(unknown)
            )
        )

    missing_framework = (
        set(FRAMEWORK_METRICS)
        - set(metrics)
    )

    if missing_framework:
        raise ConvergenceConfigurationError(
            "All framework metrics are required: "
            + ", ".join(
                sorted(
                    missing_framework
                )
            )
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
        raise ConvergenceConfigurationError(
            "analysis.confidence_level "
            "must be in (0, 1)."
        )

    _non_empty_string(
        _mapping(
            config,
            "outputs",
        ),
        "directory",
    )

    if study_type == "time_convergence":
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

        _positive_integer(
            _mapping(
                config,
                "simulation",
            ),
            "population_size",
        )

        maximum_horizon = (
            _positive_integer(
                config,
                "maximum_horizon",
            )
        )

        horizons = (
            _sorted_positive_integers(
                config,
                "candidate_horizons",
            )
        )

        if max(
            horizons
        ) > maximum_horizon:
            raise ConvergenceConfigurationError(
                "candidate_horizons cannot exceed "
                "maximum_horizon."
            )

        detection = _mapping(
            config,
            "detection",
        )

        window = _positive_integer(
            detection,
            "rolling_window",
        )

        stable_windows = (
            _positive_integer(
                detection,
                "required_stable_windows",
            )
        )

        minimum_step = (
            _positive_integer(
                detection,
                "minimum_step",
            )
        )

        _non_negative_number(
            detection,
            "slope_tolerance",
        )

        _non_negative_number(
            detection,
            "mean_difference_tolerance",
        )

        _non_negative_integer(
            detection,
            "required_observation_steps",
        )

        if (
            window
            * (
                stable_windows
                + 1
            )
            > maximum_horizon
        ):
            raise ConvergenceConfigurationError(
                "maximum_horizon is too small "
                "for the stable-window rule."
            )

        if minimum_step > maximum_horizon:
            raise ConvergenceConfigurationError(
                "detection.minimum_step cannot "
                "exceed maximum_horizon."
            )

    elif study_type == "seed_convergence":
        seed_plan = _mapping(
            config,
            "seed_plan",
        )

        _non_negative_integer(
            seed_plan,
            "start_seed",
        )

        counts = (
            _sorted_positive_integers(
                config,
                "candidate_seed_counts",
            )
        )

        if max(counts) < 2:
            raise ConvergenceConfigurationError(
                "At least two seeds are required "
                "for confidence intervals."
            )

        simulation = _mapping(
            config,
            "simulation",
        )

        _positive_integer(
            simulation,
            "population_size",
        )

        horizon = _positive_integer(
            simulation,
            "horizon",
        )

        terminal_window = (
            _positive_integer(
                simulation,
                "terminal_window",
            )
        )

        if terminal_window > horizon:
            raise ConvergenceConfigurationError(
                "simulation.terminal_window cannot "
                "exceed simulation.horizon."
            )

        if (
            _non_negative_number(
                analysis,
                "maximum_relative_ci_width",
            )
            <= 0.0
        ):
            raise ConvergenceConfigurationError(
                "analysis.maximum_relative_ci_width "
                "must be greater than zero."
            )

        _non_negative_number(
            analysis,
            "relative_ci_denominator_floor",
        )

        _non_negative_number(
            analysis,
            "ranking_tolerance",
        )

    else:
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

        sizes = (
            _sorted_positive_integers(
                config,
                "candidate_population_sizes",
            )
        )

        if min(sizes) <= 1:
            raise ConvergenceConfigurationError(
                "Population sizes must be "
                "greater than one."
            )

        simulation = _mapping(
            config,
            "simulation",
        )

        horizon = _positive_integer(
            simulation,
            "horizon",
        )

        terminal_window = (
            _positive_integer(
                simulation,
                "terminal_window",
            )
        )

        if terminal_window > horizon:
            raise ConvergenceConfigurationError(
                "simulation.terminal_window cannot "
                "exceed simulation.horizon."
            )

        _non_negative_number(
            analysis,
            "material_difference_tolerance",
        )

        _non_negative_number(
            analysis,
            "ranking_tolerance",
        )


def detect_block_convergence(
    values: Sequence[float] | pd.Series,
    *,
    rolling_window: int,
    required_stable_windows: int,
    minimum_step: int,
    slope_tolerance: float,
    mean_difference_tolerance: float,
) -> dict[str, Any]:
    """
    Detect convergence using consecutive non-overlapping blocks.

    A block is stable when:

    1. its absolute linear slope is below the configured tolerance;
    2. its mean differs from the preceding block by no more than the
       configured mean-difference tolerance;
    3. the minimum simulation step has been reached.
    """

    numeric = pd.to_numeric(
        pd.Series(values),
        errors="raise",
    ).astype(float)

    if (
        numeric.isna().any()
        or not np.isfinite(
            numeric
        ).all()
    ):
        raise ConvergenceError(
            "Convergence values must be finite."
        )

    stable_count = 0
    previous_mean: float | None = None
    last_mean: float | None = None
    last_slope: float | None = None

    for start in range(
        0,
        len(numeric)
        - rolling_window
        + 1,
        rolling_window,
    ):
        block = numeric.iloc[
            start:
            start + rolling_window
        ].to_numpy(
            dtype=float
        )

        block_mean = float(
            block.mean()
        )

        block_slope = float(
            np.polyfit(
                np.arange(
                    rolling_window
                ),
                block,
                1,
            )[0]
        )

        end_step = (
            start
            + rolling_window
            - 1
        )

        stable = (
            end_step + 1
            >= minimum_step
            and previous_mean
            is not None
            and abs(
                block_mean
                - previous_mean
            )
            <= mean_difference_tolerance
            and abs(
                block_slope
            )
            <= slope_tolerance
        )

        stable_count = (
            stable_count + 1
            if stable
            else 0
        )

        previous_mean = block_mean
        last_mean = block_mean
        last_slope = block_slope

        if (
            stable_count
            >= required_stable_windows
        ):
            return {
                "converged": True,
                "convergence_step": (
                    end_step
                ),
                "stable_window_count": (
                    stable_count
                ),
                "last_window_mean": (
                    block_mean
                ),
                "last_window_slope": (
                    block_slope
                ),
            }

    return {
        "converged": False,
        "convergence_step": None,
        "stable_window_count": (
            stable_count
        ),
        "last_window_mean": (
            last_mean
        ),
        "last_window_slope": (
            last_slope
        ),
    }


def run_single_trajectory(
    *,
    base_config: Mapping[str, Any],
    dynamics_config: Mapping[str, Any],
    population_size: int,
    horizon: int,
    seed: int,
) -> pd.DataFrame:
    """
    Run one framework-neutral trajectory and score every framework.

    Event records are cleared after each step because convergence studies
    require step-level metrics rather than millions of retained events.
    """

    model_config = deepcopy(
        dict(base_config)
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
    ]["max_steps"] = horizon

    attachment_m = int(
        model_config[
            "network"
        ]["attachment_m"]
    )

    if attachment_m >= population_size:
        raise ConvergenceConfigurationError(
            f"population_size {population_size} "
            f"must exceed attachment_m {attachment_m}."
        )

    dynamics = (
        NeutralOperationalDynamics(
            model_config,
            dynamics_config,
        )
    )

    model = SGBModel(
        model_config,
        seed=seed,
        dynamics=dynamics,
    )

    for _ in range(
        horizon
    ):
        model.step()

        model.event_records.clear()

    definitions = (
        build_framework_definitions(
            model_config
        )
    )

    return evaluate_trajectory(
        model.get_results(),
        definitions,
    )


def summarize_values(
    values: Sequence[float] | pd.Series,
    *,
    confidence_level: float,
) -> dict[str, float | int]:
    """Calculate mean, standard deviation, and Student-t confidence interval."""

    numeric = pd.to_numeric(
        pd.Series(values),
        errors="raise",
    ).astype(float)

    if (
        numeric.empty
        or numeric.isna().any()
        or not np.isfinite(
            numeric
        ).all()
    ):
        raise ConvergenceError(
            "Replicate values must be "
            "non-empty and finite."
        )

    count = len(
        numeric
    )

    mean = float(
        numeric.mean()
    )

    if count == 1:
        standard_deviation = 0.0
        half_width = 0.0
    else:
        standard_deviation = float(
            numeric.std(
                ddof=1
            )
        )

        critical = float(
            student_t.ppf(
                (
                    1.0
                    + confidence_level
                )
                / 2.0,
                df=count - 1,
            )
        )

        half_width = (
            critical
            * standard_deviation
            / math.sqrt(
                count
            )
        )

    return {
        "sample_count": int(
            count
        ),
        "mean": mean,
        "standard_deviation": (
            standard_deviation
        ),
        "ci_lower": (
            mean - half_width
        ),
        "ci_upper": (
            mean + half_width
        ),
        "ci_width": (
            2.0 * half_width
        ),
    }


def framework_ranking(
    scores: Mapping[str, float],
    *,
    tolerance: float,
) -> str:
    """Create a stable ranking string with tolerance-based ties."""

    ordered = sorted(
        scores.items(),
        key=lambda item: (
            -item[1],
            item[0],
        ),
    )

    groups: list[
        list[str]
    ] = []

    previous_score: float | None = None

    for metric, score in ordered:
        framework = metric.removeprefix(
            "E_GMI_"
        )

        if (
            previous_score is None
            or abs(
                score
                - previous_score
            )
            > tolerance
        ):
            groups.append(
                [framework]
            )
        else:
            groups[-1].append(
                framework
            )

        previous_score = score

    return ">".join(
        "=".join(
            sorted(group)
        )
        for group in groups
    )


def save_convergence_result(
    result: ConvergenceStudyResult,
    output_directory: str | Path,
) -> dict[str, Path]:
    """Save CSV tables, metadata, and one convergence figure."""

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

    paths: dict[
        str,
        Path,
    ] = {}

    for name, table in (
        result.tables.items()
    ):
        path = (
            output_path
            / f"{name}.csv"
        )

        table.to_csv(
            path,
            index=False,
        )

        paths[name] = path

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
            f"{result.study_type}"
            ".png"
        )
    )

    if (
        result.study_type
        == "time_convergence"
    ):
        _plot_time(
            result.tables[
                "trajectory_summary"
            ],
            figure_path,
        )

    elif (
        result.study_type
        == "seed_convergence"
    ):
        _plot_seed(
            result.tables[
                "ci_summary"
            ],
            figure_path,
        )

    else:
        _plot_population(
            result.tables[
                "population_summary"
            ],
            figure_path,
        )

    paths[
        "figure"
    ] = figure_path

    return paths


def _plot_time(
    summary: pd.DataFrame,
    path: Path,
) -> None:
    figure, axis = plt.subplots(
        figsize=(
            10,
            6,
        )
    )

    for metric in FRAMEWORK_METRICS:
        data = summary.loc[
            summary[
                "metric"
            ]
            == metric
        ].sort_values(
            "step"
        )

        axis.plot(
            data["step"],
            data["mean"],
            label=metric,
        )

        axis.fill_between(
            data["step"],
            data["ci_lower"],
            data["ci_upper"],
            alpha=0.15,
        )

    axis.set(
        xlabel="Simulation step",
        ylabel="Mean E-GMI",
        title="E-GMI time convergence",
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


def _plot_seed(
    summary: pd.DataFrame,
    path: Path,
) -> None:
    figure, axis = plt.subplots(
        figsize=(
            10,
            6,
        )
    )

    for metric in summary[
        "metric"
    ].drop_duplicates():
        data = summary.loc[
            summary[
                "metric"
            ]
            == metric
        ].sort_values(
            "candidate_seed_count"
        )

        axis.plot(
            data[
                "candidate_seed_count"
            ],
            data[
                "relative_ci_width"
            ],
            marker="o",
            label=metric,
        )

    axis.set(
        xlabel="Seed count",
        ylabel="Relative 95% CI width",
        title="Seed convergence",
    )

    axis.legend(
        fontsize="small",
        ncol=2,
    )

    figure.tight_layout()

    figure.savefig(
        path,
        dpi=200,
    )

    plt.close(
        figure
    )


def _plot_population(
    summary: pd.DataFrame,
    path: Path,
) -> None:
    figure, axis = plt.subplots(
        figsize=(
            10,
            6,
        )
    )

    for metric in FRAMEWORK_METRICS:
        data = summary.loc[
            summary[
                "metric"
            ]
            == metric
        ].sort_values(
            "population_size"
        )

        axis.errorbar(
            data[
                "population_size"
            ],
            data["mean"],
            yerr=(
                data["ci_upper"]
                - data["ci_lower"]
            )
            / 2.0,
            marker="o",
            capsize=3,
            label=metric,
        )

    axis.set(
        xlabel="Population size",
        ylabel="Mean terminal E-GMI",
        title="Population convergence",
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
        raise ConvergenceConfigurationError(
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
        raise ConvergenceConfigurationError(
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
        isinstance(
            value,
            bool,
        )
        or not isinstance(
            value,
            int,
        )
        or value <= 0
    ):
        raise ConvergenceConfigurationError(
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
        isinstance(
            value,
            bool,
        )
        or not isinstance(
            value,
            int,
        )
        or value < 0
    ):
        raise ConvergenceConfigurationError(
            f"{key} must be a non-negative integer."
        )

    return value


def _finite_number(
    container: Mapping[str, Any],
    key: str,
) -> float:
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
            (
                int,
                float,
            ),
        )
    ):
        raise ConvergenceConfigurationError(
            f"{key} must be numeric."
        )

    numeric = float(
        value
    )

    if not math.isfinite(
        numeric
    ):
        raise ConvergenceConfigurationError(
            f"{key} must be finite."
        )

    return numeric


def _non_negative_number(
    container: Mapping[str, Any],
    key: str,
) -> float:
    value = _finite_number(
        container,
        key,
    )

    if value < 0.0:
        raise ConvergenceConfigurationError(
            f"{key} must be non-negative."
        )

    return value


def _sorted_positive_integers(
    container: Mapping[str, Any],
    key: str,
) -> list[int]:
    values = container.get(
        key
    )

    if (
        not isinstance(
            values,
            list,
        )
        or not values
    ):
        raise ConvergenceConfigurationError(
            f"{key} must be a non-empty list."
        )

    if any(
        isinstance(
            value,
            bool,
        )
        or not isinstance(
            value,
            int,
        )
        or value <= 0
        for value in values
    ):
        raise ConvergenceConfigurationError(
            f"Every value in {key} must be positive."
        )

    if values != sorted(
        set(values)
    ):
        raise ConvergenceConfigurationError(
            f"{key} must be sorted and unique."
        )

    return values


def _metric_list(
    config: Mapping[str, Any],
) -> list[str]:
    metrics = config.get(
        "metrics"
    )

    if (
        not isinstance(
            metrics,
            list,
        )
        or not metrics
    ):
        raise ConvergenceConfigurationError(
            "metrics must be a non-empty list."
        )

    if any(
        not isinstance(
            metric,
            str,
        )
        or not metric.strip()
        for metric in metrics
    ):
        raise ConvergenceConfigurationError(
            "Every metric must be a "
            "non-empty string."
        )

    if len(metrics) != len(
        set(metrics)
    ):
        raise ConvergenceConfigurationError(
            "metrics cannot contain duplicates."
        )

    return metrics


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
        "Cannot serialize "
        f"{type(value).__name__}."
    )