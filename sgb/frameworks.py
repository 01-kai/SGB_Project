from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import pandas as pd

from sgb.agents import OrganizationAgent
from sgb.config import EXPECTED_FRAMEWORK_DIMENSIONS
from sgb.dimensions import ALL_DIMENSIONS


class FrameworkConfigurationError(ValueError):
    """Raised when a framework definition is invalid."""


class FrameworkEvaluationError(ValueError):
    """Raised when operational state cannot be evaluated."""


@dataclass(
    frozen=True,
    slots=True,
)
class FrameworkDefinition:
    """
    Immutable framework-scoring definition.

    A framework contains only evaluation weights. It does not contain
    operational behavior, recovery rules, exchange probabilities, or
    perturbation responses.
    """

    name: str
    label: str
    weights: Mapping[str, float]

    def __post_init__(self) -> None:
        if self.name not in EXPECTED_FRAMEWORK_DIMENSIONS:
            raise FrameworkConfigurationError(
                f"Unknown framework: {self.name!r}."
            )

        if not isinstance(self.label, str) or not self.label.strip():
            raise FrameworkConfigurationError(
                f"{self.name}.label must be a non-empty string."
            )

        if not isinstance(self.weights, Mapping):
            raise FrameworkConfigurationError(
                f"{self.name}.weights must be a mapping."
            )

        expected_dimensions = EXPECTED_FRAMEWORK_DIMENSIONS[
            self.name
        ]
        actual_dimensions = set(self.weights)

        missing = expected_dimensions - actual_dimensions
        unexpected = actual_dimensions - expected_dimensions

        if missing or unexpected:
            details: list[str] = []

            if missing:
                details.append(
                    "missing="
                    + ",".join(sorted(missing))
                )

            if unexpected:
                details.append(
                    "unexpected="
                    + ",".join(sorted(unexpected))
                )

            raise FrameworkConfigurationError(
                f"Invalid dimensions for {self.name}: "
                + "; ".join(details)
            )

        validated_weights: dict[str, float] = {}

        for dimension, weight in self.weights.items():
            validated_weights[dimension] = (
                _require_unit_interval(
                    value=weight,
                    name=(
                        f"{self.name}.weights."
                        f"{dimension}"
                    ),
                    error_type=(
                        FrameworkConfigurationError
                    ),
                )
            )

        total_weight = sum(
            validated_weights.values()
        )

        if not math.isclose(
            total_weight,
            1.0,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise FrameworkConfigurationError(
                f"{self.name} weights must sum to 1.0; "
                f"received {total_weight}."
            )

        object.__setattr__(
            self,
            "label",
            self.label.strip(),
        )

        object.__setattr__(
            self,
            "weights",
            MappingProxyType(
                validated_weights
            ),
        )

    def score(
        self,
        dimensions: Mapping[str, float],
    ) -> float:
        """
        Calculate one framework-specific Governance Maturity Index.

        Extra dimensions are allowed. For example, DBL and DMM ignore
        SVC while ISF includes it.
        """

        if not isinstance(dimensions, Mapping):
            raise FrameworkEvaluationError(
                "dimensions must be a mapping."
            )

        missing_dimensions = (
            set(self.weights)
            - set(dimensions)
        )

        if missing_dimensions:
            raise FrameworkEvaluationError(
                f"Missing dimensions for {self.name}: "
                + ", ".join(
                    sorted(missing_dimensions)
                )
            )

        score = 0.0

        for dimension, weight in self.weights.items():
            dimension_value = (
                _require_unit_interval(
                    value=dimensions[dimension],
                    name=dimension,
                    error_type=(
                        FrameworkEvaluationError
                    ),
                )
            )

            score += (
                weight
                * dimension_value
            )

        return min(
            1.0,
            max(0.0, float(score)),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable framework definition."""

        return {
            "name": self.name,
            "label": self.label,
            "weights": dict(self.weights),
        }


def build_framework_definitions(
    base_config: Mapping[str, Any],
) -> dict[str, FrameworkDefinition]:
    """
    Build validated framework definitions from the base configuration.
    """

    if not isinstance(base_config, Mapping):
        raise FrameworkConfigurationError(
            "base_config must be a mapping."
        )

    frameworks = base_config.get(
        "frameworks"
    )

    if not isinstance(frameworks, Mapping):
        raise FrameworkConfigurationError(
            "base_config.frameworks must be a mapping."
        )

    expected_frameworks = set(
        EXPECTED_FRAMEWORK_DIMENSIONS
    )
    actual_frameworks = set(
        frameworks
    )

    missing = (
        expected_frameworks
        - actual_frameworks
    )
    unexpected = (
        actual_frameworks
        - expected_frameworks
    )

    if missing or unexpected:
        details: list[str] = []

        if missing:
            details.append(
                "missing="
                + ",".join(sorted(missing))
            )

        if unexpected:
            details.append(
                "unexpected="
                + ",".join(sorted(unexpected))
            )

        raise FrameworkConfigurationError(
            "Invalid framework configuration: "
            + "; ".join(details)
        )

    definitions: dict[
        str,
        FrameworkDefinition,
    ] = {}

    for framework_name in (
        EXPECTED_FRAMEWORK_DIMENSIONS
    ):
        framework_config = frameworks[
            framework_name
        ]

        if not isinstance(
            framework_config,
            Mapping,
        ):
            raise FrameworkConfigurationError(
                f"{framework_name} must be a mapping."
            )

        definitions[framework_name] = (
            FrameworkDefinition(
                name=framework_name,
                label=str(
                    framework_config.get(
                        "label",
                        "",
                    )
                ),
                weights=framework_config.get(
                    "weights",
                    {},
                ),
            )
        )

    return definitions


def score_all_frameworks(
    dimensions: Mapping[str, float],
    definitions: Mapping[
        str,
        FrameworkDefinition,
    ],
) -> dict[str, float]:
    """
    Score one operational state using every framework.
    """

    validated_definitions = (
        _validate_definitions(
            definitions
        )
    )

    return {
        framework_name: definition.score(
            dimensions
        )
        for framework_name, definition
        in validated_definitions.items()
    }


def score_agent(
    agent: OrganizationAgent,
    definitions: Mapping[
        str,
        FrameworkDefinition,
    ],
) -> dict[str, float]:
    """
    Score one organization without modifying its state.
    """

    if not isinstance(
        agent,
        OrganizationAgent,
    ):
        raise FrameworkEvaluationError(
            "agent must be an OrganizationAgent."
        )

    return score_all_frameworks(
        dimensions=agent.dimensions,
        definitions=definitions,
    )


def evaluate_agents(
    agents: Iterable[OrganizationAgent],
    definitions: Mapping[
        str,
        FrameworkDefinition,
    ],
) -> pd.DataFrame:
    """
    Evaluate multiple organization agents.

    Each returned row contains the same operational dimensions followed
    by one framework-specific GMI column.
    """

    if isinstance(
        agents,
        (str, bytes),
    ):
        raise FrameworkEvaluationError(
            "agents must be an iterable "
            "of OrganizationAgent objects."
        )

    try:
        agent_list = list(agents)
    except TypeError as error:
        raise FrameworkEvaluationError(
            "agents must be iterable."
        ) from error

    snapshots: list[dict[str, Any]] = []

    for agent in agent_list:
        if not isinstance(
            agent,
            OrganizationAgent,
        ):
            raise FrameworkEvaluationError(
                "Every agent must be an "
                "OrganizationAgent."
            )

        snapshots.append(
            agent.snapshot()
        )

    return evaluate_snapshots(
        snapshots=snapshots,
        definitions=definitions,
    )


def evaluate_snapshots(
    snapshots: Iterable[
        Mapping[str, Any]
    ],
    definitions: Mapping[
        str,
        FrameworkDefinition,
    ],
) -> pd.DataFrame:
    """
    Evaluate serialized organization states.

    Framework evaluation is read-only. The supplied snapshots are not
    modified.
    """

    validated_definitions = (
        _validate_definitions(
            definitions
        )
    )

    if isinstance(
        snapshots,
        (str, bytes),
    ):
        raise FrameworkEvaluationError(
            "snapshots must be an iterable "
            "of mappings."
        )

    try:
        snapshot_list = list(
            snapshots
        )
    except TypeError as error:
        raise FrameworkEvaluationError(
            "snapshots must be iterable."
        ) from error

    framework_columns = [
        f"GMI_{framework_name}"
        for framework_name
        in validated_definitions
    ]

    columns = [
        "uid",
        "org_type",
        "exchange_volume",
        *ALL_DIMENSIONS,
        *framework_columns,
    ]

    if not snapshot_list:
        return pd.DataFrame(
            columns=columns
        )

    rows: list[dict[str, Any]] = []
    observed_uids: set[int] = set()

    for snapshot in snapshot_list:
        if not isinstance(
            snapshot,
            Mapping,
        ):
            raise FrameworkEvaluationError(
                "Every snapshot must be a mapping."
            )

        uid = _require_non_negative_integer(
            snapshot.get("uid"),
            "snapshot.uid",
        )

        if uid in observed_uids:
            raise FrameworkEvaluationError(
                f"Duplicate snapshot uid: {uid}."
            )

        observed_uids.add(uid)

        org_type = snapshot.get(
            "org_type"
        )

        if (
            not isinstance(org_type, str)
            or not org_type.strip()
        ):
            raise FrameworkEvaluationError(
                "snapshot.org_type must be "
                "a non-empty string."
            )

        exchange_volume = (
            _require_positive_integer(
                snapshot.get(
                    "exchange_volume"
                ),
                "snapshot.exchange_volume",
            )
        )

        dimensions = snapshot.get(
            "dimensions"
        )

        if not isinstance(
            dimensions,
            Mapping,
        ):
            raise FrameworkEvaluationError(
                "snapshot.dimensions must "
                "be a mapping."
            )

        validated_dimensions = (
            _validate_complete_dimensions(
                dimensions
            )
        )

        framework_scores = (
            score_all_frameworks(
                dimensions=(
                    validated_dimensions
                ),
                definitions=(
                    validated_definitions
                ),
            )
        )

        row: dict[str, Any] = {
            "uid": uid,
            "org_type": org_type.strip(),
            "exchange_volume": (
                exchange_volume
            ),
        }

        for dimension in ALL_DIMENSIONS:
            row[dimension] = (
                validated_dimensions[
                    dimension
                ]
            )

        for (
            framework_name,
            score,
        ) in framework_scores.items():
            row[
                f"GMI_{framework_name}"
            ] = score

        rows.append(row)

    rows.sort(
        key=lambda row: row["uid"]
    )

    return pd.DataFrame(
        rows,
        columns=columns,
    )


def aggregate_agent_scores(
    scored_agents: pd.DataFrame,
    definitions: Mapping[
        str,
        FrameworkDefinition,
    ],
) -> pd.DataFrame:
    """
    Aggregate agent-level GMI values into ecosystem-level E-GMI values.

    E-GMI is the unweighted arithmetic mean across organizations.
    """

    validated_definitions = (
        _validate_definitions(
            definitions
        )
    )

    if not isinstance(
        scored_agents,
        pd.DataFrame,
    ):
        raise FrameworkEvaluationError(
            "scored_agents must be a pandas DataFrame."
        )

    if scored_agents.empty:
        raise FrameworkEvaluationError(
            "Cannot aggregate an empty agent-score table."
        )

    records: list[dict[str, Any]] = []
    mean_scores: dict[str, float] = {}

    for framework_name in (
        validated_definitions
    ):
        column = (
            f"GMI_{framework_name}"
        )

        if column not in scored_agents.columns:
            raise FrameworkEvaluationError(
                f"Missing score column: {column}."
            )

        values = _validated_series(
            scored_agents[column],
            column,
        )

        mean_score = float(
            values.mean()
        )

        mean_scores[
            framework_name
        ] = mean_score

        records.append(
            {
                "framework": (
                    framework_name
                ),
                "label": (
                    validated_definitions[
                        framework_name
                    ].label
                ),
                "E_GMI": mean_score,
                "standard_deviation": (
                    float(
                        values.std(
                            ddof=1
                        )
                    )
                    if len(values) > 1
                    else 0.0
                ),
                "minimum": float(
                    values.min()
                ),
                "maximum": float(
                    values.max()
                ),
                "organization_count": (
                    int(len(values))
                ),
            }
        )

    ranking = {
        item["framework"]: item["rank"]
        for item in rank_framework_scores(
            mean_scores
        )
    }

    for record in records:
        record["rank"] = ranking[
            record["framework"]
        ]

    records.sort(
        key=lambda record: (
            record["rank"],
            record["framework"],
        )
    )

    return pd.DataFrame(
        records,
        columns=[
            "rank",
            "framework",
            "label",
            "E_GMI",
            "standard_deviation",
            "minimum",
            "maximum",
            "organization_count",
        ],
    )


def evaluate_trajectory(
    operational_results: pd.DataFrame,
    definitions: Mapping[
        str,
        FrameworkDefinition,
    ],
) -> pd.DataFrame:
    """
    Score each step of one shared operational trajectory.

    The returned DataFrame is a copy. The original operational results
    are not modified.
    """

    validated_definitions = (
        _validate_definitions(
            definitions
        )
    )

    if not isinstance(
        operational_results,
        pd.DataFrame,
    ):
        raise FrameworkEvaluationError(
            "operational_results must be "
            "a pandas DataFrame."
        )

    scored_results = (
        operational_results.copy(
            deep=True
        )
    )

    required_dimensions = set().union(
        *(
            set(definition.weights)
            for definition
            in validated_definitions.values()
        )
    )

    missing_columns = (
        required_dimensions
        - set(scored_results.columns)
    )

    if missing_columns:
        raise FrameworkEvaluationError(
            "Operational results are missing dimensions: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    validated_columns: dict[
        str,
        pd.Series,
    ] = {}

    for dimension in sorted(
        required_dimensions
    ):
        validated_columns[dimension] = (
            _validated_series(
                scored_results[
                    dimension
                ],
                dimension,
            )
        )

    for (
        framework_name,
        definition,
    ) in validated_definitions.items():
        score_series = pd.Series(
            0.0,
            index=scored_results.index,
            dtype=float,
        )

        for (
            dimension,
            weight,
        ) in definition.weights.items():
            score_series = (
                score_series
                + (
                    validated_columns[
                        dimension
                    ]
                    * weight
                )
            )

        scored_results[
            f"E_GMI_{framework_name}"
        ] = score_series.clip(
            lower=0.0,
            upper=1.0,
        )

    return scored_results


def rank_framework_scores(
    scores: Mapping[str, float],
    tolerance: float = 1e-12,
) -> list[dict[str, Any]]:
    """
    Rank framework scores using dense ranking.

    Scores within ``tolerance`` receive the same rank.
    """

    if not isinstance(scores, Mapping) or not scores:
        raise FrameworkEvaluationError(
            "scores must be a non-empty mapping."
        )

    tolerance_value = (
        _require_non_negative_number(
            tolerance,
            "tolerance",
        )
    )

    validated_scores: dict[
        str,
        float,
    ] = {}

    for framework_name, score in scores.items():
        if (
            not isinstance(framework_name, str)
            or not framework_name.strip()
        ):
            raise FrameworkEvaluationError(
                "Framework names must be "
                "non-empty strings."
            )

        validated_scores[
            framework_name
        ] = _require_unit_interval(
            value=score,
            name=(
                f"score for "
                f"{framework_name}"
            ),
            error_type=(
                FrameworkEvaluationError
            ),
        )

    sorted_scores = sorted(
        validated_scores.items(),
        key=lambda item: (
            -item[1],
            item[0],
        ),
    )

    records: list[
        dict[str, Any]
    ] = []

    current_rank = 0
    previous_score: float | None = None

    for framework_name, score in (
        sorted_scores
    ):
        if (
            previous_score is None
            or not math.isclose(
                score,
                previous_score,
                rel_tol=0.0,
                abs_tol=tolerance_value,
            )
        ):
            current_rank += 1

        records.append(
            {
                "rank": current_rank,
                "framework": (
                    framework_name
                ),
                "score": score,
            }
        )

        previous_score = score

    return records


def _validate_definitions(
    definitions: Mapping[
        str,
        FrameworkDefinition,
    ],
) -> dict[
    str,
    FrameworkDefinition,
]:
    if not isinstance(
        definitions,
        Mapping,
    ):
        raise FrameworkConfigurationError(
            "definitions must be a mapping."
        )

    expected_frameworks = set(
        EXPECTED_FRAMEWORK_DIMENSIONS
    )
    actual_frameworks = set(
        definitions
    )

    missing = (
        expected_frameworks
        - actual_frameworks
    )
    unexpected = (
        actual_frameworks
        - expected_frameworks
    )

    if missing or unexpected:
        details: list[str] = []

        if missing:
            details.append(
                "missing="
                + ",".join(sorted(missing))
            )

        if unexpected:
            details.append(
                "unexpected="
                + ",".join(sorted(unexpected))
            )

        raise FrameworkConfigurationError(
            "Invalid framework definitions: "
            + "; ".join(details)
        )

    validated: dict[
        str,
        FrameworkDefinition,
    ] = {}

    for framework_name in (
        EXPECTED_FRAMEWORK_DIMENSIONS
    ):
        definition = definitions[
            framework_name
        ]

        if not isinstance(
            definition,
            FrameworkDefinition,
        ):
            raise FrameworkConfigurationError(
                f"{framework_name} must be a "
                "FrameworkDefinition."
            )

        if definition.name != framework_name:
            raise FrameworkConfigurationError(
                f"Definition key {framework_name!r} "
                f"does not match {definition.name!r}."
            )

        validated[
            framework_name
        ] = definition

    return validated


def _validate_complete_dimensions(
    dimensions: Mapping[str, Any],
) -> dict[str, float]:
    expected_dimensions = set(
        ALL_DIMENSIONS
    )
    actual_dimensions = set(
        dimensions
    )

    missing = (
        expected_dimensions
        - actual_dimensions
    )
    unexpected = (
        actual_dimensions
        - expected_dimensions
    )

    if missing or unexpected:
        details: list[str] = []

        if missing:
            details.append(
                "missing="
                + ",".join(sorted(missing))
            )

        if unexpected:
            details.append(
                "unexpected="
                + ",".join(sorted(unexpected))
            )

        raise FrameworkEvaluationError(
            "Invalid snapshot dimensions: "
            + "; ".join(details)
        )

    return {
        dimension: _require_unit_interval(
            value=dimensions[
                dimension
            ],
            name=dimension,
            error_type=(
                FrameworkEvaluationError
            ),
        )
        for dimension in ALL_DIMENSIONS
    }


def _validated_series(
    series: pd.Series,
    name: str,
) -> pd.Series:
    try:
        numeric_series = pd.to_numeric(
            series,
            errors="raise",
        ).astype(float)
    except (TypeError, ValueError) as error:
        raise FrameworkEvaluationError(
            f"{name} must contain numeric values."
        ) from error

    if numeric_series.isna().any():
        raise FrameworkEvaluationError(
            f"{name} cannot contain missing values."
        )

    if (
        (numeric_series < 0.0).any()
        or (numeric_series > 1.0).any()
    ):
        raise FrameworkEvaluationError(
            f"{name} values must be in [0, 1]."
        )

    return numeric_series


def _require_unit_interval(
    *,
    value: Any,
    name: str,
    error_type: type[ValueError],
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(
            value,
            (int, float),
        )
    ):
        raise error_type(
            f"{name} must be numeric."
        )

    numeric_value = float(value)

    if not math.isfinite(
        numeric_value
    ):
        raise error_type(
            f"{name} must be finite."
        )

    if not 0.0 <= numeric_value <= 1.0:
        raise error_type(
            f"{name} must be in [0, 1]."
        )

    return numeric_value


def _require_non_negative_integer(
    value: Any,
    name: str,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
    ):
        raise FrameworkEvaluationError(
            f"{name} must be an integer."
        )

    if value < 0:
        raise FrameworkEvaluationError(
            f"{name} must be zero or greater."
        )

    return value


def _require_positive_integer(
    value: Any,
    name: str,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
    ):
        raise FrameworkEvaluationError(
            f"{name} must be an integer."
        )

    if value <= 0:
        raise FrameworkEvaluationError(
            f"{name} must be greater than zero."
        )

    return value


def _require_non_negative_number(
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
        raise FrameworkEvaluationError(
            f"{name} must be numeric."
        )

    numeric_value = float(value)

    if not math.isfinite(
        numeric_value
    ):
        raise FrameworkEvaluationError(
            f"{name} must be finite."
        )

    if numeric_value < 0.0:
        raise FrameworkEvaluationError(
            f"{name} must be zero or greater."
        )

    return numeric_value