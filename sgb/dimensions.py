from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any


STANDARD_DIMENSIONS = (
    "D1",
    "D2",
    "D3",
    "D4",
    "D5",
)

ALL_DIMENSIONS = STANDARD_DIMENSIONS + ("SVC",)


class DimensionValueError(ValueError):
    """Raised when a dimension submetric is missing or invalid."""


class DimensionConfigurationError(ValueError):
    """Raised when dimension formula configuration is invalid."""


def compute_d1(
    submetrics: Mapping[str, float],
    weights: Mapping[str, float],
) -> float:
    """
    Calculate Data Quality.

    D1 =
        w1 * completeness_ratio
        + w2 * (1 - duplicate_rate)
        + w3 * schema_pass_rate
    """

    raw_values = _read_submetrics(
        submetrics,
        (
            "completeness_ratio",
            "duplicate_rate",
            "schema_pass_rate",
        ),
    )

    effective_values = {
        "completeness_ratio": (
            raw_values["completeness_ratio"]
        ),
        "duplicate_rate": (
            1.0 - raw_values["duplicate_rate"]
        ),
        "schema_pass_rate": (
            raw_values["schema_pass_rate"]
        ),
    }

    return _weighted_score(
        dimension="D1",
        effective_values=effective_values,
        weights=weights,
    )


def compute_d2(
    submetrics: Mapping[str, float],
    weights: Mapping[str, float],
) -> float:
    """
    Calculate Security Posture.

    D2 =
        w1 * encryption_coverage
        + w2 * (1 - normalized_breach_latency)
        + w3 * (1 - policy_violation_rate)
    """

    raw_values = _read_submetrics(
        submetrics,
        (
            "encryption_coverage",
            "normalized_breach_latency",
            "policy_violation_rate",
        ),
    )

    effective_values = {
        "encryption_coverage": (
            raw_values["encryption_coverage"]
        ),
        "normalized_breach_latency": (
            1.0 - raw_values["normalized_breach_latency"]
        ),
        "policy_violation_rate": (
            1.0 - raw_values["policy_violation_rate"]
        ),
    }

    return _weighted_score(
        dimension="D2",
        effective_values=effective_values,
        weights=weights,
    )


def compute_d3(
    submetrics: Mapping[str, float],
    weights: Mapping[str, float],
) -> float:
    """
    Calculate Metadata Completeness.

    D3 =
        w1 * lineage_rate
        + w2 * catalog_coverage
        + w3 * catalog_freshness
    """

    values = _read_submetrics(
        submetrics,
        (
            "lineage_rate",
            "catalog_coverage",
            "catalog_freshness",
        ),
    )

    return _weighted_score(
        dimension="D3",
        effective_values=values,
        weights=weights,
    )


def compute_d4(
    submetrics: Mapping[str, float],
    weights: Mapping[str, float],
) -> float:
    """
    Calculate Regulatory Compliance.

    D4 =
        w1 * rule_pass_rate
        + w2 * consent_propagation_rate
    """

    values = _read_submetrics(
        submetrics,
        (
            "rule_pass_rate",
            "consent_propagation_rate",
        ),
    )

    return _weighted_score(
        dimension="D4",
        effective_values=values,
        weights=weights,
    )


def compute_d5(
    submetrics: Mapping[str, float],
    weights: Mapping[str, float],
) -> float:
    """
    Calculate Interoperability.

    D5 =
        w1 * schema_reconciliation_rate
        + w2 * api_adherence_rate
        + w3 * format_translation_rate
    """

    values = _read_submetrics(
        submetrics,
        (
            "schema_reconciliation_rate",
            "api_adherence_rate",
            "format_translation_rate",
        ),
    )

    return _weighted_score(
        dimension="D5",
        effective_values=values,
        weights=weights,
    )


def compute_svc(
    submetrics: Mapping[str, float],
    weights: Mapping[str, float],
) -> float:
    """
    Calculate Sovereignty Compliance.

    SVC =
        w1 * localization_compliance_rate
        + w2 * cross_border_flagging_rate
        + w3 * audit_trail_completeness
    """

    values = _read_submetrics(
        submetrics,
        (
            "localization_compliance_rate",
            "cross_border_flagging_rate",
            "audit_trail_completeness",
        ),
    )

    return _weighted_score(
        dimension="SVC",
        effective_values=values,
        weights=weights,
    )


DIMENSION_COMPUTERS: dict[
    str,
    Callable[
        [Mapping[str, float], Mapping[str, float]],
        float,
    ],
] = {
    "D1": compute_d1,
    "D2": compute_d2,
    "D3": compute_d3,
    "D4": compute_d4,
    "D5": compute_d5,
    "SVC": compute_svc,
}


def compute_dimension_scores(
    submetrics_by_dimension: Mapping[
        str,
        Mapping[str, float],
    ],
    formula_config: Mapping[str, Mapping[str, Any]],
) -> dict[str, float]:
    """
    Compute D1-D5 and optionally SVC.

    D1-D5 are mandatory. SVC is calculated only when its
    submetrics are supplied.
    """

    missing_standard_dimensions = (
        set(STANDARD_DIMENSIONS)
        - set(submetrics_by_dimension)
    )

    if missing_standard_dimensions:
        raise DimensionValueError(
            "Missing standard dimension submetrics: "
            + ", ".join(
                sorted(missing_standard_dimensions)
            )
        )

    dimensions_to_compute = list(
        STANDARD_DIMENSIONS
    )

    if "SVC" in submetrics_by_dimension:
        dimensions_to_compute.append("SVC")

    scores: dict[str, float] = {}

    for dimension in dimensions_to_compute:
        formula = formula_config.get(dimension)

        if not isinstance(formula, Mapping):
            raise DimensionConfigurationError(
                f"Missing formula configuration for {dimension}."
            )

        weights = formula.get("weights")

        if not isinstance(weights, Mapping):
            raise DimensionConfigurationError(
                f"Missing weight configuration for {dimension}."
            )

        computer = DIMENSION_COMPUTERS[dimension]

        scores[dimension] = computer(
            submetrics_by_dimension[dimension],
            weights,
        )

    return scores


def _read_submetrics(
    submetrics: Mapping[str, float],
    required_names: tuple[str, ...],
) -> dict[str, float]:
    values: dict[str, float] = {}

    for name in required_names:
        if name not in submetrics:
            raise DimensionValueError(
                f"Missing required submetric: {name}."
            )

        values[name] = _require_unit_interval(
            name,
            submetrics[name],
        )

    return values


def _weighted_score(
    *,
    dimension: str,
    effective_values: Mapping[str, float],
    weights: Mapping[str, float],
) -> float:
    expected_names = set(effective_values)
    actual_names = set(weights)

    if actual_names != expected_names:
        missing = expected_names - actual_names
        unexpected = actual_names - expected_names

        details = []

        if missing:
            details.append(
                "missing=" + ",".join(sorted(missing))
            )

        if unexpected:
            details.append(
                "unexpected=" + ",".join(sorted(unexpected))
            )

        raise DimensionConfigurationError(
            f"Invalid weights for {dimension}: "
            + "; ".join(details)
        )

    numeric_weights: dict[str, float] = {}

    for name, weight in weights.items():
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
        ):
            raise DimensionConfigurationError(
                f"{dimension}.{name} weight must be numeric."
            )

        numeric_weight = float(weight)

        if not 0.0 <= numeric_weight <= 1.0:
            raise DimensionConfigurationError(
                f"{dimension}.{name} weight must be in [0, 1]."
            )

        numeric_weights[name] = numeric_weight

    total_weight = sum(
        numeric_weights.values()
    )

    if not math.isclose(
        total_weight,
        1.0,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise DimensionConfigurationError(
            f"{dimension} weights must sum to 1.0; "
            f"received {total_weight}."
        )

    score = sum(
        numeric_weights[name] * effective_values[name]
        for name in effective_values
    )

    return min(
        1.0,
        max(0.0, float(score)),
    )


def _require_unit_interval(
    name: str,
    value: float,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
    ):
        raise DimensionValueError(
            f"{name} must be numeric."
        )

    numeric_value = float(value)

    if not 0.0 <= numeric_value <= 1.0:
        raise DimensionValueError(
            f"{name} must be in [0, 1]; "
            f"received {numeric_value}."
        )

    return numeric_value