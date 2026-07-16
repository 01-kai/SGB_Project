from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.random import Generator


class DistributionError(ValueError):
    """Raised when a probability distribution is configured incorrectly."""


def truncated_pareto_quantile(
    probability: float,
    minimum: float,
    maximum: float,
    alpha: float,
) -> float:
    """
    Calculate the quantile of a bounded Pareto Type I distribution.

    The untruncated Pareto Type I survival function is:

        P(X > x) = (minimum / x) ** alpha

    This implementation truncates the distribution to:

        minimum <= X <= maximum

    Parameters
    ----------
    probability:
        Cumulative probability in [0, 1].
    minimum:
        Lower distribution bound.
    maximum:
        Upper distribution bound.
    alpha:
        Positive Pareto shape parameter.

    Returns
    -------
    float
        The requested bounded Pareto quantile.
    """

    probability_value = _require_probability(
        probability,
        "probability",
    )

    minimum_value, maximum_value, alpha_value = (
        _validate_pareto_parameters(
            minimum=minimum,
            maximum=maximum,
            alpha=alpha,
        )
    )

    truncation_ratio = (
        minimum_value / maximum_value
    ) ** alpha_value

    denominator = (
        1.0
        - probability_value
        * (1.0 - truncation_ratio)
    )

    quantile = minimum_value / (
        denominator ** (1.0 / alpha_value)
    )

    return min(
        maximum_value,
        max(minimum_value, float(quantile)),
    )


def sample_truncated_pareto(
    rng: Generator,
    minimum: float,
    maximum: float,
    alpha: float,
) -> float:
    """
    Draw one sample from a bounded Pareto Type I distribution.

    Inverse-transform sampling is used so the upper truncation is
    handled directly rather than generating and rejecting samples.
    """

    _require_generator(rng)

    probability = float(rng.random())

    return truncated_pareto_quantile(
        probability=probability,
        minimum=minimum,
        maximum=maximum,
        alpha=alpha,
    )


def sample_truncated_pareto_integer(
    rng: Generator,
    minimum: int,
    maximum: int,
    alpha: float,
) -> int:
    """
    Draw one integer from a bounded Pareto Type I distribution.

    The continuous sample is rounded down and then constrained to the
    declared integer bounds.
    """

    if isinstance(minimum, bool) or not isinstance(minimum, int):
        raise DistributionError(
            "minimum must be an integer."
        )

    if isinstance(maximum, bool) or not isinstance(maximum, int):
        raise DistributionError(
            "maximum must be an integer."
        )

    sample = sample_truncated_pareto(
        rng=rng,
        minimum=float(minimum),
        maximum=float(maximum),
        alpha=alpha,
    )

    integer_sample = math.floor(sample)

    return min(
        maximum,
        max(minimum, integer_sample),
    )


def sample_beta(
    rng: Generator,
    alpha: float,
    beta: float,
) -> float:
    """Draw one normalized value from a Beta distribution."""

    _require_generator(rng)

    alpha_value = _require_positive_number(
        alpha,
        "alpha",
    )
    beta_value = _require_positive_number(
        beta,
        "beta",
    )

    return float(
        rng.beta(
            alpha_value,
            beta_value,
        )
    )


def sample_categorical(
    rng: Generator,
    probabilities: Mapping[str, float],
) -> str:
    """
    Select one label from a categorical probability distribution.

    Mapping insertion order is preserved so seeded runs remain
    deterministic.
    """

    _require_generator(rng)

    if not isinstance(probabilities, Mapping) or not probabilities:
        raise DistributionError(
            "probabilities must be a non-empty mapping."
        )

    labels: list[str] = []
    numeric_probabilities: list[float] = []

    for label, probability in probabilities.items():
        if not isinstance(label, str) or not label.strip():
            raise DistributionError(
                "Every category label must be a non-empty string."
            )

        labels.append(label)
        numeric_probabilities.append(
            _require_probability(
                probability,
                f"probability for {label}",
            )
        )

    total_probability = sum(numeric_probabilities)

    if not math.isclose(
        total_probability,
        1.0,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise DistributionError(
            "Categorical probabilities must sum to 1.0; "
            f"received {total_probability}."
        )

    selected = rng.choice(
        labels,
        p=numeric_probabilities,
    )

    return str(selected)


def _validate_pareto_parameters(
    *,
    minimum: Any,
    maximum: Any,
    alpha: Any,
) -> tuple[float, float, float]:
    minimum_value = _require_positive_number(
        minimum,
        "minimum",
    )
    maximum_value = _require_positive_number(
        maximum,
        "maximum",
    )
    alpha_value = _require_positive_number(
        alpha,
        "alpha",
    )

    if maximum_value <= minimum_value:
        raise DistributionError(
            "maximum must be greater than minimum."
        )

    if alpha_value <= 1.0:
        raise DistributionError(
            "alpha must be greater than 1.0."
        )

    return (
        minimum_value,
        maximum_value,
        alpha_value,
    )


def _require_generator(rng: Any) -> Generator:
    if not isinstance(rng, Generator):
        raise DistributionError(
            "rng must be a numpy.random.Generator."
        )

    return rng


def _require_positive_number(
    value: Any,
    name: str,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
    ):
        raise DistributionError(
            f"{name} must be numeric."
        )

    numeric_value = float(value)

    if not math.isfinite(numeric_value):
        raise DistributionError(
            f"{name} must be finite."
        )

    if numeric_value <= 0.0:
        raise DistributionError(
            f"{name} must be greater than zero."
        )

    return numeric_value


def _require_probability(
    value: Any,
    name: str,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(
            value,
            (int, float, np.integer, np.floating),
        )
    ):
        raise DistributionError(
            f"{name} must be numeric."
        )

    numeric_value = float(value)

    if not math.isfinite(numeric_value):
        raise DistributionError(
            f"{name} must be finite."
        )

    if not 0.0 <= numeric_value <= 1.0:
        raise DistributionError(
            f"{name} must be in [0, 1]."
        )

    return numeric_value