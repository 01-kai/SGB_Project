from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import mesa

from sgb.config import EXPECTED_DIMENSION_SUBMETRICS
from sgb.dimensions import (
    ALL_DIMENSIONS,
    compute_dimension_scores,
)


class AgentStateError(ValueError):
    """Raised when an organization agent receives invalid state."""


class OrganizationAgent(mesa.Agent):
    """
    Synthetic organization participating in the governance ecosystem.

    The agent stores operational submetrics rather than directly storing
    arbitrary D1-D5 values. Dimension scores are always derived from the
    Appendix A and Appendix C formulas defined in ``sgb.dimensions``.

    Framework-specific GMI values are intentionally not stored here.
    Framework scoring will be performed separately so that the selected
    framework does not alter the underlying operational trajectory.
    """

    def __init__(
        self,
        model: mesa.Model,
        uid: int,
        org_type: str,
        exchange_volume: int,
        submetrics_by_dimension: Mapping[
            str,
            Mapping[str, float],
        ],
        formula_config: Mapping[
            str,
            Mapping[str, Any],
        ],
    ) -> None:
        super().__init__(model)

        self.uid = _require_non_negative_integer(
            uid,
            "uid",
        )

        self.org_type = _require_non_empty_string(
            org_type,
            "org_type",
        )

        self.exchange_volume = _require_positive_integer(
            exchange_volume,
            "exchange_volume",
        )

        self._formula_config = deepcopy(
            dict(formula_config)
        )

        self._submetrics = _validate_and_copy_submetrics(
            submetrics_by_dimension
        )

        self._dimensions: dict[str, float] = {}

        self.recompute_dimensions()

    @property
    def submetrics(self) -> dict[str, dict[str, float]]:
        """
        Return a defensive copy of all operational submetrics.

        External code cannot silently mutate the internal state without
        triggering dimension recomputation.
        """

        return deepcopy(self._submetrics)

    @property
    def dimensions(self) -> dict[str, float]:
        """Return a copy of the current D1-D5 and SVC scores."""

        return dict(self._dimensions)

    def recompute_dimensions(self) -> dict[str, float]:
        """
        Recalculate every dimension from the current submetrics.

        Returns
        -------
        dict
            A copy of the newly computed dimension scores.
        """

        scores = compute_dimension_scores(
            self._submetrics,
            self._formula_config,
        )

        expected_dimensions = set(ALL_DIMENSIONS)
        actual_dimensions = set(scores)

        if actual_dimensions != expected_dimensions:
            missing = expected_dimensions - actual_dimensions
            unexpected = actual_dimensions - expected_dimensions

            details = []

            if missing:
                details.append(
                    "missing=" + ",".join(sorted(missing))
                )

            if unexpected:
                details.append(
                    "unexpected=" + ",".join(sorted(unexpected))
                )

            raise AgentStateError(
                "Dimension calculation returned an invalid set: "
                + "; ".join(details)
            )

        self._dimensions = {
            dimension: float(value)
            for dimension, value in scores.items()
        }

        return dict(self._dimensions)

    def get_dimension_score(
        self,
        dimension: str,
    ) -> float:
        """Return one derived dimension score."""

        if dimension not in self._dimensions:
            raise AgentStateError(
                f"Unknown dimension: {dimension!r}."
            )

        return self._dimensions[dimension]

    def get_submetric(
        self,
        dimension: str,
        submetric: str,
    ) -> float:
        """Return one operational submetric."""

        self._validate_dimension_and_submetric(
            dimension,
            submetric,
        )

        return self._submetrics[dimension][submetric]

    def set_submetric(
        self,
        dimension: str,
        submetric: str,
        value: float,
    ) -> float:
        """
        Set one submetric to an explicit normalized value.

        Explicit assignments outside [0, 1] are rejected rather than
        silently corrected.
        """

        self._validate_dimension_and_submetric(
            dimension,
            submetric,
        )

        normalized_value = _require_unit_interval(
            value,
            f"{dimension}.{submetric}",
        )

        self._submetrics[dimension][submetric] = (
            normalized_value
        )

        self.recompute_dimensions()

        return normalized_value

    def adjust_submetric(
        self,
        dimension: str,
        submetric: str,
        delta: float,
    ) -> float:
        """
        Adjust one operational submetric by a signed amount.

        Dynamic changes are clamped to [0, 1]. This allows perturbations,
        exchange outcomes, and recovery processes to remain normalized
        without directly manipulating D1-D5.
        """

        self._validate_dimension_and_submetric(
            dimension,
            submetric,
        )

        numeric_delta = _require_finite_number(
            delta,
            f"delta for {dimension}.{submetric}",
        )

        current_value = self._submetrics[
            dimension
        ][submetric]

        updated_value = _clamp_unit_interval(
            current_value + numeric_delta
        )

        self._submetrics[dimension][submetric] = (
            updated_value
        )

        self.recompute_dimensions()

        return updated_value

    def apply_submetric_changes(
        self,
        changes: Mapping[
            str,
            Mapping[str, float],
        ],
    ) -> dict[str, float]:
        """
        Apply multiple signed submetric changes atomically.

        All requested changes are validated before the agent state is
        replaced. If one change is invalid, none of the changes are
        committed.

        Parameters
        ----------
        changes:
            Nested mapping in the following form::

                {
                    "D1": {
                        "schema_pass_rate": -0.05,
                    },
                    "D5": {
                        "schema_reconciliation_rate": -0.03,
                    },
                }

        Returns
        -------
        dict
            Updated dimension scores.
        """

        if not isinstance(changes, Mapping):
            raise AgentStateError(
                "Submetric changes must be a mapping."
            )

        candidate_state = deepcopy(self._submetrics)

        for dimension, dimension_changes in changes.items():
            if not isinstance(dimension_changes, Mapping):
                raise AgentStateError(
                    f"Changes for {dimension!r} must be a mapping."
                )

            for submetric, delta in dimension_changes.items():
                self._validate_dimension_and_submetric(
                    dimension,
                    submetric,
                )

                numeric_delta = _require_finite_number(
                    delta,
                    f"delta for {dimension}.{submetric}",
                )

                current_value = candidate_state[
                    dimension
                ][submetric]

                candidate_state[
                    dimension
                ][submetric] = _clamp_unit_interval(
                    current_value + numeric_delta
                )

        candidate_dimensions = compute_dimension_scores(
            candidate_state,
            self._formula_config,
        )

        if set(candidate_dimensions) != set(ALL_DIMENSIONS):
            raise AgentStateError(
                "Atomic update produced incomplete dimensions."
            )

        self._submetrics = candidate_state
        self._dimensions = {
            dimension: float(value)
            for dimension, value
            in candidate_dimensions.items()
        }

        return dict(self._dimensions)

    def snapshot(self) -> dict[str, Any]:
        """
        Return a serializable copy of the complete organization state.

        No framework-specific GMI is included because framework evaluation
        is intentionally separated from operational behavior.
        """

        return {
            "uid": self.uid,
            "org_type": self.org_type,
            "exchange_volume": self.exchange_volume,
            "submetrics": deepcopy(self._submetrics),
            "dimensions": dict(self._dimensions),
        }

    def step(self) -> None:
        """
        Delegate one agent step to the ecosystem model.

        The model is responsible for partner selection, exchange outcomes,
        perturbations, and recovery. This prevents the agent from embedding
        framework-specific or experiment-specific behavior.
        """

        executor = getattr(
            self.model,
            "execute_agent_step",
            None,
        )

        if not callable(executor):
            raise AgentStateError(
                "The model must implement "
                "execute_agent_step(agent)."
            )

        executor(self)

    def _validate_dimension_and_submetric(
        self,
        dimension: str,
        submetric: str,
    ) -> None:
        if dimension not in self._submetrics:
            raise AgentStateError(
                f"Unknown dimension: {dimension!r}."
            )

        if submetric not in self._submetrics[dimension]:
            raise AgentStateError(
                f"Unknown submetric for {dimension}: "
                f"{submetric!r}."
            )


def _validate_and_copy_submetrics(
    submetrics_by_dimension: Mapping[
        str,
        Mapping[str, float],
    ],
) -> dict[str, dict[str, float]]:
    if not isinstance(submetrics_by_dimension, Mapping):
        raise AgentStateError(
            "submetrics_by_dimension must be a mapping."
        )

    expected_dimensions = set(ALL_DIMENSIONS)
    actual_dimensions = set(submetrics_by_dimension)

    missing_dimensions = (
        expected_dimensions - actual_dimensions
    )
    unexpected_dimensions = (
        actual_dimensions - expected_dimensions
    )

    if missing_dimensions or unexpected_dimensions:
        details = []

        if missing_dimensions:
            details.append(
                "missing="
                + ",".join(sorted(missing_dimensions))
            )

        if unexpected_dimensions:
            details.append(
                "unexpected="
                + ",".join(sorted(unexpected_dimensions))
            )

        raise AgentStateError(
            "Invalid agent dimension state: "
            + "; ".join(details)
        )

    validated_state: dict[
        str,
        dict[str, float],
    ] = {}

    for dimension in ALL_DIMENSIONS:
        dimension_submetrics = (
            submetrics_by_dimension[dimension]
        )

        if not isinstance(
            dimension_submetrics,
            Mapping,
        ):
            raise AgentStateError(
                f"Submetrics for {dimension} "
                "must be a mapping."
            )

        expected_submetrics = (
            EXPECTED_DIMENSION_SUBMETRICS[dimension]
        )
        actual_submetrics = set(
            dimension_submetrics
        )

        missing_submetrics = (
            expected_submetrics - actual_submetrics
        )
        unexpected_submetrics = (
            actual_submetrics - expected_submetrics
        )

        if missing_submetrics or unexpected_submetrics:
            details = []

            if missing_submetrics:
                details.append(
                    "missing="
                    + ",".join(
                        sorted(missing_submetrics)
                    )
                )

            if unexpected_submetrics:
                details.append(
                    "unexpected="
                    + ",".join(
                        sorted(unexpected_submetrics)
                    )
                )

            raise AgentStateError(
                f"Invalid submetrics for {dimension}: "
                + "; ".join(details)
            )

        validated_state[dimension] = {
            submetric: _require_unit_interval(
                value,
                f"{dimension}.{submetric}",
            )
            for submetric, value
            in dimension_submetrics.items()
        }

    return validated_state


def _require_non_negative_integer(
    value: Any,
    name: str,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AgentStateError(
            f"{name} must be an integer."
        )

    if value < 0:
        raise AgentStateError(
            f"{name} must be zero or greater."
        )

    return value


def _require_positive_integer(
    value: Any,
    name: str,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AgentStateError(
            f"{name} must be an integer."
        )

    if value <= 0:
        raise AgentStateError(
            f"{name} must be greater than zero."
        )

    return value


def _require_non_empty_string(
    value: Any,
    name: str,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AgentStateError(
            f"{name} must be a non-empty string."
        )

    return value.strip()


def _require_finite_number(
    value: Any,
    name: str,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
    ):
        raise AgentStateError(
            f"{name} must be numeric."
        )

    numeric_value = float(value)

    if not math.isfinite(numeric_value):
        raise AgentStateError(
            f"{name} must be finite."
        )

    return numeric_value


def _require_unit_interval(
    value: Any,
    name: str,
) -> float:
    numeric_value = _require_finite_number(
        value,
        name,
    )

    if not 0.0 <= numeric_value <= 1.0:
        raise AgentStateError(
            f"{name} must be in [0, 1]; "
            f"received {numeric_value}."
        )

    return numeric_value


def _clamp_unit_interval(value: float) -> float:
    return min(
        1.0,
        max(0.0, float(value)),
    )