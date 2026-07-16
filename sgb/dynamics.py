from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from sgb.agents import OrganizationAgent
from sgb.config import (
    EXPECTED_DIMENSION_SUBMETRICS,
    PROJECT_ROOT,
)
from sgb.dimensions import ALL_DIMENSIONS


DYNAMICS_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "dynamics.yaml"
)

EXPECTED_FAILURE_REASONS = {
    "schema",
    "security",
    "metadata",
    "regulatory",
}


class DynamicsConfigurationError(ValueError):
    """Raised when dynamics configuration is invalid."""


class DynamicsStateError(ValueError):
    """Raised when an invalid exchange event is supplied."""


def load_dynamics_config(
    path: str | Path = DYNAMICS_CONFIG_PATH,
) -> dict[str, Any]:
    """
    Load and validate the neutral operational-dynamics configuration.
    """

    config_path = Path(path)

    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    config_path = config_path.resolve()

    if not config_path.exists():
        raise DynamicsConfigurationError(
            "Dynamics configuration does not exist: "
            f"{config_path}"
        )

    if not config_path.is_file():
        raise DynamicsConfigurationError(
            "Dynamics configuration path is not a file: "
            f"{config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise DynamicsConfigurationError(
            "Dynamics configuration root must be a mapping."
        )

    validate_dynamics_config(config)

    config["_metadata"] = {
        "source_path": str(config_path),
    }

    return config


def validate_dynamics_config(
    config: Mapping[str, Any],
) -> None:
    """
    Validate all exchange, degradation, and recovery parameters.
    """

    if config.get("schema_version") != 1:
        raise DynamicsConfigurationError(
            "schema_version must be 1."
        )

    if config.get("config_kind") != "dynamics":
        raise DynamicsConfigurationError(
            "config_kind must be 'dynamics'."
        )

    configuration = _require_mapping(
        config,
        "configuration",
    )
    success = _require_mapping(
        config,
        "success",
    )
    failure = _require_mapping(
        config,
        "failure",
    )
    recovery = _require_mapping(
        config,
        "recovery",
    )

    status = configuration.get("status")

    if status not in {"candidate", "final"}:
        raise DynamicsConfigurationError(
            "configuration.status must be "
            "'candidate' or 'final'."
        )

    if (
        configuration.get("parameter_category")
        != "modeling_judgment"
    ):
        raise DynamicsConfigurationError(
            "configuration.parameter_category must be "
            "'modeling_judgment'."
        )

    _require_non_empty_string(
        configuration,
        "description",
    )

    success_magnitude = _require_unit_fraction(
        success,
        "magnitude",
    )

    if success_magnitude <= 0.0:
        raise DynamicsConfigurationError(
            "success.magnitude must be greater than zero."
        )

    _require_non_empty_string(
        success,
        "rationale",
    )

    success_changes = _require_mapping(
        success,
        "changes",
    )

    _validate_change_profile(
        success_changes,
        profile_name="success",
    )

    failure_magnitude = _require_unit_fraction(
        failure,
        "magnitude",
    )

    if failure_magnitude <= 0.0:
        raise DynamicsConfigurationError(
            "failure.magnitude must be greater than zero."
        )

    _require_non_empty_string(
        failure,
        "rationale",
    )

    failure_changes = _require_mapping(
        failure,
        "changes",
    )

    actual_failure_reasons = set(
        failure_changes
    )

    if actual_failure_reasons != EXPECTED_FAILURE_REASONS:
        missing = (
            EXPECTED_FAILURE_REASONS
            - actual_failure_reasons
        )
        unexpected = (
            actual_failure_reasons
            - EXPECTED_FAILURE_REASONS
        )

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

        raise DynamicsConfigurationError(
            "Invalid failure reasons: "
            + "; ".join(details)
        )

    for failure_reason, profile in (
        failure_changes.items()
    ):
        if not isinstance(profile, Mapping):
            raise DynamicsConfigurationError(
                f"Failure profile {failure_reason!r} "
                "must be a mapping."
            )

        _validate_change_profile(
            profile,
            profile_name=(
                f"failure.{failure_reason}"
            ),
        )

    recovery_gap_fraction = (
        _require_unit_fraction(
            recovery,
            "gap_fraction_per_step",
        )
    )

    maximum_adjustment = (
        _require_unit_fraction(
            recovery,
            "maximum_adjustment_per_step",
        )
    )

    if recovery_gap_fraction <= 0.0:
        raise DynamicsConfigurationError(
            "recovery.gap_fraction_per_step "
            "must be greater than zero."
        )

    if maximum_adjustment <= 0.0:
        raise DynamicsConfigurationError(
            "recovery.maximum_adjustment_per_step "
            "must be greater than zero."
        )

    _require_non_empty_string(
        recovery,
        "rationale",
    )


class NeutralOperationalDynamics:
    """
    Framework-neutral learning, degradation, and recovery.

    This class never reads framework names or framework weights.
    The same operational trajectory can therefore be evaluated later
    by DBL, DMM, and ISF without allowing a framework to influence
    the trajectory that it is used to score.
    """

    def __init__(
        self,
        base_config: Mapping[str, Any],
        dynamics_config: Mapping[str, Any],
    ) -> None:
        if not isinstance(base_config, Mapping):
            raise DynamicsConfigurationError(
                "base_config must be a mapping."
            )

        if not isinstance(
            dynamics_config,
            Mapping,
        ):
            raise DynamicsConfigurationError(
                "dynamics_config must be a mapping."
            )

        validate_dynamics_config(
            dynamics_config
        )

        self.base_config = deepcopy(
            dict(base_config)
        )
        self.config = deepcopy(
            dict(dynamics_config)
        )

        self.success_magnitude = float(
            self.config[
                "success"
            ]["magnitude"]
        )

        self.failure_magnitude = float(
            self.config[
                "failure"
            ]["magnitude"]
        )

        self.success_changes = deepcopy(
            self.config[
                "success"
            ]["changes"]
        )

        self.failure_changes = deepcopy(
            self.config[
                "failure"
            ]["changes"]
        )

        recovery = self.config["recovery"]

        self.recovery_gap_fraction = float(
            recovery[
                "gap_fraction_per_step"
            ]
        )

        self.maximum_recovery_adjustment = float(
            recovery[
                "maximum_adjustment_per_step"
            ]
        )

        self.baseline_targets = (
            self._calculate_baseline_targets()
        )

    def apply_exchange_outcome(
        self,
        model: Any,
        sender: OrganizationAgent,
        receiver: OrganizationAgent,
        event: Mapping[str, Any],
    ) -> None:
        """
        Apply one exchange outcome to both participants.

        Successful and failed exchanges use identical rules for every
        organization and every later evaluation framework.
        """

        if not isinstance(event, Mapping):
            raise DynamicsStateError(
                "event must be a mapping."
            )

        success = event.get("success")

        if not isinstance(success, bool):
            raise DynamicsStateError(
                "event.success must be Boolean."
            )

        if success:
            profile = self.success_changes
            magnitude = self.success_magnitude
        else:
            failure_reason = event.get(
                "failure_reason"
            )

            if (
                failure_reason
                not in self.failure_changes
            ):
                raise DynamicsStateError(
                    "Unknown or missing failure reason: "
                    f"{failure_reason!r}."
                )

            profile = self.failure_changes[
                failure_reason
            ]
            magnitude = self.failure_magnitude

        changes = self._scale_change_profile(
            profile=profile,
            magnitude=magnitude,
        )

        sender.apply_submetric_changes(
            changes
        )

        if receiver.uid != sender.uid:
            receiver.apply_submetric_changes(
                changes
            )

    def recover_agent(
        self,
        model: Any,
        agent: OrganizationAgent,
    ) -> None:
        """
        Move each operational submetric toward its baseline mean.

        For a Beta(alpha, beta) distribution, the target mean is:

            alpha / (alpha + beta)
        """

        current_state = agent.submetrics

        recovery_changes: dict[
            str,
            dict[str, float],
        ] = {}

        for dimension in ALL_DIMENSIONS:
            dimension_changes: dict[
                str,
                float,
            ] = {}

            for submetric, current_value in (
                current_state[
                    dimension
                ].items()
            ):
                target = self.baseline_targets[
                    submetric
                ]

                gap = target - current_value

                proposed_adjustment = (
                    self.recovery_gap_fraction
                    * gap
                )

                bounded_adjustment = min(
                    self.maximum_recovery_adjustment,
                    max(
                        -self.maximum_recovery_adjustment,
                        proposed_adjustment,
                    ),
                )

                if not math.isclose(
                    bounded_adjustment,
                    0.0,
                    abs_tol=1e-15,
                ):
                    dimension_changes[
                        submetric
                    ] = bounded_adjustment

            if dimension_changes:
                recovery_changes[
                    dimension
                ] = dimension_changes

        if recovery_changes:
            agent.apply_submetric_changes(
                recovery_changes
            )

    def _calculate_baseline_targets(
        self,
    ) -> dict[str, float]:
        try:
            distributions = self.base_config[
                "baseline"
            ]["initialization"][
                "submetric_distributions"
            ]
        except (KeyError, TypeError) as error:
            raise DynamicsConfigurationError(
                "Base configuration is missing baseline "
                "submetric distributions."
            ) from error

        if not isinstance(distributions, Mapping):
            raise DynamicsConfigurationError(
                "Baseline submetric distributions "
                "must be a mapping."
            )

        expected_submetrics = set().union(
            *EXPECTED_DIMENSION_SUBMETRICS.values()
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
                    + ",".join(sorted(missing))
                )

            if unexpected:
                details.append(
                    "unexpected="
                    + ",".join(
                        sorted(unexpected)
                    )
                )

            raise DynamicsConfigurationError(
                "Invalid baseline distributions: "
                + "; ".join(details)
            )

        targets: dict[str, float] = {}

        for submetric, parameters in (
            distributions.items()
        ):
            if not isinstance(
                parameters,
                Mapping,
            ):
                raise DynamicsConfigurationError(
                    f"Distribution for {submetric} "
                    "must be a mapping."
                )

            alpha = (
                _require_positive_number_value(
                    parameters.get("alpha"),
                    f"{submetric}.alpha",
                )
            )
            beta = (
                _require_positive_number_value(
                    parameters.get("beta"),
                    f"{submetric}.beta",
                )
            )

            targets[submetric] = (
                alpha
                / (alpha + beta)
            )

        return targets

    @staticmethod
    def _scale_change_profile(
        *,
        profile: Mapping[
            str,
            Mapping[str, float],
        ],
        magnitude: float,
    ) -> dict[
        str,
        dict[str, float],
    ]:
        scaled: dict[
            str,
            dict[str, float],
        ] = {}

        for dimension, submetrics in (
            profile.items()
        ):
            scaled[dimension] = {
                submetric: (
                    float(coefficient)
                    * magnitude
                )
                for submetric, coefficient
                in submetrics.items()
            }

        return scaled


def _validate_change_profile(
    profile: Mapping[str, Any],
    *,
    profile_name: str,
) -> None:
    if not profile:
        raise DynamicsConfigurationError(
            f"{profile_name} changes cannot be empty."
        )

    for dimension, changes in profile.items():
        if (
            dimension
            not in EXPECTED_DIMENSION_SUBMETRICS
        ):
            raise DynamicsConfigurationError(
                f"Unknown dimension in {profile_name}: "
                f"{dimension!r}."
            )

        if (
            not isinstance(changes, Mapping)
            or not changes
        ):
            raise DynamicsConfigurationError(
                f"{profile_name}.{dimension} "
                "must be a non-empty mapping."
            )

        for submetric, coefficient in (
            changes.items()
        ):
            if (
                submetric
                not in EXPECTED_DIMENSION_SUBMETRICS[
                    dimension
                ]
            ):
                raise DynamicsConfigurationError(
                    f"Unknown submetric in {profile_name}: "
                    f"{dimension}.{submetric}."
                )

            numeric_coefficient = (
                _require_finite_number_value(
                    coefficient,
                    (
                        f"{profile_name}."
                        f"{dimension}."
                        f"{submetric}"
                    ),
                )
            )

            if not (
                -1.0
                <= numeric_coefficient
                <= 1.0
            ):
                raise DynamicsConfigurationError(
                    f"{profile_name}.{dimension}."
                    f"{submetric} coefficient "
                    "must be in [-1, 1]."
                )

            if math.isclose(
                numeric_coefficient,
                0.0,
                abs_tol=1e-15,
            ):
                raise DynamicsConfigurationError(
                    f"{profile_name}.{dimension}."
                    f"{submetric} coefficient "
                    "cannot be zero."
                )


def _require_mapping(
    container: Mapping[str, Any],
    key: str,
) -> Mapping[str, Any]:
    value = container.get(key)

    if not isinstance(value, Mapping):
        raise DynamicsConfigurationError(
            f"{key} must be a mapping."
        )

    return value


def _require_non_empty_string(
    container: Mapping[str, Any],
    key: str,
) -> str:
    value = container.get(key)

    if (
        not isinstance(value, str)
        or not value.strip()
    ):
        raise DynamicsConfigurationError(
            f"{key} must be a non-empty string."
        )

    return value.strip()


def _require_unit_fraction(
    container: Mapping[str, Any],
    key: str,
) -> float:
    numeric_value = (
        _require_finite_number_value(
            container.get(key),
            key,
        )
    )

    if not 0.0 <= numeric_value <= 1.0:
        raise DynamicsConfigurationError(
            f"{key} must be in [0, 1]."
        )

    return numeric_value


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
        raise DynamicsConfigurationError(
            f"{name} must be greater than zero."
        )

    return numeric_value


def _require_finite_number_value(
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
        raise DynamicsConfigurationError(
            f"{name} must be numeric."
        )

    numeric_value = float(value)

    if not math.isfinite(
        numeric_value
    ):
        raise DynamicsConfigurationError(
            f"{name} must be finite."
        )

    return numeric_value