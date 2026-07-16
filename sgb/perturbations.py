from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

import networkx as nx

from sgb.agents import OrganizationAgent
from sgb.model import (
    ModelStateError,
    SGBModel,
)


class PerturbationError(ValueError):
    """Raised when a perturbation request is invalid."""


def apply_data_quality_shock(
    model: SGBModel,
    impact: float,
    target_uids: Iterable[int] | None = None,
) -> list[dict[str, Any]]:
    """
    Reduce completeness and schema conformance while increasing
    duplicate rates.
    """

    normalized_impact = _require_impact(
        impact
    )

    return _apply_uniform_changes(
        model=model,
        perturbation_name=(
            "data_quality_shock"
        ),
        changes={
            "D1": {
                "completeness_ratio": (
                    -normalized_impact
                ),
                "duplicate_rate": (
                    normalized_impact
                ),
                "schema_pass_rate": (
                    -normalized_impact
                ),
            }
        },
        target_uids=target_uids,
    )


def apply_metadata_outage(
    model: SGBModel,
    impact: float,
    target_uids: Iterable[int] | None = None,
) -> list[dict[str, Any]]:
    """
    Reduce lineage, catalog coverage, and catalog freshness.
    """

    normalized_impact = _require_impact(
        impact
    )

    return _apply_uniform_changes(
        model=model,
        perturbation_name=(
            "metadata_outage"
        ),
        changes={
            "D3": {
                "lineage_rate": (
                    -normalized_impact
                ),
                "catalog_coverage": (
                    -normalized_impact
                ),
                "catalog_freshness": (
                    -normalized_impact
                ),
            }
        },
        target_uids=target_uids,
    )


def apply_regulatory_change(
    model: SGBModel,
    impact: float,
    target_uids: Iterable[int] | None = None,
) -> list[dict[str, Any]]:
    """
    Reduce regulatory-rule satisfaction and consent propagation.
    """

    normalized_impact = _require_impact(
        impact
    )

    return _apply_uniform_changes(
        model=model,
        perturbation_name=(
            "regulatory_change"
        ),
        changes={
            "D4": {
                "rule_pass_rate": (
                    -normalized_impact
                ),
                "consent_propagation_rate": (
                    -normalized_impact
                ),
            }
        },
        target_uids=target_uids,
    )


def apply_interoperability_disruption(
    model: SGBModel,
    impact: float,
    target_uids: Iterable[int] | None = None,
) -> list[dict[str, Any]]:
    """
    Reduce schema reconciliation, API adherence, and format translation.
    """

    normalized_impact = _require_impact(
        impact
    )

    return _apply_uniform_changes(
        model=model,
        perturbation_name=(
            "interoperability_disruption"
        ),
        changes={
            "D5": {
                "schema_reconciliation_rate": (
                    -normalized_impact
                ),
                "api_adherence_rate": (
                    -normalized_impact
                ),
                "format_translation_rate": (
                    -normalized_impact
                ),
            }
        },
        target_uids=target_uids,
    )


def apply_sovereignty_policy_shock(
    model: SGBModel,
    impact: float,
    target_uids: Iterable[int] | None = None,
) -> list[dict[str, Any]]:
    """
    Reduce sovereignty-compliance submetrics.

    The operational state is changed once and may later be evaluated
    by every framework. No framework-specific trajectory is created.
    """

    normalized_impact = _require_impact(
        impact
    )

    return _apply_uniform_changes(
        model=model,
        perturbation_name=(
            "sovereignty_policy_shock"
        ),
        changes={
            "SVC": {
                "localization_compliance_rate": (
                    -normalized_impact
                ),
                "cross_border_flagging_rate": (
                    -normalized_impact
                ),
                "audit_trail_completeness": (
                    -normalized_impact
                ),
            }
        },
        target_uids=target_uids,
    )


def apply_security_incident(
    model: SGBModel,
    target_uid: int,
    impact: float,
    max_distance: int,
    decay: float,
) -> list[dict[str, Any]]:
    """
    Apply a network-propagating security incident.

    Impact at network distance d is:

        impact * decay ** d
    """

    normalized_impact = _require_impact(
        impact
    )
    normalized_decay = _require_decay(
        decay
    )
    validated_distance = (
        _require_non_negative_integer(
            max_distance,
            "max_distance",
        )
    )
    validated_target_uid = (
        _require_non_negative_integer(
            target_uid,
            "target_uid",
        )
    )

    _get_agent_or_raise(
        model,
        validated_target_uid,
    )

    shortest_paths = (
        nx.single_source_shortest_path_length(
            model.graph,
            validated_target_uid,
            cutoff=validated_distance,
        )
    )

    records: list[
        dict[str, Any]
    ] = []

    for affected_uid, distance in sorted(
        shortest_paths.items()
    ):
        distance_impact = (
            normalized_impact
            * (
                normalized_decay
                ** distance
            )
        )

        agent = _get_agent_or_raise(
            model,
            affected_uid,
        )

        before = agent.dimensions

        agent.apply_submetric_changes(
            {
                "D2": {
                    "encryption_coverage": (
                        -distance_impact
                    ),
                    "normalized_breach_latency": (
                        distance_impact
                    ),
                    "policy_violation_rate": (
                        distance_impact
                    ),
                }
            }
        )

        after = agent.dimensions

        records.append(
            {
                "perturbation": (
                    "security_incident"
                ),
                "step": model.current_step,
                "target_uid": (
                    validated_target_uid
                ),
                "affected_uid": affected_uid,
                "network_distance": distance,
                "applied_impact": (
                    distance_impact
                ),
                "D2_before": before["D2"],
                "D2_after": after["D2"],
            }
        )

    return records


def _apply_uniform_changes(
    *,
    model: SGBModel,
    perturbation_name: str,
    changes: Mapping[
        str,
        Mapping[str, float],
    ],
    target_uids: Iterable[int] | None,
) -> list[dict[str, Any]]:
    target_agents = (
        _resolve_target_agents(
            model=model,
            target_uids=target_uids,
        )
    )

    records: list[
        dict[str, Any]
    ] = []

    for agent in target_agents:
        before = agent.dimensions

        agent.apply_submetric_changes(
            changes
        )

        after = agent.dimensions

        record: dict[str, Any] = {
            "perturbation": (
                perturbation_name
            ),
            "step": model.current_step,
            "affected_uid": agent.uid,
        }

        for dimension in changes:
            record[
                f"{dimension}_before"
            ] = before[dimension]

            record[
                f"{dimension}_after"
            ] = after[dimension]

        records.append(record)

    return records


def _resolve_target_agents(
    *,
    model: SGBModel,
    target_uids: Iterable[int] | None,
) -> list[OrganizationAgent]:
    if target_uids is None:
        return [
            _get_agent_or_raise(
                model,
                uid,
            )
            for uid in sorted(
                model.agents_by_uid
            )
        ]

    if isinstance(
        target_uids,
        (str, bytes),
    ):
        raise PerturbationError(
            "target_uids must be an iterable "
            "of integer identifiers."
        )

    try:
        supplied_uids = list(
            target_uids
        )
    except TypeError as error:
        raise PerturbationError(
            "target_uids must be iterable."
        ) from error

    validated_uids: set[int] = set()

    for uid in supplied_uids:
        validated_uids.add(
            _require_non_negative_integer(
                uid,
                "target uid",
            )
        )

    return [
        _get_agent_or_raise(
            model,
            uid,
        )
        for uid in sorted(
            validated_uids
        )
    ]


def _get_agent_or_raise(
    model: SGBModel,
    uid: int,
) -> OrganizationAgent:
    try:
        return model.get_agent(uid)
    except ModelStateError as error:
        raise PerturbationError(
            f"Unknown organization uid: {uid}."
        ) from error


def _require_impact(
    impact: Any,
) -> float:
    if (
        isinstance(impact, bool)
        or not isinstance(
            impact,
            (int, float),
        )
    ):
        raise PerturbationError(
            "impact must be numeric."
        )

    numeric_impact = float(impact)

    if not math.isfinite(
        numeric_impact
    ):
        raise PerturbationError(
            "impact must be finite."
        )

    if not 0.0 <= numeric_impact <= 1.0:
        raise PerturbationError(
            "impact must be in [0, 1]."
        )

    return numeric_impact


def _require_decay(
    decay: Any,
) -> float:
    if (
        isinstance(decay, bool)
        or not isinstance(
            decay,
            (int, float),
        )
    ):
        raise PerturbationError(
            "decay must be numeric."
        )

    numeric_decay = float(decay)

    if not math.isfinite(
        numeric_decay
    ):
        raise PerturbationError(
            "decay must be finite."
        )

    if not 0.0 < numeric_decay <= 1.0:
        raise PerturbationError(
            "decay must be in (0, 1]."
        )

    return numeric_decay


def _require_non_negative_integer(
    value: Any,
    name: str,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
    ):
        raise PerturbationError(
            f"{name} must be an integer."
        )

    if value < 0:
        raise PerturbationError(
            f"{name} must be zero or greater."
        )

    return value