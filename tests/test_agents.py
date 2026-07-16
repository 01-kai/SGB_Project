from __future__ import annotations

from copy import deepcopy
from typing import Any

import mesa
import pytest

from sgb.agents import (
    AgentStateError,
    OrganizationAgent,
)
from sgb.config import (
    BASE_CONFIG_PATH,
    load_config,
)


class DummyModel(mesa.Model):
    """Minimal model used to verify agent delegation."""

    def __init__(self) -> None:
        super().__init__(rng=42)
        self.executed_agent_uids: list[int] = []

    def execute_agent_step(
        self,
        agent: OrganizationAgent,
    ) -> None:
        self.executed_agent_uids.append(
            agent.uid
        )


@pytest.fixture(scope="module")
def formula_config() -> dict[str, dict[str, Any]]:
    config = load_config(BASE_CONFIG_PATH)
    return config["dimension_formulas"]


@pytest.fixture
def balanced_submetrics() -> dict[
    str,
    dict[str, float],
]:
    return {
        "D1": {
            "completeness_ratio": 0.50,
            "duplicate_rate": 0.50,
            "schema_pass_rate": 0.50,
        },
        "D2": {
            "encryption_coverage": 0.50,
            "normalized_breach_latency": 0.50,
            "policy_violation_rate": 0.50,
        },
        "D3": {
            "lineage_rate": 0.50,
            "catalog_coverage": 0.50,
            "catalog_freshness": 0.50,
        },
        "D4": {
            "rule_pass_rate": 0.50,
            "consent_propagation_rate": 0.50,
        },
        "D5": {
            "schema_reconciliation_rate": 0.50,
            "api_adherence_rate": 0.50,
            "format_translation_rate": 0.50,
        },
        "SVC": {
            "localization_compliance_rate": 0.50,
            "cross_border_flagging_rate": 0.50,
            "audit_trail_completeness": 0.50,
        },
    }


@pytest.fixture
def agent(
    formula_config: dict[str, dict[str, Any]],
    balanced_submetrics: dict[
        str,
        dict[str, float],
    ],
) -> OrganizationAgent:
    return OrganizationAgent(
        model=DummyModel(),
        uid=0,
        org_type="government_ministry",
        exchange_volume=100,
        submetrics_by_dimension=balanced_submetrics,
        formula_config=formula_config,
    )


def test_agent_initializes_from_submetrics(
    agent: OrganizationAgent,
) -> None:
    assert agent.uid == 0
    assert agent.org_type == "government_ministry"
    assert agent.exchange_volume == 100

    assert set(agent.dimensions) == {
        "D1",
        "D2",
        "D3",
        "D4",
        "D5",
        "SVC",
    }

    for score in agent.dimensions.values():
        assert score == pytest.approx(0.50)


def test_agent_copies_constructor_state(
    formula_config: dict[str, dict[str, Any]],
    balanced_submetrics: dict[
        str,
        dict[str, float],
    ],
) -> None:
    original = deepcopy(balanced_submetrics)

    created_agent = OrganizationAgent(
        model=DummyModel(),
        uid=1,
        org_type="state_enterprise",
        exchange_volume=200,
        submetrics_by_dimension=original,
        formula_config=formula_config,
    )

    original["D1"]["completeness_ratio"] = 1.0

    assert created_agent.get_submetric(
        "D1",
        "completeness_ratio",
    ) == pytest.approx(0.50)


def test_submetrics_property_returns_copy(
    agent: OrganizationAgent,
) -> None:
    returned_state = agent.submetrics

    returned_state["D1"][
        "completeness_ratio"
    ] = 1.0

    assert agent.get_submetric(
        "D1",
        "completeness_ratio",
    ) == pytest.approx(0.50)


def test_dimensions_property_returns_copy(
    agent: OrganizationAgent,
) -> None:
    returned_dimensions = agent.dimensions
    returned_dimensions["D1"] = 1.0

    assert agent.get_dimension_score(
        "D1"
    ) == pytest.approx(0.50)


def test_set_submetric_recomputes_dimension(
    agent: OrganizationAgent,
) -> None:
    agent.set_submetric(
        "D1",
        "duplicate_rate",
        0.70,
    )

    assert agent.get_submetric(
        "D1",
        "duplicate_rate",
    ) == pytest.approx(0.70)

    assert agent.get_dimension_score(
        "D1"
    ) == pytest.approx(0.43)


def test_adjust_submetric_clamps_upper_bound(
    agent: OrganizationAgent,
) -> None:
    updated_value = agent.adjust_submetric(
        "D1",
        "completeness_ratio",
        10.0,
    )

    assert updated_value == pytest.approx(1.0)

    assert agent.get_submetric(
        "D1",
        "completeness_ratio",
    ) == pytest.approx(1.0)


def test_adjust_submetric_clamps_lower_bound(
    agent: OrganizationAgent,
) -> None:
    updated_value = agent.adjust_submetric(
        "D2",
        "encryption_coverage",
        -10.0,
    )

    assert updated_value == pytest.approx(0.0)

    assert agent.get_submetric(
        "D2",
        "encryption_coverage",
    ) == pytest.approx(0.0)


def test_explicit_out_of_range_value_is_rejected(
    agent: OrganizationAgent,
) -> None:
    original_value = agent.get_submetric(
        "D4",
        "rule_pass_rate",
    )

    with pytest.raises(
        AgentStateError,
        match=r"\[0, 1\]",
    ):
        agent.set_submetric(
            "D4",
            "rule_pass_rate",
            1.20,
        )

    assert agent.get_submetric(
        "D4",
        "rule_pass_rate",
    ) == pytest.approx(original_value)


def test_bulk_changes_recompute_dimensions(
    agent: OrganizationAgent,
) -> None:
    updated_dimensions = (
        agent.apply_submetric_changes(
            {
                "D1": {
                    "schema_pass_rate": -0.10,
                },
                "D5": {
                    "schema_reconciliation_rate": -0.20,
                },
            }
        )
    )

    assert agent.get_submetric(
        "D1",
        "schema_pass_rate",
    ) == pytest.approx(0.40)

    assert agent.get_submetric(
        "D5",
        "schema_reconciliation_rate",
    ) == pytest.approx(0.30)

    assert updated_dimensions["D1"] == pytest.approx(
        agent.get_dimension_score("D1")
    )

    assert updated_dimensions["D5"] == pytest.approx(
        agent.get_dimension_score("D5")
    )


def test_bulk_changes_are_atomic(
    agent: OrganizationAgent,
) -> None:
    original_state = agent.submetrics
    original_dimensions = agent.dimensions

    with pytest.raises(
        AgentStateError,
        match="unknown_metric",
    ):
        agent.apply_submetric_changes(
            {
                "D1": {
                    "schema_pass_rate": -0.10,
                },
                "D5": {
                    "unknown_metric": -0.20,
                },
            }
        )

    assert agent.submetrics == original_state
    assert agent.dimensions == original_dimensions


def test_missing_dimension_is_rejected(
    formula_config: dict[str, dict[str, Any]],
    balanced_submetrics: dict[
        str,
        dict[str, float],
    ],
) -> None:
    invalid_state = deepcopy(
        balanced_submetrics
    )
    invalid_state.pop("SVC")

    with pytest.raises(
        AgentStateError,
        match="missing=SVC",
    ):
        OrganizationAgent(
            model=DummyModel(),
            uid=2,
            org_type="regulated_private_entity",
            exchange_volume=50,
            submetrics_by_dimension=invalid_state,
            formula_config=formula_config,
        )


def test_unknown_submetric_is_rejected(
    agent: OrganizationAgent,
) -> None:
    with pytest.raises(
        AgentStateError,
        match="Unknown submetric",
    ):
        agent.get_submetric(
            "D3",
            "unknown_metric",
        )


def test_invalid_exchange_volume_is_rejected(
    formula_config: dict[str, dict[str, Any]],
    balanced_submetrics: dict[
        str,
        dict[str, float],
    ],
) -> None:
    with pytest.raises(
        AgentStateError,
        match="exchange_volume",
    ):
        OrganizationAgent(
            model=DummyModel(),
            uid=3,
            org_type="government_ministry",
            exchange_volume=0,
            submetrics_by_dimension=balanced_submetrics,
            formula_config=formula_config,
        )


def test_snapshot_is_independent_copy(
    agent: OrganizationAgent,
) -> None:
    snapshot = agent.snapshot()

    snapshot["submetrics"]["D1"][
        "completeness_ratio"
    ] = 0.0

    snapshot["dimensions"]["D1"] = 0.0

    assert agent.get_submetric(
        "D1",
        "completeness_ratio",
    ) != pytest.approx(0.0)

    assert agent.get_dimension_score(
        "D1"
    ) != pytest.approx(0.0)


def test_step_delegates_to_model(
    formula_config: dict[str, dict[str, Any]],
    balanced_submetrics: dict[
        str,
        dict[str, float],
    ],
) -> None:
    model = DummyModel()

    delegated_agent = OrganizationAgent(
        model=model,
        uid=10,
        org_type="state_enterprise",
        exchange_volume=100,
        submetrics_by_dimension=balanced_submetrics,
        formula_config=formula_config,
    )

    delegated_agent.step()

    assert model.executed_agent_uids == [10]


def test_step_rejects_model_without_executor(
    formula_config: dict[str, dict[str, Any]],
    balanced_submetrics: dict[
        str,
        dict[str, float],
    ],
) -> None:
    model = mesa.Model(rng=42)

    invalid_agent = OrganizationAgent(
        model=model,
        uid=11,
        org_type="state_enterprise",
        exchange_volume=100,
        submetrics_by_dimension=balanced_submetrics,
        formula_config=formula_config,
    )

    with pytest.raises(
        AgentStateError,
        match="execute_agent_step",
    ):
        invalid_agent.step()