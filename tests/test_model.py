from __future__ import annotations

from copy import deepcopy
from typing import Any

import networkx as nx
import pandas as pd
import pytest

from sgb.config import (
    BASE_CONFIG_PATH,
    load_config,
)
from sgb.model import (
    FAILURE_REASONS,
    ModelStateError,
    SGBModel,
)


@pytest.fixture
def small_config() -> dict[str, Any]:
    config = load_config(BASE_CONFIG_PATH)

    config = deepcopy(config)
    config.pop("_metadata", None)

    config["simulation"]["population_size"] = 12
    config["simulation"]["max_steps"] = 5
    config["simulation"]["seed_count"] = 2

    config["network"]["attachment_m"] = 2

    config["exchange"]["minimum_volume"] = 10
    config["exchange"]["maximum_volume"] = 200

    return config


class RecordingDynamics:
    def __init__(self) -> None:
        self.outcome_count = 0
        self.recovery_count = 0

    def apply_exchange_outcome(
        self,
        model: SGBModel,
        sender: Any,
        receiver: Any,
        event: dict[str, Any],
    ) -> None:
        self.outcome_count += 1

    def recover_agent(
        self,
        model: SGBModel,
        agent: Any,
    ) -> None:
        self.recovery_count += 1


def test_model_creates_expected_agents_and_network(
    small_config: dict[str, Any],
) -> None:
    model = SGBModel(
        config=small_config,
        seed=42,
    )

    assert len(model.agents) == 12
    assert len(model.agents_by_uid) == 12

    assert model.graph.number_of_nodes() == 12
    assert nx.is_connected(model.graph)


def test_agents_have_complete_submetric_state(
    small_config: dict[str, Any],
) -> None:
    model = SGBModel(
        config=small_config,
        seed=42,
    )

    for snapshot in model.agent_snapshots():
        assert set(snapshot["dimensions"]) == {
            "D1",
            "D2",
            "D3",
            "D4",
            "D5",
            "SVC",
        }

        for value in snapshot[
            "dimensions"
        ].values():
            assert 0.0 <= value <= 1.0


def test_exchange_volumes_respect_bounds(
    small_config: dict[str, Any],
) -> None:
    model = SGBModel(
        config=small_config,
        seed=42,
    )

    volumes = [
        agent.exchange_volume
        for agent in model.agents.to_list()
    ]

    assert min(volumes) >= 10
    assert max(volumes) <= 200


def test_model_is_reproducible_for_same_seed(
    small_config: dict[str, Any],
) -> None:
    first = SGBModel(
        config=small_config,
        seed=123,
    )
    second = SGBModel(
        config=small_config,
        seed=123,
    )

    assert (
        first.agent_snapshots()
        == second.agent_snapshots()
    )

    assert list(first.graph.edges()) == list(
        second.graph.edges()
    )

    first.run(steps=3)
    second.run(steps=3)

    pd.testing.assert_frame_equal(
        first.get_results(),
        second.get_results(),
    )

    pd.testing.assert_frame_equal(
        first.get_events(),
        second.get_events(),
    )


def test_different_seeds_change_population(
    small_config: dict[str, Any],
) -> None:
    first = SGBModel(
        config=small_config,
        seed=1,
    )
    second = SGBModel(
        config=small_config,
        seed=2,
    )

    assert (
        first.agent_snapshots()
        != second.agent_snapshots()
    )


def test_framework_weights_do_not_change_trajectory(
    small_config: dict[str, Any],
) -> None:
    first_config = deepcopy(small_config)
    second_config = deepcopy(small_config)

    second_config["frameworks"]["DBL"]["weights"] = {
        "D1": 0.00,
        "D2": 0.00,
        "D3": 0.00,
        "D4": 0.00,
        "D5": 1.00,
    }

    second_config["frameworks"]["DMM"]["weights"] = {
        "D1": 1.00,
        "D2": 0.00,
        "D3": 0.00,
        "D4": 0.00,
        "D5": 0.00,
    }

    first = SGBModel(
        config=first_config,
        seed=99,
    )
    second = SGBModel(
        config=second_config,
        seed=99,
    )

    first.run(steps=3)
    second.run(steps=3)

    pd.testing.assert_frame_equal(
        first.get_results(),
        second.get_results(),
    )

    pd.testing.assert_frame_equal(
        first.get_events(),
        second.get_events(),
    )


def test_one_event_is_initiated_per_agent(
    small_config: dict[str, Any],
) -> None:
    model = SGBModel(
        config=small_config,
        seed=42,
    )

    model.step()

    events = model.get_events()

    assert len(events) == 12
    assert set(events["step"]) == {0}


def test_event_records_have_valid_values(
    small_config: dict[str, Any],
) -> None:
    model = SGBModel(
        config=small_config,
        seed=42,
    )

    model.run(steps=3)

    events = model.get_events()

    assert events[
        "success_probability"
    ].between(0.0, 1.0).all()

    assert events["volume"].between(
        10,
        200,
    ).all()

    failure_rows = events[
        ~events["success"]
    ]

    assert set(
        failure_rows["failure_reason"]
    ).issubset(
        set(FAILURE_REASONS)
    )


def test_failure_probabilities_sum_to_one(
    small_config: dict[str, Any],
) -> None:
    model = SGBModel(
        config=small_config,
        seed=42,
    )

    sender = model.get_agent(0)
    receiver = model.select_partner(sender)

    assert receiver is not None

    probabilities = (
        model.failure_reason_probabilities(
            sender,
            receiver,
        )
    )

    assert set(probabilities) == set(
        FAILURE_REASONS
    )

    assert sum(
        probabilities.values()
    ) == pytest.approx(1.0)


def test_operational_readiness_excludes_svc(
    small_config: dict[str, Any],
) -> None:
    model = SGBModel(
        config=small_config,
        seed=42,
    )

    agent = model.get_agent(0)

    readiness_before = (
        model.operational_readiness(agent)
    )

    agent.set_submetric(
        "SVC",
        "localization_compliance_rate",
        1.0,
    )
    agent.set_submetric(
        "SVC",
        "cross_border_flagging_rate",
        1.0,
    )
    agent.set_submetric(
        "SVC",
        "audit_trail_completeness",
        1.0,
    )

    readiness_after = (
        model.operational_readiness(agent)
    )

    assert readiness_after == pytest.approx(
        readiness_before
    )


def test_step_collects_consistent_metrics(
    small_config: dict[str, Any],
) -> None:
    model = SGBModel(
        config=small_config,
        seed=42,
    )

    model.step()

    results = model.get_results()

    assert len(results) == 1

    row = results.iloc[0]

    assert row["total_events"] == 12

    assert (
        row["success_rate"]
        == pytest.approx(
            row["successful_events"]
            / row["total_events"]
        )
    )

    assert (
        row["volume_success_rate"]
        == pytest.approx(
            row["successful_volume"]
            / row["total_volume"]
        )
    )


def test_dynamics_hooks_are_called(
    small_config: dict[str, Any],
) -> None:
    dynamics = RecordingDynamics()

    model = SGBModel(
        config=small_config,
        seed=42,
        dynamics=dynamics,
    )

    model.step()

    assert dynamics.outcome_count == 12
    assert dynamics.recovery_count == 12


def test_run_cannot_exceed_max_steps(
    small_config: dict[str, Any],
) -> None:
    model = SGBModel(
        config=small_config,
        seed=42,
    )

    with pytest.raises(
        ModelStateError,
        match="exceed",
    ):
        model.run(steps=6)


def test_network_summary_is_consistent(
    small_config: dict[str, Any],
) -> None:
    model = SGBModel(
        config=small_config,
        seed=42,
    )

    summary = model.network_summary()

    assert summary["node_count"] == 12
    assert summary["edge_count"] == (
        model.graph.number_of_edges()
    )
    assert summary["connected"] is True
    assert summary["average_degree"] > 0.0
    assert 0.0 < summary["density"] <= 1.0


def test_model_does_not_select_a_framework(
    small_config: dict[str, Any],
) -> None:
    model = SGBModel(
        config=small_config,
        seed=42,
    )

    assert not hasattr(
        model,
        "framework_name",
    )