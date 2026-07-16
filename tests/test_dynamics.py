from __future__ import annotations

from copy import deepcopy
from typing import Any

import pandas as pd
import pytest

from sgb.config import (
    BASE_CONFIG_PATH,
    load_config,
)
from sgb.dynamics import (
    DYNAMICS_CONFIG_PATH,
    DynamicsConfigurationError,
    DynamicsStateError,
    NeutralOperationalDynamics,
    load_dynamics_config,
    validate_dynamics_config,
)
from sgb.model import SGBModel


@pytest.fixture
def small_config() -> dict[str, Any]:
    config = deepcopy(
        load_config(
            BASE_CONFIG_PATH
        )
    )
    config.pop(
        "_metadata",
        None,
    )

    config[
        "simulation"
    ]["population_size"] = 10
    config[
        "simulation"
    ]["max_steps"] = 10
    config[
        "network"
    ]["attachment_m"] = 2
    config[
        "exchange"
    ]["maximum_volume"] = 100

    return config


@pytest.fixture
def dynamics_config() -> dict[str, Any]:
    config = deepcopy(
        load_dynamics_config(
            DYNAMICS_CONFIG_PATH
        )
    )
    config.pop(
        "_metadata",
        None,
    )

    return config


@pytest.fixture
def model(
    small_config: dict[str, Any],
    dynamics_config: dict[str, Any],
) -> SGBModel:
    dynamics = NeutralOperationalDynamics(
        base_config=small_config,
        dynamics_config=dynamics_config,
    )

    return SGBModel(
        config=small_config,
        seed=42,
        dynamics=dynamics,
    )


def test_dynamics_configuration_loads(
    dynamics_config: dict[str, Any],
) -> None:
    assert (
        dynamics_config[
            "config_kind"
        ]
        == "dynamics"
    )

    assert (
        dynamics_config[
            "configuration"
        ]["status"]
        == "candidate"
    )


def test_success_changes_both_participants(
    model: SGBModel,
) -> None:
    sender = model.get_agent(0)
    receiver = model.get_agent(1)

    sender_completeness_before = (
        sender.get_submetric(
            "D1",
            "completeness_ratio",
        )
    )
    receiver_completeness_before = (
        receiver.get_submetric(
            "D1",
            "completeness_ratio",
        )
    )

    sender_duplicate_before = (
        sender.get_submetric(
            "D1",
            "duplicate_rate",
        )
    )
    receiver_duplicate_before = (
        receiver.get_submetric(
            "D1",
            "duplicate_rate",
        )
    )

    model.dynamics.apply_exchange_outcome(
        model=model,
        sender=sender,
        receiver=receiver,
        event={
            "success": True,
            "failure_reason": None,
        },
    )

    assert sender.get_submetric(
        "D1",
        "completeness_ratio",
    ) > sender_completeness_before

    assert receiver.get_submetric(
        "D1",
        "completeness_ratio",
    ) > receiver_completeness_before

    assert sender.get_submetric(
        "D1",
        "duplicate_rate",
    ) < sender_duplicate_before

    assert receiver.get_submetric(
        "D1",
        "duplicate_rate",
    ) < receiver_duplicate_before


def test_security_failure_changes_only_d2(
    model: SGBModel,
) -> None:
    sender = model.get_agent(0)
    receiver = model.get_agent(1)

    dimensions_before = (
        sender.dimensions
    )

    model.dynamics.apply_exchange_outcome(
        model=model,
        sender=sender,
        receiver=receiver,
        event={
            "success": False,
            "failure_reason": (
                "security"
            ),
        },
    )

    dimensions_after = (
        sender.dimensions
    )

    assert (
        dimensions_after["D2"]
        < dimensions_before["D2"]
    )

    for dimension in {
        "D1",
        "D3",
        "D4",
        "D5",
        "SVC",
    }:
        assert (
            dimensions_after[
                dimension
            ]
            == pytest.approx(
                dimensions_before[
                    dimension
                ]
            )
        )


def test_recovery_moves_positive_metric_toward_baseline(
    model: SGBModel,
) -> None:
    agent = model.get_agent(0)

    agent.set_submetric(
        "D2",
        "encryption_coverage",
        0.0,
    )

    before = agent.get_submetric(
        "D2",
        "encryption_coverage",
    )

    model.dynamics.recover_agent(
        model=model,
        agent=agent,
    )

    after = agent.get_submetric(
        "D2",
        "encryption_coverage",
    )

    assert after > before

    assert after <= (
        model.dynamics.baseline_targets[
            "encryption_coverage"
        ]
    )


def test_recovery_moves_inverse_metric_toward_baseline(
    model: SGBModel,
) -> None:
    agent = model.get_agent(0)

    agent.set_submetric(
        "D2",
        "policy_violation_rate",
        1.0,
    )

    before = agent.get_submetric(
        "D2",
        "policy_violation_rate",
    )

    model.dynamics.recover_agent(
        model=model,
        agent=agent,
    )

    after = agent.get_submetric(
        "D2",
        "policy_violation_rate",
    )

    assert after < before

    assert after >= (
        model.dynamics.baseline_targets[
            "policy_violation_rate"
        ]
    )


def test_recovery_adjustment_respects_maximum(
    model: SGBModel,
) -> None:
    agent = model.get_agent(0)

    agent.set_submetric(
        "D1",
        "completeness_ratio",
        0.0,
    )

    before = agent.get_submetric(
        "D1",
        "completeness_ratio",
    )

    model.dynamics.recover_agent(
        model=model,
        agent=agent,
    )

    after = agent.get_submetric(
        "D1",
        "completeness_ratio",
    )

    actual_adjustment = (
        after - before
    )

    assert actual_adjustment <= (
        model.dynamics
        .maximum_recovery_adjustment
        + 1e-12
    )


def test_invalid_failure_reason_is_rejected(
    model: SGBModel,
) -> None:
    with pytest.raises(
        DynamicsStateError,
        match="failure reason",
    ):
        model.dynamics.apply_exchange_outcome(
            model=model,
            sender=model.get_agent(0),
            receiver=model.get_agent(1),
            event={
                "success": False,
                "failure_reason": (
                    "unknown"
                ),
            },
        )


def test_framework_weights_do_not_change_dynamics(
    small_config: dict[str, Any],
    dynamics_config: dict[str, Any],
) -> None:
    first_config = deepcopy(
        small_config
    )
    second_config = deepcopy(
        small_config
    )

    second_config[
        "frameworks"
    ]["DBL"]["weights"] = {
        "D1": 1.0,
        "D2": 0.0,
        "D3": 0.0,
        "D4": 0.0,
        "D5": 0.0,
    }

    second_config[
        "frameworks"
    ]["DMM"]["weights"] = {
        "D1": 0.0,
        "D2": 0.0,
        "D3": 0.0,
        "D4": 0.0,
        "D5": 1.0,
    }

    first_dynamics = (
        NeutralOperationalDynamics(
            base_config=first_config,
            dynamics_config=(
                dynamics_config
            ),
        )
    )

    second_dynamics = (
        NeutralOperationalDynamics(
            base_config=second_config,
            dynamics_config=(
                dynamics_config
            ),
        )
    )

    first_model = SGBModel(
        config=first_config,
        seed=99,
        dynamics=first_dynamics,
    )

    second_model = SGBModel(
        config=second_config,
        seed=99,
        dynamics=second_dynamics,
    )

    first_model.run(
        steps=5
    )
    second_model.run(
        steps=5
    )

    assert (
        first_model.agent_snapshots()
        == second_model.agent_snapshots()
    )

    pd.testing.assert_frame_equal(
        first_model.get_results(),
        second_model.get_results(),
    )

    pd.testing.assert_frame_equal(
        first_model.get_events(),
        second_model.get_events(),
    )


def test_model_state_changes_with_dynamics(
    model: SGBModel,
) -> None:
    before = model.agent_snapshots()

    model.run(
        steps=5
    )

    after = model.agent_snapshots()

    assert before != after


def test_invalid_success_submetric_is_rejected(
    dynamics_config: dict[str, Any],
) -> None:
    invalid_config = deepcopy(
        dynamics_config
    )

    invalid_config[
        "success"
    ]["changes"]["D1"][
        "unknown_submetric"
    ] = 1.0

    with pytest.raises(
        DynamicsConfigurationError,
        match="Unknown submetric",
    ):
        validate_dynamics_config(
            invalid_config
        )