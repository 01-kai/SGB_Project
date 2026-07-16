from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

import networkx as nx
import pytest

from sgb.config import (
    BASE_CONFIG_PATH,
    load_config,
)
from sgb.model import SGBModel
from sgb.perturbations import (
    PerturbationError,
    apply_data_quality_shock,
    apply_interoperability_disruption,
    apply_metadata_outage,
    apply_regulatory_change,
    apply_security_incident,
    apply_sovereignty_policy_shock,
)


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
    ]["population_size"] = 15
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
def model(
    small_config: dict[str, Any],
) -> SGBModel:
    return SGBModel(
        config=small_config,
        seed=42,
    )


def test_data_quality_shock_affects_all_agents(
    model: SGBModel,
) -> None:
    before = {
        uid: (
            model.get_agent(uid)
            .get_dimension_score("D1")
        )
        for uid in model.agents_by_uid
    }

    records = (
        apply_data_quality_shock(
            model=model,
            impact=0.10,
        )
    )

    assert len(records) == len(
        model.agents_by_uid
    )

    for uid in model.agents_by_uid:
        after = (
            model.get_agent(uid)
            .get_dimension_score("D1")
        )

        assert after < before[uid]


def test_subset_targets_are_deduplicated(
    model: SGBModel,
) -> None:
    before = {
        uid: model.get_agent(
            uid
        ).dimensions
        for uid in model.agents_by_uid
    }

    records = (
        apply_data_quality_shock(
            model=model,
            impact=0.10,
            target_uids=[
                0,
                1,
                1,
                0,
            ],
        )
    )

    affected_uids = {
        record["affected_uid"]
        for record in records
    }

    assert affected_uids == {
        0,
        1,
    }

    assert len(records) == 2

    assert (
        model.get_agent(0)
        .get_dimension_score("D1")
        < before[0]["D1"]
    )

    assert (
        model.get_agent(2)
        .get_dimension_score("D1")
        == pytest.approx(
            before[2]["D1"]
        )
    )


@pytest.mark.parametrize(
    (
        "shock_function",
        "dimension",
    ),
    [
        (
            apply_metadata_outage,
            "D3",
        ),
        (
            apply_regulatory_change,
            "D4",
        ),
        (
            apply_interoperability_disruption,
            "D5",
        ),
        (
            apply_sovereignty_policy_shock,
            "SVC",
        ),
    ],
)
def test_named_uniform_shocks_reduce_target_dimension(
    model: SGBModel,
    shock_function: Callable[..., list[dict[str, Any]]],
    dimension: str,
) -> None:
    agent = model.get_agent(0)

    before = (
        agent.get_dimension_score(
            dimension
        )
    )

    records = shock_function(
        model=model,
        impact=0.10,
        target_uids=[0],
    )

    after = (
        agent.get_dimension_score(
            dimension
        )
    )

    assert len(records) == 1
    assert after < before


def test_regulatory_change_does_not_modify_svc(
    model: SGBModel,
) -> None:
    svc_before = {
        uid: (
            model.get_agent(uid)
            .get_dimension_score("SVC")
        )
        for uid in model.agents_by_uid
    }

    apply_regulatory_change(
        model=model,
        impact=0.10,
    )

    for uid in model.agents_by_uid:
        after = (
            model.get_agent(uid)
            .get_dimension_score("SVC")
        )

        assert after == pytest.approx(
            svc_before[uid]
        )


def test_security_incident_decays_with_distance(
    model: SGBModel,
) -> None:
    target_uid = max(
        model.graph.degree,
        key=lambda item: item[1],
    )[0]

    records = apply_security_incident(
        model=model,
        target_uid=target_uid,
        impact=0.30,
        max_distance=2,
        decay=0.50,
    )

    assert records

    for record in records:
        distance = record[
            "network_distance"
        ]

        expected_impact = (
            0.30
            * (0.50 ** distance)
        )

        assert (
            record["applied_impact"]
            == pytest.approx(
                expected_impact
            )
        )


def test_security_incident_uses_network_cutoff(
    model: SGBModel,
) -> None:
    target_uid = 0

    records = apply_security_incident(
        model=model,
        target_uid=target_uid,
        impact=0.20,
        max_distance=1,
        decay=0.50,
    )

    expected_nodes = set(
        nx.single_source_shortest_path_length(
            model.graph,
            target_uid,
            cutoff=1,
        )
    )

    affected_nodes = {
        record["affected_uid"]
        for record in records
    }

    assert affected_nodes == expected_nodes


def test_security_incident_reduces_target_d2(
    model: SGBModel,
) -> None:
    target_uid = 0

    before = (
        model.get_agent(target_uid)
        .get_dimension_score("D2")
    )

    records = apply_security_incident(
        model=model,
        target_uid=target_uid,
        impact=0.20,
        max_distance=0,
        decay=0.50,
    )

    after = (
        model.get_agent(target_uid)
        .get_dimension_score("D2")
    )

    assert len(records) == 1
    assert after < before


def test_perturbation_values_remain_bounded(
    model: SGBModel,
) -> None:
    apply_data_quality_shock(
        model=model,
        impact=1.0,
    )

    apply_metadata_outage(
        model=model,
        impact=1.0,
    )

    apply_regulatory_change(
        model=model,
        impact=1.0,
    )

    apply_interoperability_disruption(
        model=model,
        impact=1.0,
    )

    apply_sovereignty_policy_shock(
        model=model,
        impact=1.0,
    )

    apply_security_incident(
        model=model,
        target_uid=0,
        impact=1.0,
        max_distance=3,
        decay=1.0,
    )

    for snapshot in (
        model.agent_snapshots()
    ):
        for dimension_state in (
            snapshot[
                "submetrics"
            ].values()
        ):
            for value in (
                dimension_state.values()
            ):
                assert (
                    0.0
                    <= value
                    <= 1.0
                )


@pytest.mark.parametrize(
    "impact",
    [
        -0.10,
        1.10,
    ],
)
def test_invalid_impact_is_rejected(
    model: SGBModel,
    impact: float,
) -> None:
    with pytest.raises(
        PerturbationError,
        match=r"\[0, 1\]",
    ):
        apply_data_quality_shock(
            model=model,
            impact=impact,
        )


def test_invalid_decay_is_rejected(
    model: SGBModel,
) -> None:
    with pytest.raises(
        PerturbationError,
        match=r"\(0, 1\]",
    ):
        apply_security_incident(
            model=model,
            target_uid=0,
            impact=0.20,
            max_distance=2,
            decay=0.0,
        )


def test_unknown_target_is_rejected(
    model: SGBModel,
) -> None:
    with pytest.raises(
        PerturbationError,
        match="Unknown organization",
    ):
        apply_data_quality_shock(
            model=model,
            impact=0.10,
            target_uids=[999],
        )


def test_framework_weights_do_not_change_shock_results(
    small_config: dict[str, Any],
) -> None:
    first_config = deepcopy(
        small_config
    )
    second_config = deepcopy(
        small_config
    )

    second_config[
        "frameworks"
    ]["ISF"]["weights"] = {
        "D1": 0.0,
        "D2": 0.0,
        "D3": 0.0,
        "D4": 0.0,
        "D5": 0.0,
        "SVC": 1.0,
    }

    first_model = SGBModel(
        config=first_config,
        seed=123,
    )

    second_model = SGBModel(
        config=second_config,
        seed=123,
    )

    first_records = (
        apply_security_incident(
            model=first_model,
            target_uid=0,
            impact=0.20,
            max_distance=2,
            decay=0.50,
        )
    )

    second_records = (
        apply_security_incident(
            model=second_model,
            target_uid=0,
            impact=0.20,
            max_distance=2,
            decay=0.50,
        )
    )

    assert (
        first_records
        == second_records
    )

    assert (
        first_model.agent_snapshots()
        == second_model.agent_snapshots()
    )