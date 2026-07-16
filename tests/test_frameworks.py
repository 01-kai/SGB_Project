from __future__ import annotations

from copy import deepcopy
from typing import Any

import pandas as pd
import pytest

from sgb.config import (
    BASE_CONFIG_PATH,
    load_config,
)
from sgb.frameworks import (
    FrameworkConfigurationError,
    FrameworkDefinition,
    FrameworkEvaluationError,
    aggregate_agent_scores,
    build_framework_definitions,
    evaluate_agents,
    evaluate_snapshots,
    evaluate_trajectory,
    rank_framework_scores,
    score_all_frameworks,
)
from sgb.model import SGBModel


@pytest.fixture
def base_config() -> dict[str, Any]:
    config = deepcopy(
        load_config(
            BASE_CONFIG_PATH
        )
    )
    config.pop(
        "_metadata",
        None,
    )

    return config


@pytest.fixture
def definitions(
    base_config: dict[str, Any],
) -> dict[
    str,
    FrameworkDefinition,
]:
    return build_framework_definitions(
        base_config
    )


@pytest.fixture
def sample_dimensions() -> dict[str, float]:
    return {
        "D1": 0.40,
        "D2": 0.50,
        "D3": 0.60,
        "D4": 0.70,
        "D5": 0.80,
        "SVC": 0.90,
    }


@pytest.fixture
def small_model(
    base_config: dict[str, Any],
) -> SGBModel:
    base_config[
        "simulation"
    ]["population_size"] = 10
    base_config[
        "simulation"
    ]["max_steps"] = 5
    base_config[
        "network"
    ]["attachment_m"] = 2
    base_config[
        "exchange"
    ]["maximum_volume"] = 100

    return SGBModel(
        config=base_config,
        seed=42,
    )


def test_framework_definitions_load(
    definitions: dict[
        str,
        FrameworkDefinition,
    ],
) -> None:
    assert set(definitions) == {
        "DBL",
        "DMM",
        "ISF",
    }

    assert set(
        definitions[
            "DBL"
        ].weights
    ) == {
        "D1",
        "D2",
        "D3",
        "D4",
        "D5",
    }

    assert set(
        definitions[
            "ISF"
        ].weights
    ) == {
        "D1",
        "D2",
        "D3",
        "D4",
        "D5",
        "SVC",
    }


def test_known_framework_scores(
    definitions: dict[
        str,
        FrameworkDefinition,
    ],
    sample_dimensions: dict[
        str,
        float,
    ],
) -> None:
    scores = score_all_frameworks(
        dimensions=sample_dimensions,
        definitions=definitions,
    )

    assert scores["DBL"] == pytest.approx(
        0.580
    )
    assert scores["DMM"] == pytest.approx(
        0.582
    )
    assert scores["ISF"] == pytest.approx(
        0.610
    )


def test_dbl_and_dmm_ignore_svc(
    definitions: dict[
        str,
        FrameworkDefinition,
    ],
    sample_dimensions: dict[
        str,
        float,
    ],
) -> None:
    first_dimensions = deepcopy(
        sample_dimensions
    )
    second_dimensions = deepcopy(
        sample_dimensions
    )

    first_dimensions["SVC"] = 0.0
    second_dimensions["SVC"] = 1.0

    first_scores = (
        score_all_frameworks(
            first_dimensions,
            definitions,
        )
    )
    second_scores = (
        score_all_frameworks(
            second_dimensions,
            definitions,
        )
    )

    assert first_scores[
        "DBL"
    ] == pytest.approx(
        second_scores["DBL"]
    )

    assert first_scores[
        "DMM"
    ] == pytest.approx(
        second_scores["DMM"]
    )

    assert second_scores[
        "ISF"
    ] > first_scores["ISF"]


def test_framework_weights_are_immutable(
    definitions: dict[
        str,
        FrameworkDefinition,
    ],
) -> None:
    with pytest.raises(
        TypeError
    ):
        definitions[
            "DBL"
        ].weights["D1"] = 1.0


def test_missing_dimension_is_rejected(
    definitions: dict[
        str,
        FrameworkDefinition,
    ],
    sample_dimensions: dict[
        str,
        float,
    ],
) -> None:
    invalid_dimensions = deepcopy(
        sample_dimensions
    )
    invalid_dimensions.pop("D5")

    with pytest.raises(
        FrameworkEvaluationError,
        match="Missing dimensions",
    ):
        score_all_frameworks(
            invalid_dimensions,
            definitions,
        )


def test_out_of_range_dimension_is_rejected(
    definitions: dict[
        str,
        FrameworkDefinition,
    ],
    sample_dimensions: dict[
        str,
        float,
    ],
) -> None:
    invalid_dimensions = deepcopy(
        sample_dimensions
    )
    invalid_dimensions["D1"] = 1.20

    with pytest.raises(
        FrameworkEvaluationError,
        match=r"\[0, 1\]",
    ):
        score_all_frameworks(
            invalid_dimensions,
            definitions,
        )


def test_invalid_framework_dimensions_are_rejected() -> None:
    with pytest.raises(
        FrameworkConfigurationError,
        match="Invalid dimensions",
    ):
        FrameworkDefinition(
            name="DBL",
            label="Invalid DBL",
            weights={
                "D1": 0.25,
                "D2": 0.25,
                "D3": 0.25,
                "D4": 0.25,
            },
        )


def test_evaluate_agents_returns_one_row_per_agent(
    small_model: SGBModel,
    definitions: dict[
        str,
        FrameworkDefinition,
    ],
) -> None:
    scored_agents = evaluate_agents(
        agents=(
            small_model.agents.to_list()
        ),
        definitions=definitions,
    )

    assert len(scored_agents) == 10

    assert {
        "GMI_DBL",
        "GMI_DMM",
        "GMI_ISF",
    }.issubset(
        scored_agents.columns
    )

    assert scored_agents[
        "uid"
    ].tolist() == list(
        range(10)
    )

    for column in {
        "GMI_DBL",
        "GMI_DMM",
        "GMI_ISF",
    }:
        assert scored_agents[
            column
        ].between(
            0.0,
            1.0,
        ).all()


def test_agent_evaluation_does_not_modify_model(
    small_model: SGBModel,
    definitions: dict[
        str,
        FrameworkDefinition,
    ],
) -> None:
    before = (
        small_model.agent_snapshots()
    )

    evaluate_agents(
        agents=(
            small_model.agents.to_list()
        ),
        definitions=definitions,
    )

    after = (
        small_model.agent_snapshots()
    )

    assert after == before


def test_duplicate_snapshot_uid_is_rejected(
    small_model: SGBModel,
    definitions: dict[
        str,
        FrameworkDefinition,
    ],
) -> None:
    snapshot = (
        small_model.get_agent(
            0
        ).snapshot()
    )

    with pytest.raises(
        FrameworkEvaluationError,
        match="Duplicate snapshot uid",
    ):
        evaluate_snapshots(
            snapshots=[
                snapshot,
                deepcopy(snapshot),
            ],
            definitions=definitions,
        )


def test_aggregate_agent_scores(
    small_model: SGBModel,
    definitions: dict[
        str,
        FrameworkDefinition,
    ],
) -> None:
    scored_agents = evaluate_agents(
        agents=(
            small_model.agents.to_list()
        ),
        definitions=definitions,
    )

    summary = aggregate_agent_scores(
        scored_agents=(
            scored_agents
        ),
        definitions=definitions,
    )

    assert set(
        summary["framework"]
    ) == {
        "DBL",
        "DMM",
        "ISF",
    }

    assert summary[
        "E_GMI"
    ].between(
        0.0,
        1.0,
    ).all()

    assert set(
        summary[
            "organization_count"
        ]
    ) == {10}

    for _, row in summary.iterrows():
        expected_mean = float(
            scored_agents[
                f"GMI_{row['framework']}"
            ].mean()
        )

        assert row[
            "E_GMI"
        ] == pytest.approx(
            expected_mean
        )


def test_rank_framework_scores_uses_dense_ranking() -> None:
    ranking = rank_framework_scores(
        {
            "DBL": 0.60,
            "DMM": 0.60,
            "ISF": 0.50,
        }
    )

    assert ranking == [
        {
            "rank": 1,
            "framework": "DBL",
            "score": 0.60,
        },
        {
            "rank": 1,
            "framework": "DMM",
            "score": 0.60,
        },
        {
            "rank": 2,
            "framework": "ISF",
            "score": 0.50,
        },
    ]


def test_evaluate_trajectory_scores_same_rows(
    small_model: SGBModel,
    definitions: dict[
        str,
        FrameworkDefinition,
    ],
) -> None:
    small_model.run(
        steps=3
    )

    operational_results = (
        small_model.get_results()
    )

    scored_results = (
        evaluate_trajectory(
            operational_results=(
                operational_results
            ),
            definitions=definitions,
        )
    )

    assert len(
        scored_results
    ) == len(
        operational_results
    )

    assert {
        "E_GMI_DBL",
        "E_GMI_DMM",
        "E_GMI_ISF",
    }.issubset(
        scored_results.columns
    )

    assert scored_results[
        "step"
    ].tolist() == operational_results[
        "step"
    ].tolist()

    for column in {
        "E_GMI_DBL",
        "E_GMI_DMM",
        "E_GMI_ISF",
    }:
        assert scored_results[
            column
        ].between(
            0.0,
            1.0,
        ).all()


def test_trajectory_evaluation_does_not_modify_input(
    small_model: SGBModel,
    definitions: dict[
        str,
        FrameworkDefinition,
    ],
) -> None:
    small_model.run(
        steps=2
    )

    operational_results = (
        small_model.get_results()
    )

    original_copy = (
        operational_results.copy(
            deep=True
        )
    )

    evaluate_trajectory(
        operational_results=(
            operational_results
        ),
        definitions=definitions,
    )

    pd.testing.assert_frame_equal(
        operational_results,
        original_copy,
    )


def test_framework_weight_changes_only_scores(
    base_config: dict[str, Any],
) -> None:
    first_config = deepcopy(
        base_config
    )
    second_config = deepcopy(
        base_config
    )

    first_config[
        "simulation"
    ]["population_size"] = 10
    first_config[
        "simulation"
    ]["max_steps"] = 3
    first_config[
        "network"
    ]["attachment_m"] = 2

    second_config[
        "simulation"
    ]["population_size"] = 10
    second_config[
        "simulation"
    ]["max_steps"] = 3
    second_config[
        "network"
    ]["attachment_m"] = 2

    second_config[
        "frameworks"
    ]["DBL"]["weights"] = {
        "D1": 1.0,
        "D2": 0.0,
        "D3": 0.0,
        "D4": 0.0,
        "D5": 0.0,
    }

    first_model = SGBModel(
        config=first_config,
        seed=123,
    )
    second_model = SGBModel(
        config=second_config,
        seed=123,
    )

    first_model.run(
        steps=3
    )
    second_model.run(
        steps=3
    )

    pd.testing.assert_frame_equal(
        first_model.get_results(),
        second_model.get_results(),
    )

    pd.testing.assert_frame_equal(
        first_model.get_events(),
        second_model.get_events(),
    )

    first_definitions = (
        build_framework_definitions(
            first_config
        )
    )
    second_definitions = (
        build_framework_definitions(
            second_config
        )
    )

    first_scores = evaluate_agents(
        first_model.agents.to_list(),
        first_definitions,
    )

    second_scores = evaluate_agents(
        second_model.agents.to_list(),
        second_definitions,
    )

    assert not first_scores[
        "GMI_DBL"
    ].equals(
        second_scores[
            "GMI_DBL"
        ]
    )

    for column in [
        "uid",
        "org_type",
        "exchange_volume",
        "D1",
        "D2",
        "D3",
        "D4",
        "D5",
        "SVC",
    ]:
        pd.testing.assert_series_equal(
            first_scores[column],
            second_scores[column],
        )