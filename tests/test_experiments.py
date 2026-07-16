from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from sgb.config import (
    BASE_CONFIG_PATH,
    load_config,
)
from sgb.dynamics import load_dynamics_config
from sgb.experiments import (
    ExperimentResult,
    classify_maturity,
    load_experiment_config,
    run_experiment_1,
    run_experiment_2,
    run_experiment_3,
    run_experiment_4,
    save_experiment_result,
)


@pytest.fixture(scope="module")
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


@pytest.fixture(scope="module")
def dynamics_config() -> dict[str, Any]:
    config = deepcopy(
        load_dynamics_config()
    )

    config.pop(
        "_metadata",
        None,
    )

    return config


@pytest.mark.parametrize(
    (
        "path",
        "experiment_id",
    ),
    [
        (
            "configs/experiments/experiment_1.yaml",
            1,
        ),
        (
            "configs/experiments/experiment_2.yaml",
            2,
        ),
        (
            "configs/experiments/experiment_3.yaml",
            3,
        ),
        (
            "configs/experiments/experiment_4.yaml",
            4,
        ),
    ],
)
def test_experiment_configuration_loads(
    path: str,
    experiment_id: int,
) -> None:
    config = load_experiment_config(
        path
    )

    assert config[
        "experiment_id"
    ] == experiment_id


def relaxed_steady_state(
    config: dict[str, Any],
) -> None:
    config[
        "steady_state"
    ] = {
        "maximum_steps": 60,
        "rolling_window": 10,
        "required_stable_windows": 2,
        "minimum_step": 30,
        "slope_tolerance": 1.0,
        "mean_difference_tolerance": 1.0,
        "stabilization_buffer": 0,
        "metrics": [
            "mean_operational_readiness",
            "D1",
            "D2",
            "D3",
            "D4",
            "D5",
            "SVC",
            "E_GMI_DBL",
            "E_GMI_DMM",
            "E_GMI_ISF",
        ],
    }


def test_maturity_classification() -> None:
    cutoffs = [
        0.20,
        0.40,
        0.60,
        0.80,
    ]

    labels = [
        "initial",
        "developing",
        "defined",
        "managed",
        "optimized",
    ]

    assert classify_maturity(
        0.10,
        cutoffs,
        labels,
    ) == "initial"

    assert classify_maturity(
        0.50,
        cutoffs,
        labels,
    ) == "defined"

    assert classify_maturity(
        0.90,
        cutoffs,
        labels,
    ) == "optimized"


def test_reduced_experiment_1(
    base_config: dict[str, Any],
    dynamics_config: dict[str, Any],
) -> None:
    config = deepcopy(
        load_experiment_config(
            "configs/experiments/experiment_1.yaml"
        )
    )

    config.pop(
        "_metadata",
        None,
    )

    config[
        "seed_plan"
    ]["seed_count"] = 2

    config[
        "simulation"
    ]["population_size"] = 8

    config[
        "simulation"
    ]["terminal_window"] = 5

    relaxed_steady_state(
        config
    )

    result = run_experiment_1(
        base_config,
        dynamics_config,
        config,
    )

    assert isinstance(
        result,
        ExperimentResult,
    )

    assert set(
        result.tables[
            "framework_summary"
        ]["framework"]
    ) == {
        "DBL",
        "DMM",
        "ISF",
    }

    assert len(
        result.tables[
            "seed_framework_scores"
        ]
    ) == 6

    assert result.metadata[
        "shared_trajectory"
    ] is True


def test_reduced_experiment_2(
    base_config: dict[str, Any],
    dynamics_config: dict[str, Any],
) -> None:
    config = deepcopy(
        load_experiment_config(
            "configs/experiments/experiment_2.yaml"
        )
    )

    config.pop(
        "_metadata",
        None,
    )

    config[
        "seed_plan"
    ]["seed_count"] = 2

    config[
        "simulation"
    ]["population_size"] = 8

    config[
        "shock"
    ]["maximum_network_distance"] = 2

    config[
        "recovery"
    ]["maximum_observation_steps"] = 10

    config[
        "recovery"
    ]["required_consecutive_steps"] = 2

    config[
        "recovery"
    ]["recovery_tolerance"] = 1.0

    relaxed_steady_state(
        config
    )

    result = run_experiment_2(
        base_config,
        dynamics_config,
        config,
    )

    assert set(
        result.tables[
            "recovery_results"
        ]["framework"]
    ) == {
        "DBL",
        "DMM",
        "ISF",
    }

    assert len(
        result.tables[
            "propagation_results"
        ]
    ) == 2

    assert result.metadata[
        "shared_shock"
    ] is True


def test_reduced_experiment_3(
    base_config: dict[str, Any],
    dynamics_config: dict[str, Any],
) -> None:
    config = deepcopy(
        load_experiment_config(
            "configs/experiments/experiment_3.yaml"
        )
    )

    config.pop(
        "_metadata",
        None,
    )

    config[
        "seed_plan"
    ]["seed_count"] = 2

    config[
        "simulation"
    ]["population_size"] = 8

    config[
        "simulation"
    ]["observation_steps"] = 10

    relaxed_steady_state(
        config
    )

    result = run_experiment_3(
        base_config,
        dynamics_config,
        config,
    )

    recommendations = result.tables[
        "threshold_recommendations"
    ]

    assert set(
        recommendations[
            "framework"
        ]
    ) == {
        "DBL",
        "DMM",
        "ISF",
    }

    assert result.metadata[
        "coarse_full_range_scanned"
    ] is True

    threshold_results = result.tables[
        "seed_threshold_results"
    ]

    assert threshold_results[
        "threshold"
    ].min() == pytest.approx(
        0.0
    )

    assert threshold_results[
        "threshold"
    ].max() == pytest.approx(
        1.0
    )


def test_reduced_experiment_4(
    base_config: dict[str, Any],
    dynamics_config: dict[str, Any],
) -> None:
    config = deepcopy(
        load_experiment_config(
            "configs/experiments/experiment_4.yaml"
        )
    )

    config.pop(
        "_metadata",
        None,
    )

    config[
        "seed_plan"
    ]["seed_count"] = 2

    config[
        "simulation"
    ]["population_size"] = 8

    config[
        "simulation"
    ]["horizon"] = 10

    config[
        "simulation"
    ]["terminal_window"] = 5

    result = run_experiment_4(
        base_config,
        dynamics_config,
        config,
    )

    assert result.metadata[
        "low_endpoint_passed"
    ] is True

    assert result.metadata[
        "high_endpoint_passed"
    ] is True

    assert result.metadata[
        "all_frameworks_evaluated"
    ] is True

    expected_rows = (
        len(
            config[
                "maturity_levels"
            ]
        )
        * 2
        * 3
    )

    assert len(
        result.tables[
            "level_seed_results"
        ]
    ) == expected_rows


def test_experiment_1_is_reproducible(
    base_config: dict[str, Any],
    dynamics_config: dict[str, Any],
) -> None:
    config = deepcopy(
        load_experiment_config(
            "configs/experiments/experiment_1.yaml"
        )
    )

    config.pop(
        "_metadata",
        None,
    )

    config[
        "seed_plan"
    ]["seed_count"] = 1

    config[
        "simulation"
    ]["population_size"] = 8

    config[
        "simulation"
    ]["terminal_window"] = 3

    relaxed_steady_state(
        config
    )

    first = run_experiment_1(
        base_config,
        dynamics_config,
        config,
    )

    second = run_experiment_1(
        base_config,
        dynamics_config,
        config,
    )

    pd.testing.assert_frame_equal(
        first.tables[
            "seed_framework_scores"
        ],
        second.tables[
            "seed_framework_scores"
        ],
    )

    assert first.metadata == second.metadata


def test_experiment_outputs_are_saved(
    base_config: dict[str, Any],
    dynamics_config: dict[str, Any],
    tmp_path: Path,
) -> None:
    config = deepcopy(
        load_experiment_config(
            "configs/experiments/experiment_4.yaml"
        )
    )

    config.pop(
        "_metadata",
        None,
    )

    config[
        "seed_plan"
    ]["seed_count"] = 1

    config[
        "simulation"
    ]["population_size"] = 8

    config[
        "simulation"
    ]["horizon"] = 5

    config[
        "simulation"
    ]["terminal_window"] = 2

    result = run_experiment_4(
        base_config,
        dynamics_config,
        config,
    )

    paths = save_experiment_result(
        result,
        tmp_path,
    )

    assert paths[
        "metadata"
    ].exists()

    assert paths[
        "figure"
    ].exists()

    assert paths[
        "level_summary"
    ].exists()