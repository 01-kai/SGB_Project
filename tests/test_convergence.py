from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from sgb.config import (
    BASE_CONFIG_PATH,
    load_config,
)
from sgb.convergence import (
    ConvergenceStudyResult,
    detect_block_convergence,
    load_convergence_config,
    run_population_convergence,
    run_seed_convergence,
    run_time_convergence,
    save_convergence_result,
)
from sgb.dynamics import (
    load_dynamics_config,
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
        "study_type",
    ),
    [
        (
            "configs/studies/time_convergence.yaml",
            "time_convergence",
        ),
        (
            "configs/studies/seed_convergence.yaml",
            "seed_convergence",
        ),
        (
            "configs/studies/population_convergence.yaml",
            "population_convergence",
        ),
    ],
)
def test_convergence_configuration_loads(
    path: str,
    study_type: str,
) -> None:
    config = (
        load_convergence_config(
            path
        )
    )

    assert (
        config[
            "study_type"
        ]
        == study_type
    )


def test_stable_series_converges() -> None:
    values = np.full(
        200,
        0.45,
    )

    result = (
        detect_block_convergence(
            values,
            rolling_window=20,
            required_stable_windows=3,
            minimum_step=80,
            slope_tolerance=1e-12,
            mean_difference_tolerance=1e-12,
        )
    )

    assert (
        result[
            "converged"
        ]
        is True
    )

    assert (
        result[
            "convergence_step"
        ]
        is not None
    )


def test_trending_series_does_not_converge() -> None:
    values = np.linspace(
        0.20,
        0.80,
        200,
    )

    result = (
        detect_block_convergence(
            values,
            rolling_window=20,
            required_stable_windows=3,
            minimum_step=80,
            slope_tolerance=1e-4,
            mean_difference_tolerance=1e-3,
        )
    )

    assert (
        result[
            "converged"
        ]
        is False
    )

    assert (
        result[
            "convergence_step"
        ]
        is None
    )


def test_reduced_time_convergence_study(
    base_config: dict[str, Any],
    dynamics_config: dict[str, Any],
) -> None:
    study = deepcopy(
        load_convergence_config(
            "configs/studies/"
            "time_convergence.yaml"
        )
    )

    study.pop(
        "_metadata",
        None,
    )

    study[
        "seed_plan"
    ]["seed_count"] = 3

    study[
        "simulation"
    ]["population_size"] = 8

    study[
        "maximum_horizon"
    ] = 80

    study[
        "candidate_horizons"
    ] = [
        40,
        80,
    ]

    study["detection"] = {
        "rolling_window": 10,
        "required_stable_windows": 2,
        "minimum_step": 30,
        "slope_tolerance": 1.0,
        "mean_difference_tolerance": 1.0,
        "required_observation_steps": 10,
    }

    result = (
        run_time_convergence(
            base_config,
            dynamics_config,
            study,
        )
    )

    assert isinstance(
        result,
        ConvergenceStudyResult,
    )

    assert len(
        result.tables[
            "replicate_trajectories"
        ]
    ) == (
        3
        * 80
    )

    assert set(
        result.tables[
            "metric_convergence"
        ]["metric"]
    ) == set(
        study[
            "metrics"
        ]
    )

    assert result.metadata[
        "recommended_horizon"
    ] in {
        40,
        80,
    }


def test_reduced_seed_convergence_study(
    base_config: dict[str, Any],
    dynamics_config: dict[str, Any],
) -> None:
    study = deepcopy(
        load_convergence_config(
            "configs/studies/"
            "seed_convergence.yaml"
        )
    )

    study.pop(
        "_metadata",
        None,
    )

    study[
        "candidate_seed_counts"
    ] = [
        2,
        3,
        4,
    ]

    study[
        "simulation"
    ]["population_size"] = 8

    study[
        "simulation"
    ]["horizon"] = 30

    study[
        "simulation"
    ]["terminal_window"] = 10

    study[
        "analysis"
    ]["maximum_relative_ci_width"] = 10.0

    study[
        "analysis"
    ]["ranking_tolerance"] = 1.0

    result = (
        run_seed_convergence(
            base_config,
            dynamics_config,
            study,
        )
    )

    assert len(
        result.tables[
            "seed_metric_values"
        ]
    ) == 4

    assert set(
        result.tables[
            "ranking_summary"
        ][
            "candidate_seed_count"
        ]
    ) == {
        2,
        3,
        4,
    }

    assert result.metadata[
        "recommended_seed_count"
    ] in {
        2,
        3,
        4,
    }


def test_reduced_population_convergence_study(
    base_config: dict[str, Any],
    dynamics_config: dict[str, Any],
) -> None:
    study = deepcopy(
        load_convergence_config(
            "configs/studies/"
            "population_convergence.yaml"
        )
    )

    study.pop(
        "_metadata",
        None,
    )

    study[
        "seed_plan"
    ]["seed_count"] = 2

    study[
        "candidate_population_sizes"
    ] = [
        6,
        8,
        10,
    ]

    study[
        "simulation"
    ]["horizon"] = 20

    study[
        "simulation"
    ]["terminal_window"] = 5

    study[
        "analysis"
    ][
        "material_difference_tolerance"
    ] = 1.0

    study[
        "analysis"
    ]["ranking_tolerance"] = 1.0

    result = (
        run_population_convergence(
            base_config,
            dynamics_config,
            study,
        )
    )

    assert len(
        result.tables[
            "population_metric_values"
        ]
    ) == (
        3
        * 2
    )

    assert set(
        result.tables[
            "population_acceptance"
        ][
            "population_size"
        ]
    ) == {
        6,
        8,
        10,
    }

    assert result.metadata[
        "recommended_population_size"
    ] in {
        6,
        8,
        10,
    }


def test_time_convergence_is_reproducible(
    base_config: dict[str, Any],
    dynamics_config: dict[str, Any],
) -> None:
    study = deepcopy(
        load_convergence_config(
            "configs/studies/"
            "time_convergence.yaml"
        )
    )

    study.pop(
        "_metadata",
        None,
    )

    study[
        "seed_plan"
    ]["seed_count"] = 2

    study[
        "simulation"
    ]["population_size"] = 8

    study[
        "maximum_horizon"
    ] = 50

    study[
        "candidate_horizons"
    ] = [
        50
    ]

    study["detection"] = {
        "rolling_window": 10,
        "required_stable_windows": 2,
        "minimum_step": 30,
        "slope_tolerance": 1.0,
        "mean_difference_tolerance": 1.0,
        "required_observation_steps": 0,
    }

    first = (
        run_time_convergence(
            base_config,
            dynamics_config,
            study,
        )
    )

    second = (
        run_time_convergence(
            base_config,
            dynamics_config,
            study,
        )
    )

    pd.testing.assert_frame_equal(
        first.tables[
            "replicate_trajectories"
        ],
        second.tables[
            "replicate_trajectories"
        ],
    )

    assert (
        first.metadata
        == second.metadata
    )


def test_convergence_outputs_are_saved(
    base_config: dict[str, Any],
    dynamics_config: dict[str, Any],
    tmp_path: Path,
) -> None:
    study = deepcopy(
        load_convergence_config(
            "configs/studies/"
            "seed_convergence.yaml"
        )
    )

    study.pop(
        "_metadata",
        None,
    )

    study[
        "candidate_seed_counts"
    ] = [
        2
    ]

    study[
        "simulation"
    ]["population_size"] = 8

    study[
        "simulation"
    ]["horizon"] = 10

    study[
        "simulation"
    ]["terminal_window"] = 5

    study[
        "analysis"
    ]["maximum_relative_ci_width"] = 10.0

    study[
        "analysis"
    ]["ranking_tolerance"] = 1.0

    result = (
        run_seed_convergence(
            base_config,
            dynamics_config,
            study,
        )
    )

    paths = (
        save_convergence_result(
            result,
            tmp_path,
        )
    )

    assert paths[
        "metadata"
    ].exists()

    assert paths[
        "figure"
    ].exists()

    assert paths[
        "ci_summary"
    ].exists()