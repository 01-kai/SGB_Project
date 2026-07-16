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
from sgb.dynamics import (
    load_dynamics_config,
)
from sgb.sensitivity import (
    DIRECT_CODES,
    SENSITIVITY_CONFIG_PATH,
    SensitivityResult,
    apply_scenario,
    build_coverage_matrix,
    load_sensitivity_config,
    run_sensitivity_study,
    save_sensitivity_result,
    transform_catalog_freshness,
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


@pytest.fixture(scope="module")
def study_config() -> dict[str, Any]:
    config = deepcopy(
        load_sensitivity_config(
            SENSITIVITY_CONFIG_PATH
        )
    )

    config.pop(
        "_metadata",
        None,
    )

    return config


def reduced_study(
    study_config: dict[str, Any],
) -> dict[str, Any]:
    study = deepcopy(
        study_config
    )

    study[
        "profiles"
    ]["smoke"] = {
        "start_seed": 9100,
        "seed_count": 2,
        "population_size": 8,
        "burn_in_steps": 20,
        "observation_steps": 10,
        "terminal_window": 5,
    }

    return study


def test_sensitivity_configuration_loads(
    study_config: dict[str, Any],
) -> None:
    assert (
        study_config[
            "study_type"
        ]
        == "sensitivity"
    )

    assert set(
        scenario[
            "code"
        ]
        for scenario
        in study_config[
            "scenarios"
        ]
    ) == DIRECT_CODES


def test_coverage_matrix_contains_p1_to_p21(
    study_config: dict[str, Any],
) -> None:
    coverage = build_coverage_matrix(
        study_config
    )

    assert coverage[
        "code"
    ].tolist() == [
        f"P{number}"
        for number in range(
            1,
            22,
        )
    ]

    assert len(
        coverage
    ) == 21

    p9_status = str(
        coverage.loc[
            coverage[
                "code"
            ]
            == "P9",
            "resolution_status",
        ].iloc[0]
    )

    assert "pending" in p9_status


def test_freshness_transform_preserves_baseline_window() -> None:
    transformed = transform_catalog_freshness(
        0.45,
        baseline_window=20,
        target_window=20,
    )

    assert transformed == pytest.approx(
        0.45
    )


def test_freshness_transform_increases_with_window() -> None:
    k10 = transform_catalog_freshness(
        0.45,
        baseline_window=20,
        target_window=10,
    )

    k20 = transform_catalog_freshness(
        0.45,
        baseline_window=20,
        target_window=20,
    )

    k40 = transform_catalog_freshness(
        0.45,
        baseline_window=20,
        target_window=40,
    )

    assert k10 < k20 < k40


def test_apply_dimension_weight_scenario_does_not_modify_source(
    base_config: dict[str, Any],
    study_config: dict[str, Any],
) -> None:
    source = deepcopy(
        base_config
    )

    scenario = next(
        scenario
        for scenario
        in study_config[
            "scenarios"
        ]
        if (
            scenario[
                "code"
            ]
            == "P2"
            and scenario[
                "name"
            ]
            == "D1_equal"
        )
    )

    modified = apply_scenario(
        base_config,
        scenario,
    )

    assert base_config == source
    assert modified != source


def test_beta_concentration_preserves_means(
    base_config: dict[str, Any],
    study_config: dict[str, Any],
) -> None:
    scenario = next(
        scenario
        for scenario
        in study_config[
            "scenarios"
        ]
        if (
            scenario[
                "code"
            ]
            == "P10"
            and scenario[
                "name"
            ]
            == "diffuse_beta_shapes"
        )
    )

    modified = apply_scenario(
        base_config,
        scenario,
    )

    original = base_config[
        "baseline"
    ]["initialization"][
        "submetric_distributions"
    ]

    changed = modified[
        "baseline"
    ]["initialization"][
        "submetric_distributions"
    ]

    for submetric in original:
        original_mean = (
            original[
                submetric
            ]["alpha"]
            / (
                original[
                    submetric
                ]["alpha"]
                + original[
                    submetric
                ]["beta"]
            )
        )

        changed_mean = (
            changed[
                submetric
            ]["alpha"]
            / (
                changed[
                    submetric
                ]["alpha"]
                + changed[
                    submetric
                ]["beta"]
            )
        )

        assert changed_mean == pytest.approx(
            original_mean
        )


def test_reduced_weight_and_freshness_sensitivity(
    base_config: dict[str, Any],
    dynamics_config: dict[str, Any],
    study_config: dict[str, Any],
) -> None:
    study = reduced_study(
        study_config
    )

    result = run_sensitivity_study(
        base_config=base_config,
        dynamics_config=(
            dynamics_config
        ),
        study_config=study,
        profile_name="smoke",
        codes=[
            "P1",
            "P2",
            "P13",
            "P21",
        ],
    )

    assert isinstance(
        result,
        SensitivityResult,
    )

    metrics = result.tables[
        "scenario_metrics"
    ]

    assert {
        "P1",
        "P2",
        "P21",
    }.issubset(
        set(
            metrics[
                "code"
            ]
        )
    )

    assert set(
        metrics[
            "framework"
        ]
    ) == {
        "DBL",
        "DMM",
        "ISF",
    }

    rank_stability = result.tables[
        "dimension_rank_stability"
    ]

    assert set(
        rank_stability[
            "dimension"
        ]
    ) == {
        "D1",
    }

    maturity = result.tables[
        "maturity_cutoff_stability"
    ]

    assert not maturity.empty

    freshness = result.tables[
        "freshness_window_stability"
    ]

    assert set(
        freshness[
            "window"
        ]
    ) == {
        10,
        20,
        40,
    }


def test_reduced_shock_sensitivity(
    base_config: dict[str, Any],
    dynamics_config: dict[str, Any],
    study_config: dict[str, Any],
) -> None:
    study = reduced_study(
        study_config
    )

    result = run_sensitivity_study(
        base_config=base_config,
        dynamics_config=(
            dynamics_config
        ),
        study_config=study,
        profile_name="smoke",
        codes=[
            "P14",
        ],
    )

    shocks = result.tables[
        "shock_sensitivity"
    ]

    assert not shocks.empty

    assert set(
        shocks[
            "shock_type"
        ]
    ) == {
        "security",
        "data_quality",
        "regulatory",
    }

    assert shocks[
        "affected_fraction"
    ].between(
        0.0,
        1.0,
    ).all()

    assert (
        shocks[
            "immediate_E_GMI_loss"
        ]
        >= 0.0
    ).all()


def test_reduced_network_propagation_sensitivity(
    base_config: dict[str, Any],
    dynamics_config: dict[str, Any],
    study_config: dict[str, Any],
) -> None:
    study = reduced_study(
        study_config
    )

    result = run_sensitivity_study(
        base_config=base_config,
        dynamics_config=(
            dynamics_config
        ),
        study_config=study,
        profile_name="smoke",
        codes=[
            "P12",
        ],
    )

    shocks = result.tables[
        "shock_sensitivity"
    ]

    assert set(
        shocks[
            "scenario"
        ]
    ) == {
        "network_m_2",
        "network_m_3",
        "network_m_5",
    }

    assert set(
        shocks[
            "shock_type"
        ]
    ) == {
        "security",
    }


def test_sensitivity_is_reproducible(
    base_config: dict[str, Any],
    dynamics_config: dict[str, Any],
    study_config: dict[str, Any],
) -> None:
    study = reduced_study(
        study_config
    )

    first = run_sensitivity_study(
        base_config=base_config,
        dynamics_config=(
            dynamics_config
        ),
        study_config=study,
        profile_name="smoke",
        codes=[
            "P1",
        ],
    )

    second = run_sensitivity_study(
        base_config=base_config,
        dynamics_config=(
            dynamics_config
        ),
        study_config=study,
        profile_name="smoke",
        codes=[
            "P1",
        ],
    )

    pd.testing.assert_frame_equal(
        first.tables[
            "scenario_metrics"
        ],
        second.tables[
            "scenario_metrics"
        ],
    )

    assert (
        first.metadata
        == second.metadata
    )


def test_sensitivity_outputs_are_saved(
    base_config: dict[str, Any],
    dynamics_config: dict[str, Any],
    study_config: dict[str, Any],
    tmp_path: Path,
) -> None:
    study = reduced_study(
        study_config
    )

    result = run_sensitivity_study(
        base_config=base_config,
        dynamics_config=(
            dynamics_config
        ),
        study_config=study,
        profile_name="smoke",
        codes=[
            "P1",
        ],
    )

    paths = save_sensitivity_result(
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
        "scenario_metrics"
    ].exists()

    assert paths[
        "coverage_matrix"
    ].exists()