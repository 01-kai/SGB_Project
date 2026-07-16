from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from sgb.calibration import (
    CALIBRATION_CONFIG_PATH,
    BaselineCalibrationResult,
    CalibrationConfigurationError,
    beta_mean,
    calculate_theoretical_dimension_means,
    calculate_theoretical_submetric_means,
    load_calibration_config,
    run_baseline_calibration,
    save_calibration_result,
)
from sgb.config import (
    BASE_CONFIG_PATH,
    load_config,
)
from sgb.dimensions import (
    ALL_DIMENSIONS,
    STANDARD_DIMENSIONS,
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
def calibration_config() -> dict[str, Any]:
    config = deepcopy(
        load_calibration_config(
            CALIBRATION_CONFIG_PATH
        )
    )
    config.pop(
        "_metadata",
        None,
    )

    return config


@pytest.fixture(scope="module")
def calibration_result(
    base_config: dict[str, Any],
    calibration_config: dict[str, Any],
) -> BaselineCalibrationResult:
    return run_baseline_calibration(
        base_config=base_config,
        calibration_config=(
            calibration_config
        ),
    )


def test_calibration_configuration_loads(
    calibration_config: dict[str, Any],
) -> None:
    assert (
        calibration_config[
            "config_kind"
        ]
        == "baseline_calibration"
    )

    assert calibration_config[
        "dimensions"
    ] == list(
        ALL_DIMENSIONS
    )


def test_beta_mean() -> None:
    assert beta_mean(
        9.0,
        11.0,
    ) == pytest.approx(
        0.45
    )

    assert beta_mean(
        11.0,
        9.0,
    ) == pytest.approx(
        0.55
    )


def test_theoretical_submetric_means(
    base_config: dict[str, Any],
) -> None:
    means = (
        calculate_theoretical_submetric_means(
            base_config
        )
    )

    assert means[
        "completeness_ratio"
    ] == pytest.approx(
        0.45
    )

    assert means[
        "duplicate_rate"
    ] == pytest.approx(
        0.55
    )

    assert means[
        "normalized_breach_latency"
    ] == pytest.approx(
        0.55
    )

    assert means[
        "policy_violation_rate"
    ] == pytest.approx(
        0.55
    )


def test_all_theoretical_dimensions_equal_point_four_five(
    base_config: dict[str, Any],
) -> None:
    means = (
        calculate_theoretical_dimension_means(
            base_config
        )
    )

    assert set(
        means
    ) == set(
        ALL_DIMENSIONS
    )

    for dimension in ALL_DIMENSIONS:
        assert means[
            dimension
        ] == pytest.approx(
            0.45
        )


def test_calibration_uses_all_requested_seeds(
    calibration_result: BaselineCalibrationResult,
    calibration_config: dict[str, Any],
) -> None:
    expected_seed_count = (
        calibration_config[
            "seed_plan"
        ]["seed_count"]
    )

    assert len(
        calibration_result.seed_dimension_means
    ) == expected_seed_count

    assert set(
        calibration_result.seed_dimension_means[
            "seed"
        ]
    ) == set(
        range(
            calibration_config[
                "seed_plan"
            ]["start_seed"],
            calibration_config[
                "seed_plan"
            ]["start_seed"]
            + expected_seed_count,
        )
    )


def test_calibration_dimension_summary_is_complete(
    calibration_result: BaselineCalibrationResult,
) -> None:
    summary = (
        calibration_result.dimension_summary
    )

    assert summary[
        "dimension"
    ].tolist() == list(
        ALL_DIMENSIONS
    )

    assert summary[
        "theoretical_mean"
    ].between(
        0.40,
        0.50,
    ).all()

    assert summary[
        "empirical_mean"
    ].between(
        0.40,
        0.50,
    ).all()


def test_default_calibration_passes(
    calibration_result: BaselineCalibrationResult,
) -> None:
    assert (
        calibration_result.passed
        is True
    )

    assert calibration_result.dimension_summary[
        "accepted"
    ].all()

    assert calibration_result.submetric_summary[
        "mean_error_acceptable"
    ].all()


def test_table_1_contains_only_standard_dimensions(
    calibration_result: BaselineCalibrationResult,
) -> None:
    assert calibration_result.table_1[
        "dimension"
    ].tolist() == list(
        STANDARD_DIMENSIONS
    )

    assert "SVC" not in set(
        calibration_result.table_1[
            "dimension"
        ]
    )


def test_submetric_summary_is_complete(
    calibration_result: BaselineCalibrationResult,
) -> None:
    summary = (
        calibration_result.submetric_summary
    )

    assert len(
        summary
    ) == 17

    assert set(
        summary[
            "orientation"
        ]
    ) == {
        "positive",
        "inverse",
    }

    inverse_submetrics = set(
        summary.loc[
            summary[
                "orientation"
            ]
            == "inverse",
            "submetric",
        ]
    )

    assert inverse_submetrics == {
        "duplicate_rate",
        "normalized_breach_latency",
        "policy_violation_rate",
    }


def test_calibration_is_reproducible(
    base_config: dict[str, Any],
    calibration_config: dict[str, Any],
) -> None:
    small_study = deepcopy(
        calibration_config
    )

    small_study[
        "seed_plan"
    ]["seed_count"] = 4
    small_study[
        "population_size"
    ] = 12

    small_study[
        "acceptance"
    ][
        "require_dimension_ci_within_target"
    ] = False

    first = run_baseline_calibration(
        base_config=base_config,
        calibration_config=small_study,
    )

    second = run_baseline_calibration(
        base_config=base_config,
        calibration_config=small_study,
    )

    pd.testing.assert_frame_equal(
        first.seed_dimension_means,
        second.seed_dimension_means,
    )

    pd.testing.assert_frame_equal(
        first.seed_submetric_means,
        second.seed_submetric_means,
    )

    pd.testing.assert_frame_equal(
        first.dimension_summary,
        second.dimension_summary,
    )


def test_calibration_does_not_modify_base_config(
    base_config: dict[str, Any],
    calibration_config: dict[str, Any],
) -> None:
    original = deepcopy(
        base_config
    )

    small_study = deepcopy(
        calibration_config
    )

    small_study[
        "seed_plan"
    ]["seed_count"] = 2
    small_study[
        "population_size"
    ] = 10

    small_study[
        "acceptance"
    ][
        "require_dimension_ci_within_target"
    ] = False

    run_baseline_calibration(
        base_config=base_config,
        calibration_config=small_study,
    )

    assert base_config == original


def test_target_range_must_match_base_configuration(
    base_config: dict[str, Any],
    calibration_config: dict[str, Any],
) -> None:
    invalid_study = deepcopy(
        calibration_config
    )

    invalid_study[
        "acceptance"
    ]["target_minimum"] = 0.41

    with pytest.raises(
        CalibrationConfigurationError,
        match="must match",
    ):
        run_baseline_calibration(
            base_config=base_config,
            calibration_config=invalid_study,
        )


def test_calibration_outputs_are_saved(
    calibration_result: BaselineCalibrationResult,
    tmp_path: Path,
) -> None:
    paths = save_calibration_result(
        result=calibration_result,
        output_directory=tmp_path,
    )

    assert set(
        paths
    ) == {
        "seed_dimension_means",
        "seed_submetric_means",
        "dimension_summary",
        "submetric_summary",
        "table_1",
        "metadata",
    }

    for path in paths.values():
        assert path.exists()
        assert path.is_file()

    saved_table = pd.read_csv(
        paths[
            "table_1"
        ]
    )

    assert saved_table[
        "dimension"
    ].tolist() == list(
        STANDARD_DIMENSIONS
    )

    with paths[
        "metadata"
    ].open(
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(
            file
        )

    assert metadata[
        "passed"
    ] is True