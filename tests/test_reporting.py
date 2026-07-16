from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from sgb.reporting import (
    REPORTING_CONFIG_PATH,
    FinalReportResult,
    ReportingError,
    build_final_report,
    build_reproducibility_manifest,
    calculate_sha256,
    load_reporting_config,
    save_final_report,
)


@pytest.fixture
def reporting_config() -> dict[str, Any]:
    config = deepcopy(
        load_reporting_config(
            REPORTING_CONFIG_PATH
        )
    )

    config.pop(
        "_metadata",
        None,
    )

    return config


def create_synthetic_project(
    root: Path,
) -> dict[str, Any]:
    paths = {
        "baseline_table_1": (
            root
            / "outputs/calibration/baseline/"
            "table_1_baseline_dimensions.csv"
        ),
        "baseline_metadata": (
            root
            / "outputs/calibration/baseline/"
            "baseline_calibration_metadata.json"
        ),
        "time_metadata": (
            root
            / "outputs/convergence/time/metadata.json"
        ),
        "seed_metadata": (
            root
            / "outputs/convergence/seeds/metadata.json"
        ),
        "population_metadata": (
            root
            / "outputs/convergence/population/metadata.json"
        ),
        "experiment_1_metadata": (
            root
            / "outputs/experiments/experiment_1/metadata.json"
        ),
        "experiment_1_summary": (
            root
            / "outputs/experiments/experiment_1/"
            "framework_summary.csv"
        ),
        "experiment_2_metadata": (
            root
            / "outputs/experiments/experiment_2/metadata.json"
        ),
        "experiment_2_summary": (
            root
            / "outputs/experiments/experiment_2/"
            "framework_recovery_summary.csv"
        ),
        "experiment_3_metadata": (
            root
            / "outputs/experiments/experiment_3/metadata.json"
        ),
        "experiment_3_summary": (
            root
            / "outputs/experiments/experiment_3/"
            "threshold_recommendations.csv"
        ),
        "experiment_4_metadata": (
            root
            / "outputs/experiments/experiment_4/metadata.json"
        ),
        "experiment_4_summary": (
            root
            / "outputs/experiments/experiment_4/"
            "level_summary.csv"
        ),
        "sensitivity_metadata": (
            root
            / "outputs/sensitivity/global/metadata.json"
        ),
        "sensitivity_coverage": (
            root
            / "outputs/sensitivity/global/coverage_matrix.csv"
        ),
        "sensitivity_stability": (
            root
            / "outputs/sensitivity/global/stability_summary.csv"
        ),
    }

    for path in paths.values():
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    pd.DataFrame(
        {
            "dimension": [
                "D1",
                "D2",
                "D3",
                "D4",
                "D5",
            ],
            "theoretical_mean": [
                0.45,
                0.45,
                0.45,
                0.45,
                0.45,
            ],
            "empirical_mean": [
                0.451,
                0.449,
                0.452,
                0.448,
                0.450,
            ],
            "accepted": [
                True,
                True,
                True,
                True,
                True,
            ],
        }
    ).to_csv(
        paths[
            "baseline_table_1"
        ],
        index=False,
    )

    metadata_payloads = {
        "baseline_metadata": {
            "passed": True,
            "dimension_acceptance_passed": True,
            "submetric_acceptance_passed": True,
        },
        "time_metadata": {
            "passed": True,
            "recommended_horizon": 500,
        },
        "seed_metadata": {
            "passed": True,
            "recommended_seed_count": 30,
        },
        "population_metadata": {
            "passed": True,
            "recommended_population_size": 50,
        },
        "experiment_1_metadata": {
            "passed": True,
            "overall_ranking": "DBL>DMM>ISF",
        },
        "experiment_2_metadata": {
            "passed": True,
            "all_framework_runs_recovered": True,
        },
        "experiment_3_metadata": {
            "passed": True,
            "coarse_full_range_scanned": True,
        },
        "experiment_4_metadata": {
            "passed": True,
            "low_endpoint_passed": True,
            "high_endpoint_passed": True,
        },
        "sensitivity_metadata": {
            "passed": False,
            "simulation_stability_passed": True,
            "P9_public_calibration_complete": False,
        },
    }

    for name, payload in (
        metadata_payloads.items()
    ):
        paths[
            name
        ].write_text(
            json.dumps(
                payload
            ),
            encoding="utf-8",
        )

    pd.DataFrame(
        {
            "framework": [
                "DBL",
                "DMM",
                "ISF",
            ],
            "mean": [
                0.45,
                0.44,
                0.43,
            ],
        }
    ).to_csv(
        paths[
            "experiment_1_summary"
        ],
        index=False,
    )

    pd.DataFrame(
        {
            "framework": [
                "DBL",
                "DMM",
                "ISF",
            ],
            "recovered_fraction": [
                1.0,
                1.0,
                1.0,
            ],
        }
    ).to_csv(
        paths[
            "experiment_2_summary"
        ],
        index=False,
    )

    pd.DataFrame(
        {
            "framework": [
                "DBL",
                "DMM",
                "ISF",
            ],
            "recommended_threshold": [
                0.40,
                0.41,
                0.39,
            ],
        }
    ).to_csv(
        paths[
            "experiment_3_summary"
        ],
        index=False,
    )

    pd.DataFrame(
        {
            "level": [
                "low",
                "high",
            ],
            "framework": [
                "DBL",
                "DBL",
            ],
            "E_GMI_mean": [
                0.20,
                0.80,
            ],
        }
    ).to_csv(
        paths[
            "experiment_4_summary"
        ],
        index=False,
    )

    pd.DataFrame(
        {
            "code": [
                f"P{number}"
                for number in range(
                    1,
                    22,
                )
            ],
            "category": [
                "II"
            ] * 21,
            "implementation": [
                "test"
            ] * 21,
            "config": [
                "config.yaml"
            ] * 21,
            "resolution_status": [
                (
                    "simulation_envelope_complete_"
                    "public_calibration_pending"
                    if number == 9
                    else "direct_sensitivity_implemented"
                )
                for number in range(
                    1,
                    22,
                )
            ],
        }
    ).to_csv(
        paths[
            "sensitivity_coverage"
        ],
        index=False,
    )

    pd.DataFrame(
        {
            "code": [
                "P1",
            ],
            "scenario": [
                "equal_standard_dimensions",
            ],
            "framework": [
                "DBL",
            ],
            "accepted": [
                True,
            ],
        }
    ).to_csv(
        paths[
            "sensitivity_stability"
        ],
        index=False,
    )

    source_files = {
        root / "configs/example.yaml": "value: 1\n",
        root / "sgb/example.py": "VALUE = 1\n",
        root / "tests/test_example.py": (
            "def test_example():\n"
            "    assert True\n"
        ),
        root / "pyproject.toml": (
            "[project]\n"
            'name = "example"\n'
        ),
        root / "requirements.txt": (
            "pandas==3.0.3\n"
        ),
    }

    for path, content in (
        source_files.items()
    ):
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        path.write_text(
            content,
            encoding="utf-8",
        )

    config = {
        "schema_version": 1,
        "config_kind": "reporting",
        "project": {
            "title": "Synthetic Governance Benchmark",
            "short_name": "SGB",
            "version": "0.2.0",
        },
        "strict_required_inputs": True,
        "inputs": {
            "baseline_table_1": {
                "path": str(
                    paths[
                        "baseline_table_1"
                    ].relative_to(
                        root
                    )
                ),
                "kind": "csv",
                "required": True,
                "role": "Baseline table",
            },
            "baseline_metadata": {
                "path": str(
                    paths[
                        "baseline_metadata"
                    ].relative_to(
                        root
                    )
                ),
                "kind": "json",
                "required": True,
                "role": "Baseline decision",
            },
            "time_convergence_metadata": {
                "path": str(
                    paths[
                        "time_metadata"
                    ].relative_to(
                        root
                    )
                ),
                "kind": "json",
                "required": True,
                "role": "Time convergence",
            },
            "seed_convergence_metadata": {
                "path": str(
                    paths[
                        "seed_metadata"
                    ].relative_to(
                        root
                    )
                ),
                "kind": "json",
                "required": True,
                "role": "Seed convergence",
            },
            "population_convergence_metadata": {
                "path": str(
                    paths[
                        "population_metadata"
                    ].relative_to(
                        root
                    )
                ),
                "kind": "json",
                "required": True,
                "role": "Population convergence",
            },
            "experiment_1_metadata": {
                "path": str(
                    paths[
                        "experiment_1_metadata"
                    ].relative_to(
                        root
                    )
                ),
                "kind": "json",
                "required": True,
                "role": "Experiment 1 decision",
            },
            "experiment_1_summary": {
                "path": str(
                    paths[
                        "experiment_1_summary"
                    ].relative_to(
                        root
                    )
                ),
                "kind": "csv",
                "required": True,
                "role": "Experiment 1 summary",
            },
            "experiment_2_metadata": {
                "path": str(
                    paths[
                        "experiment_2_metadata"
                    ].relative_to(
                        root
                    )
                ),
                "kind": "json",
                "required": True,
                "role": "Experiment 2 decision",
            },
            "experiment_2_summary": {
                "path": str(
                    paths[
                        "experiment_2_summary"
                    ].relative_to(
                        root
                    )
                ),
                "kind": "csv",
                "required": True,
                "role": "Experiment 2 summary",
            },
            "experiment_3_metadata": {
                "path": str(
                    paths[
                        "experiment_3_metadata"
                    ].relative_to(
                        root
                    )
                ),
                "kind": "json",
                "required": True,
                "role": "Experiment 3 decision",
            },
            "experiment_3_summary": {
                "path": str(
                    paths[
                        "experiment_3_summary"
                    ].relative_to(
                        root
                    )
                ),
                "kind": "csv",
                "required": True,
                "role": "Experiment 3 summary",
            },
            "experiment_4_metadata": {
                "path": str(
                    paths[
                        "experiment_4_metadata"
                    ].relative_to(
                        root
                    )
                ),
                "kind": "json",
                "required": True,
                "role": "Experiment 4 decision",
            },
            "experiment_4_summary": {
                "path": str(
                    paths[
                        "experiment_4_summary"
                    ].relative_to(
                        root
                    )
                ),
                "kind": "csv",
                "required": True,
                "role": "Experiment 4 summary",
            },
            "sensitivity_metadata": {
                "path": str(
                    paths[
                        "sensitivity_metadata"
                    ].relative_to(
                        root
                    )
                ),
                "kind": "json",
                "required": True,
                "role": "Sensitivity decision",
            },
            "sensitivity_coverage": {
                "path": str(
                    paths[
                        "sensitivity_coverage"
                    ].relative_to(
                        root
                    )
                ),
                "kind": "csv",
                "required": True,
                "role": "Parameter coverage",
            },
            "sensitivity_stability": {
                "path": str(
                    paths[
                        "sensitivity_stability"
                    ].relative_to(
                        root
                    )
                ),
                "kind": "csv",
                "required": True,
                "role": "Sensitivity stability",
            },
        },
        "reproducibility": {
            "hash_roots": [
                "configs",
                "sgb",
                "tests",
            ],
            "hash_files": [
                "pyproject.toml",
                "requirements.txt",
            ],
            "include_extensions": [
                ".py",
                ".yaml",
                ".toml",
                ".txt",
            ],
        },
        "outputs": {
            "directory": "outputs/final",
            "report_markdown": "final_report.md",
            "artifact_index": "artifact_index.csv",
            "decision_summary": "decision_summary.csv",
            "parameter_coverage": "parameter_coverage.csv",
            "reproducibility_manifest": (
                "reproducibility_manifest.csv"
            ),
            "metadata": "reporting_metadata.json",
        },
    }

    return config


def test_reporting_configuration_loads(
    reporting_config: dict[str, Any],
) -> None:
    assert (
        reporting_config[
            "config_kind"
        ]
        == "reporting"
    )

    assert (
        reporting_config[
            "strict_required_inputs"
        ]
        is True
    )


def test_sha256_is_deterministic(
    tmp_path: Path,
) -> None:
    path = (
        tmp_path
        / "example.txt"
    )

    path.write_text(
        "reproducible\n",
        encoding="utf-8",
    )

    first = calculate_sha256(
        path
    )
    second = calculate_sha256(
        path
    )

    assert first == second
    assert len(first) == 64


def test_final_report_builds_from_complete_inputs(
    tmp_path: Path,
) -> None:
    config = create_synthetic_project(
        tmp_path
    )

    result = build_final_report(
        config,
        project_root=tmp_path,
    )

    assert isinstance(
        result,
        FinalReportResult,
    )

    assert result.passed is True

    assert result.artifact_index[
        "exists"
    ].all()

    assert "P9" in set(
        result.parameter_coverage[
            "code"
        ]
    )

    p9_status = str(
        result.parameter_coverage.loc[
            result.parameter_coverage[
                "code"
            ]
            == "P9",
            "final_status",
        ].iloc[0]
    )

    assert p9_status == "pending"

    assert (
        "# Synthetic Governance Benchmark"
        in result.report_markdown
    )

    assert (
        "Recommended horizon: 500"
        in result.report_markdown
    )


def test_sensitivity_public_calibration_is_pending(
    tmp_path: Path,
) -> None:
    config = create_synthetic_project(
        tmp_path
    )

    result = build_final_report(
        config,
        project_root=tmp_path,
    )

    row = result.decision_summary.loc[
        result.decision_summary[
            "input_name"
        ]
        == "sensitivity_metadata"
    ].iloc[0]

    assert row[
        "status"
    ] == "pending"

    assert (
        "P9 public calibration complete: False"
        in row[
            "decision"
        ]
    )


def test_missing_required_input_raises_in_strict_mode(
    tmp_path: Path,
) -> None:
    config = create_synthetic_project(
        tmp_path
    )

    missing_path = (
        tmp_path
        / config[
            "inputs"
        ][
            "baseline_table_1"
        ]["path"]
    )

    missing_path.unlink()

    with pytest.raises(
        ReportingError,
        match="Required reporting inputs are missing",
    ):
        build_final_report(
            config,
            project_root=tmp_path,
        )


def test_missing_input_is_recorded_when_not_strict(
    tmp_path: Path,
) -> None:
    config = create_synthetic_project(
        tmp_path
    )

    config[
        "strict_required_inputs"
    ] = False

    missing_path = (
        tmp_path
        / config[
            "inputs"
        ][
            "baseline_table_1"
        ]["path"]
    )

    missing_path.unlink()

    result = build_final_report(
        config,
        project_root=tmp_path,
    )

    assert result.passed is False

    missing_row = (
        result.artifact_index.loc[
            result.artifact_index[
                "input_name"
            ]
            == "baseline_table_1"
        ].iloc[0]
    )

    assert (
        missing_row[
            "exists"
        ]
        is False
        or missing_row[
            "exists"
        ]
        == False
    )


def test_reproducibility_manifest_is_deterministic(
    tmp_path: Path,
) -> None:
    config = create_synthetic_project(
        tmp_path
    )

    first = build_reproducibility_manifest(
        config,
        project_root=tmp_path,
    )

    second = build_reproducibility_manifest(
        config,
        project_root=tmp_path,
    )

    pd.testing.assert_frame_equal(
        first,
        second,
    )

    assert {
        "relative_path",
        "size_bytes",
        "sha256",
    } == set(
        first.columns
    )


def test_final_report_outputs_are_saved(
    tmp_path: Path,
) -> None:
    config = create_synthetic_project(
        tmp_path
    )

    result = build_final_report(
        config,
        project_root=tmp_path,
    )

    paths = save_final_report(
        result=result,
        reporting_config=config,
        project_root=tmp_path,
    )

    assert set(
        paths
    ) == {
        "report_markdown",
        "artifact_index",
        "decision_summary",
        "parameter_coverage",
        "reproducibility_manifest",
        "metadata",
    }

    for path in paths.values():
        assert path.is_file()

    saved_metadata = json.loads(
        paths[
            "metadata"
        ].read_text(
            encoding="utf-8",
        )
    )

    assert (
        saved_metadata[
            "passed"
        ]
        is True
    )