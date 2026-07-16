from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from sgb.pipeline import (
    PIPELINE_CONFIG_PATH,
    PipelineConfigurationError,
    PipelineRunResult,
    build_stage_plan,
    list_stage_names,
    load_pipeline_config,
    run_pipeline,
    save_pipeline_log,
    validate_expected_artifacts,
    validate_pipeline_config,
)


@pytest.fixture
def pipeline_config() -> dict[str, Any]:
    config = deepcopy(
        load_pipeline_config(
            PIPELINE_CONFIG_PATH
        )
    )

    config.pop(
        "_metadata",
        None,
    )

    return config


def test_pipeline_configuration_loads(
    pipeline_config: dict[str, Any],
) -> None:
    assert (
        pipeline_config[
            "config_kind"
        ]
        == "pipeline"
    )

    assert (
        pipeline_config[
            "project"
        ]["short_name"]
        == "SGB"
    )


def test_pipeline_stage_order(
    pipeline_config: dict[str, Any],
) -> None:
    names = list_stage_names(
        pipeline_config
    )

    assert names[
        0
    ] == "tests"

    assert names[
        -1
    ] == "final_reporting"

    assert names.index(
        "baseline_calibration"
    ) < names.index(
        "experiment_1"
    )

    assert names.index(
        "experiment_4"
    ) < names.index(
        "sensitivity_full"
    )


def test_selected_stage_includes_dependencies(
    pipeline_config: dict[str, Any],
) -> None:
    plan = build_stage_plan(
        pipeline_config,
        [
            "final_reporting",
        ],
    )

    assert plan[
        0
    ] == "tests"

    assert plan[
        -1
    ] == "final_reporting"

    assert {
        "baseline_calibration",
        "time_convergence",
        "seed_convergence",
        "population_convergence",
        "experiment_1",
        "experiment_2",
        "experiment_3",
        "experiment_4",
        "sensitivity_full",
    }.issubset(
        set(
            plan
        )
    )


def test_dependencies_can_be_disabled(
    pipeline_config: dict[str, Any],
) -> None:
    plan = build_stage_plan(
        pipeline_config,
        [
            "experiment_3",
        ],
        include_dependencies=False,
    )

    assert plan == [
        "experiment_3",
    ]


def test_unknown_stage_is_rejected(
    pipeline_config: dict[str, Any],
) -> None:
    with pytest.raises(
        PipelineConfigurationError,
        match="Unknown pipeline stage",
    ):
        build_stage_plan(
            pipeline_config,
            [
                "unknown_stage",
            ],
        )


def test_duplicate_stage_is_rejected(
    pipeline_config: dict[str, Any],
) -> None:
    invalid = deepcopy(
        pipeline_config
    )

    invalid[
        "stages"
    ].append(
        deepcopy(
            invalid[
                "stages"
            ][0]
        )
    )

    with pytest.raises(
        PipelineConfigurationError,
        match="Duplicate pipeline stage",
    ):
        validate_pipeline_config(
            invalid
        )


def test_unknown_dependency_is_rejected(
    pipeline_config: dict[str, Any],
) -> None:
    invalid = deepcopy(
        pipeline_config
    )

    invalid[
        "stages"
    ][1][
        "dependencies"
    ] = [
        "missing_stage",
    ]

    with pytest.raises(
        PipelineConfigurationError,
        match="unknown dependency",
    ):
        validate_pipeline_config(
            invalid
        )


def test_artifact_validation(
    tmp_path: Path,
) -> None:
    existing = (
        tmp_path
        / "outputs/example.csv"
    )

    existing.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing.write_text(
        "value\n1\n",
        encoding="utf-8",
    )

    config = {
        "schema_version": 1,
        "config_kind": "pipeline",
        "project": {
            "short_name": "TEST",
            "version": "1.0.0",
        },
        "execution": {
            "stop_on_failure": True,
            "log_path": "outputs/pipeline.json",
        },
        "stages": [
            {
                "name": "example",
                "description": "Synthetic stage.",
                "dependencies": [],
                "command": [
                    "-c",
                    "print('example')",
                ],
                "expected_artifacts": [
                    "outputs/example.csv",
                    "outputs/missing.csv",
                ],
            },
        ],
    }

    records = validate_expected_artifacts(
        config,
        project_root=tmp_path,
    )

    assert len(
        records
    ) == 2

    status = {
        record[
            "artifact"
        ]: record[
            "exists"
        ]
        for record in records
    }

    assert status[
        "outputs/example.csv"
    ] is True

    assert status[
        "outputs/missing.csv"
    ] is False


def test_dry_run_does_not_execute_commands(
    tmp_path: Path,
) -> None:
    marker = (
        tmp_path
        / "marker.txt"
    )

    config = {
        "schema_version": 1,
        "config_kind": "pipeline",
        "project": {
            "short_name": "TEST",
            "version": "1.0.0",
        },
        "execution": {
            "stop_on_failure": True,
            "log_path": "outputs/pipeline.json",
        },
        "stages": [
            {
                "name": "example",
                "description": "Synthetic stage.",
                "dependencies": [],
                "command": [
                    "-c",
                    (
                        "from pathlib import Path; "
                        f"Path({str(marker)!r}).write_text('created')"
                    ),
                ],
                "expected_artifacts": [],
            },
        ],
    }

    result = run_pipeline(
        config,
        project_root=tmp_path,
        dry_run=True,
    )

    assert isinstance(
        result,
        PipelineRunResult,
    )

    assert result.passed is True

    assert (
        result.records[
            0
        ]["status"]
        == "planned"
    )

    assert not marker.exists()


def test_pipeline_executes_successful_stage(
    tmp_path: Path,
) -> None:
    marker = (
        tmp_path
        / "outputs/marker.txt"
    )

    command = (
        "from pathlib import Path; "
        f"path = Path({str(marker)!r}); "
        "path.parent.mkdir(parents=True, exist_ok=True); "
        "path.write_text('created', encoding='utf-8')"
    )

    config = {
        "schema_version": 1,
        "config_kind": "pipeline",
        "project": {
            "short_name": "TEST",
            "version": "1.0.0",
        },
        "execution": {
            "stop_on_failure": True,
            "log_path": "outputs/pipeline.json",
        },
        "stages": [
            {
                "name": "example",
                "description": "Synthetic stage.",
                "dependencies": [],
                "command": [
                    "-c",
                    command,
                ],
                "expected_artifacts": [
                    "outputs/marker.txt",
                ],
            },
        ],
    }

    result = run_pipeline(
        config,
        project_root=tmp_path,
    )

    assert result.passed is True
    assert marker.is_file()

    assert (
        result.records[
            0
        ]["status"]
        == "passed"
    )


def test_missing_expected_artifact_fails_stage(
    tmp_path: Path,
) -> None:
    config = {
        "schema_version": 1,
        "config_kind": "pipeline",
        "project": {
            "short_name": "TEST",
            "version": "1.0.0",
        },
        "execution": {
            "stop_on_failure": True,
            "log_path": "outputs/pipeline.json",
        },
        "stages": [
            {
                "name": "example",
                "description": "Synthetic stage.",
                "dependencies": [],
                "command": [
                    "-c",
                    "print('completed')",
                ],
                "expected_artifacts": [
                    "outputs/missing.txt",
                ],
            },
        ],
    }

    result = run_pipeline(
        config,
        project_root=tmp_path,
    )

    assert result.passed is False

    assert (
        result.records[
            0
        ]["status"]
        == "failed"
    )

    assert result.records[
        0
    ][
        "missing_artifacts"
    ] == [
        "outputs/missing.txt",
    ]


def test_pipeline_log_is_saved(
    pipeline_config: dict[str, Any],
    tmp_path: Path,
) -> None:
    result = run_pipeline(
        pipeline_config,
        project_root=tmp_path,
        selected_stages=[
            "tests",
        ],
        include_dependencies=False,
        dry_run=True,
    )

    log_path = save_pipeline_log(
        result,
        "outputs/pipeline.json",
        project_root=tmp_path,
    )

    assert log_path.is_file()

    payload = json.loads(
        log_path.read_text(
            encoding="utf-8",
        )
    )

    assert payload[
        "metadata"
    ][
        "dry_run"
    ] is True

    assert payload[
        "records"
    ][0][
        "stage"
    ] == "tests"


def test_pipeline_does_not_modify_configuration(
    pipeline_config: dict[str, Any],
    tmp_path: Path,
) -> None:
    original = deepcopy(
        pipeline_config
    )

    run_pipeline(
        pipeline_config,
        project_root=tmp_path,
        selected_stages=[
            "tests",
        ],
        include_dependencies=False,
        dry_run=True,
    )

    assert pipeline_config == original