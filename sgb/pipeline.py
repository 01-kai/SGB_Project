from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from sgb.config import PROJECT_ROOT


PIPELINE_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "pipeline.yaml"
)


class PipelineConfigurationError(ValueError):
    """Raised when the pipeline configuration is invalid."""


class PipelineExecutionError(RuntimeError):
    """Raised when the pipeline cannot be executed."""


@dataclass(frozen=True, slots=True)
class PipelineRunResult:
    """Result of one pipeline execution or dry run."""

    selected_stages: tuple[str, ...]
    records: tuple[dict[str, Any], ...]
    metadata: dict[str, Any]

    @property
    def passed(self) -> bool:
        """Return whether every executed stage passed."""

        return bool(
            self.metadata.get(
                "passed",
                False,
            )
        )


def load_pipeline_config(
    path: str | Path = PIPELINE_CONFIG_PATH,
) -> dict[str, Any]:
    """Load and validate the pipeline configuration."""

    config_path = Path(path)

    if not config_path.is_absolute():
        config_path = (
            PROJECT_ROOT
            / config_path
        )

    config_path = config_path.resolve()

    if not config_path.is_file():
        raise PipelineConfigurationError(
            "Pipeline configuration does not exist: "
            f"{config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(
            file
        )

    if not isinstance(
        config,
        dict,
    ):
        raise PipelineConfigurationError(
            "Pipeline configuration root must be a mapping."
        )

    validate_pipeline_config(
        config
    )

    config["_metadata"] = {
        "source_path": str(
            config_path
        ),
    }

    return config


def validate_pipeline_config(
    config: Mapping[str, Any],
) -> None:
    """Validate the complete pipeline definition."""

    if config.get(
        "schema_version"
    ) != 1:
        raise PipelineConfigurationError(
            "schema_version must be 1."
        )

    if config.get(
        "config_kind"
    ) != "pipeline":
        raise PipelineConfigurationError(
            "config_kind must be 'pipeline'."
        )

    project = _require_mapping(
        config,
        "project",
    )

    _require_non_empty_string(
        project,
        "short_name",
    )
    _require_non_empty_string(
        project,
        "version",
    )

    execution = _require_mapping(
        config,
        "execution",
    )

    stop_on_failure = execution.get(
        "stop_on_failure"
    )

    if not isinstance(
        stop_on_failure,
        bool,
    ):
        raise PipelineConfigurationError(
            "execution.stop_on_failure must be Boolean."
        )

    _require_non_empty_string(
        execution,
        "log_path",
    )

    stages = config.get(
        "stages"
    )

    if (
        not isinstance(stages, list)
        or not stages
    ):
        raise PipelineConfigurationError(
            "stages must be a non-empty list."
        )

    observed_names: set[str] = set()
    ordered_names: list[str] = []

    for stage in stages:
        if not isinstance(
            stage,
            Mapping,
        ):
            raise PipelineConfigurationError(
                "Every pipeline stage must be a mapping."
            )

        name = _require_non_empty_string(
            stage,
            "name",
        )

        if name in observed_names:
            raise PipelineConfigurationError(
                f"Duplicate pipeline stage: {name!r}."
            )

        observed_names.add(
            name
        )
        ordered_names.append(
            name
        )

        _require_non_empty_string(
            stage,
            "description",
        )

        dependencies = _require_string_list(
            stage,
            "dependencies",
            allow_empty=True,
        )

        command = _require_string_list(
            stage,
            "command",
            allow_empty=False,
        )

        if not command:
            raise PipelineConfigurationError(
                f"Stage {name!r} command cannot be empty."
            )

        _require_string_list(
            stage,
            "expected_artifacts",
            allow_empty=True,
        )

        if name in dependencies:
            raise PipelineConfigurationError(
                f"Stage {name!r} cannot depend on itself."
            )

    stage_positions = {
        name: index
        for index, name
        in enumerate(
            ordered_names
        )
    }

    for stage in stages:
        name = str(
            stage[
                "name"
            ]
        )

        for dependency in stage[
            "dependencies"
        ]:
            if dependency not in observed_names:
                raise PipelineConfigurationError(
                    f"Stage {name!r} references unknown dependency "
                    f"{dependency!r}."
                )

            if (
                stage_positions[
                    dependency
                ]
                >= stage_positions[
                    name
                ]
            ):
                raise PipelineConfigurationError(
                    f"Dependency {dependency!r} must appear before "
                    f"stage {name!r}."
                )

    _validate_acyclic_dependencies(
        stages
    )


def list_stage_names(
    pipeline_config: Mapping[str, Any],
) -> list[str]:
    """Return pipeline stage names in execution order."""

    validate_pipeline_config(
        pipeline_config
    )

    return [
        str(
            stage[
                "name"
            ]
        )
        for stage in pipeline_config[
            "stages"
        ]
    ]


def build_stage_plan(
    pipeline_config: Mapping[str, Any],
    selected_stages: Sequence[str] | None = None,
    *,
    include_dependencies: bool = True,
) -> list[str]:
    """
    Build an ordered execution plan.

    When no stages are selected, every configured stage is returned.
    """

    validate_pipeline_config(
        pipeline_config
    )

    ordered_names = list_stage_names(
        pipeline_config
    )

    stage_map = {
        str(
            stage[
                "name"
            ]
        ): stage
        for stage in pipeline_config[
            "stages"
        ]
    }

    if not selected_stages:
        return ordered_names

    normalized_selection: list[str] = []

    for stage_name in selected_stages:
        if (
            not isinstance(stage_name, str)
            or not stage_name.strip()
        ):
            raise PipelineConfigurationError(
                "Selected stage names must be non-empty strings."
            )

        normalized_name = stage_name.strip()

        if normalized_name not in stage_map:
            raise PipelineConfigurationError(
                f"Unknown pipeline stage: {normalized_name!r}."
            )

        if normalized_name not in normalized_selection:
            normalized_selection.append(
                normalized_name
            )

    required: set[str] = set(
        normalized_selection
    )

    if include_dependencies:
        pending = list(
            normalized_selection
        )

        while pending:
            current_name = pending.pop()

            for dependency in stage_map[
                current_name
            ][
                "dependencies"
            ]:
                if dependency not in required:
                    required.add(
                        dependency
                    )
                    pending.append(
                        dependency
                    )

    return [
        stage_name
        for stage_name in ordered_names
        if stage_name in required
    ]


def validate_expected_artifacts(
    pipeline_config: Mapping[str, Any],
    *,
    project_root: str | Path = PROJECT_ROOT,
    selected_stages: Sequence[str] | None = None,
    include_dependencies: bool = True,
) -> list[dict[str, Any]]:
    """Check whether configured stage artifacts currently exist."""

    validate_pipeline_config(
        pipeline_config
    )

    root = Path(
        project_root
    ).resolve()

    plan = build_stage_plan(
        pipeline_config,
        selected_stages,
        include_dependencies=(
            include_dependencies
        ),
    )

    stage_map = {
        str(
            stage[
                "name"
            ]
        ): stage
        for stage in pipeline_config[
            "stages"
        ]
    }

    records: list[
        dict[str, Any]
    ] = []

    for stage_name in plan:
        stage = stage_map[
            stage_name
        ]

        for artifact in stage[
            "expected_artifacts"
        ]:
            artifact_path = Path(
                artifact
            )

            if not artifact_path.is_absolute():
                artifact_path = (
                    root
                    / artifact_path
                )

            artifact_path = (
                artifact_path.resolve()
            )

            exists = artifact_path.is_file()

            records.append(
                {
                    "stage": stage_name,
                    "artifact": str(
                        artifact
                    ),
                    "exists": exists,
                    "size_bytes": (
                        artifact_path.stat().st_size
                        if exists
                        else None
                    ),
                }
            )

    return records


def run_pipeline(
    pipeline_config: Mapping[str, Any],
    *,
    project_root: str | Path = PROJECT_ROOT,
    selected_stages: Sequence[str] | None = None,
    include_dependencies: bool = True,
    dry_run: bool = False,
    continue_on_failure: bool | None = None,
    python_executable: str | Path = sys.executable,
) -> PipelineRunResult:
    """Execute or preview an ordered set of pipeline stages."""

    validate_pipeline_config(
        pipeline_config
    )

    root = Path(
        project_root
    ).resolve()

    if not root.is_dir():
        raise PipelineExecutionError(
            f"Project root does not exist: {root}"
        )

    plan = build_stage_plan(
        pipeline_config,
        selected_stages,
        include_dependencies=(
            include_dependencies
        ),
    )

    stage_map = {
        str(
            stage[
                "name"
            ]
        ): stage
        for stage in pipeline_config[
            "stages"
        ]
    }

    configured_stop = bool(
        pipeline_config[
            "execution"
        ]["stop_on_failure"]
    )

    should_continue = (
        continue_on_failure
        if continue_on_failure
        is not None
        else not configured_stop
    )

    executable = str(
        Path(
            python_executable
        )
    )

    started_at = (
        datetime.now(
            timezone.utc
        )
    )

    records: list[
        dict[str, Any]
    ] = []

    pipeline_failed = False

    for stage_name in plan:
        stage = stage_map[
            stage_name
        ]

        command = [
            executable,
            *[
                str(argument)
                for argument
                in stage[
                    "command"
                ]
            ],
        ]

        command_display = shlex.join(
            command
        )

        stage_started_at = (
            datetime.now(
                timezone.utc
            )
        )

        if dry_run:
            records.append(
                {
                    "stage": stage_name,
                    "description": (
                        stage[
                            "description"
                        ]
                    ),
                    "status": "planned",
                    "command": (
                        command_display
                    ),
                    "return_code": None,
                    "duration_seconds": 0.0,
                    "started_at": (
                        stage_started_at.isoformat()
                    ),
                    "finished_at": (
                        stage_started_at.isoformat()
                    ),
                    "missing_artifacts": [],
                    "stdout_tail": "",
                    "stderr_tail": "",
                }
            )
            continue

        monotonic_start = (
            time.monotonic()
        )

        try:
            completed = subprocess.run(
                command,
                cwd=root,
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError as error:
            raise PipelineExecutionError(
                f"Unable to execute stage {stage_name!r}: "
                f"{error}"
            ) from error

        duration = (
            time.monotonic()
            - monotonic_start
        )

        stage_finished_at = (
            datetime.now(
                timezone.utc
            )
        )

        missing_artifacts = (
            _missing_stage_artifacts(
                stage=stage,
                project_root=root,
            )
        )

        passed = bool(
            completed.returncode == 0
            and not missing_artifacts
        )

        status = (
            "passed"
            if passed
            else "failed"
        )

        records.append(
            {
                "stage": stage_name,
                "description": (
                    stage[
                        "description"
                    ]
                ),
                "status": status,
                "command": (
                    command_display
                ),
                "return_code": int(
                    completed.returncode
                ),
                "duration_seconds": round(
                    duration,
                    6,
                ),
                "started_at": (
                    stage_started_at.isoformat()
                ),
                "finished_at": (
                    stage_finished_at.isoformat()
                ),
                "missing_artifacts": (
                    missing_artifacts
                ),
                "stdout_tail": _tail_text(
                    completed.stdout
                ),
                "stderr_tail": _tail_text(
                    completed.stderr
                ),
            }
        )

        if not passed:
            pipeline_failed = True

            if not should_continue:
                break

    finished_at = (
        datetime.now(
            timezone.utc
        )
    )

    completed_stage_names = [
        str(
            record[
                "stage"
            ]
        )
        for record in records
    ]

    unexecuted_stages = [
        stage_name
        for stage_name in plan
        if stage_name
        not in completed_stage_names
    ]

    passed = bool(
        not pipeline_failed
        and (
            dry_run
            or len(records)
            == len(plan)
        )
    )

    metadata = {
        "project": (
            pipeline_config[
                "project"
            ]["short_name"]
        ),
        "version": (
            pipeline_config[
                "project"
            ]["version"]
        ),
        "dry_run": dry_run,
        "include_dependencies": (
            include_dependencies
        ),
        "continue_on_failure": (
            should_continue
        ),
        "selected_stages": plan,
        "executed_stage_count": (
            len(records)
        ),
        "planned_stage_count": (
            len(plan)
        ),
        "unexecuted_stages": (
            unexecuted_stages
        ),
        "started_at": (
            started_at.isoformat()
        ),
        "finished_at": (
            finished_at.isoformat()
        ),
        "duration_seconds": round(
            (
                finished_at
                - started_at
            ).total_seconds(),
            6,
        ),
        "passed": passed,
    }

    return PipelineRunResult(
        selected_stages=tuple(
            plan
        ),
        records=tuple(
            records
        ),
        metadata=metadata,
    )


def save_pipeline_log(
    result: PipelineRunResult,
    path: str | Path,
    *,
    project_root: str | Path = PROJECT_ROOT,
) -> Path:
    """Save pipeline metadata and stage records as JSON."""

    if not isinstance(
        result,
        PipelineRunResult,
    ):
        raise PipelineExecutionError(
            "result must be a PipelineRunResult."
        )

    root = Path(
        project_root
    ).resolve()

    log_path = Path(
        path
    )

    if not log_path.is_absolute():
        log_path = (
            root
            / log_path
        )

    log_path = log_path.resolve()

    log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "metadata": result.metadata,
        "records": list(
            result.records
        ),
    }

    with log_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
            sort_keys=True,
        )

    return log_path


def _missing_stage_artifacts(
    *,
    stage: Mapping[str, Any],
    project_root: Path,
) -> list[str]:
    missing: list[str] = []

    for artifact in stage[
        "expected_artifacts"
    ]:
        artifact_path = Path(
            artifact
        )

        if not artifact_path.is_absolute():
            artifact_path = (
                project_root
                / artifact_path
            )

        if not artifact_path.is_file():
            missing.append(
                str(
                    artifact
                )
            )

    return missing


def _validate_acyclic_dependencies(
    stages: Sequence[
        Mapping[str, Any]
    ],
) -> None:
    stage_map = {
        str(
            stage[
                "name"
            ]
        ): tuple(
            stage[
                "dependencies"
            ]
        )
        for stage in stages
    }

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(
        stage_name: str,
    ) -> None:
        if stage_name in visited:
            return

        if stage_name in visiting:
            raise PipelineConfigurationError(
                "Pipeline dependency cycle detected at "
                f"{stage_name!r}."
            )

        visiting.add(
            stage_name
        )

        for dependency in stage_map[
            stage_name
        ]:
            visit(
                dependency
            )

        visiting.remove(
            stage_name
        )
        visited.add(
            stage_name
        )

    for name in stage_map:
        visit(
            name
        )


def _tail_text(
    text: str | None,
    maximum_characters: int = 4000,
) -> str:
    if not text:
        return ""

    normalized = text.strip()

    if len(normalized) <= maximum_characters:
        return normalized

    return normalized[
        -maximum_characters:
    ]


def _require_mapping(
    container: Mapping[str, Any],
    key: str,
) -> Mapping[str, Any]:
    value = container.get(
        key
    )

    if not isinstance(
        value,
        Mapping,
    ):
        raise PipelineConfigurationError(
            f"{key} must be a mapping."
        )

    return value


def _require_non_empty_string(
    container: Mapping[str, Any],
    key: str,
) -> str:
    value = container.get(
        key
    )

    if (
        not isinstance(
            value,
            str,
        )
        or not value.strip()
    ):
        raise PipelineConfigurationError(
            f"{key} must be a non-empty string."
        )

    return value.strip()


def _require_string_list(
    container: Mapping[str, Any],
    key: str,
    *,
    allow_empty: bool,
) -> list[str]:
    value = container.get(
        key
    )

    if not isinstance(
        value,
        list,
    ):
        raise PipelineConfigurationError(
            f"{key} must be a list."
        )

    if (
        not allow_empty
        and not value
    ):
        raise PipelineConfigurationError(
            f"{key} cannot be empty."
        )

    if any(
        not isinstance(
            item,
            str,
        )
        or not item.strip()
        for item in value
    ):
        raise PipelineConfigurationError(
            f"{key} must contain only non-empty strings."
        )

    if len(value) != len(
        set(
            value
        )
    ):
        raise PipelineConfigurationError(
            f"{key} cannot contain duplicate values."
        )

    return value


def build_parser() -> argparse.ArgumentParser:
    """Build the pipeline command-line parser."""

    parser = argparse.ArgumentParser(
        prog="python -m sgb.pipeline",
        description=(
            "Run or validate the complete SGB research pipeline."
        ),
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=(
            PIPELINE_CONFIG_PATH
        ),
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List all configured pipeline stages.",
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help=(
            "Check expected artifacts without running stages."
        ),
    )

    parser.add_argument(
        "--stage",
        action="append",
        default=None,
        help=(
            "Run a selected stage. This option may be repeated. "
            "When omitted, all stages are selected."
        ),
    )

    parser.add_argument(
        "--no-dependencies",
        action="store_true",
        help=(
            "Do not automatically include dependencies "
            "of selected stages."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Display the execution plan without running commands."
        ),
    )

    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help=(
            "Continue running later stages after a stage failure."
        ),
    )

    parser.add_argument(
        "--output-log",
        type=Path,
        default=None,
        help=(
            "Optional pipeline-log path override."
        ),
    )

    return parser


def main(
    argv: Sequence[str] | None = None,
) -> int:
    """Run the SGB research pipeline."""

    arguments = build_parser().parse_args(
        argv
    )

    try:
        config = load_pipeline_config(
            arguments.config
        )

        if arguments.list:
            for stage in config[
                "stages"
            ]:
                print(
                    f"{stage['name']}: "
                    f"{stage['description']}"
                )

            return 0

        include_dependencies = (
            not arguments.no_dependencies
        )

        if arguments.validate:
            records = validate_expected_artifacts(
                config,
                selected_stages=(
                    arguments.stage
                ),
                include_dependencies=(
                    include_dependencies
                ),
            )

            missing = [
                record
                for record in records
                if not record[
                    "exists"
                ]
            ]

            print(
                json.dumps(
                    {
                        "artifact_count": (
                            len(records)
                        ),
                        "missing_artifact_count": (
                            len(missing)
                        ),
                        "artifacts": records,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )

            return (
                0
                if not missing
                else 1
            )

        result = run_pipeline(
            config,
            selected_stages=(
                arguments.stage
            ),
            include_dependencies=(
                include_dependencies
            ),
            dry_run=(
                arguments.dry_run
            ),
            continue_on_failure=(
                arguments.continue_on_failure
            ),
        )

        log_path = (
            arguments.output_log
            if arguments.output_log
            is not None
            else config[
                "execution"
            ]["log_path"]
        )

        saved_log = save_pipeline_log(
            result,
            log_path,
        )

        print(
            json.dumps(
                {
                    "metadata": (
                        result.metadata
                    ),
                    "records": list(
                        result.records
                    ),
                    "log_path": str(
                        saved_log
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )

        return (
            0
            if result.passed
            else 1
        )

    except (
        PipelineConfigurationError,
        PipelineExecutionError,
    ) as error:
        print(
            f"Pipeline error: {error}"
        )

        return 2


if __name__ == "__main__":
    raise SystemExit(
        main()
    )