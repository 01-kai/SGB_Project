from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from sgb.config import PROJECT_ROOT


REPORTING_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "reporting.yaml"
)


class ReportingConfigurationError(ValueError):
    """Raised when the reporting configuration is invalid."""


class ReportingError(RuntimeError):
    """Raised when the final report cannot be generated."""


@dataclass(frozen=True, slots=True)
class FinalReportResult:
    """Complete output of the final reporting process."""

    artifact_index: pd.DataFrame
    decision_summary: pd.DataFrame
    parameter_coverage: pd.DataFrame
    reproducibility_manifest: pd.DataFrame
    report_markdown: str
    metadata: dict[str, Any]

    @property
    def passed(self) -> bool:
        """Return whether all required reporting inputs were available."""

        return bool(
            self.metadata.get(
                "passed",
                False,
            )
        )


def load_reporting_config(
    path: str | Path = REPORTING_CONFIG_PATH,
) -> dict[str, Any]:
    """Load and validate the final-report configuration."""

    config_path = Path(path)

    if not config_path.is_absolute():
        config_path = (
            PROJECT_ROOT
            / config_path
        )

    config_path = config_path.resolve()

    if not config_path.is_file():
        raise ReportingConfigurationError(
            "Reporting configuration does not exist: "
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
        raise ReportingConfigurationError(
            "Reporting configuration root must be a mapping."
        )

    validate_reporting_config(
        config
    )

    config["_metadata"] = {
        "source_path": str(
            config_path
        ),
    }

    return config


def validate_reporting_config(
    config: Mapping[str, Any],
) -> None:
    """Validate the complete reporting configuration."""

    if config.get(
        "schema_version"
    ) != 1:
        raise ReportingConfigurationError(
            "schema_version must be 1."
        )

    if config.get(
        "config_kind"
    ) != "reporting":
        raise ReportingConfigurationError(
            "config_kind must be 'reporting'."
        )

    project = _require_mapping(
        config,
        "project",
    )

    _require_non_empty_string(
        project,
        "title",
    )
    _require_non_empty_string(
        project,
        "short_name",
    )
    _require_non_empty_string(
        project,
        "version",
    )

    strict = config.get(
        "strict_required_inputs"
    )

    if not isinstance(
        strict,
        bool,
    ):
        raise ReportingConfigurationError(
            "strict_required_inputs must be Boolean."
        )

    inputs = _require_mapping(
        config,
        "inputs",
    )

    if not inputs:
        raise ReportingConfigurationError(
            "inputs cannot be empty."
        )

    for input_name, specification in inputs.items():
        if (
            not isinstance(
                input_name,
                str,
            )
            or not input_name.strip()
        ):
            raise ReportingConfigurationError(
                "Input names must be non-empty strings."
            )

        if not isinstance(
            specification,
            Mapping,
        ):
            raise ReportingConfigurationError(
                f"Input {input_name!r} must be a mapping."
            )

        _require_non_empty_string(
            specification,
            "path",
        )

        kind = _require_non_empty_string(
            specification,
            "kind",
        )

        if kind not in {
            "csv",
            "json",
        }:
            raise ReportingConfigurationError(
                f"Input {input_name!r} kind must be "
                "'csv' or 'json'."
            )

        required = specification.get(
            "required"
        )

        if not isinstance(
            required,
            bool,
        ):
            raise ReportingConfigurationError(
                f"Input {input_name!r}.required "
                "must be Boolean."
            )

        _require_non_empty_string(
            specification,
            "role",
        )

    reproducibility = _require_mapping(
        config,
        "reproducibility",
    )

    _require_string_list(
        reproducibility,
        "hash_roots",
    )
    _require_string_list(
        reproducibility,
        "hash_files",
    )
    _require_string_list(
        reproducibility,
        "include_extensions",
    )

    outputs = _require_mapping(
        config,
        "outputs",
    )

    for key in {
        "directory",
        "report_markdown",
        "artifact_index",
        "decision_summary",
        "parameter_coverage",
        "reproducibility_manifest",
        "metadata",
    }:
        _require_non_empty_string(
            outputs,
            key,
        )


def build_final_report(
    reporting_config: Mapping[str, Any],
    *,
    project_root: str | Path = PROJECT_ROOT,
) -> FinalReportResult:
    """Build the final report from existing study outputs."""

    validate_reporting_config(
        reporting_config
    )

    root = Path(
        project_root
    ).resolve()

    loaded_inputs: dict[
        str,
        pd.DataFrame | dict[str, Any],
    ] = {}

    artifact_rows: list[
        dict[str, Any]
    ] = []

    missing_required: list[
        str
    ] = []

    for input_name, specification in (
        reporting_config[
            "inputs"
        ].items()
    ):
        relative_path = Path(
            specification[
                "path"
            ]
        )

        input_path = (
            relative_path
            if relative_path.is_absolute()
            else root / relative_path
        ).resolve()

        exists = input_path.is_file()
        required = bool(
            specification[
                "required"
            ]
        )
        kind = str(
            specification[
                "kind"
            ]
        )

        if exists:
            loaded_inputs[
                input_name
            ] = _load_input(
                input_path,
                kind,
            )

            size_bytes: int | None = (
                input_path.stat().st_size
            )
            digest: str | None = (
                calculate_sha256(
                    input_path
                )
            )
        else:
            size_bytes = None
            digest = None

            if required:
                missing_required.append(
                    input_name
                )

        artifact_rows.append(
            {
                "input_name": (
                    input_name
                ),
                "role": str(
                    specification[
                        "role"
                    ]
                ),
                "kind": kind,
                "required": required,
                "relative_path": (
                    _relative_display_path(
                        input_path,
                        root,
                    )
                ),
                "exists": exists,
                "size_bytes": (
                    size_bytes
                ),
                "sha256": digest,
            }
        )

    strict = bool(
        reporting_config[
            "strict_required_inputs"
        ]
    )

    if strict and missing_required:
        raise ReportingError(
            "Required reporting inputs are missing: "
            + ", ".join(
                sorted(
                    missing_required
                )
            )
        )

    artifact_index = pd.DataFrame(
        artifact_rows
    )

    decision_summary = (
        _build_decision_summary(
            reporting_config=(
                reporting_config
            ),
            loaded_inputs=(
                loaded_inputs
            ),
        )
    )

    parameter_coverage = (
        _build_parameter_coverage(
            loaded_inputs
        )
    )

    reproducibility_manifest = (
        build_reproducibility_manifest(
            reporting_config=(
                reporting_config
            ),
            project_root=root,
        )
    )

    report_markdown = (
        _build_markdown_report(
            reporting_config=(
                reporting_config
            ),
            loaded_inputs=(
                loaded_inputs
            ),
            artifact_index=(
                artifact_index
            ),
            decision_summary=(
                decision_summary
            ),
            parameter_coverage=(
                parameter_coverage
            ),
            reproducibility_manifest=(
                reproducibility_manifest
            ),
            missing_required=(
                missing_required
            ),
        )
    )

    all_required_available = (
        len(
            missing_required
        )
        == 0
    )

    failed_decisions = int(
        (
            decision_summary[
                "status"
            ]
            == "not_passed"
        ).sum()
    ) if not decision_summary.empty else 0

    pending_decisions = int(
        (
            decision_summary[
                "status"
            ]
            == "pending"
        ).sum()
    ) if not decision_summary.empty else 0

    metadata = {
        "project": (
            reporting_config[
                "project"
            ]["short_name"]
        ),
        "version": (
            reporting_config[
                "project"
            ]["version"]
        ),
        "strict_required_inputs": (
            strict
        ),
        "required_input_count": int(
            artifact_index[
                "required"
            ].sum()
        ),
        "available_required_input_count": int(
            artifact_index.loc[
                artifact_index[
                    "required"
                ],
                "exists",
            ].sum()
        ),
        "missing_required_inputs": (
            sorted(
                missing_required
            )
        ),
        "failed_decision_count": (
            failed_decisions
        ),
        "pending_decision_count": (
            pending_decisions
        ),
        "source_file_count": int(
            len(
                reproducibility_manifest
            )
        ),
        "passed": (
            all_required_available
        ),
    }

    return FinalReportResult(
        artifact_index=(
            artifact_index
        ),
        decision_summary=(
            decision_summary
        ),
        parameter_coverage=(
            parameter_coverage
        ),
        reproducibility_manifest=(
            reproducibility_manifest
        ),
        report_markdown=(
            report_markdown
        ),
        metadata=metadata,
    )


def build_reproducibility_manifest(
    reporting_config: Mapping[str, Any],
    *,
    project_root: str | Path = PROJECT_ROOT,
) -> pd.DataFrame:
    """Hash configured source, test, and configuration files."""

    validate_reporting_config(
        reporting_config
    )

    root = Path(
        project_root
    ).resolve()

    reproducibility = reporting_config[
        "reproducibility"
    ]

    extensions = {
        str(extension).lower()
        for extension in reproducibility[
            "include_extensions"
        ]
    }

    collected_files: set[
        Path
    ] = set()

    for relative_root in reproducibility[
        "hash_roots"
    ]:
        hash_root = (
            root
            / relative_root
        ).resolve()

        if not hash_root.exists():
            continue

        if hash_root.is_file():
            if (
                not extensions
                or hash_root.suffix.lower()
                in extensions
            ):
                collected_files.add(
                    hash_root
                )

            continue

        for candidate in hash_root.rglob(
            "*"
        ):
            if (
                candidate.is_file()
                and (
                    not extensions
                    or candidate.suffix.lower()
                    in extensions
                )
                and "__pycache__"
                not in candidate.parts
                and ".pytest_cache"
                not in candidate.parts
            ):
                collected_files.add(
                    candidate.resolve()
                )

    for relative_file in reproducibility[
        "hash_files"
    ]:
        candidate = (
            root
            / relative_file
        ).resolve()

        if candidate.is_file():
            collected_files.add(
                candidate
            )

    rows = [
        {
            "relative_path": (
                _relative_display_path(
                    path,
                    root,
                )
            ),
            "size_bytes": (
                path.stat().st_size
            ),
            "sha256": (
                calculate_sha256(
                    path
                )
            ),
        }
        for path in sorted(
            collected_files,
            key=lambda item: str(
                item
            ),
        )
    ]

    return pd.DataFrame(
        rows,
        columns=[
            "relative_path",
            "size_bytes",
            "sha256",
        ],
    )


def calculate_sha256(
    path: str | Path,
) -> str:
    """Calculate the SHA-256 digest of one file."""

    file_path = Path(
        path
    )

    if not file_path.is_file():
        raise ReportingError(
            f"Cannot hash missing file: {file_path}"
        )

    digest = hashlib.sha256()

    with file_path.open(
        "rb"
    ) as file:
        while True:
            block = file.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(
                block
            )

    return digest.hexdigest()


def save_final_report(
    result: FinalReportResult,
    reporting_config: Mapping[str, Any],
    *,
    project_root: str | Path = PROJECT_ROOT,
    output_directory: str | Path | None = None,
) -> dict[str, Path]:
    """Save the final report package."""

    if not isinstance(
        result,
        FinalReportResult,
    ):
        raise ReportingError(
            "result must be a FinalReportResult."
        )

    validate_reporting_config(
        reporting_config
    )

    root = Path(
        project_root
    ).resolve()

    outputs = reporting_config[
        "outputs"
    ]

    configured_directory = (
        output_directory
        if output_directory
        is not None
        else outputs[
            "directory"
        ]
    )

    output_path = Path(
        configured_directory
    )

    if not output_path.is_absolute():
        output_path = (
            root
            / output_path
        )

    output_path = (
        output_path.resolve()
    )

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = {
        "report_markdown": (
            output_path
            / outputs[
                "report_markdown"
            ]
        ),
        "artifact_index": (
            output_path
            / outputs[
                "artifact_index"
            ]
        ),
        "decision_summary": (
            output_path
            / outputs[
                "decision_summary"
            ]
        ),
        "parameter_coverage": (
            output_path
            / outputs[
                "parameter_coverage"
            ]
        ),
        "reproducibility_manifest": (
            output_path
            / outputs[
                "reproducibility_manifest"
            ]
        ),
        "metadata": (
            output_path
            / outputs[
                "metadata"
            ]
        ),
    }

    paths[
        "report_markdown"
    ].write_text(
        result.report_markdown,
        encoding="utf-8",
    )

    result.artifact_index.to_csv(
        paths[
            "artifact_index"
        ],
        index=False,
    )

    result.decision_summary.to_csv(
        paths[
            "decision_summary"
        ],
        index=False,
    )

    result.parameter_coverage.to_csv(
        paths[
            "parameter_coverage"
        ],
        index=False,
    )

    result.reproducibility_manifest.to_csv(
        paths[
            "reproducibility_manifest"
        ],
        index=False,
    )

    with paths[
        "metadata"
    ].open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result.metadata,
            file,
            indent=2,
            sort_keys=True,
        )

    return paths


def _load_input(
    path: Path,
    kind: str,
) -> pd.DataFrame | dict[str, Any]:
    if kind == "csv":
        try:
            return pd.read_csv(
                path
            )
        except Exception as error:
            raise ReportingError(
                f"Unable to read CSV input: {path}"
            ) from error

    if kind == "json":
        try:
            with path.open(
                "r",
                encoding="utf-8",
            ) as file:
                payload = json.load(
                    file
                )
        except Exception as error:
            raise ReportingError(
                f"Unable to read JSON input: {path}"
            ) from error

        if not isinstance(
            payload,
            dict,
        ):
            raise ReportingError(
                f"JSON input must contain an object: {path}"
            )

        return payload

    raise ReportingError(
        f"Unsupported input kind: {kind!r}."
    )


def _build_decision_summary(
    *,
    reporting_config: Mapping[str, Any],
    loaded_inputs: Mapping[
        str,
        pd.DataFrame | dict[str, Any],
    ],
) -> pd.DataFrame:
    rows: list[
        dict[str, Any]
    ] = []

    for input_name, specification in (
        reporting_config[
            "inputs"
        ].items()
    ):
        if specification[
            "kind"
        ] != "json":
            continue

        payload = loaded_inputs.get(
            input_name
        )

        if payload is None:
            rows.append(
                {
                    "input_name": (
                        input_name
                    ),
                    "role": (
                        specification[
                            "role"
                        ]
                    ),
                    "status": "missing",
                    "passed": None,
                    "decision": (
                        "Metadata file is unavailable."
                    ),
                }
            )
            continue

        if not isinstance(
            payload,
            dict,
        ):
            raise ReportingError(
                f"Loaded JSON input {input_name!r} "
                "is not a mapping."
            )

        passed = payload.get(
            "passed"
        )

        if passed is True:
            status = "passed"
        elif passed is False:
            status = "not_passed"
        else:
            status = "informational"

        decision = _describe_decision(
            input_name,
            payload,
        )

        if (
            input_name
            == "sensitivity_metadata"
            and payload.get(
                "P9_public_calibration_complete"
            )
            is False
        ):
            status = "pending"

        rows.append(
            {
                "input_name": (
                    input_name
                ),
                "role": (
                    specification[
                        "role"
                    ]
                ),
                "status": status,
                "passed": passed,
                "decision": decision,
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "input_name",
            "role",
            "status",
            "passed",
            "decision",
        ],
    )


def _describe_decision(
    input_name: str,
    payload: Mapping[str, Any],
) -> str:
    if (
        "recommended_horizon"
        in payload
    ):
        return (
            "Recommended horizon: "
            f"{payload['recommended_horizon']}"
        )

    if (
        "recommended_seed_count"
        in payload
    ):
        return (
            "Recommended seed count: "
            f"{payload['recommended_seed_count']}"
        )

    if (
        "recommended_population_size"
        in payload
    ):
        return (
            "Recommended population size: "
            f"{payload['recommended_population_size']}"
        )

    if (
        "overall_ranking"
        in payload
    ):
        return (
            "Framework ranking: "
            f"{payload['overall_ranking']}"
        )

    if (
        "all_framework_runs_recovered"
        in payload
    ):
        return (
            "All framework views recovered: "
            f"{payload['all_framework_runs_recovered']}"
        )

    if (
        "coarse_full_range_scanned"
        in payload
    ):
        return (
            "Full [0,1] threshold range scanned: "
            f"{payload['coarse_full_range_scanned']}"
        )

    if (
        "low_endpoint_passed"
        in payload
        or "high_endpoint_passed"
        in payload
    ):
        return (
            "Low endpoint passed: "
            f"{payload.get('low_endpoint_passed')}; "
            "high endpoint passed: "
            f"{payload.get('high_endpoint_passed')}"
        )

    if (
        input_name
        == "sensitivity_metadata"
    ):
        return (
            "Simulation stability passed: "
            f"{payload.get('simulation_stability_passed')}; "
            "P9 public calibration complete: "
            f"{payload.get('P9_public_calibration_complete')}"
        )

    if (
        "dimension_acceptance_passed"
        in payload
    ):
        return (
            "Dimension acceptance passed: "
            f"{payload.get('dimension_acceptance_passed')}; "
            "submetric acceptance passed: "
            f"{payload.get('submetric_acceptance_passed')}"
        )

    if "passed" in payload:
        return (
            "Reported decision: "
            f"{payload['passed']}"
        )

    return "See source metadata for details."


def _build_parameter_coverage(
    loaded_inputs: Mapping[
        str,
        pd.DataFrame | dict[str, Any],
    ],
) -> pd.DataFrame:
    coverage = loaded_inputs.get(
        "sensitivity_coverage"
    )

    if coverage is None:
        return pd.DataFrame(
            columns=[
                "code",
                "category",
                "implementation",
                "config",
                "resolution_status",
                "final_status",
            ]
        )

    if not isinstance(
        coverage,
        pd.DataFrame,
    ):
        raise ReportingError(
            "sensitivity_coverage must be a CSV table."
        )

    required_columns = {
        "code",
        "category",
        "implementation",
        "config",
        "resolution_status",
    }

    missing = (
        required_columns
        - set(
            coverage.columns
        )
    )

    if missing:
        raise ReportingError(
            "Sensitivity coverage is missing columns: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    result = coverage.copy(
        deep=True
    )

    result[
        "final_status"
    ] = result[
        "resolution_status"
    ].astype(
        str
    ).map(
        lambda status: (
            "pending"
            if "pending"
            in status.lower()
            else "implemented"
        )
    )

    return result


def _build_markdown_report(
    *,
    reporting_config: Mapping[str, Any],
    loaded_inputs: Mapping[
        str,
        pd.DataFrame | dict[str, Any],
    ],
    artifact_index: pd.DataFrame,
    decision_summary: pd.DataFrame,
    parameter_coverage: pd.DataFrame,
    reproducibility_manifest: pd.DataFrame,
    missing_required: Sequence[str],
) -> str:
    project = reporting_config[
        "project"
    ]

    sections = [
        f"# {project['title']}",
        "",
        "## Final Computational Report",
        "",
        (
            f"**Project:** {project['short_name']}  \n"
            f"**Version:** {project['version']}"
        ),
        "",
        (
            "This report consolidates calibration, convergence, "
            "framework comparison, perturbation, threshold-selection, "
            "maturity-sweep, and sensitivity-analysis outputs."
        ),
        "",
        "## 1. Artifact Availability",
        "",
        _markdown_table(
            artifact_index,
            columns=[
                "input_name",
                "role",
                "required",
                "exists",
                "relative_path",
            ],
        ),
        "",
        "## 2. Study Decisions",
        "",
        _markdown_table(
            decision_summary,
            columns=[
                "role",
                "status",
                "decision",
            ],
        ),
        "",
    ]

    baseline_table = loaded_inputs.get(
        "baseline_table_1"
    )

    sections.extend(
        [
            "## 3. Baseline Calibration",
            "",
        ]
    )

    if isinstance(
        baseline_table,
        pd.DataFrame,
    ):
        sections.append(
            _markdown_table(
                baseline_table,
                max_rows=20,
            )
        )
    else:
        sections.append(
            "Baseline Table 1 was not available."
        )

    sections.extend(
        [
            "",
            "## 4. Parameter Resolution Matrix",
            "",
            _markdown_table(
                parameter_coverage,
                columns=[
                    "code",
                    "category",
                    "implementation",
                    "resolution_status",
                    "final_status",
                ],
                max_rows=30,
            ),
            "",
            "## 5. Principal Experiment Results",
            "",
        ]
    )

    summary_inputs = [
        (
            "experiment_1_summary",
            "Experiment 1 — Framework comparison",
        ),
        (
            "experiment_2_summary",
            "Experiment 2 — Security recovery",
        ),
        (
            "experiment_3_summary",
            "Experiment 3 — Threshold recommendations",
        ),
        (
            "experiment_4_summary",
            "Experiment 4 — Maturity sweep",
        ),
        (
            "sensitivity_stability",
            "Global sensitivity stability",
        ),
    ]

    for input_name, heading in summary_inputs:
        sections.extend(
            [
                f"### {heading}",
                "",
            ]
        )

        table = loaded_inputs.get(
            input_name
        )

        if isinstance(
            table,
            pd.DataFrame,
        ):
            sections.append(
                _markdown_table(
                    table,
                    max_rows=25,
                )
            )
        else:
            sections.append(
                "The corresponding output table was not available."
            )

        sections.append(
            ""
        )

    sections.extend(
        [
            "## 6. Reproducibility",
            "",
            (
                "The reproducibility manifest contains SHA-256 hashes "
                "for source code, tests, configuration files, dependency "
                "files, and the report inputs."
            ),
            "",
            (
                f"Number of hashed source and configuration files: "
                f"**{len(reproducibility_manifest)}**"
            ),
            "",
            "## 7. Remaining Limitations",
            "",
            (
                "- Framework and dimension weights remain modeling "
                "choices and must be interpreted through the sensitivity "
                "results rather than as empirically estimated coefficients."
            ),
            (
                "- Equal-width maturity categories are definitional "
                "partitions rather than externally validated maturity "
                "boundaries."
            ),
            (
                "- P9 remains incomplete until organization-type "
                "proportions are calibrated against a documented public "
                "source and denominator."
            ),
            (
                "- Synthetic results demonstrate computational behavior "
                "and robustness; they do not independently establish "
                "real-world causal validity."
            ),
            "",
        ]
    )

    if missing_required:
        sections.extend(
            [
                "## 8. Missing Required Inputs",
                "",
                *[
                    f"- `{input_name}`"
                    for input_name
                    in sorted(
                        missing_required
                    )
                ],
                "",
            ]
        )

    return "\n".join(
        sections
    ).strip() + "\n"


def _markdown_table(
    table: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    max_rows: int = 50,
) -> str:
    if table.empty:
        return "_No rows available._"

    selected = table.copy(
        deep=True
    )

    if columns is not None:
        existing_columns = [
            column
            for column in columns
            if column in selected.columns
        ]

        selected = selected[
            existing_columns
        ]

    selected = selected.head(
        max_rows
    )

    headers = [
        str(column)
        for column in selected.columns
    ]

    lines = [
        "| "
        + " | ".join(
            _escape_markdown_cell(
                header
            )
            for header in headers
        )
        + " |",
        "| "
        + " | ".join(
            "---"
            for _ in headers
        )
        + " |",
    ]

    for _, row in selected.iterrows():
        lines.append(
            "| "
            + " | ".join(
                _format_markdown_value(
                    row[
                        column
                    ]
                )
                for column
                in selected.columns
            )
            + " |"
        )

    if len(
        table
    ) > max_rows:
        lines.append(
            f"\n_Only the first {max_rows} of "
            f"{len(table)} rows are shown._"
        )

    return "\n".join(
        lines
    )


def _format_markdown_value(
    value: Any,
) -> str:
    if pd.isna(
        value
    ):
        return ""

    if isinstance(
        value,
        float,
    ):
        return (
            f"{value:.6f}"
        ).rstrip(
            "0"
        ).rstrip(
            "."
        )

    return _escape_markdown_cell(
        str(
            value
        )
    )


def _escape_markdown_cell(
    value: str,
) -> str:
    return (
        value
        .replace(
            "|",
            "\\|",
        )
        .replace(
            "\n",
            " ",
        )
    )


def _relative_display_path(
    path: Path,
    root: Path,
) -> str:
    try:
        return str(
            path.relative_to(
                root
            )
        )
    except ValueError:
        return str(
            path
        )


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
        raise ReportingConfigurationError(
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
        raise ReportingConfigurationError(
            f"{key} must be a non-empty string."
        )

    return value.strip()


def _require_string_list(
    container: Mapping[str, Any],
    key: str,
) -> list[str]:
    value = container.get(
        key
    )

    if (
        not isinstance(
            value,
            list,
        )
        or any(
            not isinstance(
                item,
                str,
            )
            or not item.strip()
            for item in value
        )
    ):
        raise ReportingConfigurationError(
            f"{key} must be a list of non-empty strings."
        )

    if len(
        value
    ) != len(
        set(
            value
        )
    ):
        raise ReportingConfigurationError(
            f"{key} cannot contain duplicates."
        )

    return value


def build_parser() -> argparse.ArgumentParser:
    """Build the final-report command-line parser."""

    parser = argparse.ArgumentParser(
        prog="python -m sgb.reporting",
        description=(
            "Generate the final SGB computational report."
        ),
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=(
            REPORTING_CONFIG_PATH
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help=(
            "Generate a partial report even when required "
            "study outputs are missing."
        ),
    )

    return parser


def main(
    argv: Sequence[str] | None = None,
) -> int:
    """Generate the final research report."""

    arguments = build_parser().parse_args(
        argv
    )

    try:
        reporting_config = (
            load_reporting_config(
                arguments.config
            )
        )

        if arguments.allow_missing:
            reporting_config = dict(
                reporting_config
            )

            reporting_config[
                "strict_required_inputs"
            ] = False

        result = build_final_report(
            reporting_config
        )

        paths = save_final_report(
            result=result,
            reporting_config=(
                reporting_config
            ),
            output_directory=(
                arguments.output
            ),
        )

        print(
            json.dumps(
                result.metadata,
                indent=2,
                sort_keys=True,
            )
        )

        print(
            "Final report: "
            f"{paths['report_markdown']}"
        )

        return 0

    except (
        ReportingConfigurationError,
        ReportingError,
    ) as error:
        print(
            f"Reporting error: {error}"
        )

        return 2


if __name__ == "__main__":
    raise SystemExit(
        main()
    )