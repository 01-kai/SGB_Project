from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIRECTORY = PROJECT_ROOT / "configs"
BASE_CONFIG_PATH = CONFIG_DIRECTORY / "base.yaml"
STUDY_CONFIG_DIRECTORY = CONFIG_DIRECTORY / "studies"


EXPECTED_DIMENSION_SUBMETRICS = {
    "D1": {
        "completeness_ratio",
        "duplicate_rate",
        "schema_pass_rate",
    },
    "D2": {
        "encryption_coverage",
        "normalized_breach_latency",
        "policy_violation_rate",
    },
    "D3": {
        "lineage_rate",
        "catalog_coverage",
        "catalog_freshness",
    },
    "D4": {
        "rule_pass_rate",
        "consent_propagation_rate",
    },
    "D5": {
        "schema_reconciliation_rate",
        "api_adherence_rate",
        "format_translation_rate",
    },
    "SVC": {
        "localization_compliance_rate",
        "cross_border_flagging_rate",
        "audit_trail_completeness",
    },
}


EXPECTED_FRAMEWORK_DIMENSIONS = {
    "DBL": {"D1", "D2", "D3", "D4", "D5"},
    "DMM": {"D1", "D2", "D3", "D4", "D5"},
    "ISF": {"D1", "D2", "D3", "D4", "D5", "SVC"},
}


REQUIRED_BASELINE_SUBMETRICS = set().union(
    *EXPECTED_DIMENSION_SUBMETRICS.values()
)


class ConfigurationError(ValueError):
    """Raised when an SGB configuration is missing or invalid."""


def load_config(path: str | Path) -> dict[str, Any]:
    """
    Load and validate an SGB YAML configuration.

    Relative paths are resolved against the project root.
    """

    config_path = Path(path)

    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    config_path = config_path.resolve()

    if not config_path.exists():
        raise ConfigurationError(
            f"Configuration file does not exist: {config_path}"
        )

    if not config_path.is_file():
        raise ConfigurationError(
            f"Configuration path is not a file: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ConfigurationError(
            "The configuration root must be a YAML mapping."
        )

    validate_config(config)

    config["_metadata"] = {
        "source_path": str(config_path),
    }

    return config


def validate_config(config: dict[str, Any]) -> None:
    """Validate a base configuration or study configuration."""

    if config.get("schema_version") != 2:
        raise ConfigurationError(
            "schema_version must be 2."
        )

    config_kind = config.get("config_kind")

    if config_kind == "base":
        validate_base_config(config)
        return

    if config_kind == "study":
        validate_study_config(config)
        return

    raise ConfigurationError(
        "config_kind must be either 'base' or 'study'."
    )


def validate_base_config(config: dict[str, Any]) -> None:
    """Validate the main SGB model configuration."""

    configuration = require_mapping(config, "configuration")
    simulation = require_mapping(config, "simulation")
    network = require_mapping(config, "network")
    exchange = require_mapping(config, "exchange")
    population = require_mapping(config, "population")
    baseline = require_mapping(config, "baseline")
    metadata = require_mapping(config, "metadata")
    dimension_formulas = require_mapping(
        config,
        "dimension_formulas",
    )
    frameworks = require_mapping(config, "frameworks")
    outputs = require_mapping(config, "outputs")

    validate_configuration_metadata(configuration)

    population_size = require_positive_integer(
        simulation,
        "population_size",
    )
    require_positive_integer(
        simulation,
        "max_steps",
    )
    require_positive_integer(
        simulation,
        "seed_count",
    )

    validate_network(
        network=network,
        population_size=population_size,
    )
    validate_exchange(exchange)
    validate_population(population)
    validate_baseline(baseline)
    validate_metadata(metadata)
    validate_dimension_formulas(dimension_formulas)
    validate_frameworks(frameworks)

    root_directory = outputs.get("root_directory")

    if not isinstance(root_directory, str) or not root_directory.strip():
        raise ConfigurationError(
            "outputs.root_directory must be a non-empty string."
        )


def validate_configuration_metadata(
    configuration: dict[str, Any],
) -> None:
    status = configuration.get("status")

    if status not in {"candidate", "final"}:
        raise ConfigurationError(
            "configuration.status must be 'candidate' or 'final'."
        )

    name = configuration.get("name")

    if not isinstance(name, str) or not name.strip():
        raise ConfigurationError(
            "configuration.name must be a non-empty string."
        )

    description = configuration.get("description")

    if not isinstance(description, str) or not description.strip():
        raise ConfigurationError(
            "configuration.description must be a non-empty string."
        )

    if status == "final":
        selection_evidence = configuration.get(
            "selection_evidence"
        )

        if not isinstance(selection_evidence, dict):
            raise ConfigurationError(
                "A final configuration must contain "
                "configuration.selection_evidence."
            )


def validate_network(
    *,
    network: dict[str, Any],
    population_size: int,
) -> None:
    if network.get("model") != "barabasi_albert":
        raise ConfigurationError(
            "network.model must currently be 'barabasi_albert'."
        )

    attachment_m = require_positive_integer(
        network,
        "attachment_m",
    )

    if attachment_m >= population_size:
        raise ConfigurationError(
            "network.attachment_m must be smaller than "
            "simulation.population_size."
        )


def validate_exchange(exchange: dict[str, Any]) -> None:
    if exchange.get("volume_distribution") != "truncated_pareto":
        raise ConfigurationError(
            "exchange.volume_distribution must be "
            "'truncated_pareto'."
        )

    minimum_volume = require_positive_number(
        exchange,
        "minimum_volume",
    )
    maximum_volume = require_positive_number(
        exchange,
        "maximum_volume",
    )
    pareto_alpha = require_positive_number(
        exchange,
        "pareto_alpha",
    )

    if maximum_volume <= minimum_volume:
        raise ConfigurationError(
            "exchange.maximum_volume must be greater than "
            "exchange.minimum_volume."
        )

    if pareto_alpha <= 1.0:
        raise ConfigurationError(
            "exchange.pareto_alpha must be greater than 1.0."
        )


def validate_population(population: dict[str, Any]) -> None:
    organization_types = require_mapping(
        population,
        "organization_type_distribution",
    )

    validate_parameter_metadata(
        organization_types,
        expected_category="public_documented_source",
    )

    probabilities = require_mapping(
        organization_types,
        "probabilities",
    )

    if not probabilities:
        raise ConfigurationError(
            "Organization-type probabilities cannot be empty."
        )

    numeric_probabilities = []

    for organization_type, probability in probabilities.items():
        if not isinstance(organization_type, str) or not organization_type:
            raise ConfigurationError(
                "Organization-type names must be non-empty strings."
            )

        numeric_probability = require_normalized_value(
            probability,
            f"population probability {organization_type}",
        )

        numeric_probabilities.append(numeric_probability)

    total = sum(numeric_probabilities)

    if not math.isclose(
        total,
        1.0,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise ConfigurationError(
            "Organization-type probabilities must sum to 1.0; "
            f"received {total}."
        )


def validate_baseline(baseline: dict[str, Any]) -> None:
    target_range = require_mapping(
        baseline,
        "target_dimension_range",
    )

    target_minimum = require_normalized_number(
        target_range,
        "minimum",
    )
    target_maximum = require_normalized_number(
        target_range,
        "maximum",
    )

    if target_maximum <= target_minimum:
        raise ConfigurationError(
            "baseline target maximum must be greater than "
            "the minimum."
        )

    initialization = require_mapping(
        baseline,
        "initialization",
    )

    validate_parameter_metadata(
        initialization,
        expected_category="simulation_internal",
    )

    distributions = require_mapping(
        initialization,
        "submetric_distributions",
    )

    actual_submetrics = set(distributions)

    missing = REQUIRED_BASELINE_SUBMETRICS - actual_submetrics
    unexpected = actual_submetrics - REQUIRED_BASELINE_SUBMETRICS

    if missing:
        raise ConfigurationError(
            "Missing baseline submetric distributions: "
            + ", ".join(sorted(missing))
        )

    if unexpected:
        raise ConfigurationError(
            "Unexpected baseline submetric distributions: "
            + ", ".join(sorted(unexpected))
        )

    for submetric_name, distribution in distributions.items():
        if not isinstance(distribution, dict):
            raise ConfigurationError(
                f"Distribution for {submetric_name} must be a mapping."
            )

        require_positive_number(
            distribution,
            "alpha",
        )
        require_positive_number(
            distribution,
            "beta",
        )


def validate_metadata(metadata: dict[str, Any]) -> None:
    freshness = require_mapping(
        metadata,
        "catalog_freshness_window",
    )

    validate_parameter_metadata(
        freshness,
        expected_category="simulation_internal",
    )

    require_positive_integer(
        freshness,
        "value",
    )


def validate_dimension_formulas(
    formulas: dict[str, Any],
) -> None:
    expected_dimensions = set(
        EXPECTED_DIMENSION_SUBMETRICS
    )
    actual_dimensions = set(formulas)

    missing_dimensions = (
        expected_dimensions - actual_dimensions
    )
    unexpected_dimensions = (
        actual_dimensions - expected_dimensions
    )

    if missing_dimensions:
        raise ConfigurationError(
            "Missing dimension formulas: "
            + ", ".join(sorted(missing_dimensions))
        )

    if unexpected_dimensions:
        raise ConfigurationError(
            "Unexpected dimension formulas: "
            + ", ".join(sorted(unexpected_dimensions))
        )

    for dimension, expected_submetrics in (
        EXPECTED_DIMENSION_SUBMETRICS.items()
    ):
        formula = require_mapping(
            formulas,
            dimension,
        )

        expected_source = (
            "appendix_c"
            if dimension == "SVC"
            else "appendix_a"
        )

        if formula.get("source") != expected_source:
            raise ConfigurationError(
                f"{dimension}.source must be "
                f"{expected_source!r}."
            )

        validate_parameter_metadata(
            formula,
            expected_category="modeling_judgment",
            require_status=False,
        )

        name = formula.get("name")

        if not isinstance(name, str) or not name.strip():
            raise ConfigurationError(
                f"{dimension}.name must be a non-empty string."
            )

        weights = require_mapping(
            formula,
            "weights",
        )

        validate_weight_mapping(
            name=f"{dimension} sub-weights",
            weights=weights,
            expected_keys=expected_submetrics,
        )


def validate_frameworks(
    frameworks: dict[str, Any],
) -> None:
    expected_frameworks = set(
        EXPECTED_FRAMEWORK_DIMENSIONS
    )
    actual_frameworks = set(frameworks)

    missing = expected_frameworks - actual_frameworks
    unexpected = actual_frameworks - expected_frameworks

    if missing:
        raise ConfigurationError(
            "Missing frameworks: "
            + ", ".join(sorted(missing))
        )

    if unexpected:
        raise ConfigurationError(
            "Unexpected frameworks: "
            + ", ".join(sorted(unexpected))
        )

    for framework_name, expected_dimensions in (
        EXPECTED_FRAMEWORK_DIMENSIONS.items()
    ):
        framework = require_mapping(
            frameworks,
            framework_name,
        )

        validate_parameter_metadata(
            framework,
            expected_category="modeling_judgment",
        )

        label = framework.get("label")

        if not isinstance(label, str) or not label.strip():
            raise ConfigurationError(
                f"{framework_name}.label must be a non-empty string."
            )

        weights = require_mapping(
            framework,
            "weights",
        )

        validate_weight_mapping(
            name=f"{framework_name} framework weights",
            weights=weights,
            expected_keys=expected_dimensions,
        )


def validate_parameter_metadata(
    parameter: dict[str, Any],
    *,
    expected_category: str,
    require_status: bool = True,
) -> None:
    if parameter.get("parameter_category") != expected_category:
        raise ConfigurationError(
            "parameter_category must be "
            f"{expected_category!r}."
        )

    if require_status:
        status = parameter.get("status")

        if status not in {"candidate", "final"}:
            raise ConfigurationError(
                "Parameter status must be 'candidate' or 'final'."
            )

    rationale = parameter.get("rationale")

    if not isinstance(rationale, str) or not rationale.strip():
        raise ConfigurationError(
            "Parameter rationale must be a non-empty string."
        )


def validate_weight_mapping(
    *,
    name: str,
    weights: dict[str, Any],
    expected_keys: set[str],
) -> None:
    actual_keys = set(weights)

    missing = expected_keys - actual_keys
    unexpected = actual_keys - expected_keys

    if missing or unexpected:
        details = []

        if missing:
            details.append(
                "missing=" + ",".join(sorted(missing))
            )

        if unexpected:
            details.append(
                "unexpected=" + ",".join(sorted(unexpected))
            )

        raise ConfigurationError(
            f"Invalid {name}: " + "; ".join(details)
        )

    numeric_weights = []

    for key, weight in weights.items():
        numeric_weight = require_normalized_value(
            weight,
            f"{name}.{key}",
        )

        numeric_weights.append(numeric_weight)

    total = sum(numeric_weights)

    if not math.isclose(
        total,
        1.0,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise ConfigurationError(
            f"{name} must sum to 1.0; received {total}."
        )


def validate_study_config(config: dict[str, Any]) -> None:
    """Validate convergence or sensitivity study configuration."""

    study_type = config.get("study_type")

    supported_studies = {
        "seed_convergence",
        "time_convergence",
        "population_convergence",
        "sensitivity",
    }

    if study_type not in supported_studies:
        raise ConfigurationError(
            f"Unsupported study_type: {study_type!r}."
        )

    reference_config = config.get("reference_config")

    if not isinstance(reference_config, str) or not reference_config.strip():
        raise ConfigurationError(
            "reference_config must be a non-empty string."
        )

    outputs = require_mapping(config, "outputs")
    output_directory = outputs.get("directory")

    if (
        not isinstance(output_directory, str)
        or not output_directory.strip()
    ):
        raise ConfigurationError(
            "outputs.directory must be a non-empty string."
        )

    if study_type == "seed_convergence":
        require_sorted_unique_positive_integers(
            config,
            "candidate_seed_counts",
        )
        require_metric_list(config)

    elif study_type == "time_convergence":
        maximum_horizon = require_positive_integer(
            config,
            "maximum_horizon",
        )

        candidate_horizons = (
            require_sorted_unique_positive_integers(
                config,
                "candidate_horizons",
            )
        )

        if maximum_horizon < max(candidate_horizons):
            raise ConfigurationError(
                "maximum_horizon cannot be smaller than "
                "the largest candidate horizon."
            )

        require_metric_list(config)

        detection = require_mapping(
            config,
            "detection",
        )

        rolling_window = require_positive_integer(
            detection,
            "rolling_window",
        )

        require_positive_integer(
            detection,
            "required_stable_windows",
        )

        if rolling_window >= maximum_horizon:
            raise ConfigurationError(
                "detection.rolling_window must be smaller "
                "than maximum_horizon."
            )

        require_positive_number(
            detection,
            "slope_tolerance",
            allow_zero=True,
        )
        require_positive_number(
            detection,
            "mean_difference_tolerance",
            allow_zero=True,
        )

    elif study_type == "population_convergence":
        require_sorted_unique_positive_integers(
            config,
            "candidate_population_sizes",
        )
        require_metric_list(config)

    elif study_type == "sensitivity":
        parameters = require_mapping(
            config,
            "parameters",
        )

        if not parameters:
            raise ConfigurationError(
                "Sensitivity parameters cannot be empty."
            )

        for parameter_name, candidate_values in parameters.items():
            if (
                not isinstance(parameter_name, str)
                or not parameter_name
            ):
                raise ConfigurationError(
                    "Sensitivity parameter names must be "
                    "non-empty strings."
                )

            if (
                not isinstance(candidate_values, list)
                or not candidate_values
            ):
                raise ConfigurationError(
                    f"Sensitivity parameter {parameter_name!r} "
                    "must contain a non-empty list."
                )

            if len(candidate_values) != len(
                {repr(value) for value in candidate_values}
            ):
                raise ConfigurationError(
                    f"Sensitivity parameter {parameter_name!r} "
                    "contains duplicate values."
                )


def require_mapping(
    container: dict[str, Any],
    key: str,
) -> dict[str, Any]:
    value = container.get(key)

    if not isinstance(value, dict):
        raise ConfigurationError(
            f"{key} must be a mapping."
        )

    return value


def require_positive_integer(
    container: dict[str, Any],
    key: str,
) -> int:
    value = container.get(key)

    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(
            f"{key} must be an integer."
        )

    if value <= 0:
        raise ConfigurationError(
            f"{key} must be greater than zero."
        )

    return value


def require_positive_number(
    container: dict[str, Any],
    key: str,
    *,
    allow_zero: bool = False,
) -> float:
    value = container.get(key)

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
    ):
        raise ConfigurationError(
            f"{key} must be numeric."
        )

    numeric_value = float(value)

    if allow_zero:
        valid = numeric_value >= 0.0
    else:
        valid = numeric_value > 0.0

    if not valid:
        comparison = (
            "zero or greater"
            if allow_zero
            else "greater than zero"
        )

        raise ConfigurationError(
            f"{key} must be {comparison}."
        )

    return numeric_value


def require_normalized_number(
    container: dict[str, Any],
    key: str,
) -> float:
    return require_normalized_value(
        container.get(key),
        key,
    )


def require_normalized_value(
    value: Any,
    name: str,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
    ):
        raise ConfigurationError(
            f"{name} must be numeric."
        )

    numeric_value = float(value)

    if not 0.0 <= numeric_value <= 1.0:
        raise ConfigurationError(
            f"{name} must be within [0, 1]."
        )

    return numeric_value


def require_sorted_unique_positive_integers(
    container: dict[str, Any],
    key: str,
) -> list[int]:
    values = container.get(key)

    if not isinstance(values, list) or not values:
        raise ConfigurationError(
            f"{key} must be a non-empty list."
        )

    for value in values:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
        ):
            raise ConfigurationError(
                f"Every value in {key} must be a positive integer."
            )

    if values != sorted(set(values)):
        raise ConfigurationError(
            f"{key} must be sorted and contain no duplicates."
        )

    return values


def require_metric_list(
    config: dict[str, Any],
) -> list[str]:
    metrics = config.get("metrics")

    if not isinstance(metrics, list) or not metrics:
        raise ConfigurationError(
            "metrics must be a non-empty list."
        )

    if any(
        not isinstance(metric, str) or not metric.strip()
        for metric in metrics
    ):
        raise ConfigurationError(
            "Every metric name must be a non-empty string."
        )

    if len(metrics) != len(set(metrics)):
        raise ConfigurationError(
            "metrics cannot contain duplicate values."
        )

    return metrics