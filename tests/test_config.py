from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from sgb.config import (
    BASE_CONFIG_PATH,
    ConfigurationError,
    load_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def write_yaml(
    path: Path,
    data: dict,
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            data,
            file,
            sort_keys=False,
        )


def load_clean_base_config() -> dict:
    config = load_config(BASE_CONFIG_PATH)
    clean_config = deepcopy(config)
    clean_config.pop("_metadata", None)
    return clean_config


def test_base_configuration_is_valid() -> None:
    config = load_config(BASE_CONFIG_PATH)

    assert config["config_kind"] == "base"
    assert config["configuration"]["status"] == "candidate"


@pytest.mark.parametrize(
    "relative_path, expected_study_type",
    [
        (
            "configs/studies/seed_convergence.yaml",
            "seed_convergence",
        ),
        (
            "configs/studies/time_convergence.yaml",
            "time_convergence",
        ),
        (
            "configs/studies/population_convergence.yaml",
            "population_convergence",
        ),
        (
            "configs/studies/sensitivity.yaml",
            "sensitivity",
        ),
    ],
)
def test_study_configuration_is_valid(
    relative_path: str,
    expected_study_type: str,
) -> None:
    config = load_config(
        PROJECT_ROOT / relative_path
    )

    assert config["config_kind"] == "study"
    assert config["study_type"] == expected_study_type


def test_attachment_m_must_be_smaller_than_population(
    tmp_path: Path,
) -> None:
    config = load_clean_base_config()

    config["network"]["attachment_m"] = (
        config["simulation"]["population_size"]
    )

    invalid_path = (
        tmp_path / "invalid_attachment.yaml"
    )
    write_yaml(invalid_path, config)

    with pytest.raises(
        ConfigurationError,
        match="attachment_m",
    ):
        load_config(invalid_path)


def test_final_configuration_requires_selection_evidence(
    tmp_path: Path,
) -> None:
    config = load_clean_base_config()

    config["configuration"]["status"] = "final"
    config["configuration"].pop(
        "selection_evidence",
        None,
    )

    invalid_path = (
        tmp_path / "invalid_final.yaml"
    )
    write_yaml(invalid_path, config)

    with pytest.raises(
        ConfigurationError,
        match="selection_evidence",
    ):
        load_config(invalid_path)


def test_pareto_alpha_must_be_greater_than_one(
    tmp_path: Path,
) -> None:
    config = load_clean_base_config()
    config["exchange"]["pareto_alpha"] = 1.0

    invalid_path = (
        tmp_path / "invalid_pareto.yaml"
    )
    write_yaml(invalid_path, config)

    with pytest.raises(
        ConfigurationError,
        match="pareto_alpha",
    ):
        load_config(invalid_path)


def test_target_dimension_range_is_point_four_to_point_five() -> None:
    config = load_config(BASE_CONFIG_PATH)
    target_range = config[
        "baseline"
    ]["target_dimension_range"]

    assert target_range["minimum"] == pytest.approx(
        0.40
    )
    assert target_range["maximum"] == pytest.approx(
        0.50
    )


@pytest.mark.parametrize(
    "dimension",
    ["D1", "D2", "D3", "D4", "D5", "SVC"],
)
def test_dimension_subweights_sum_to_one(
    dimension: str,
) -> None:
    config = load_config(BASE_CONFIG_PATH)
    weights = config[
        "dimension_formulas"
    ][dimension]["weights"]

    assert sum(weights.values()) == pytest.approx(
        1.0
    )


@pytest.mark.parametrize(
    "framework",
    ["DBL", "DMM", "ISF"],
)
def test_framework_weights_sum_to_one(
    framework: str,
) -> None:
    config = load_config(BASE_CONFIG_PATH)
    weights = config[
        "frameworks"
    ][framework]["weights"]

    assert sum(weights.values()) == pytest.approx(
        1.0
    )


def test_organization_type_distribution_sums_to_one() -> None:
    config = load_config(BASE_CONFIG_PATH)

    probabilities = config[
        "population"
    ]["organization_type_distribution"]["probabilities"]

    assert sum(
        probabilities.values()
    ) == pytest.approx(1.0)


def test_all_submetric_distributions_are_present() -> None:
    config = load_config(BASE_CONFIG_PATH)

    distributions = config[
        "baseline"
    ]["initialization"]["submetric_distributions"]

    expected_names = {
        "completeness_ratio",
        "duplicate_rate",
        "schema_pass_rate",
        "encryption_coverage",
        "normalized_breach_latency",
        "policy_violation_rate",
        "lineage_rate",
        "catalog_coverage",
        "catalog_freshness",
        "rule_pass_rate",
        "consent_propagation_rate",
        "schema_reconciliation_rate",
        "api_adherence_rate",
        "format_translation_rate",
        "localization_compliance_rate",
        "cross_border_flagging_rate",
        "audit_trail_completeness",
    }

    assert set(distributions) == expected_names