from collections.abc import Callable, Mapping
from typing import Any

import pytest

from sgb.config import (
    BASE_CONFIG_PATH,
    load_config,
)
from sgb.dimensions import (
    DimensionConfigurationError,
    DimensionValueError,
    compute_d1,
    compute_d2,
    compute_d3,
    compute_d4,
    compute_d5,
    compute_dimension_scores,
    compute_svc,
)


@pytest.fixture(scope="module")
def formula_config() -> dict[str, dict[str, Any]]:
    config = load_config(BASE_CONFIG_PATH)
    return config["dimension_formulas"]


@pytest.mark.parametrize(
    (
        "dimension",
        "computer",
        "submetrics",
        "expected",
    ),
    [
        (
            "D1",
            compute_d1,
            {
                "completeness_ratio": 0.80,
                "duplicate_rate": 0.10,
                "schema_pass_rate": 0.60,
            },
            0.785,
        ),
        (
            "D2",
            compute_d2,
            {
                "encryption_coverage": 0.75,
                "normalized_breach_latency": 0.20,
                "policy_violation_rate": 0.10,
            },
            0.805,
        ),
        (
            "D3",
            compute_d3,
            {
                "lineage_rate": 0.60,
                "catalog_coverage": 0.50,
                "catalog_freshness": 0.80,
            },
            0.610,
        ),
        (
            "D4",
            compute_d4,
            {
                "rule_pass_rate": 0.70,
                "consent_propagation_rate": 0.40,
            },
            0.580,
        ),
        (
            "D5",
            compute_d5,
            {
                "schema_reconciliation_rate": 0.50,
                "api_adherence_rate": 0.80,
                "format_translation_rate": 0.60,
            },
            0.630,
        ),
        (
            "SVC",
            compute_svc,
            {
                "localization_compliance_rate": 0.90,
                "cross_border_flagging_rate": 0.70,
                "audit_trail_completeness": 0.80,
            },
            0.820,
        ),
    ],
)
def test_dimension_formula(
    dimension: str,
    computer: Callable[
        [Mapping[str, float], Mapping[str, float]],
        float,
    ],
    submetrics: dict[str, float],
    expected: float,
    formula_config: dict[str, dict[str, Any]],
) -> None:
    score = computer(
        submetrics,
        formula_config[dimension]["weights"],
    )

    assert score == pytest.approx(expected)


def test_compute_all_dimension_scores(
    formula_config: dict[str, dict[str, Any]],
) -> None:
    submetrics = {
        "D1": {
            "completeness_ratio": 0.50,
            "duplicate_rate": 0.50,
            "schema_pass_rate": 0.50,
        },
        "D2": {
            "encryption_coverage": 0.50,
            "normalized_breach_latency": 0.50,
            "policy_violation_rate": 0.50,
        },
        "D3": {
            "lineage_rate": 0.50,
            "catalog_coverage": 0.50,
            "catalog_freshness": 0.50,
        },
        "D4": {
            "rule_pass_rate": 0.50,
            "consent_propagation_rate": 0.50,
        },
        "D5": {
            "schema_reconciliation_rate": 0.50,
            "api_adherence_rate": 0.50,
            "format_translation_rate": 0.50,
        },
        "SVC": {
            "localization_compliance_rate": 0.50,
            "cross_border_flagging_rate": 0.50,
            "audit_trail_completeness": 0.50,
        },
    }

    scores = compute_dimension_scores(
        submetrics,
        formula_config,
    )

    assert set(scores) == {
        "D1",
        "D2",
        "D3",
        "D4",
        "D5",
        "SVC",
    }

    for score in scores.values():
        assert score == pytest.approx(0.50)


def test_svc_is_optional(
    formula_config: dict[str, dict[str, Any]],
) -> None:
    submetrics = {
        "D1": {
            "completeness_ratio": 0.50,
            "duplicate_rate": 0.50,
            "schema_pass_rate": 0.50,
        },
        "D2": {
            "encryption_coverage": 0.50,
            "normalized_breach_latency": 0.50,
            "policy_violation_rate": 0.50,
        },
        "D3": {
            "lineage_rate": 0.50,
            "catalog_coverage": 0.50,
            "catalog_freshness": 0.50,
        },
        "D4": {
            "rule_pass_rate": 0.50,
            "consent_propagation_rate": 0.50,
        },
        "D5": {
            "schema_reconciliation_rate": 0.50,
            "api_adherence_rate": 0.50,
            "format_translation_rate": 0.50,
        },
    }

    scores = compute_dimension_scores(
        submetrics,
        formula_config,
    )

    assert set(scores) == {
        "D1",
        "D2",
        "D3",
        "D4",
        "D5",
    }


def test_out_of_range_submetric_is_rejected(
    formula_config: dict[str, dict[str, Any]],
) -> None:
    with pytest.raises(
        DimensionValueError,
        match=r"\[0, 1\]",
    ):
        compute_d1(
            {
                "completeness_ratio": 1.20,
                "duplicate_rate": 0.10,
                "schema_pass_rate": 0.70,
            },
            formula_config["D1"]["weights"],
        )


def test_missing_submetric_is_rejected(
    formula_config: dict[str, dict[str, Any]],
) -> None:
    with pytest.raises(
        DimensionValueError,
        match="schema_pass_rate",
    ):
        compute_d1(
            {
                "completeness_ratio": 0.80,
                "duplicate_rate": 0.10,
            },
            formula_config["D1"]["weights"],
        )


def test_weights_must_sum_to_one() -> None:
    with pytest.raises(
        DimensionConfigurationError,
        match="sum to 1.0",
    ):
        compute_d4(
            {
                "rule_pass_rate": 0.70,
                "consent_propagation_rate": 0.80,
            },
            {
                "rule_pass_rate": 0.20,
                "consent_propagation_rate": 0.20,
            },
        )