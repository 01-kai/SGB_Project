import numpy as np
import pytest

from sgb.distributions import (
    DistributionError,
    sample_beta,
    sample_categorical,
    sample_truncated_pareto,
    sample_truncated_pareto_integer,
    truncated_pareto_quantile,
)


def test_truncated_pareto_endpoints() -> None:
    assert truncated_pareto_quantile(
        probability=0.0,
        minimum=10.0,
        maximum=2000.0,
        alpha=2.1,
    ) == pytest.approx(10.0)

    assert truncated_pareto_quantile(
        probability=1.0,
        minimum=10.0,
        maximum=2000.0,
        alpha=2.1,
    ) == pytest.approx(2000.0)


def test_truncated_pareto_quantiles_are_increasing() -> None:
    probabilities = [
        0.10,
        0.25,
        0.50,
        0.75,
        0.90,
        0.99,
    ]

    quantiles = [
        truncated_pareto_quantile(
            probability=probability,
            minimum=10.0,
            maximum=2000.0,
            alpha=2.1,
        )
        for probability in probabilities
    ]

    assert quantiles == sorted(quantiles)
    assert quantiles[-1] > quantiles[0]


def test_truncated_pareto_samples_respect_bounds() -> None:
    rng = np.random.default_rng(42)

    samples = [
        sample_truncated_pareto(
            rng=rng,
            minimum=10.0,
            maximum=2000.0,
            alpha=2.1,
        )
        for _ in range(10_000)
    ]

    assert min(samples) >= 10.0
    assert max(samples) <= 2000.0


def test_integer_pareto_samples_respect_bounds() -> None:
    rng = np.random.default_rng(42)

    samples = [
        sample_truncated_pareto_integer(
            rng=rng,
            minimum=10,
            maximum=2000,
            alpha=2.1,
        )
        for _ in range(1_000)
    ]

    assert all(
        isinstance(value, int)
        for value in samples
    )

    assert min(samples) >= 10
    assert max(samples) <= 2000


def test_truncated_pareto_is_reproducible() -> None:
    first_rng = np.random.default_rng(123)
    second_rng = np.random.default_rng(123)

    first_samples = [
        sample_truncated_pareto(
            first_rng,
            10,
            2000,
            2.1,
        )
        for _ in range(100)
    ]

    second_samples = [
        sample_truncated_pareto(
            second_rng,
            10,
            2000,
            2.1,
        )
        for _ in range(100)
    ]

    assert first_samples == second_samples


def test_invalid_pareto_parameters_are_rejected() -> None:
    with pytest.raises(
        DistributionError,
        match="greater than minimum",
    ):
        truncated_pareto_quantile(
            probability=0.5,
            minimum=10,
            maximum=10,
            alpha=2.1,
        )

    with pytest.raises(
        DistributionError,
        match="greater than 1.0",
    ):
        truncated_pareto_quantile(
            probability=0.5,
            minimum=10,
            maximum=2000,
            alpha=1.0,
        )


def test_beta_sample_is_normalized() -> None:
    rng = np.random.default_rng(42)

    samples = [
        sample_beta(
            rng=rng,
            alpha=9.0,
            beta=11.0,
        )
        for _ in range(1_000)
    ]

    assert all(
        0.0 <= sample <= 1.0
        for sample in samples
    )


def test_categorical_sampling_is_reproducible() -> None:
    probabilities = {
        "a": 0.40,
        "b": 0.35,
        "c": 0.25,
    }

    first_rng = np.random.default_rng(42)
    second_rng = np.random.default_rng(42)

    first = [
        sample_categorical(
            first_rng,
            probabilities,
        )
        for _ in range(100)
    ]

    second = [
        sample_categorical(
            second_rng,
            probabilities,
        )
        for _ in range(100)
    ]

    assert first == second


def test_categorical_probabilities_must_sum_to_one() -> None:
    rng = np.random.default_rng(42)

    with pytest.raises(
        DistributionError,
        match="sum to 1.0",
    ):
        sample_categorical(
            rng,
            {
                "a": 0.40,
                "b": 0.40,
            },
        )