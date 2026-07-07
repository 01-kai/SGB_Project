# sgb/metrics.py

from scipy import stats
from sgb.frameworks import FRAMEWORKS


def clamp(x, low=0.0, high=1.0):
    return max(low, min(high, x))


def compute_gmi(dimensions, framework_name):
    """
    Compute Governance Maturity Index for one organization.
    """
    weights = FRAMEWORKS[framework_name]["weights"]
    return clamp(sum(weights[d] * dimensions.get(d, 0.0) for d in weights))


def compute_egmi(gmi_values):
    """
    Compute Ecosystem-level GMI.
    """
    if len(gmi_values) == 0:
        return 0.0

    return sum(gmi_values) / len(gmi_values)


def confidence_interval_95(values):
    """
    Compute mean and 95% confidence interval.
    """
    values = list(values)
    n = len(values)

    if n == 0:
        return 0.0, 0.0

    mean = sum(values) / n

    if n < 2:
        return mean, 0.0

    sem = stats.sem(values)
    ci = sem * stats.t.ppf((1 + 0.95) / 2, n - 1)

    return mean, ci


def compute_recovery_time(series, shock_step, baseline_window=20, tolerance=0.02):
    """
    Compute how many steps are needed for the system to recover after a shock.

    Recovery means the metric returns to at least:
    pre-shock baseline - tolerance.
    """

    if len(series) <= shock_step:
        return None

    baseline_start = max(0, shock_step - baseline_window)
    baseline_values = series.iloc[baseline_start:shock_step]

    if len(baseline_values) == 0:
        return None

    baseline = baseline_values.mean()
    recovery_target = baseline - tolerance

    for i in range(shock_step + 1, len(series)):
        if series.iloc[i] >= recovery_target:
            return i - shock_step

    return None