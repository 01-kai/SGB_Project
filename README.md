# Synthetic Governance Benchmark (SGB)

<p align="center">
  <strong>A reproducible agent-based simulation framework for benchmarking data governance maturity across synthetic multi-organizational ecosystems.</strong>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.12%2B-blue">
  <img alt="Mesa" src="https://img.shields.io/badge/ABM-Mesa-green">
  <img alt="Tests" src="https://img.shields.io/badge/tests-pytest-brightgreen">
  <img alt="Status" src="https://img.shields.io/badge/status-research%20prototype-orange">
  <img alt="License" src="https://img.shields.io/badge/license-not%20specified-lightgrey">
</p>

---

## Overview

**Synthetic Governance Benchmark (SGB)** is a computational research framework for evaluating data governance maturity models in simulated multi-organizational ecosystems.

The framework represents organizations as autonomous agents with configurable governance profiles and simulates inter-organizational data exchange over a network. It supports reproducible experimentation without requiring access to confidential organizational data.

SGB was developed for the research project:

> **Synthetic Benchmarking of Data Governance Maturity Across Simulated Multi-Organizational Ecosystems: A Computational Framework and Agent-Based Evaluation Approach**

The framework is designed to support:

* governance maturity framework comparison,
* failure and shock propagation analysis,
* resilience and recovery evaluation,
* governance-based onboarding threshold estimation,
* convergence analysis,
* and global parameter sensitivity analysis.

---

## Research Motivation

Evaluating data governance maturity in real multi-organizational environments is difficult because of:

* data sensitivity,
* restricted institutional access,
* organizational heterogeneity,
* lack of comparable governance benchmarks,
* limited longitudinal governance data,
* and difficulty reproducing real-world incidents.

SGB addresses these constraints through a **fully synthetic and reproducible simulation environment**.

The framework enables researchers to investigate questions such as:

* How do alternative governance maturity frameworks score the same organizations?
* How do governance failures propagate through connected organizations?
* What maturity threshold could be used for ecosystem onboarding?
* How quickly do organizations recover after operational or security shocks?
* Are the conclusions stable across seeds, population sizes, and simulation horizons?
* How sensitive are framework rankings to modeling assumptions and weights?

---

## Core Methodological Principle

SGB separates:

1. **operational simulation**, and
2. **governance framework scoring**.

Organizations, exchanges, failures, shocks, and recovery processes are simulated independently of the evaluated governance frameworks.

DBL, DMM, and ISF then score the **same organizational states and trajectories**.

This design prevents a framework from receiving an artificial advantage through framework-specific simulation behavior.

---

## Governance Maturity Index

SGB evaluates each organization using a **Governance Maturity Index (GMI)** derived from governance dimensions and lower-level submetrics.

### Standard dimensions

| Code | Dimension             | Representative Submetrics                                       |
| ---- | --------------------- | --------------------------------------------------------------- |
| `D1` | Data Quality          | Completeness, duplicate rate, schema validation                 |
| `D2` | Security and Privacy  | Encryption coverage, breach-response latency, policy violations |
| `D3` | Metadata and Lineage  | Lineage coverage, catalog coverage, catalog freshness           |
| `D4` | Regulatory Compliance | Rule satisfaction, consent propagation                          |
| `D5` | Interoperability      | Schema reconciliation, API adherence, format translation        |

### Sovereignty dimension

The Iran-Specific Framework adds an additional dimension:

| Code  | Dimension                     | Representative Submetrics                                                |
| ----- | ----------------------------- | ------------------------------------------------------------------------ |
| `SVC` | Sovereignty and Value Control | Localization compliance, cross-border flagging, audit-trail completeness |

All submetrics and dimension scores are bounded to:

```text
[0, 1]
```

Higher values consistently represent stronger governance maturity.

Inverse submetrics, such as duplicate rate or policy-violation rate, are transformed before dimension aggregation.

---

## Compared Frameworks

SGB currently evaluates three governance maturity framework profiles:

| Framework | Description                                                     |
| --------- | --------------------------------------------------------------- |
| `DBL`     | DAMA-DMBOK-like baseline governance profile                     |
| `DMM`     | Data Management Maturity-style process-oriented profile         |
| `ISF`     | Iran-Specific Framework including Sovereignty and Value Control |

The frameworks differ in their configured dimension weights.

They do **not** receive different operational dynamics, exchange-success logic, shock effects, or recovery advantages.

Framework definitions are stored in:

```text
configs/base.yaml
```

---

## Agent-Based Simulation

The simulation models a population of heterogeneous organizations connected through a scale-free network.

The default network is generated using the **Barabási–Albert model**, representing an ecosystem in which some organizations become more highly connected than others.

At each simulation step:

1. an organization selects an exchange partner,
2. an exchange volume is sampled,
3. exchange readiness is calculated,
4. the exchange succeeds or fails,
5. organizational submetrics are updated,
6. recovery dynamics are applied,
7. and system-level metrics are recorded.

Exchange volumes are sampled from a **bounded Pareto Type I distribution**, providing a heavy-tailed distribution while enforcing configurable lower and upper limits.

---

## Perturbations

SGB includes framework-neutral perturbations for studying ecosystem resilience.

Supported perturbations include:

* data-quality degradation,
* metadata degradation,
* regulatory changes,
* interoperability failures,
* sovereignty-compliance failures,
* and network security incidents.

Security incidents can propagate through the organizational network according to:

* network distance,
* configured impact,
* maximum propagation distance,
* and distance-decay parameters.

The same perturbation is evaluated under all governance framework scoring views.

---

## Experiments

SGB implements four primary experiments.

| Experiment   | Objective                                                       | Main Outputs                                               |
| ------------ | --------------------------------------------------------------- | ---------------------------------------------------------- |
| Experiment 1 | Compare DBL, DMM, and ISF on shared trajectories                | Framework scores, rankings, confidence intervals           |
| Experiment 2 | Inject a security incident and measure propagation and recovery | Immediate loss, affected organizations, recovery time      |
| Experiment 3 | Evaluate governance maturity thresholds                         | Precision, recall, specificity, F1, recommended thresholds |
| Experiment 4 | Evaluate low-to-high maturity distributions                     | Framework response across maturity regimes                 |

### Experiment 1 — Framework Comparison

Compares all frameworks on identical steady-state organizational trajectories.

```bash
python -m sgb.experiments \
  --config configs/experiments/experiment_1.yaml
```

### Experiment 2 — Security Propagation and Recovery

Injects a network security incident after steady-state detection and measures propagation and recovery.

```bash
python -m sgb.experiments \
  --config configs/experiments/experiment_2.yaml
```

### Experiment 3 — Threshold Selection

Scans the complete maturity threshold interval from `0.0` to `1.0`, followed by local refinement.

```bash
python -m sgb.experiments \
  --config configs/experiments/experiment_3.yaml
```

### Experiment 4 — Maturity Distribution Sweep

Evaluates all frameworks across low, moderate, and high synthetic maturity regimes.

```bash
python -m sgb.experiments \
  --config configs/experiments/experiment_4.yaml
```

---

## Calibration and Convergence

SGB does not assume that the default simulation settings are automatically valid.

It includes dedicated studies for:

### Baseline calibration

Validates whether sampled submetrics and dimensions match their configured theoretical expectations.

```bash
python -m sgb.calibration \
  --study configs/studies/baseline_calibration.yaml
```

### Time convergence

Evaluates whether the simulation horizon is sufficiently long.

```bash
python -m sgb.convergence \
  --study configs/studies/time_convergence.yaml
```

### Seed convergence

Evaluates whether the number of random seeds produces stable estimates and rankings.

```bash
python -m sgb.convergence \
  --study configs/studies/seed_convergence.yaml
```

### Population convergence

Evaluates whether conclusions remain stable across alternative organization population sizes.

```bash
python -m sgb.convergence \
  --study configs/studies/population_convergence.yaml
```

---

## Sensitivity Analysis

The global sensitivity study evaluates the robustness of conclusions to alternative parameter choices.

Direct sensitivity analysis covers:

* framework weights,
* dimension subweights,
* organization-type proportions,
* Beta-distribution concentration,
* bounded Pareto parameters,
* network attachment values,
* maturity-category cutoffs,
* perturbation magnitudes,
* and catalog-freshness windows.

Run a lightweight smoke analysis:

```bash
python -m sgb.sensitivity \
  --study configs/studies/sensitivity.yaml \
  --profile smoke \
  --codes P1 P2 P8 P12 P13 P14 P21
```

Run the complete sensitivity analysis:

```bash
python -m sgb.sensitivity \
  --study configs/studies/sensitivity.yaml \
  --profile full \
  --codes ALL
```

---

## Project Structure

```text
SGB_Project/
├── configs/
│   ├── base.yaml
│   ├── dynamics.yaml
│   ├── pipeline.yaml
│   ├── reporting.yaml
│   │
│   ├── experiments/
│   │   ├── experiment_1.yaml
│   │   ├── experiment_2.yaml
│   │   ├── experiment_3.yaml
│   │   └── experiment_4.yaml
│   │
│   └── studies/
│       ├── baseline_calibration.yaml
│       ├── time_convergence.yaml
│       ├── seed_convergence.yaml
│       ├── population_convergence.yaml
│       └── sensitivity.yaml
│
├── sgb/
│   ├── __init__.py
│   ├── agents.py
│   ├── calibration.py
│   ├── cli.py
│   ├── config.py
│   ├── convergence.py
│   ├── convergence_utils.py
│   ├── dimensions.py
│   ├── distributions.py
│   ├── dynamics.py
│   ├── experiments.py
│   ├── frameworks.py
│   ├── model.py
│   ├── perturbations.py
│   ├── pipeline.py
│   ├── reporting.py
│   └── sensitivity.py
│
├── tests/
│   ├── test_agents.py
│   ├── test_calibration.py
│   ├── test_config.py
│   ├── test_convergence.py
│   ├── test_dimensions.py
│   ├── test_distributions.py
│   ├── test_dynamics.py
│   ├── test_experiments.py
│   ├── test_frameworks.py
│   ├── test_model.py
│   ├── test_perturbations.py
│   ├── test_pipeline.py
│   ├── test_reporting.py
│   └── test_sensitivity.py
│
├── outputs/
├── README.md
├── RUNBOOK.md
├── pyproject.toml
├── requirements.txt
└── .gitignore
```

---

## Requirements

* Python `3.12+`
* Mesa
* NumPy
* pandas
* NetworkX
* SciPy
* Matplotlib
* PyYAML
* pytest

The full dependency list is available in:

```text
requirements.txt
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/01-kai/SGB_Project.git
cd SGB_Project
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

---

## Testing

Run the complete test suite:

```bash
python -m pytest -v
```

Run a syntax check:

```bash
python -m compileall -q sgb tests
```

Research outputs should not be interpreted while the test suite is failing.

---

## Pipeline

SGB includes an end-to-end research pipeline.

List all stages:

```bash
python -m sgb.pipeline --list
```

Preview the pipeline without executing simulations:

```bash
python -m sgb.pipeline --dry-run
```

Validate generated artifacts:

```bash
python -m sgb.pipeline --validate
```

Run one stage with its dependencies:

```bash
python -m sgb.pipeline \
  --stage experiment_1
```

Run one stage without repeating dependencies:

```bash
python -m sgb.pipeline \
  --stage experiment_1 \
  --no-dependencies
```

Run the complete pipeline:

```bash
python -m sgb.pipeline
```

The complete pipeline runs:

1. automated tests,
2. baseline calibration,
3. time convergence,
4. seed convergence,
5. population convergence,
6. Experiments 1–4,
7. global sensitivity analysis,
8. and final reporting.

---

## Reporting

Generate a partial report when some experiment outputs are not yet available:

```bash
python -m sgb.reporting \
  --config configs/reporting.yaml \
  --allow-missing
```

Generate the strict final report:

```bash
python -m sgb.reporting \
  --config configs/reporting.yaml
```

Expected final outputs:

```text
outputs/final/
├── final_report.md
├── artifact_index.csv
├── decision_summary.csv
├── parameter_coverage.csv
├── reproducibility_manifest.csv
├── reporting_metadata.json
└── pipeline_run.json
```

---

## Reproducibility

Each final research run should preserve:

* the Git commit identifier,
* YAML configuration files,
* dependency versions,
* seed ranges,
* CSV result tables,
* JSON metadata,
* generated figures,
* pipeline logs,
* and the reproducibility manifest.

Before using outputs in an academic paper, run:

```bash
python -m compileall -q sgb tests
python -m pytest -v
python -m sgb.pipeline --validate
git status
git rev-parse HEAD
```

Do not combine outputs generated from different commits or incompatible configurations.

---

## Current Status

The current implementation includes:

* configurable governance submetrics,
* Appendix-based governance dimension calculations,
* heterogeneous organization agents,
* Barabási–Albert network generation,
* bounded Pareto exchange-volume sampling,
* framework-neutral operational dynamics,
* framework-neutral perturbations,
* recovery modeling,
* DBL, DMM, and ISF scoring,
* baseline calibration,
* time, seed, and population convergence studies,
* four primary experiments,
* global sensitivity analysis,
* reproducibility reporting,
* and end-to-end pipeline orchestration.

### Pending calibration task

The organization-type distribution sensitivity study is implemented, but final organization proportions still require calibration against documented public data.

This issue is tracked as **P9** and should remain reported as pending until a defensible public denominator and classification method are established.

---

## Scientific Limitations

SGB is a synthetic research environment.

Its outputs should be interpreted with the following limitations:

* framework weights are modeling assumptions,
* dimension weights require sensitivity analysis,
* maturity categories are definitional partitions,
* synthetic exchanges do not prove real-world causal relationships,
* network topology is an approximation,
* perturbation effects depend on configured assumptions,
* statistical stability does not establish construct validity,
* and external calibration is still required for organization-type proportions.

---

## Development Workflow

After changing code or configuration:

```bash
python -m pytest -v
git status
git add -A
git commit -m "Describe the implementation or research change"
git pull --rebase origin main
git push origin main
```

Recommended `.gitignore` entries:

```gitignore
.venv/
__pycache__/
*.py[cod]
.pytest_cache/
.coverage
htmlcov/
.DS_Store
.vscode/
outputs/
```

---

## Repository

* **Source code:** [github.com/01-kai/SGB_Project](https://github.com/01-kai/SGB_Project)
* **Issues:** [github.com/01-kai/SGB_Project/issues](https://github.com/01-kai/SGB_Project/issues)
