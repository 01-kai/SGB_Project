# Synthetic Governance Benchmark (SGB)

<p align="center">
  <strong>A reproducible agent-based simulation framework for benchmarking data-governance maturity across synthetic multi-organizational ecosystems.</strong>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.12-blue">
  <img alt="Version" src="https://img.shields.io/badge/version-0.3.0-blueviolet">
  <img alt="ABM" src="https://img.shields.io/badge/ABM-Mesa-green">
  <img alt="Tests" src="https://img.shields.io/badge/tests-pytest-brightgreen">
  <img alt="Status" src="https://img.shields.io/badge/status-research%20prototype-orange">
</p>

---

## Contents

- [Overview](#overview)
- [Research Motivation](#research-motivation)
- [Research Objectives](#research-objectives)
- [Core Methodological Principle](#core-methodological-principle)
- [Governance Maturity Index](#governance-maturity-index)
- [Compared Frameworks](#compared-frameworks)
- [Agent-Based Simulation](#agent-based-simulation)
- [Perturbations and Recovery](#perturbations-and-recovery)
- [Experiments](#experiments)
- [Calibration and Convergence](#calibration-and-convergence)
- [Sensitivity Analysis](#sensitivity-analysis)
- [P9 Public-Source Calibration](#p9-public-source-calibration)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Testing](#testing)
- [Running Individual Studies](#running-individual-studies)
- [Running the Full Pipeline](#running-the-full-pipeline)
- [Outputs](#outputs)
- [Reporting](#reporting)
- [Reproducibility](#reproducibility)
- [Current Status](#current-status)
- [Scientific Limitations](#scientific-limitations)
- [Development Workflow](#development-workflow)
- [Citation](#citation)
- [License](#license)
- [Repository](#repository)

---

## Overview

**Synthetic Governance Benchmark (SGB)** is a computational research framework for evaluating data-governance maturity models in simulated multi-organizational ecosystems.

The framework represents organizations as autonomous agents with configurable governance profiles and simulates inter-organizational data exchange over a synthetic network. It supports reproducible experimentation without requiring access to confidential organizational data.

SGB was developed for the research project:

> **Synthetic Benchmarking of Data Governance Maturity Across Simulated Multi-Organizational Ecosystems: A Computational Framework and Agent-Based Evaluation Approach**

The framework supports:

- governance maturity framework comparison,
- failure and shock propagation analysis,
- resilience and recovery evaluation,
- governance-based onboarding threshold estimation,
- time, seed, and population convergence analysis,
- parameter sensitivity analysis,
- reproducible generation of tables, figures, metadata, and reports.

---

## Research Motivation

Evaluating data-governance maturity in real multi-organizational environments is difficult because of:

- data sensitivity,
- restricted institutional access,
- organizational heterogeneity,
- lack of comparable governance benchmarks,
- limited longitudinal governance data,
- and difficulty reproducing operational failures or security incidents.

SGB addresses these constraints through a **fully synthetic and reproducible simulation environment**.

The framework enables researchers to investigate questions such as:

- How do alternative governance maturity frameworks score the same organizations?
- How do governance failures propagate through connected organizations?
- What maturity threshold could be used for ecosystem onboarding?
- How quickly do organizations recover after operational or security shocks?
- Are conclusions stable across random seeds, population sizes, and simulation horizons?
- How sensitive are framework rankings to modeling assumptions and parameter choices?

---

## Research Objectives

The principal objectives of SGB are to:

1. construct a reproducible synthetic ecosystem of heterogeneous organizations;
2. model governance maturity through measurable submetrics and dimensions;
3. simulate inter-organizational data exchange over a connected network;
4. evaluate multiple governance frameworks on identical operational trajectories;
5. measure the propagation and recovery of governance-related failures;
6. estimate maturity thresholds for exchange-system onboarding;
7. evaluate the stability of results across time horizons, seed counts, and population sizes;
8. assess the robustness of conclusions under parameter variation;
9. generate reproducible research artifacts suitable for academic analysis.

---

## Core Methodological Principle

SGB separates:

1. **operational simulation**, and
2. **governance framework scoring**.

Organizations, exchanges, failures, shocks, and recovery processes are simulated independently of the evaluated governance frameworks.

DBL, DMM, and ISF then score the **same organizational states, events, and trajectories**.

This design prevents any framework from receiving an artificial advantage through framework-specific simulation dynamics.

### Shared-trajectory evaluation

The framework uses shared trajectories wherever comparisons are required. This means:

- the same organizations are evaluated by all frameworks;
- the same exchange events are observed;
- the same perturbations are applied;
- the same random seeds are reused where appropriate;
- framework differences arise from scoring definitions rather than different operational histories.

---

## Governance Maturity Index

SGB evaluates each organization using a **Governance Maturity Index (GMI)** derived from governance dimensions and lower-level submetrics.

### Standard dimensions

| Code | Dimension | Representative Submetrics |
|---|---|---|
| `D1` | Data Quality | Completeness, duplicate rate, schema validation |
| `D2` | Security and Privacy | Encryption coverage, response latency, policy violations |
| `D3` | Metadata and Lineage | Lineage coverage, catalog coverage, catalog freshness |
| `D4` | Regulatory Compliance | Rule satisfaction, consent propagation |
| `D5` | Interoperability | Schema reconciliation, API adherence, format translation |

### Sovereignty dimension

The Iran-Specific Framework adds an additional dimension:

| Code | Dimension | Representative Submetrics |
|---|---|---|
| `SVC` | Sovereignty and Value Control | Localization compliance, cross-border flagging, audit-trail completeness |

All submetrics and derived dimension scores are bounded to:

```text
[0, 1]
```

Higher values consistently represent stronger governance maturity.

Inverse submetrics, such as duplicate rate or policy-violation rate, are transformed before dimension aggregation.

The exact formulas, weights, and baseline distributions are defined in:

```text
configs/base.yaml
```

---

## Compared Frameworks

SGB currently evaluates three governance maturity framework profiles:

| Framework | Description |
|---|---|
| `DBL` | DAMA-DMBOK-like baseline governance profile |
| `DMM` | Data Management Maturity-style process-oriented profile |
| `ISF` | Iran-Specific Framework including Sovereignty and Value Control |

The frameworks differ in their configured dimension weights.

They do **not** receive different:

- exchange-success logic,
- operational dynamics,
- shock propagation rules,
- recovery rules,
- network structures,
- or random trajectories.

---

## Agent-Based Simulation

The simulation models a population of heterogeneous organizations connected through a scale-free network.

The default network is generated using the **Barabási–Albert model**, representing an ecosystem in which some organizations become more highly connected than others.

At each simulation step:

1. organizations select exchange partners;
2. exchange volumes are sampled;
3. operational readiness is evaluated;
4. exchanges succeed or fail;
5. organizational submetrics are updated;
6. perturbation and recovery rules are applied;
7. state and event metrics are recorded.

### Exchange-volume distribution

Exchange volumes are sampled from a **bounded Pareto Type I distribution**.

This provides a heavy-tailed distribution while enforcing configurable lower and upper limits.

### Organization heterogeneity

Organizations may differ in:

- organization type,
- initial governance submetrics,
- network degree,
- exposure to exchange failures,
- baseline maturity,
- recovery trajectory,
- and perturbation impact.

### Baseline-preserving recovery

Version `0.3.0` preserves organization-specific baseline heterogeneity during recovery.

Degraded submetrics recover toward each organization’s sampled baseline rather than toward one shared global constant.

---

## Perturbations and Recovery

SGB includes framework-neutral perturbations for studying ecosystem resilience.

Supported perturbations include:

- data-quality degradation,
- metadata degradation,
- regulatory changes,
- interoperability failures,
- sovereignty-compliance failures,
- and network security incidents.

Security incidents can propagate according to:

- network distance,
- configured impact magnitude,
- propagation depth,
- distance decay,
- and affected-organization eligibility.

The same perturbation trajectory is evaluated under every framework.

Recovery is measured using the post-shock restoration of affected organizations relative to their pre-shock condition.

---

## Experiments

SGB implements four primary experiments.

| Experiment | Objective | Main Outputs |
|---|---|---|
| Experiment 1 | Compare DBL, DMM, and ISF on shared trajectories | Framework scores, rankings, confidence intervals |
| Experiment 2 | Inject a security incident and measure propagation and recovery | Immediate loss, affected organizations, recovery time |
| Experiment 3 | Evaluate governance maturity thresholds | Precision, recall, specificity, F1, balanced accuracy |
| Experiment 4 | Evaluate low-to-high maturity distributions | Framework response across maturity regimes |

### Experiment 1 — Framework Comparison

Experiment 1 compares all frameworks on identical steady-state organizational trajectories.

```bash
python -m sgb.experiments \
  --config configs/experiments/experiment_1.yaml
```

Primary outputs include:

- framework-level E-GMI estimates,
- confidence intervals,
- framework rankings,
- maturity-category distributions,
- per-seed summaries.

### Experiment 2 — Security Propagation and Recovery

Experiment 2 injects a framework-neutral security incident after stabilization.

```bash
python -m sgb.experiments \
  --config configs/experiments/experiment_2.yaml
```

Primary outputs include:

- number of affected organizations,
- propagation distance,
- immediate maturity loss,
- recovery time,
- recovery confidence intervals,
- framework-level post-hoc comparisons.

### Experiment 3 — Threshold Selection

Experiment 3 evaluates maturity thresholds over the full configured range.

```bash
python -m sgb.experiments \
  --config configs/experiments/experiment_3.yaml
```

Threshold selection uses event-time governance scores rather than one terminal score map.

Thresholds are evaluated using:

- precision,
- recall,
- specificity,
- F1 score,
- balanced accuracy,
- event coverage,
- volume coverage.

The current implementation uses **balanced accuracy** as the primary threshold-optimization criterion to avoid trivial all-eligible or all-ineligible solutions.

### Experiment 4 — Maturity Distribution Sweep

Experiment 4 evaluates all frameworks across low-to-high synthetic maturity regimes.

```bash
python -m sgb.experiments \
  --config configs/experiments/experiment_4.yaml
```

The experiment verifies that configured low and high Beta-distribution endpoints represent sufficiently distinct maturity conditions.

---

## Calibration and Convergence

SGB does not assume that default simulation settings are automatically valid.

It includes dedicated calibration and convergence studies.

### Baseline calibration

The baseline calibration study verifies that sampled submetrics and derived dimensions match their configured theoretical expectations.

```bash
python -m sgb.calibration \
  --study configs/studies/baseline_calibration.yaml
```

### Time convergence

The time-convergence study evaluates whether the simulation horizon is sufficiently long.

```bash
python -m sgb.convergence \
  --study configs/studies/time_convergence.yaml
```

### Seed convergence

The seed-convergence study evaluates:

- confidence-interval stability,
- framework-ranking stability,
- and the supported number of random seeds.

```bash
python -m sgb.convergence \
  --study configs/studies/seed_convergence.yaml
```

### Population convergence

The population-convergence study evaluates whether conclusions remain stable across candidate organization counts.

```bash
python -m sgb.convergence \
  --study configs/studies/population_convergence.yaml
```

A rejected convergence candidate is a scientific result and does not necessarily indicate a software error.

---

## Sensitivity Analysis

The sensitivity study evaluates the robustness of conclusions under alternative modeling assumptions.

Direct sensitivity analysis covers:

```text
P1–P8, P10–P14, and P21
```

P9 is resolved through public-source calibration.

P15–P20 are evaluated through linked convergence and experiment studies.

Direct sensitivity scenarios include variations in:

- framework weights,
- dimension subweights,
- Beta-distribution concentration,
- bounded Pareto parameters,
- network attachment values,
- maturity-category boundaries,
- perturbation magnitude,
- catalog-freshness windows,
- and selected operational assumptions.

Run a lightweight smoke profile:

```bash
python -m sgb.sensitivity \
  --study configs/studies/sensitivity.yaml \
  --profile smoke \
  --codes P1 P2 P8 P12 P13 P14 P21
```

Run the complete direct sensitivity study:

```bash
python -m sgb.sensitivity \
  --study configs/studies/sensitivity.yaml \
  --profile full \
  --codes ALL
```

Sensitivity rejection is not automatically a software failure. It may indicate that conclusions are genuinely unstable under one or more parameter alternatives.

---

## P9 Public-Source Calibration

P9 represents the organization-type distribution used in the synthetic ecosystem.

The calibrated baseline is:

| Organization Type | Probability |
|---|---:|
| Ministry | `0.02` |
| State enterprise | `0.33` |
| Private entity | `0.65` |

This distribution is implemented as a **public-source proxy** for Iranian organizations participating in or interacting with the national information-exchange environment.

Configuration metadata records:

```yaml
public_calibration:
  P9_complete: true
  calibration_type: public_source_proxy
  source_scope: iran_nix_connected_organizations
```

P9 is not treated as a behaviorally active simulation parameter because organization type currently serves as a population-classification variable rather than an independent operational-dynamics mechanism.

Alternative organization mixes may still be retained as descriptive or stress-test scenarios, but they are not interpreted as direct causal sensitivity parameters unless organization type is connected to explicit behavioral rules.

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
│   ├── check_convergence.py
│   ├── cli.py
│   ├── config.py
│   ├── convergence.py
│   ├── convergence_utils.py
│   ├── dimensions.py
│   ├── distributions.py
│   ├── dynamics.py
│   ├── experiments.py
│   ├── frameworks.py
│   ├── metrics.py
│   ├── model.py
│   ├── perturbations.py
│   ├── pipeline.py
│   ├── reporting.py
│   ├── sensitivity.py
│   └── visualize.py
│
├── tests/
├── outputs/
├── FIX_NOTES.md
├── README.md
├── RUNBOOK.md
├── pyproject.toml
├── requirements.txt
└── .gitignore
```

---

## Requirements

- Python `3.12` — validated research environment
- Mesa
- NumPy
- pandas
- NetworkX
- SciPy
- Matplotlib
- PyYAML
- tqdm
- pytest

The exact package versions used in the validated environment are pinned in:

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
python3.12 -m venv .venv
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

Verify the environment:

```bash
python --version
python -m pip check
```

---

## Testing

Run the complete test suite before generating research results:

```bash
python -m pytest -q
```

Run a syntax check:

```bash
python -m compileall -q sgb tests
```

Research outputs should not be interpreted while the test suite is failing.

---

## Running Individual Studies

### Baseline calibration

```bash
python -m sgb.calibration \
  --study configs/studies/baseline_calibration.yaml
```

### Time convergence

```bash
python -m sgb.convergence \
  --study configs/studies/time_convergence.yaml
```

### Seed convergence

```bash
python -m sgb.convergence \
  --study configs/studies/seed_convergence.yaml
```

### Population convergence

```bash
python -m sgb.convergence \
  --study configs/studies/population_convergence.yaml
```

### Experiment 1

```bash
python -m sgb.experiments \
  --config configs/experiments/experiment_1.yaml
```

### Experiment 2

```bash
python -m sgb.experiments \
  --config configs/experiments/experiment_2.yaml
```

### Experiment 3

```bash
python -m sgb.experiments \
  --config configs/experiments/experiment_3.yaml
```

### Experiment 4

```bash
python -m sgb.experiments \
  --config configs/experiments/experiment_4.yaml
```

### Full sensitivity analysis

```bash
python -m sgb.sensitivity \
  --study configs/studies/sensitivity.yaml \
  --profile full \
  --codes ALL
```

---

## Running the Full Pipeline

List all configured stages:

```bash
python -m sgb.pipeline --list
```

Preview the execution plan:

```bash
python -m sgb.pipeline --dry-run
```

Validate currently available artifacts:

```bash
python -m sgb.pipeline --validate
```

Run one stage with dependencies:

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
PYTHONUNBUFFERED=1 \
python -m sgb.pipeline \
2>&1 | tee outputs/full_pipeline_console.log
```

The complete pipeline executes:

1. automated tests;
2. baseline calibration;
3. time convergence;
4. seed convergence;
5. population convergence;
6. Experiments 1–4;
7. full direct sensitivity analysis;
8. final reporting.

The pipeline stops on software or configuration failures according to:

```text
configs/pipeline.yaml
```

A scientific rejection may still be recorded as a valid result depending on the stage and command configuration.

---

## Outputs

Generated research artifacts are written under:

```text
outputs/
├── calibration/
├── convergence/
├── experiments/
├── sensitivity/
├── figures/
├── tables/
├── raw/
├── final/
└── full_pipeline_console.log
```

Typical artifacts include:

- CSV result tables,
- JSON metadata,
- PNG figures,
- convergence summaries,
- experiment-level outputs,
- sensitivity matrices,
- reporting metadata,
- pipeline execution records.

Generated outputs should not be edited manually.

Outputs from different code versions or incompatible configurations must not be combined.

---

## Reporting

Generate a partial report when some large-study artifacts are missing:

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

Expected reporting outputs include:

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

The reporting layer reads current calibration and sensitivity metadata, including the completed P9 public-source calibration status.

---

## Reproducibility

Each final research run should preserve:

- the Git commit identifier,
- YAML configuration files,
- exact dependency versions,
- random-seed ranges,
- CSV result tables,
- JSON metadata,
- generated figures,
- pipeline logs,
- artifact indexes,
- and the reproducibility manifest.

Before using outputs in an academic paper, run:

```bash
python -m compileall -q sgb tests
python -m pytest -q
python -m sgb.pipeline --validate
git status
git rev-parse HEAD
```

Do not combine outputs generated from different commits or configurations.

---

## Current Status

Version `0.3.0` includes:

- configurable governance submetrics,
- governance dimension calculations,
- heterogeneous organization agents,
- Barabási–Albert network generation,
- bounded Pareto exchange-volume sampling,
- framework-neutral operational dynamics,
- baseline-preserving recovery,
- framework-neutral perturbations,
- event-time threshold scoring,
- DBL, DMM, and ISF evaluation,
- baseline calibration,
- time, seed, and population convergence studies,
- four primary experiments,
- public-source P9 calibration,
- direct and linked sensitivity coverage,
- progress reporting,
- reproducibility reporting,
- and end-to-end pipeline orchestration.

See:

```text
FIX_NOTES.md
```

for the methodological and implementation changes introduced in version `0.3.0`.

---


## Development Workflow

After changing code or configuration:

```bash
python -m compileall -q sgb tests
python -m pytest -q

git status
git add -A
git commit -m "Describe the implementation or research change"
git pull --rebase origin main
git push origin main
```

Recommended `.gitignore` entries:

```gitignore
.venv/
.venv-clean/
__pycache__/
*.py[cod]
.pytest_cache/
.coverage
htmlcov/
.DS_Store
.vscode/

outputs/
outputs_*/

*.zip
*.tar.gz
*.log
```

---


## Repository

- **Source code:** [github.com/01-kai/SGB_Project](https://github.com/01-kai/SGB_Project)
- **Issues:** [github.com/01-kai/SGB_Project/issues](https://github.com/01-kai/SGB_Project/issues)
