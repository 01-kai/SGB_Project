# Synthetic Governance Benchmark (SGB)

<p align="center">
  <strong>A reproducible agent-based simulation framework for benchmarking data governance maturity across synthetic multi-organizational ecosystems.</strong>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.12%2B-blue">
  <img alt="Mesa" src="https://img.shields.io/badge/ABM-Mesa-green">
  <img alt="Status" src="https://img.shields.io/badge/status-research%20prototype-orange">
  <img alt="License" src="https://img.shields.io/badge/license-academic-lightgrey">
</p>

---

## Overview

**Synthetic Governance Benchmark (SGB)** is a computational framework for evaluating data governance maturity models in simulated multi-organizational ecosystems.

The framework models organizations as autonomous agents with configurable governance profiles and simulates inter-organizational data exchange events under different governance maturity frameworks. It is designed to support reproducible experimentation without requiring access to real organizational data.

SGB was developed for the research project:

> **Synthetic Benchmarking of Data Governance Maturity Across Simulated Multi-Organizational Ecosystems: A Computational Framework and Agent-Based Evaluation Approach**

The simulator is used to compare governance maturity frameworks, evaluate failure propagation, estimate onboarding thresholds for the National Data Exchange Layer (NDEL), and analyze the trade-off between resilience and exchange efficiency.

---

## Research Motivation

Assessing data governance maturity in real multi-organizational environments is difficult because of:

- data sensitivity,
- access restrictions,
- institutional heterogeneity,
- lack of comparable real-world benchmarks,
- and limited availability of longitudinal governance data.

SGB addresses this problem by replacing direct empirical access with a **fully synthetic, reproducible, simulation-based evaluation environment**.

The framework allows researchers and policymakers to ask questions such as:

- Which maturity framework produces better ecosystem-level governance outcomes?
- How do governance failures propagate through connected organizations?
- What minimum GMI threshold should be required for NDEL onboarding?
- Does higher maturity always improve resilience and efficiency?

---

## Core Concepts

SGB evaluates organizations using a **Governance Maturity Index (GMI)**.

The standard GMI includes five dimensions:

| Code | Dimension | Description |
|---|---|---|
| D1 | Data Quality | Completeness, duplicate rate, schema validation |
| D2 | Security Posture | Encryption coverage, breach detection, policy violations |
| D3 | Metadata Completeness | Lineage, catalog coverage, catalog freshness |
| D4 | Regulatory Compliance | Rule satisfaction and consent propagation |
| D5 | Interoperability | Schema reconciliation, API adherence, format translation |

The Iran-Specific Framework (**ISF**) adds a sixth dimension:

| Code | Dimension | Description |
|---|---|---|
| SVC | Sovereignty Compliance | Localization, cross-border flags, audit trail retention |

---

## Compared Frameworks

SGB currently compares three maturity framework profiles:

| Framework | Description |
|---|---|
| DBL | DAMA-DMBOK-like baseline framework |
| DMM | Data Management Maturity-style process-oriented framework |
| ISF | Iran-Specific Framework with Sovereignty Compliance |

Each framework uses different GMI weights and recovery behavior.

---

## Experiments

The simulator runs four experiments.

| Experiment | Purpose | Main Output |
|---|---|---|
| Experiment 1 | Compare DBL, DMM, and ISF under normal conditions | Table 2 |
| Experiment 2 | Inject a Security Incident and measure failure propagation | Figure 2, Table 3 |
| Experiment 3 | Evaluate NDEL onboarding thresholds | Figure 3, Table 4 |
| Experiment 4 | Analyze resilience-efficiency trade-off | Figure 4, Table 5 |

---

## Project Structure

```text
SGB_Project/
├── sgb/
│   ├── __init__.py
│   ├── agents.py
│   ├── model.py
│   ├── perturbations.py
│   ├── frameworks.py
│   ├── metrics.py
│   ├── experiments.py
│   ├── visualize.py
│   └── config.yaml
│
├── outputs/
│   ├── raw/
│   ├── tables/
│   └── figures/
│
├── requirements.txt
├── README.md
└── .gitignore
