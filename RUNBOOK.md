# Synthetic Governance Benchmark — Execution Runbook

## 1. Purpose

This runbook defines the reproducible execution order for the Synthetic
Governance Benchmark project.

The operational simulation is framework-independent. DBL, DMM, and ISF
evaluate the same organization states, exchange events, shocks, and recovery
trajectories.

## 2. Environment Setup

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .