# Synthetic Benchmarking of Data Governance Maturity Across Simulated Multi-Organizational Ecosystems


## Final Computational Report

**Project:** SGB  
**Version:** 0.2.0

This report consolidates calibration, convergence, framework comparison, perturbation, threshold-selection, maturity-sweep, and sensitivity-analysis outputs.

## 1. Artifact Availability

| input_name | role | required | exists | relative_path |
| --- | --- | --- | --- | --- |
| baseline_table_1 | Baseline dimension calibration and paper Table 1 | True | True | outputs/calibration/baseline/table_1_baseline_dimensions.csv |
| baseline_metadata | Baseline-calibration decision | True | True | outputs/calibration/baseline/baseline_calibration_metadata.json |
| time_convergence_metadata | Time-horizon convergence decision | True | True | outputs/convergence/time/metadata.json |
| seed_convergence_metadata | Seed-count convergence decision | True | True | outputs/convergence/seeds/metadata.json |
| population_convergence_metadata | Population-size convergence decision | True | True | outputs/convergence/population/metadata.json |
| experiment_1_metadata | Framework-comparison experiment decision | True | True | outputs/experiments/experiment_1/metadata.json |
| experiment_1_summary | Framework-comparison summary | True | True | outputs/experiments/experiment_1/framework_summary.csv |
| experiment_2_metadata | Security propagation and recovery decision | True | True | outputs/experiments/experiment_2/metadata.json |
| experiment_2_summary | Security recovery summary | True | True | outputs/experiments/experiment_2/framework_recovery_summary.csv |
| experiment_3_metadata | Maturity-threshold experiment decision | True | True | outputs/experiments/experiment_3/metadata.json |
| experiment_3_summary | Recommended framework thresholds | True | True | outputs/experiments/experiment_3/threshold_recommendations.csv |
| experiment_4_metadata | Maturity-distribution endpoint decision | True | True | outputs/experiments/experiment_4/metadata.json |
| experiment_4_summary | Maturity-distribution sweep summary | True | True | outputs/experiments/experiment_4/level_summary.csv |
| sensitivity_metadata | Global sensitivity decision | True | True | outputs/sensitivity/global/metadata.json |
| sensitivity_coverage | P1-P21 implementation and resolution matrix | True | True | outputs/sensitivity/global/coverage_matrix.csv |
| sensitivity_stability | Sensitivity robustness decisions | True | True | outputs/sensitivity/global/stability_summary.csv |

## 2. Study Decisions

| role | status | decision |
| --- | --- | --- |
| Baseline-calibration decision | passed | Dimension acceptance passed: True; submetric acceptance passed: True |
| Time-horizon convergence decision | passed | Recommended horizon: 1000 |
| Seed-count convergence decision | passed | Recommended seed count: 50 |
| Population-size convergence decision | passed | Recommended population size: 50 |
| Framework-comparison experiment decision | passed | Framework ranking: ISF>DBL>DMM |
| Security propagation and recovery decision | passed | All framework views recovered: True |
| Maturity-threshold experiment decision | passed | Full [0,1] threshold range scanned: True |
| Maturity-distribution endpoint decision | passed | Low endpoint passed: True; high endpoint passed: True |
| Global sensitivity decision | not_passed | Simulation stability passed: False; P9 public calibration complete: True |

## 3. Baseline Calibration

| dimension | theoretical_mean | empirical_mean | ci_lower | ci_upper | absolute_mean_error | target_minimum | target_maximum | accepted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D1 | 0.45 | 0.450665 | 0.447342 | 0.453989 | 0.000665 | 0.4 | 0.5 | True |
| D2 | 0.45 | 0.450674 | 0.446977 | 0.454371 | 0.000674 | 0.4 | 0.5 | True |
| D3 | 0.45 | 0.448057 | 0.444689 | 0.451424 | 0.001943 | 0.4 | 0.5 | True |
| D4 | 0.45 | 0.450944 | 0.446343 | 0.455545 | 0.000944 | 0.4 | 0.5 | True |
| D5 | 0.45 | 0.450776 | 0.447226 | 0.454327 | 0.000776 | 0.4 | 0.5 | True |

## 4. Parameter Resolution Matrix

| code | category | implementation | resolution_status | final_status |
| --- | --- | --- | --- | --- |
| P1 | III | sgb.sensitivity | direct_sensitivity_implemented | implemented |
| P2 | III | sgb.sensitivity | direct_sensitivity_implemented | implemented |
| P3 | III | sgb.sensitivity | direct_sensitivity_implemented | implemented |
| P4 | III | sgb.sensitivity | direct_sensitivity_implemented | implemented |
| P5 | III | sgb.sensitivity | direct_sensitivity_implemented | implemented |
| P6 | III | sgb.sensitivity | direct_sensitivity_implemented | implemented |
| P7 | III | sgb.sensitivity | direct_sensitivity_implemented | implemented |
| P8 | III | sgb.sensitivity | direct_sensitivity_implemented | implemented |
| P9 | I | sgb.sensitivity | simulation_envelope_complete_public_calibration_complete | implemented |
| P10 | II | sgb.sensitivity | direct_sensitivity_implemented | implemented |
| P11 | II | sgb.sensitivity | direct_sensitivity_implemented | implemented |
| P12 | II | sgb.sensitivity | direct_sensitivity_implemented | implemented |
| P13 | III | sgb.sensitivity | direct_sensitivity_implemented | implemented |
| P14 | II | sgb.sensitivity | direct_sensitivity_implemented | implemented |
| P15 | II | population_convergence | linked_study_implemented | implemented |
| P16 | II | time_convergence | linked_study_implemented | implemented |
| P17 | II | seed_convergence | linked_study_implemented | implemented |
| P18 | II | steady_state_shock_and_recovery | linked_study_implemented | implemented |
| P19 | II | full_range_threshold_scan | linked_study_implemented | implemented |
| P20 | II | maturity_endpoint_verification | linked_study_implemented | implemented |
| P21 | II | sgb.sensitivity | direct_sensitivity_implemented | implemented |

## 5. Principal Experiment Results

### Experiment 1 — Framework comparison

| framework | sample_count | mean | standard_deviation | ci_lower | ci_upper | ci_width |
| --- | --- | --- | --- | --- | --- | --- |
| DBL | 30 | 0.143075 | 0.002217 | 0.142247 | 0.143903 | 0.001656 |
| DMM | 30 | 0.12307 | 0.001971 | 0.122335 | 0.123806 | 0.001472 |
| ISF | 30 | 0.154007 | 0.001722 | 0.153364 | 0.15465 | 0.001286 |

### Experiment 2 — Security recovery

| framework | mean_immediate_loss | recovered_fraction | mean_recovery_time | recovery_time_ci_lower | recovery_time_ci_upper |
| --- | --- | --- | --- | --- | --- |
| DBL | 0.001169 | 1 | 1 | 1 | 1 |
| DMM | 0.001169 | 1 | 1 | 1 | 1 |
| ISF | 0.001169 | 1 | 1 | 1 | 1 |

### Experiment 3 — Threshold recommendations

| framework | recommended_threshold | mean_f1_score | mean_balanced_accuracy | mean_event_coverage | mean_eligible_success_rate |
| --- | --- | --- | --- | --- | --- |
| DBL | 0 | 0.239598 | 0.5 | 1 | 0.136113 |
| DMM | 0 | 0.239598 | 0.5 | 1 | 0.136113 |
| ISF | 0 | 0.239598 | 0.5 | 1 | 0.136113 |

### Experiment 4 — Maturity sweep

| level_index | level | framework | theoretical_effective_mean | E_GMI_mean | E_GMI_ci_lower | E_GMI_ci_upper | success_rate_mean | success_rate_ci_lower | success_rate_ci_upper |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | low | DBL | 0.166667 | 0.052355 | 0.052239 | 0.05247 | 0.0502 | 0.048458 | 0.051942 |
| 0 | low | DMM | 0.166667 | 0.04555 | 0.045439 | 0.04566 | 0.0502 | 0.048458 | 0.051942 |
| 0 | low | ISF | 0.166667 | 0.057221 | 0.057116 | 0.057327 | 0.0502 | 0.048458 | 0.051942 |
| 1 | low_moderate | DBL | 0.333333 | 0.112023 | 0.11123 | 0.112816 | 0.10528 | 0.102735 | 0.107825 |
| 1 | low_moderate | DMM | 0.333333 | 0.097855 | 0.097053 | 0.098658 | 0.10528 | 0.102735 | 0.107825 |
| 1 | low_moderate | ISF | 0.333333 | 0.120161 | 0.119442 | 0.12088 | 0.10528 | 0.102735 | 0.107825 |
| 2 | neutral | DBL | 0.5 | 0.263539 | 0.260295 | 0.266782 | 0.237813 | 0.233089 | 0.242537 |
| 2 | neutral | DMM | 0.5 | 0.245733 | 0.242248 | 0.249217 | 0.237813 | 0.233089 | 0.242537 |
| 2 | neutral | ISF | 0.5 | 0.267814 | 0.264716 | 0.270911 | 0.237813 | 0.233089 | 0.242537 |
| 3 | high_moderate | DBL | 0.666667 | 0.538625 | 0.535526 | 0.541725 | 0.50408 | 0.498794 | 0.509366 |
| 3 | high_moderate | DMM | 0.666667 | 0.52673 | 0.523412 | 0.530047 | 0.50408 | 0.498794 | 0.509366 |
| 3 | high_moderate | ISF | 0.666667 | 0.535491 | 0.532465 | 0.538518 | 0.50408 | 0.498794 | 0.509366 |
| 4 | high | DBL | 0.833333 | 0.830514 | 0.828614 | 0.832413 | 0.81968 | 0.814821 | 0.824539 |
| 4 | high | DMM | 0.833333 | 0.825871 | 0.82386 | 0.827882 | 0.81968 | 0.814821 | 0.824539 |
| 4 | high | ISF | 0.833333 | 0.822772 | 0.820944 | 0.824599 | 0.81968 | 0.814821 | 0.824539 |

### Global sensitivity stability

| code | scenario | framework | absolute_E_GMI_difference | maximum_allowed_E_GMI_difference | absolute_threshold_difference | maximum_allowed_threshold_difference | ranking_match_fraction | minimum_required_ranking_match_fraction | mean_dimension_rank_correlation | minimum_required_dimension_rank_correlation | accepted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P1 | equal_standard_dimensions | DBL | 0.00276 | 0.02 | 0 | 0.1 | 0 | 0.9 |  | 0.9 | False |
| P1 | equal_standard_dimensions | DMM | 0.017373 | 0.02 | 0 | 0.1 | 0 | 0.9 |  | 0.9 | False |
| P1 | equal_standard_dimensions | ISF | 0.01746 | 0.02 | 0 | 0.1 | 0 | 0.9 |  | 0.9 | False |
| P1 | quality_heavy | DBL | 0.054691 | 0.02 | 0 | 0.1 | 0 | 0.9 |  | 0.9 | False |
| P1 | quality_heavy | DMM | 0.074824 | 0.02 | 0 | 0.1 | 0 | 0.9 |  | 0.9 | False |
| P1 | quality_heavy | ISF | 0.077253 | 0.02 | 0 | 0.1 | 0 | 0.9 |  | 0.9 | False |
| P1 | security_heavy | DBL | 0.036823 | 0.02 | 0 | 0.1 | 0 | 0.9 |  | 0.9 | False |
| P1 | security_heavy | DMM | 0.01669 | 0.02 | 0 | 0.1 | 0 | 0.9 |  | 0.9 | False |
| P1 | security_heavy | ISF | 0.003279 | 0.02 | 0 | 0.1 | 0 | 0.9 |  | 0.9 | False |
| P10 | concentrated_beta_shapes | DBL | 0.000092 | 0.02 | 0 | 0.1 | 1 | 0.9 |  | 0.9 | True |
| P10 | concentrated_beta_shapes | DMM | 0.000005 | 0.02 | 0 | 0.1 | 1 | 0.9 |  | 0.9 | True |
| P10 | concentrated_beta_shapes | ISF | 0.000019 | 0.02 | 0 | 0.1 | 1 | 0.9 |  | 0.9 | True |
| P10 | diffuse_beta_shapes | DBL | 0.001265 | 0.02 | 0 | 0.1 | 1 | 0.9 |  | 0.9 | True |
| P10 | diffuse_beta_shapes | DMM | 0.001095 | 0.02 | 0 | 0.1 | 1 | 0.9 |  | 0.9 | True |
| P10 | diffuse_beta_shapes | ISF | 0.000942 | 0.02 | 0 | 0.1 | 1 | 0.9 |  | 0.9 | True |
| P10 | moderate_beta_shapes | DBL | 0.000942 | 0.02 | 0 | 0.1 | 1 | 0.9 |  | 0.9 | True |
| P10 | moderate_beta_shapes | DMM | 0.00085 | 0.02 | 0 | 0.1 | 1 | 0.9 |  | 0.9 | True |
| P10 | moderate_beta_shapes | ISF | 0.000731 | 0.02 | 0 | 0.1 | 1 | 0.9 |  | 0.9 | True |
| P11 | narrow_volume_bounds | DBL | 0 | 0.02 | 0 | 0.1 | 1 | 0.9 |  | 0.9 | True |
| P11 | narrow_volume_bounds | DMM | 0 | 0.02 | 0 | 0.1 | 1 | 0.9 |  | 0.9 | True |
| P11 | narrow_volume_bounds | ISF | 0 | 0.02 | 0 | 0.1 | 1 | 0.9 |  | 0.9 | True |
| P11 | pareto_alpha_1_5 | DBL | 0 | 0.02 | 0 | 0.1 | 1 | 0.9 |  | 0.9 | True |
| P11 | pareto_alpha_1_5 | DMM | 0 | 0.02 | 0 | 0.1 | 1 | 0.9 |  | 0.9 | True |
| P11 | pareto_alpha_1_5 | ISF | 0 | 0.02 | 0 | 0.1 | 1 | 0.9 |  | 0.9 | True |
| P11 | pareto_alpha_3_0 | DBL | 0 | 0.02 | 0 | 0.1 | 1 | 0.9 |  | 0.9 | True |

_Only the first 25 of 129 rows are shown._

## 6. Reproducibility

The reproducibility manifest contains SHA-256 hashes for source code, tests, configuration files, dependency files, and the report inputs.

Number of hashed source and configuration files: **49**

## 7. Remaining Limitations

- Framework and dimension weights remain modeling choices and must be interpreted through the sensitivity results rather than as empirically estimated coefficients.
- Equal-width maturity categories are definitional partitions rather than externally validated maturity boundaries.
- P9 remains incomplete until organization-type proportions are calibrated against a documented public source and denominator.
- Synthetic results demonstrate computational behavior and robustness; they do not independently establish real-world causal validity.
