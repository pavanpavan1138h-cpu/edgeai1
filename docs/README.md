# Edge AI Battery-Life Prediction — Documentation Index

Welcome to the technical documentation repository for the **Edge AI Module 1 Assignment: Neural Networks from Scratch to Deployment**. This repository documents the end-to-end machine learning engineering workflow for **Team 3 (Phoenix)**, focusing on predicting the remaining battery life of an IoT smart doorbell.

---

## 📌 Project Information

* **Institution & Course:** Edge AI (5th Semester) — Module 1
* **Team:** 3 — **Phoenix**
* **Team Members:** Pavan R, Shreyank D K, Adarsh K R, Ankitha A
* **Dataset Assigned:** `data/team_03_train.csv` (2,000 raw telemetry observations)
* **Target Feature:** `days_left` (continuous remaining battery life in physical days)
* **Execution Environment:** Conda environment `edgeai1` (Python 3.11, PyTorch 2.15, FastAPI, Pandas, NumPy)

> [!IMPORTANT]
> ### 🎯 Confirmed Project Scope & Boundaries
> **Physical deployment onto an ESP32-S3 microcontroller is NOT required for this assignment.**
> * **In-Scope Implementation Deliverables:**
>   1. Telemetry data inspection, quality audit, and provisional sensor glitch cleaning.
>   2. Classical baseline modeling (Mean heuristic and OLS Linear Regression).
>   3. Neural network training, hyperparameter tuning, and convergence analysis.
>   4. Systematic architecture sweeps (width & depth trade-offs) and Pareto selection.
>   5. Multi-seed repeatability validation across five training random seeds.
>   6. Rigorous diagnostic residual and error distribution audits.
>   7. Multi-objective model selection and mathematical methodology derivation.
>   8. Necessary API endpoint integration via **FastAPI** (`app/main.py`) with Pydantic glitch refusal (HTTP 422).
>   9. API inference testing and latency stage profiling using `client/bench.py`.
>   10. Comprehensive, reproducible technical documentation across 11 formal modules.
> * **Out of Scope (Unless explicitly required later):**
>   - Physical ESP32-S3 flashing, hardware wiring, or lab oscilloscope power profiling.
>   - Embedded C/C++ firmware development or FreeRTOS task coding.
>   - Hardware-specific model conversions (ESP-DL / byte arrays) or deployment-only quantization.
> * **Role of Edge AI Constraints:** Microcontroller hardware constraints (e.g. ESP32-S3 $\approx 300\text{ KB}$ SRAM and clock cycles) are evaluated **theoretically** via parameter counts, Multiply-Accumulate operations (MACs), and memory footprint arithmetic to motivate and justify model compactness.

---

## 🧭 Comprehensive 11-Document Modular Architecture

The documentation is organized into 11 dedicated, cross-referenced technical reports providing complete experimental transparency:

| Document | Topic | Status | Scope & Content Summary |
| :--- | :--- | :---: | :--- |
| [**Experiment Log**](experiment_log.md) | Chronological Register (EXP-001 to EXP-006) | **Active** | Formal chronological ledger of all baselines, MLP runs, architecture sweeps, and multi-seed validations. |
| [**01. Problem Definition**](01_problem_definition.md) | Problem Scope & Edge AI Framing | **Completed** | Formulates doorbell battery life prediction as a supervised tabular regression problem with edge constraints. |
| [**02. Dataset & Data Quality**](02_dataset_and_data_quality.md) | Quality Audit & Glitch Filtering | **Completed** | Audits raw data (2,000 rows), verifies zero nulls/duplicates, and documents 24 provisional sensor glitches (1,976 clean). |
| [**03. EDA & Baselines**](03_eda_and_baselines.md) | Exploratory Analysis & Benchmarks | **Completed** | Analyzes distributions, non-linear discharge knees, and establishes Baseline 1 (Mean: 47.58d) and Baseline 2 (OLS: 20.54d). |
| [**04. Model Development**](04_model_development.md) | Neural Network Fundamentals | **Completed** | Details multilayer perceptron concepts, training-only z-score standardisation, Adam optimization, and loss formulations. |
| [**05. Experiment 1: Initial NN**](05_experiment_1_initial_nn.md) | Initial $5 \rightarrow 32 \rightarrow 32 \rightarrow 1$ Network | **Completed** | Reports baseline MLP achieving 4.13 days validation MAE (79.9% error reduction vs OLS) with 1,281 parameters. |
| [**06. Diagnostic Audit**](06_diagnostics.md) | Residual Audit & Error Distributions | **Completed** | 14-point audit on frozen checkpoint reproducing MAE 4.13d, MedAE 2.42d, low-battery MAE 1.19d, and 12 negative outputs. |
| [**07. Architecture Experiments**](07_architecture_experiments.md) | Architecture Sweep 1 (EXP-005) | **Completed** | Sweeps 10 models (1-layer vs 2-layer, widths 4..64); isolates 5 Pareto-optimal topologies; documents complexity reductions. |
| [**08. Repeatability Validation**](08_repeatability.md) | Multi-Seed Repeatability (EXP-006) | **Completed** | Benchmarks 3 shortlisted models across 5 random seeds (15 runs); proves 16–16 matches 32–32 within 0.20d (4.74 hours). |
| [**09. Model Selection**](09_model_selection.md) | Multi-Objective Engineering Decision | **Completed** | 9-step selection reasoning chain; decision matrix establishing 16–16 as recommended compromise and 32–32 as reference. |
| [**10. Mathematical Methodology**](10_mathematical_methodology.md) | Formulas, MACs & Derivations | **Completed** | Definitive mathematical manual: MAE, MSE, RMSE, MedAE, bias, parameter derivations, MAC calculations, FP32/INT8 RAM. |
| [**11. Assumptions & Reproducibility**](11_assumptions_and_reproducibility.md) | Assumptions Matrix & Checklist | **Completed** | 16-point assumptions table, epistemology of claims, team scope boundaries, and verified command execution protocol. |
| [**Teammate Deployment Protocol**](teammate_edge_deployment.md) | FastAPI Serving & Client Benchmarking | **Active** | Outlines teammate's FastAPI serving (`app/main.py`), Pydantic validation (HTTP 422), and benchmark client (`client/bench.py`). |

---

## 📊 Summary of Verified Results Across Experimental Phases

| Stage / Experiment ID | Model / Architecture | Trainable Params | MACs / Pass | Validation MAE | Multi-Seed Stability ($s$) | Low-Battery MAE ($y \le 10\text{d}$) | Key Engineering Finding |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **EXP-001 (Baseline 1)** | Predict Mean ($\bar{y} = 65.98\text{d}$) | 0 | 0 | $47.58\text{ days}$ | Deterministic | $49.32\text{ days}$ | Baseline floor; ignores all sensor telemetry. |
| **EXP-002 (Baseline 2)** | Linear Regression (OLS) | 6 | 5 | $20.54\text{ days}$ | Deterministic | $14.21\text{ days}$ | Linear baseline; fails at non-linear discharge knee. |
| **EXP-003 (Exp 1 MLP)** | $5 \rightarrow 32 \rightarrow 32 \rightarrow 1$ (Seed 0) | 1,281 | 1,216 | **$4.13\text{ days}$** | Single run | $1.19\text{ days}$ | **79.9% error reduction** over Linear Regression. |
| **EXP-004 (Audit)** | Diagnostic Residual Audit | 1,281 | 1,216 | **$4.13\text{ days}$** | Single run | $1.19\text{ days}$ (MedAE: $1.04\text{d}$) | Verified checkpoint reproducibility; identified 12 negative outputs. |
| **EXP-005 (Sweep 1)** | 10 Architectures (Widths 4..64) | 29 to 4,609 | 24 to 4,480 | $4.13\text{d}$ to $15.62\text{d}$ | Single run | Segmented | Two hidden layers dominate single layers; width 64 overfits. |
| **EXP-006 (Repeatability)**| 2-Layer (8, 8) across 5 seeds | 129 | 112 | $5.66\text{ days}$ | $\pm 1.32\text{ days}$ | $1.80\text{ days}$ | Extreme compression ($90\%$), but seed 4 optimization fragility. |
| **EXP-006 (Repeatability)**| **2-Layer (16, 16) across 5 seeds** | **385** | **352** | **$4.45\text{ days}$** | **$\pm 0.20\text{ days}$** | **$1.22\text{ days}$ (Best)** | **🏆 Recommended Edge Compromise:** Saves $70\%$ RAM & MACs for 4.7h MAE diff. |
| **EXP-006 (Repeatability)**| **2-Layer (32, 32) across 5 seeds** | **1,281** | **1,216** | **$4.26\text{ days}$** | **$\pm 0.16\text{ days}$** | **$1.28\text{ days}$** | **Accuracy Reference:** Lowest overall error; smooth overparameterized surface. |

---

## 🛠️ Reproducibility Quick Links

* **Experiment Logs & Tracking:**
  * [Chronological Experiment Log](experiment_log.md): Formal register of all baselines, MLP runs, and audits.
  * [Assumptions & Reproducibility Checklist](11_assumptions_and_reproducibility.md): Verified command execution guide.
  * [Team Mission Log (Raw)](../notebook/log.txt): Working experiment notes and glitch records.
* **Notebooks:**
  * [Dataset Inspection Notebook](../notebook/dataset_inspection.ipynb): Pure data auditing, zero modifications.
  * [EDA & Baselines Notebook](../notebook/eda.ipynb): Visualizations, correlations, baselines, and Experiment 1.
* **Executable Experiment Scripts:**
  * Architecture Sweep Script: [`experiments/architecture_sweep_1.py`](../experiments/architecture_sweep_1.py)
  * Repeatability Validation Script: [`experiments/shortlist_repeatability.py`](../experiments/shortlist_repeatability.py)
  * Baseline Training Pipeline: [`train.py`](../train.py)
  * API Microservice Server: [`app/main.py`](../app/main.py)
  * Client Benchmark Tool: [`client/bench.py`](../client/bench.py)
* **Model Checkpoints & Artifacts:**
  * Production Checkpoint (32-32): [`model/model.pt`](../model/model.pt)
  * Production Configuration: [`model/config.json`](../model/config.json)
  * Sweep 1 Artifacts: [`experiments/sweep_1_results/`](../experiments/sweep_1_results/)
  * Repeatability Artifacts: [`experiments/shortlist_repeatability/`](../experiments/shortlist_repeatability/)
