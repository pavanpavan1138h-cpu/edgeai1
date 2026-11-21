# Edge AI Battery-Life Prediction: Neural Networks from Scratch to Deployment

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.15-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.165-009688.svg)](https://fastapi.tiangolo.com/)
[![Team](https://img.shields.io/badge/Team-3%20(Phoenix)-orange.svg)](TEAMS.csv)
[![Validation%20MAE](https://img.shields.io/badge/Validation%20MAE-4.27%20days-success.svg)](model/config.json)

---

## 1. Project Title & Overview

**AI255TBD Edge AI — Module 1 Assignment**  
**Team 3 — Phoenix:** Pavan R, Shreyank D K, Adarsh K R, Ankitha A  

This project develops an end-to-end Edge AI solution to predict the remaining battery life of a smart doorbell from operational sensor telemetry. The workflow spans raw telemetry auditing, exploratory data analysis, classical baseline benchmarking, neural network design and training, model complexity evaluation, and lightweight edge microservice deployment via a FastAPI serving endpoint, accompanied by theoretical feasibility analysis for ultra-low-power microcontrollers (such as the Espressif ESP32-S3).

> [!IMPORTANT]
> ### 🎯 Project Scope Clarification: Physical ESP32 Deployment Not Required
> Physical deployment of the model onto an ESP32-S3 microcontroller is **NOT** a required deliverable for this assignment.
> * **Required Implementation Scope:** Telemetry data cleaning, baseline benchmarking, neural network experiments, architecture sweeps, multi-seed repeatability, model evaluation and selection, necessary FastAPI microservice integration (`app/main.py`), API inference testing and latency profiling (`client/bench.py`), and comprehensive documentation across 11 formal modules.
> * **Out of Scope (Unless explicitly required later):** Physical ESP32-S3 flashing, hardware wiring, C++ firmware development, on-device memory profiling, and hardware-specific model conversion.
> * **Role of Edge AI Discussion:** Embedded microcontroller constraints ($\approx 300\text{ KB}$ SRAM, clock cycles) are evaluated **theoretically** (parameter count, MACs, memory footprint) to motivate and justify model compactness during model selection.

---

## 2. Problem Statement & Motivation

Traditional smart doorbells report ambiguous battery status warnings such as "Low Battery", leaving homeowners uncertain whether the device will last several days or shut down within hours. 

The goal of this project is to replace coarse status flags with an accurate, continuous numerical prediction: **how many operational days remain before battery depletion (`days_left`)?**

### The Edge AI Motivation
While predictions could theoretically be computed in the cloud, smart doorbells benefit enormously from **on-device / edge inference**:
* **Connectivity Resilience:** The device maintains battery health monitoring even during home Wi-Fi or ISP outages.
* **Energy Savings:** Firing up a Wi-Fi radio to transmit telemetry to the cloud consumes significant battery energy; computing inference locally takes microjoules.
* **Privacy & Cost:** Telemetry remains on the device, eliminating ongoing cloud compute and API gateway costs.

---

## 3. Project Objectives

1. **Audit & Sanitize Telemetry:** Identify and isolate corrupt hardware readings in the Team 3 dataset (`data/team_03_train.csv`) without destroying valid operational extremes.
2. **Establish Benchmarks:** Measure classical statistical baselines (constant mean prediction and Ordinary Least Squares linear regression).
3. **Develop a Compact Neural Network:** Implement, standardize, and train a Multilayer Perceptron ($5 \rightarrow 32 \rightarrow 32 \rightarrow 1$) capable of modeling non-linear battery discharge curves.
4. **Evaluate Edge Efficiency (Theoretical):** Analyze memory footprint, Multiply-Accumulate operations (MACs), and theoretical execution feasibility against edge microcontroller constraints ($\approx 300\text{ KB}$ SRAM).
5. **Systematic Architecture Sweeps & Repeatability:** Sweep single- and two-hidden-layer architectures (widths 4..64) and evaluate repeatability across 5 random seeds to determine the Pareto-optimal edge architecture.
6. **API Integration & Testing:** Serve the model via a high-performance FastAPI endpoint with sub-millisecond stage timing, Pydantic glitch refusal (HTTP 422), and client latency benchmarking.

---

## 4. Machine Learning Formulation

* **Task Type:** Supervised Tabular Regression
* **Input Vector ($x \in \mathbb{R}^5$):** 5 continuous and discrete telemetry features
* **Target ($y \in \mathbb{R}^+$):** Continuous remaining battery lifespan in days (`days_left`)
* **Optimization Objective:** Mean Squared Error (MSE) loss
* **Evaluation Metric:** Mean Absolute Error (MAE), measured in physical days

---

## 5. Dataset Overview: Features & Target

The project uses the verified Team 3 dataset: `data/team_03_train.csv` (2,000 raw rows).

| Column Name | Role | Type | Unit | Physical Meaning | Normal Range |
| :--- | :--- | :---: | :---: | :--- | :---: |
| `battery_v` | Input Feature | `float64` | $V$ | Battery terminal voltage | $3.35\text{ V} - 4.20\text{ V}$ |
| `temp_c` | Input Feature | `float64` | $^\circ C$ | Operating ambient temperature | $8.0^\circ\text{C} - 46.0^\circ\text{C}$ |
| `rings_per_day` | Input Feature | `int64` | Count | Daily doorbell button presses | $0 - 38\text{ rings}$ |
| `wifi_rssi_dbm` | Input Feature | `int64` | $dBm$ | Wi-Fi Signal Strength Indicator | $-88\text{ dBm} - -45\text{ dBm}$ |
| `age_months` | Input Feature | `int64` | Months | Battery service installation age | $0 - 36\text{ months}$ |
| `days_left` | **Target** | `float64` | Days | Remaining operational battery life | $0.1 - 317.1\text{ days}$ |

---

## 6. Project Engineering Workflow

```
┌─────────────────────────────────┐
│     1. Data Inspection          │ ──► Audited 2,000 rows, 0 nulls, 0 duplicates
└────────────────┬────────────────┘     Identified 24 provisional sensor glitches (1.2%)
                 │
┌────────────────▼────────────────┐
│   2. EDA & Baseline Models      │ ──► Uncovered non-linear discharge knee
└────────────────┬────────────────┘     Baseline 1 (Mean): 47.58d MAE | Baseline 2 (LinReg): 20.54d MAE
                 │
┌────────────────▼────────────────┐
│  3. Neural Network Dev (Exp 1)  │ ──► Standardized features on training set only (80/20 split)
└────────────────┬────────────────┘     Trained 5 → 32 → 32 → 1 (ReLU) ──► Validation MAE: 4.13d
                 │
┌────────────────▼────────────────┐
│      4. Diagnostic Audit        │ ──► Residual analysis across voltage & lifespan bins [Completed]
└────────────────┬────────────────┘     Verified 12 raw negative predictions near cutoff (-2.68d min)
                 │
┌────────────────▼────────────────┐
│    5. Architecture Sweeps       │ ──► Hidden widths 4..64, depth trade-offs [EXP-005 Completed]
└────────────────┬────────────────┘     Identified 5 Pareto-optimal models
                 │
┌────────────────▼────────────────┐
│  6. Multi-Seed Repeatability    │ ──► Benchmarked 3 shortlisted models over 5 seeds [EXP-006 Completed]
└────────────────┬────────────────┘     16–16 matches 32–32 within 0.20d (4.74h) with 70% fewer params
                 │
┌────────────────▼────────────────┐
│  7. API Integration & Serving   │ ──► FastAPI endpoint (/health, /predict, /model_info)
└─────────────────────────────────┘     Pydantic HTTP 422 refusal, client benchmark, edge feasibility
```

---

## 7. Data Quality & Preprocessing Summary

* **Missing Values:** Exactly **0 missing values** across all columns (`0.00%`).
* **Duplicates:** Exactly **0 duplicate records**.
* **Provisional Sensor Glitch Filtering:**
  * `battery_v < 3.0 V`: 9 rows (all read $0.0\text{ V}$ due to ADC dropouts during transmission).
  * `temp_c < -10.0 °C`: 5 rows (all read $-127.0^\circ\text{C}$ due to 1-Wire bus disconnect error codes).
  * `wifi_rssi_dbm >= 0 dBm`: 10 rows (all read $0\text{ dBm}$ due to unassociated Wi-Fi stack defaults).
  * **Filtering Result:** 24 distinct corrupt rows dropped ($1.2\%$), leaving **1,976 clean records** ($98.8\%$).
  * *Integrity Note:* The raw CSV file remains completely untouched.
* **Train / Validation Split:** Clean records split 80% training (**1,581 rows**) and 20% validation (**395 rows**) with reproducible seed `0`.
* **Feature Standardisation ($z = \frac{x - \mu}{\sigma}$):** $\mu$ and $\sigma$ calculated **strictly on training rows** to prevent data leakage. Target `days_left` kept unnormalized.

---

## 8. Exploratory Data Analysis & Baseline Benchmarks

* **Target Characteristics:** Clean `days_left` spans $0.1$ to $317.1\text{ days}$ (Mean: $65.93\text{d}$, Median: $55.25\text{d}$, Std: $57.88\text{d}$, Skewness: $+1.173$).
* **Critical Non-Linearity:** `battery_v` vs `days_left` follows a non-linear Lithium-ion discharge curve with a sharp **discharge knee** around $3.5\text{ V} - 3.6\text{ V}$ where battery life drops exponentially to zero.
* **Correlations:** `battery_v` ($+0.730$), `rings_per_day` ($-0.264$), `age_months` ($-0.218$), `wifi_rssi_dbm` ($+0.198$), `temp_c` ($-0.065$). (Note: correlation measures linear association and does not establish physical causality).

### Baseline Benchmark Results:
* **Baseline 1 (Predict Mean $\bar{y} = 65.98\text{d}$):** Validation MAE = **$47.58\text{ days}$** (MSE: $3,354.21$).
* **Baseline 2 (Linear Regression via OLS):** Validation MAE = **$20.54\text{ days}$** (MSE: $792.14$).
  * *Failure Mode:* A straight hyperplane cannot bend at the $3.5\text{ V}$ knee, causing severe overestimation near battery depletion.

---

## 9. Experiment 1: Baseline Neural Network ($5 \rightarrow 32 \rightarrow 32 \rightarrow 1$)

* **Architecture:** Input (5) $\rightarrow$ Dense (32, ReLU) $\rightarrow$ Dense (32, ReLU) $\rightarrow$ Linear Output (1).
* **Training Setup:** Adam ($\alpha = 0.003$), Batch size $128$, Epochs $1,500$, Seed `0`, Loss: MSE.
* **Convergence Behavior:** Rapid convergence within 200 epochs; smooth descent to an asymptote.
* **Generalization Check:** Final Training MAE ($3.60\text{d}$) vs Final Validation MAE ($4.13\text{d}$) yields a minimal gap of $0.53\text{ days}$, showing no overt memorization.
* **Diagnostic Audit Findings (EXP-004):** Evaluated frozen `model/model.pt` ($4.1275\text{d}$ MAE, MedAE: $2.42\text{d}$). Identified 12 raw negative predictions near cutoff ($-2.68\text{d}$ min, $0.4-1.6\text{d}$ actual), which are clipped to zero in teammate serving code (`app/main.py`).

---

## 10. Performance Comparison Table

The table below synthesizes the experimental progression across classical baselines, initial neural network, architecture sweeps (EXP-005), and multi-seed repeatability testing (EXP-006):

| Stage / Model | Architecture | Trainable Params | MACs / Pass | Validation MAE | Multi-Seed Stability ($s$) | Low-Battery MAE ($y \le 10\text{d}$) | Selection Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Baseline 1** | Predict Mean | 0 | 0 | $47.58\text{ days}$ | Deterministic | $49.32\text{ days}$ | Rejected (Floor) |
| **Baseline 2** | Linear Regression (OLS) | 6 | 5 | $20.54\text{ days}$ | Deterministic | $14.21\text{ days}$ | Rejected (Cannot fit knee) |
| **Experiment 1 (Historical Ref)** | $5 \rightarrow 32 \rightarrow 32 \rightarrow 1$ (Seed 0) | 1,281 | 1,216 | **$4.13\text{ days}$** | Single run | $1.19\text{ days}$ | Historical accuracy reference |
| **Sweep 1: 1L-4** | $5 \rightarrow 4 \rightarrow 1$ | 29 | 24 | $10.53\text{ days}$ | Single run | $3.45\text{ days}$ | Pareto frontier; micro-baseline |
| **Sweep 1: 2L-4-4** | $5 \rightarrow 4 \rightarrow 4 \rightarrow 1$ | 49 | 40 | $8.08\text{ days}$ | Single run | $2.89\text{ days}$ | Pareto frontier; sub-50 parameter |
| **Repeatability: 8–8** | $5 \rightarrow 8 \rightarrow 8 \rightarrow 1$ (5 Seeds) | 129 | 112 | $5.66\text{ days}$ | $\pm 1.32\text{ days}$ | $1.80\text{ days}$ | Lightweight fallback (Seed 4 sensitivity) |
| **Repeatability: 16–16**| **$5 \rightarrow 16 \rightarrow 16 \rightarrow 1$ (5 Seeds)**| **385** | **352** | **$4.45\text{ days}$** | **$\pm 0.20\text{ days}$** | **$1.22\text{ days}$ (Best)** | **🏆 Recommended Engineering Compromise** |
| **Repeatability: 32–32**| **$5 \rightarrow 32 \rightarrow 32 \rightarrow 1$ (5 Seeds)**| **1,281** | **1,216** | **$4.26\text{ days}$** | **$\pm 0.16\text{ days}$** | **$1.28\text{ days}$** | **Accuracy Reference Benchmark** |

**Active API checkpoint:** the recommended **2-Layer (16, 16), seed 1** model is served from `model/model.pt`. Its seed-specific validation MAE is **4.27 days**; the five-seed mean is **4.45 ± 0.20 days**. It has **385 parameters** and **352 MACs**.

*The recommended **2-Layer (16, 16)** model saves **$69.95\%$ of parameters** and **$71.05\%$ of MACs** compared to the 32–32 reference, for an average trade-off of only **$0.1974\text{ days}$ ($4.74\text{ hours}$)** in validation MAE, while achieving superior precision in the critical $0 - 10\text{ day}$ near-depletion operating regime.*

---

## 11. Model Complexity & Theoretical Edge AI Feasibility (ESP32-S3 Context)

> [!NOTE]
> **Theoretical Feasibility Study:** The ESP32-S3 microcontroller figures below represent theoretical hardware feasibility analysis to evaluate model efficiency against realistic Edge AI constraints. Physical flashing, C++ firmware development, and on-chip hardware profiling are out of scope.

### Efficiency Comparison Across Shortlisted Models:
| Metric | 2-Layer (8, 8) | **2-Layer (16, 16) [Recommended]** | 2-Layer (32, 32) [Reference] | ESP32-S3 Application Budget |
| :--- | :---: | :---: | :---: | :---: |
| **Trainable Parameters** | 129 | **385 (-69.95%)** | 1,281 (0%) | N/A |
| **Multiply-Accumulates (MACs)** | 112 | **352 (-71.05%)** | 1,216 (0%) | $\approx 240\times 10^6\text{ cycles/s}$ |
| **Raw FP32 Storage** | $516\text{ bytes}$ | **$1,540\text{ bytes}$ ($1.50\text{ KB}$)** | $5,124\text{ bytes}$ ($5.00\text{ KB}$)| $\approx 300\text{ KB}$ Free SRAM |
| **Theoretical INT8 Storage** | $129\text{ bytes}$ | **$385\text{ bytes}$ ($0.38\text{ KB}$)** | $1,281\text{ bytes}$ ($1.25\text{ KB}$)| $\approx 300\text{ KB}$ Free SRAM |
| **SRAM Share (FP32)** | $0.17\%$ | **$0.51\%$** | $1.71\%$ | $100\%$ ($300\text{ KB}$) |
| **Mean Validation MAE** | $5.66\text{ days}$ | **$4.45\text{ days}$** | **$4.26\text{ days}$** | Low-error threshold |
| **Near-Depletion MAE ($0-10\text{d}$)**| $1.80\text{ days}$ | **$1.22\text{ days}$ (Best)** | $1.28\text{ days}$ | Critical warning threshold |

* **ESP32-S3 Microcontroller Fit:** The recommended 16–16 model consumes barely **$0.51\%$** of available application SRAM ($1.50\text{ KB}$), leaving ample headroom for Wi-Fi stack buffers, FreeRTOS task allocations, and peripheral drivers.
* **Compute Budget:** At $352\text{ MACs}$, a forward inference pass executes in under **$0.01\text{ milliseconds}$** on a $240\text{ MHz}$ dual-core Xtensa processor.

---

## 12. Current Project Status & Remaining Work

### Current Completed Milestones:
- [x] **Mission 1 Completed:** Telemetry audited, provisional sensor glitches isolated (24 rows dropped), Mean and Linear Regression baselines established ([docs/02_dataset_and_data_quality.md](docs/02_dataset_and_data_quality.md), [docs/03_eda_and_baselines.md](docs/03_eda_and_baselines.md)).
- [x] **Mission 2 (Experiment 1) Completed:** $5 \rightarrow 32 \rightarrow 32 \rightarrow 1$ trained, input standardization verified, validation MAE reached $4.13\text{ days}$, model artifacts saved (`model/model.pt`, `model/config.json`) ([docs/05_experiment_1_initial_nn.md](docs/05_experiment_1_initial_nn.md)).
- [x] **Diagnostic Audit Completed (EXP-004):** Executed the 14-point residual and error distribution audit ([docs/06_diagnostics.md](docs/06_diagnostics.md)): confirmed reproduced MAE 4.13d, Median AE 2.42d, 12 negative near-cutoff outputs, and low-battery MAE 1.19d.
- [x] **Architecture Sweep Completed (EXP-005):** Evaluated 10 architectures across widths 4..64 and depths 1..2 ([docs/07_architecture_experiments.md](docs/07_architecture_experiments.md)): identified 5 Pareto-optimal models.
- [x] **Multi-Seed Repeatability Completed (EXP-006):** Benchmarked 3 shortlisted architectures across 5 random seeds (15 runs) ([docs/08_repeatability.md](docs/08_repeatability.md)): verified 16–16 matches 32–32 within 0.20d (4.74 hours) with superior low-battery precision ($1.22\text{d}$).
- [x] **Model Selection Completed:** Formulated 9-step decision chain designating `2-Layer (16, 16)` as the recommended engineering compromise ([docs/09_model_selection.md](docs/09_model_selection.md)).
- [x] **Mathematical Methodology Published:** Documented formulas, parameter derivations, and MAC counts across all 10 models ([docs/10_mathematical_methodology.md](docs/10_mathematical_methodology.md)).
- [x] **Assumptions & Reproducibility Matrix Published:** 16-point assumptions matrix, team scope boundaries, and verified command execution protocol ([docs/11_assumptions_and_reproducibility.md](docs/11_assumptions_and_reproducibility.md)).

### Teammate Upcoming Tasks:
- [x] **API Serving Implementation:** Pydantic validation, dynamic `/model_info`, stage timings, and single/batch prediction endpoints are implemented ([app/main.py](app/main.py)).
- [x] **API Profiling:** Executed `client/bench.py` with 200 requests against the active 16–16 checkpoint ([benchmark_results_16x16_selected_model.csv](benchmark_results_16x16_selected_model.csv)).

---

## 13. Repository Structure

```
edge_ai_assignment1/
├── README.md                      # Main project entry point (this document)
├── TEAMS.csv                      # Official team roster
├── requirements.txt               # Project dependencies
├── train.py                       # Training pipeline & model exporter
├── app.py                         # Convenience launcher for the FastAPI service
├── app/
│   ├── model.py                   # PyTorch BatteryNet architecture
│   └── main.py                    # FastAPI live serving microservice
├── client/
│   └── bench.py                   # Latency benchmarking client
├── data/
│   └── team_03_train.csv          # Team 3 raw training dataset
├── experiments/
│   ├── architecture_sweep_1.py    # Architecture sweep runner (10 models)
│   ├── sweep_1_results/           # CSV results, weights, and Pareto plots
│   ├── shortlist_repeatability.py # Multi-seed repeatability study (5 seeds)
│   └── shortlist_repeatability/   # Multi-seed CSVs, weights, and stability plots
├── model/
│   ├── model.pt                   # Exported PyTorch weights (7,869 bytes)
│   └── config.json                # Normalization stats (mu, sd) & configuration
├── notebook/
│   ├── dataset_inspection.ipynb   # Pure Data Inspection notebook
│   ├── eda.ipynb                  # EDA, Baselines & Experiment 1 notebook
│   └── log.txt                    # Project experiment & conversation history log
└── docs/                          # Detailed modular documentation (11 Modules + Logs)
    ├── README.md                  # Documentation index & progress tracker
    ├── experiment_log.md          # Chronological experiment log (EXP-001 to EXP-006)
    ├── 01_problem_definition.md   # Problem statement & Edge AI motivation
    ├── 02_dataset_and_data_quality.md # Data auditing & glitch investigation
    ├── 03_eda_and_baselines.md    # Distributions, correlations & baselines
    ├── 04_model_development.md    # Neural network theory & standardization
    ├── 05_experiment_1_initial_nn.md # Experiment 1 detailed scientific report
    ├── 06_diagnostics.md          # 14-point diagnostic audit report (Completed)
    ├── 07_architecture_experiments.md # Architecture sweep report & Pareto analysis (EXP-005)
    ├── 08_repeatability.md        # Shortlist multi-seed repeatability validation (EXP-006)
    ├── 09_model_selection.md      # Multi-objective model selection decision
    ├── 10_mathematical_methodology.md # Mathematical formulas, MACs & parameter derivations
    ├── 11_assumptions_and_reproducibility.md # Assumptions matrix & reproducibility checklist
    └── teammate_edge_deployment.md# FastAPI serving, testing & theoretical edge feasibility
```

---

## 14. How to Set Up & Run the Project

### Environment Setup
Create and activate the repository-local virtual environment.

**GitHub Codespaces / Linux:**
```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

**Windows PowerShell:**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Retrain the Selected Architecture
The checked-in serving checkpoint is the repeatability experiment's 16-16 seed-1 model. Retraining writes new weights and configuration to `model/` and replaces that checkpoint; the following command trains a new seed-0 16-16 model, so its validation metric will differ from 4.27 days:
```bash
python train.py --team 3 --hidden 16,16 --epochs 1500
```

### Reproduce Architecture Experiments
Run the 10-architecture sweep (EXP-005):
```bash
python experiments/architecture_sweep_1.py
```

Run the 5-seed shortlist repeatability validation (EXP-006):
```bash
python experiments/shortlist_repeatability.py
```

### Start the Live FastAPI Server
Launch the local API server:
```bash
python app.py
```
* Interactive API Documentation: `http://localhost:8000/docs`
* Health Check: `http://localhost:8000/health`
* Edge Model Info: `http://localhost:8000/model_info`

### Run Client Benchmarks
Benchmark latency percentiles (p50/p95/p99) from a client terminal:
```bash
python client/bench.py --url http://localhost:8000 --n 200
```

---

## 15. Modular Documentation Links

For comprehensive technical deep-dives into each project phase, consult the 11 dedicated modular documents in [`docs/`](docs/):

* [**Chronological Experiment Log**](docs/experiment_log.md)
* [**01. Problem Definition & Formulation**](docs/01_problem_definition.md)
* [**02. Dataset Audit & Data Quality**](docs/02_dataset_and_data_quality.md)
* [**03. EDA & Baseline Benchmarks**](docs/03_eda_and_baselines.md)
* [**04. Neural Network Development Methodology**](docs/04_model_development.md)
* [**05. Experiment 1: Baseline Neural Network Report**](docs/05_experiment_1_initial_nn.md)
* [**06. Diagnostic Audit Report (Completed)**](docs/06_diagnostics.md)
* [**07. Architecture Exploration Report (EXP-005)**](docs/07_architecture_experiments.md)
* [**08. Shortlist Multi-Seed Repeatability Validation (EXP-006)**](docs/08_repeatability.md)
* [**09. Model Selection & Multi-Objective Engineering Decision**](docs/09_model_selection.md)
* [**10. Mathematical Methodology, Formulas, MACs & Derivations**](docs/10_mathematical_methodology.md)
* [**11. Assumptions, Limitations & Reproducibility Checklist**](docs/11_assumptions_and_reproducibility.md)
* [**Teammate Serving: API Integration & Latency Benchmarking**](docs/teammate_edge_deployment.md)

