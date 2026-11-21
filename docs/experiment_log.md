# Edge AI Smart Doorbell — Chronological Experiment Log

---

## 📌 Document Overview & Purpose

This document serves as the single source of truth for all quantitative experiments conducted by **Team 3 (Phoenix)** for the **Edge AI Module 1 Assignment: Neural Networks from Scratch to Deployment**. 

Every experiment, baseline benchmark, model configuration, and diagnostic evaluation is systematically recorded here in chronological order. Another engineer or researcher can follow this log to reproduce our historical trajectory, inspect exact metrics, understand technical decisions, and avoid repeating discarded approaches.

> [!IMPORTANT]
> ### 🎯 Project Scope Clarification (Recorded Nov 21, 2026)
> **Physical ESP32-S3 deployment is NOT a required deliverable for this assignment.**
> * **Required Scope:** Data analysis & cleaning, baseline benchmarking, neural network training/sweeps, model evaluation & selection, necessary FastAPI microservice integration (`app/main.py`), API inference testing with `client/bench.py`, and comprehensive documentation.
> * **Out of Scope:** Physical ESP32-S3 flashing, wiring, C++ firmware development, on-device memory profiling, and hardware-specific model conversion. Microcontroller constraints ($\approx 300\text{ KB}$ SRAM) are evaluated theoretically to guide model compactness.

---

## 🧭 Experiment Registry Index

| Experiment ID | Date | Category | Model / Description | Validation MAE | Validation MSE | Edge Status | Status | Document Link |
| :---: | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **EXP-001** | Nov 20, 2026 | Baseline | Mean Target Predictor ($\bar{y} \approx 65.98\text{d}$) | $47.58\text{ days}$ | $3,354.21$ | 1 param (0 MACs) | **Completed** | [Jump to EXP-001](#exp-001-baseline-1--constant-mean-predictor) |
| **EXP-002** | Nov 20, 2026 | Baseline | Linear Regression (OLS via `np.linalg.lstsq`) | $20.54\text{ days}$ | $792.14$ | 6 params (5 MACs) | **Completed** | [Jump to EXP-002](#exp-002-baseline-2--ordinary-least-squares-linear-regression) |
| **EXP-003** | Nov 20, 2026 | Neural Net | Initial MLP ($5 \rightarrow 32 \rightarrow 32 \rightarrow 1$, ReLU) | **$4.13\text{ days}$** | **$36.24$** | 1,281 params (1,216 MACs) | **Completed** | [Jump to EXP-003](#exp-003-experiment-1--initial-neural-network-5--32--32--1) |
| **EXP-004** | Nov 21, 2026 | Audit | Diagnostic Residual & Distribution Audit | **$4.13\text{ days}$** (MedAE: $2.42\text{d}$) | **$36.24$** | 12 raw negative outputs | **Completed** | [Jump to EXP-004](#exp-004-diagnostic-audit-of-frozen-saved-checkpoint) |
| **EXP-005** | Nov 21, 2026 | Architecture | Sweep 1 (10 Models: 1-Layer vs 2-Layer, widths 4..64) | **$4.13\text{ days}$** (Best: 32-32) | **$36.24$** | 5 Pareto models identified | **Completed** | [Jump to EXP-005](#exp-005-architecture-sweep-1--model-sizing--pareto-analysis) |
| **EXP-006** | Nov 21, 2026 | Repeatability | Shortlist Multi-Seed Validation (3 Models x 5 Seeds) | **$4.26\text{ days}$** (32-32) / **$4.45\text{ days}$** (16-16) | **$39.70$** (32-32) | 16-16 matches 32-32 within 0.20d | **Completed** | [Jump to EXP-006](#exp-006-shortlist-architecture-repeatability--multi-seed-validation) |
| *EXP-007* | *Upcoming* | Integration | API Endpoint Integration & Latency Benchmarking | *TBD* | *TBD* | FastAPI / client bench | *Planned* | [`docs/teammate_edge_deployment.md`](teammate_edge_deployment.md) |

---

## EXP-001: Baseline 1 — Constant Mean Predictor

* **Experiment ID:** `EXP-001`
* **Date:** November 20, 2026
* **Investigator:** Team 3 (Phoenix)
* **Status:** Completed & Verified
* **Related Notebook Cell:** `notebook/eda.ipynb` (Section 3)
* **Detailed Documentation:** [`docs/03_eda_and_baselines.md`](03_eda_and_baselines.md)

### 1. Objective
Establish the theoretical floor for predictive accuracy by measuring the error of a trivial model that ignores all input features and constantly predicts the arithmetic mean of the training target ($\bar{y}_{\text{train}}$).

### 2. Methodology & Configuration
* **Dataset:** `data/team_03_train.csv` (1,976 clean records after filtering 24 hardware glitches).
* **Split:** 80% train ($1,581\text{ records}$), 20% validation ($395\text{ records}$), random seed 0 (`np.random.default_rng(0)`).
* **Predictor Formulation:**
  $$\hat{y}_i = \bar{y}_{\text{train}} = \frac{1}{N_{\text{train}}} \sum_{j=1}^{N_{\text{train}}} y_j \approx 65.9814\text{ days}$$
* **Evaluation:** Evaluate $\hat{y}$ on the 395 unseen validation samples.

### 3. Quantitative Results
* **Training Mean ($\bar{y}_{\text{train}}$):** $65.98\text{ days}$
* **Validation Target Standard Deviation ($\sigma_{\text{val}}$):** $57.92\text{ days}$
* **Validation Mean Absolute Error (MAE):** **$47.58\text{ days}$**
* **Validation Mean Squared Error (MSE):** **$3,354.21$**
* **Validation Root Mean Squared Error (RMSE):** **$57.92\text{ days}$**

### 4. Interpretation & Key Takeaways
* An average error of $47.58\text{ days}$ (over $1.5\text{ months}$) confirms that a non-predictive heuristic is useless for real-world IoT operations.
* Any valid machine learning model must achieve a validation MAE substantially below $47.58\text{ days}$ to prove learned utility.

---

## EXP-002: Baseline 2 — Ordinary Least Squares Linear Regression

* **Experiment ID:** `EXP-002`
* **Date:** November 20, 2026
* **Investigator:** Team 3 (Phoenix)
* **Status:** Completed & Verified
* **Related Notebook Cell:** `notebook/eda.ipynb` (Section 4)
* **Detailed Documentation:** [`docs/03_eda_and_baselines.md`](03_eda_and_baselines.md)

### 1. Objective
Quantify how much predictive accuracy can be achieved using a purely linear hyper-plane, and determine whether linear regression is sufficient for smart doorbell battery life prediction.

### 2. Methodology & Configuration
* **Dataset & Split:** Same 1,976 clean records, 80/20 train/val split, seed 0.
* **Algorithm:** Ordinary Least Squares (OLS) closed-form solution via `np.linalg.lstsq`.
* **Model Equation:**
  $$\hat{y} = w_0 + w_1 \cdot \text{battery\_v} + w_2 \cdot \text{temp\_c} + w_3 \cdot \text{rings\_per\_day} + w_4 \cdot \text{wifi\_rssi\_dbm} + w_5 \cdot \text{age\_months}$$

### 3. Quantitative Results & Learned Coefficients
* **Learned Weights ($w$):**
  * Intercept ($w_0$): $-477.7202$
  * `battery_v` ($w_1$): **$+181.9049$** (Strongest positive driver; $+1\text{V} \implies +181.9\text{ days}$)
  * `temp_c` ($w_2$): $-0.2486$
  * `rings_per_day` ($w_3$): **$-2.5595$** (Significant negative driver; each ring costs $\approx 2.56\text{ days}$)
  * `wifi_rssi_dbm` ($w_4$): **$+1.5008$** (Poorer signal / more negative dBm reduces lifespan)
  * `age_months` ($w_5$): **$-1.0167$** (Older cells lose capacity; each month costs $\approx 1.02\text{ days}$)
* **Validation MAE:** **$20.54\text{ days}$**
* **Validation MSE:** **$792.14$**
* **Validation RMSE:** **$28.14\text{ days}$**
* **Improvement over Baseline 1:** **$56.8\%$ error reduction** ($47.58\text{d} \rightarrow 20.54\text{d}$).

### 4. Interpretation & Failure Modes
* While OLS captures the broad linear relationships (voltage dominates, rings/age drain), an error of **$20.54\text{ days}$ ($\sim 3\text{ weeks}$)** remains unacceptable for user notifications.
* **Failure Mode 1 (The Discharge Knee):** Electrochemical discharge curves are non-linear. Near battery depletion ($< 3.6\text{ V}$), voltage drops exponentially. A linear slope severely overestimates lifespan here, leading to sudden unannounced shutdowns.
* **Failure Mode 2 (Feature Interactions):** OLS cannot model coupling (e.g., high ring activity combined with weak Wi-Fi causes the radio to retransmit packets longer, multiplying battery drain).

---

## EXP-003: Experiment 1 — Initial Neural Network ($5 \rightarrow 32 \rightarrow 32 \rightarrow 1$)

* **Experiment ID:** `EXP-003`
* **Date:** November 20, 2026
* **Investigator:** Team 3 (Phoenix)
* **Status:** Completed & Checkpoint Saved
* **Executable Script:** `train.py --team 3 --hidden 32,32 --epochs 1500`
* **Exported Checkpoint:** `model/model.pt` (7,869 bytes), `model/config.json`
* **Detailed Documentation:** [`docs/05_experiment_1_initial_nn.md`](05_experiment_1_initial_nn.md)

### 1. Objective
Design, train, and validate a compact Multilayer Perceptron (MLP) capable of learning the non-linear battery discharge curve and feature interactions, while maintaining a tiny memory footprint suitable for an ESP32-S3 microcontroller ($\approx 300\text{ KB}$ SRAM).

### 2. Methodology & Configuration
* **Dataset & Cleaning:** Same 1,976 clean records from `data/team_03_train.csv`.
* **Split:** 80/20 train/validation split with `np.random.default_rng(0)` ($1,581\text{ train}$, $395\text{ val}$).
* **Feature Normalization:** Standardized using z-score parameters derived **strictly from the training partition**:
  * `battery_v`: $\mu = 3.7828\text{ V}$, $\sigma = 0.2448\text{ V}$
  * `temp_c`: $\mu = 27.9880^\circ\text{C}$, $\sigma = 6.8524^\circ\text{C}$
  * `rings_per_day`: $\mu = 7.8722$, $\sigma = 5.4824$
  * `wifi_rssi_dbm`: $\mu = -66.2296\text{ dBm}$, $\sigma = 12.4832\text{ dBm}$
  * `age_months`: $\mu = 18.0145\text{ m}$, $\sigma = 10.8142\text{ m}$
* **Architecture:** $5 \rightarrow 32 \rightarrow 32 \rightarrow 1$
  * Layer 1: Linear($5 \rightarrow 32$) + ReLU
  * Layer 2: Linear($32 \rightarrow 32$) + ReLU
  * Layer 3: Linear($32 \rightarrow 1$) (Linear output)
* **Training Hyperparameters:**
  * Optimizer: Adam ($\beta_1 = 0.9, \beta_2 = 0.999$)
  * Learning Rate: $\eta = 3 \times 10^{-3}$
  * Batch Size: $128$
  * Epochs: $1,500$
  * Loss Function: Mean Squared Error (MSE)
  * Random Seed: $0$

### 3. Quantitative Results & Training Trajectory
* **Without Feature Normalization (Ablation):** Validation $\text{MAE} = 12.05\text{ days}$ (gradients destabilized by differing feature scales).
* **With Feature Normalization:**
  * Final Training MAE: **$3.60\text{ days}$** (Training MSE: $30.29$)
  * Final Validation MAE: **$4.13\text{ days}$** ($4.1275\text{d}$, Validation MSE: $36.24$)
  * Best Historical Validation MAE: **$3.96\text{ days}$** (Observed at Epoch 979)
* **Overfitting Gap:** $\text{Val MAE} - \text{Train MAE} = 4.13 - 3.60 = 0.53\text{ days}$. Indicates excellent generalization with minimal overfitting.
* **Comparative Error Reduction:**
  * vs Baseline 1 (Mean): **$91.3\%$ error reduction** ($47.58\text{d} \rightarrow 4.13\text{d}$)
  * vs Baseline 2 (Linear Reg): **$79.9\%$ error reduction** ($20.54\text{d} \rightarrow 4.13\text{d}$)

### 4. Edge Complexity Metrics
* **Total Parameters:** $1,281$ float32 parameters ($192 + 1,056 + 33$)
* **Inference MACs:** $1,216\text{ MACs}$ per forward pass
* **Memory Footprint (FP32):** $5,124\text{ bytes}$ ($5.00\text{ KB}$)
* **Memory Footprint (INT8 Quantized):** $1,281\text{ bytes}$ ($1.25\text{ KB}$)
* **Hardware Budget Feasibility:** Fits easily within an ESP32-S3's $300\text{ KB}$ SRAM budget ($< 1.7\%$ utilization).

### 5. Checkpoint Preservation Anomaly
* The training script `train.py` saved weights at the conclusion of epoch 1499 unconditionally, rather than caching the epoch 979 checkpoint ($3.96\text{d}$).
* The saved artifact `model/model.pt` contains the epoch 1499 weights ($4.13\text{d}$).

---

## EXP-004: Diagnostic Audit of Frozen Saved Checkpoint

* **Experiment ID:** `EXP-004`
* **Date:** November 21, 2026
* **Investigator:** Team 3 (Phoenix)
* **Status:** Completed & Verified
* **Target Artifacts:** `model/model.pt`, `model/config.json`, `data/team_03_train.csv` (Val slice, $N=395$)
* **Detailed Documentation:** [`docs/06_diagnostics.md`](06_diagnostics.md)

### 1. Objective
Subject the frozen saved checkpoint (`model/model.pt`) to an exhaustive diagnostic audit to uncover physical edge cases, residual skewness, lifecyle-segmented errors, and operational edge failure modes without retraining.

### 2. Summary of Key Audit Findings
1. **Discrepancy Resolution:** Checkpoint was verified in isolation, reproducing exactly **Validation $\text{MAE} = 4.1275\text{ days}$** ($4.13\text{d}$), **$\text{MSE} = 36.2441$**, and **$\text{RMSE} = 6.0203\text{ days}$**.
2. **Median vs Mean Error:** While the mean MAE is $4.13\text{ days}$, the **Median Absolute Error is only $2.42\text{ days}$**. Over 50% of predictions are within $2.42\text{ days}$ of reality; the mean is pulled upward by high-lifespan outliers.
3. **Imminent Depletion Precision ($0 - 10\text{ days}$):** In the near-depletion evaluation subset ($0 - 10\text{ days}$, $85\text{ validation samples}$), the model achieves **$\text{MAE} = 1.193\text{ days}$** and **$\text{Median AE} = 1.035\text{ days}$**.
4. **Physical Sanity Warning (12 Negative Outputs):** 12 predictions ($3.04\%$) produced raw negative days (range: $-0.03\text{d}$ to $-2.68\text{d}$). All 12 occur exclusively when the battery is near cutoff ($0.4\text{d} - 1.6\text{d}$ actual life) due to the unconstrained linear output neuron and global negative bias ($-1.22\text{d}$).
5. **Mitigation in Production:** Clamped via `np.maximum(y, 0)` in `app/main.py`.
6. **Worst-Case Error Profile:** The top 10 worst errors occurred exclusively on devices with $> 75\text{ days}$ remaining (maximum error $26.33\text{ days}$ on a device with $137.6\text{ days}$ actual life). Zero low-battery failures were detected.

---

## EXP-005: Architecture Sweep 1 — Model Sizing & Pareto Analysis

* **Experiment ID:** `EXP-005`
* **Date:** November 21, 2026
* **Investigator:** Team 3 (Phoenix)
* **Status:** Completed & Verified
* **Executable Script:** `experiments/architecture_sweep_1.py`
* **Artifact Directory:** `experiments/sweep_1_results/`
* **Detailed Documentation:** [`docs/07_architecture_experiments.md`](07_architecture_experiments.md)

### 1. Objective
Conduct a controlled architecture sweep comparing single-hidden-layer ($5 \rightarrow w \rightarrow 1$) and two-hidden-layer ($5 \rightarrow w \rightarrow w \rightarrow 1$) neural networks of widths $w \in \{4, 8, 16, 32, 64\}$ to map the accuracy-complexity Pareto frontier.

### 2. Experimental Controls
* **Dataset & Cleaning:** Same 1,976 clean records from `data/team_03_train.csv` (24 glitches removed).
* **Split:** Exact 80/20 train/validation split (`np.random.default_rng(0)`, $1,581\text{ train} / 395\text{ val}$).
* **Normalization:** Training-only $\mu$ and $\sigma$ applied to both sets. Zero validation leakage.
* **Optimization:** Adam ($\eta = 3 \times 10^{-3}$), batch size $128$, $1,500\text{ epochs}$, loss = MSE.
* **Seed:** `torch.manual_seed(0)` initialized before each architecture.
* **Safety:** Reference checkpoint `model/model.pt` and `model/config.json` remained untouched.

### 3. Summary of Sweep Results (10 Models Evaluated)

| Architecture | Hidden Layers | Widths | Params | MACs | Val MAE (days) | Val MSE | Best Val MAE (Epoch) | Training Time | Pareto Optimal? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`1-Layer (4)`** | 1 | 4 | **29** | **24** | **$10.53\text{ d}$** | $193.45$ | $10.53\text{ d}$ (1490) | $3.37\text{ s}$ | **YES** |
| `1-Layer (8)` | 1 | 8 | 57 | 48 | $15.62\text{ d}$ | $465.37$ | $15.47\text{ d}$ (490) | $3.36\text{ s}$ | NO |
| `1-Layer (16)` | 1 | 16 | 113 | 96 | $15.53\text{ d}$ | $460.15$ | $15.49\text{ d}$ (380) | $3.51\text{ s}$ | NO |
| `1-Layer (32)` | 1 | 32 | 225 | 192 | $7.22\text{ d}$ | $94.93$ | $7.22\text{ d}$ (1499) | $3.56\text{ s}$ | NO |
| `1-Layer (64)` | 1 | 64 | 449 | 384 | $15.52\text{ d}$ | $460.02$ | $15.45\text{ d}$ (540) | $3.67\text{ s}$ | NO |
| **`2-Layer (4, 4)`** | 2 | 4, 4 | **49** | **40** | **$8.08\text{ d}$** | $126.40$ | $7.96\text{ d}$ (630) | $4.36\text{ s}$ | **YES** |
| **`2-Layer (8, 8)`** | 2 | 8, 8 | **129** | **112** | **$5.08\text{ d}$** | $53.37$ | $4.89\text{ d}$ (720) | $4.57\text{ s}$ | **YES** |
| **`2-Layer (16, 16)`**| 2 | 16, 16 | **385** | **352** | **$4.78\text{ d}$** | $49.30$ | $4.68\text{ d}$ (1470) | $4.64\text{ s}$ | **YES** |
| **`2-Layer (32, 32)`**| 2 | 32, 32 | **1,281**| **1,216**| **$4.13\text{ d}$** | **$36.24$** | $3.98\text{ d}$ (1120) | $4.77\text{ s}$ | **YES (Best)** |
| `2-Layer (64, 64)` | 2 | 64, 64 | 4,609 | 4,480 | $4.55\text{ d}$ | $44.70$ | $4.16\text{ d}$ (360) | $6.48\text{ s}$ | NO (Overfit) |

### 4. Key Takeaways & Pareto Shortlist
1. **Reference Checkpoint Reproducibility:** The reference $5 \rightarrow 32 \rightarrow 32 \rightarrow 1$ reproduced exactly $4.13\text{ days}$ ($4.1275\text{d}$) validation MAE and $36.2441$ MSE, identical to Experiment 1.
2. **Depth Dominance:** Two-layer networks consistently beat single-layer networks by resolving the non-linear battery knee with greater parameter efficiency.
3. **The 90% Compression Wonder (`2-Layer (8, 8)`):** Reached **$5.08\text{ days}$** MAE with only **129 parameters** ($516\text{ B}$ FP32, $112\text{ MACs}$), saving **$90\%$** of compute and storage versus the reference model with only a $0.95\text{d}$ accuracy difference.
4. **Overfitting at Width 64:** Training MAE reached $3.27\text{d}$, but validation MAE worsened to $4.55\text{d}$, confirming diminishing returns and over-parameterization.
5. **Shortlisted Models:** `2-Layer (32, 32)` (Accuracy Champion), `2-Layer (16, 16)` (Balanced), and `2-Layer (8, 8)` (Ultra-Lightweight).

---

## EXP-006: Shortlist Architecture Repeatability & Multi-Seed Validation

* **Experiment ID:** `EXP-006`
* **Date:** November 21, 2026
* **Investigator:** Team 3 (Phoenix)
* **Status:** Completed & Verified
* **Executable Script:** `experiments/shortlist_repeatability.py`
* **Artifact Directory:** `experiments/shortlist_repeatability/`
* **Detailed Documentation:** [`docs/08_repeatability.md`](08_repeatability.md)

### 1. Objective
Evaluate the repeatability and stability of the three shortlisted architectures from EXP-005 (`2-Layer (8, 8)`, `2-Layer (16, 16)`, and `2-Layer (32, 32)`) by varying training randomness across five distinct seeds ($s \in \{0, 1, 2, 3, 4\}$), resulting in 15 architecture/seed combinations.

### 2. Experimental Controls
* **Dataset & Cleaning:** Same 1,976 clean records from `data/team_03_train.csv` (24 glitches removed).
* **Fixed Data Split:** Seed 0 split remained strictly invariant ($1,581\text{ train} / 395\text{ val}$) across all runs.
* **Normalization:** Standardized using statistics from the training set only.
* **Varied Randomness:** `torch.manual_seed(s)`, `np.random.seed(s)`, and `random.seed(s)` set before initialization.
* **Seed-0 Reuse:** Reused verified EXP-005 results for seed 0. Seeds 1, 2, 3, and 4 trained anew.

### 3. Aggregate Performance Summary Across 5 Seeds

| Architecture | Params | MACs | Mean Val MAE | Std Val MAE | Min Val MAE | Max Val MAE | Val MAE Range ($\Delta$) | Mean Low-Bat MAE (0–10d, N=85) | Std Low-Bat MAE | Mean Signed Bias |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`2-Layer (8, 8)`** | **129** | **112** | $5.6551\text{ d}$ | $1.3166\text{ d}$ | $4.9083\text{ d}$ | $7.9941\text{ d}$ | $3.0858\text{ d}$ | $1.8028\text{ d}$ | $0.5089\text{ d}$ | $-0.1365\text{ d}$ |
| **`2-Layer (16, 16)`**| **385** | **352** | **$4.4549\text{ d}$** | **$0.2005\text{ d}$** | **$4.2742\text{ d}$** | **$4.7758\text{ d}$** | **$0.5016\text{ d}$** | **$1.2213\text{ d}$** | **$0.2526\text{ d}$** | $+0.2667\text{ d}$ |
| **`2-Layer (32, 32)`**| **1,281**| **1,216**| **$4.2575\text{ d}$** | **$0.1556\text{ d}$** | **$4.1010\text{ d}$** | **$4.4895\text{ d}$** | **$0.3885\text{ d}$** | **$1.2766\text{ d}$** | **$0.2332\text{ d}$** | $-0.3260\text{ d}$ |

### 4. Key Findings & Updated Recommendations
1. **Hierarchy Preservation:** Model hierarchy observed in Sweep 1 is confirmed: 32–32 ($4.26\text{d}$) < 16–16 ($4.45\text{d}$) < 8–8 ($5.66\text{d}$).
2. **The 16–16 Parity Discovery:** `2-Layer (16, 16)` delivers near-identical accuracy to the reference model (only $0.20\text{ days}$ mean difference) and actually achieves lower error in the low-battery / near-depletion evaluation subset ($1.22\text{d}$ vs $1.28\text{d}$), while slashing parameters by **$70\%$** ($385$ vs $1,281$).
3. **8–8 Seed Sensitivity:** 8–8 clustered tightly at $\approx 5.07\text{d}$ on seeds 0–3, but plateaued at $7.99\text{d}$ on seed 4, revealing that 8 hidden units provide narrow parameter headroom with heightened sensitivity to initial weights.
4. **Final Recommendation:** `2-Layer (16, 16)` is designated as the primary edge-optimized model, while `2-Layer (32, 32)` remains the accuracy benchmark.

---

## 🔬 Planned Future Experiments

### EXP-007: API Endpoint Integration & Latency Benchmarking
* **Scope:** Deploy the selected model to the local FastAPI microservice (`app/main.py`), verify Pydantic HTTP 422 glitch rejection, and execute `client/bench.py` to benchmark request latency percentiles (p50/p95/p99) across processing stages (`parse`, `normalise`, `inference`, `build`).
* **Scope Boundary Note:** Physical deployment onto an ESP32-S3 microcontroller is out of scope. ESP32-S3 SRAM and MAC calculations are performed theoretically to ground edge efficiency.
* **Protocol Reference:** [`docs/teammate_edge_deployment.md`](teammate_edge_deployment.md)

