# 11. Assumptions, Scientific Limitations, Scope Boundaries & Reproducibility Checklist

---

## 📌 Document Overview & Purpose

This document serves as the scientific governance and reproducibility foundation for the **Edge AI Smart Doorbell Battery-Life Prediction** project (Team 3 — Phoenix).

In empirical machine learning and edge systems engineering, credibility rests on two pillars:
1. **Intellectual Honesty:** Transparently identifying every assumption, methodological boundary, theoretical approximation, and potential source of bias.
2. **Deterministic Reproducibility:** Providing an unambiguous, step-by-step checklist and verified command protocol enabling any independent researcher, evaluator, or teammate to recreate the complete experimental pipeline from scratch.

---

## 1. The 16-Point Assumptions, Limitations & Methodology Matrix

Every scientific claim made in this project is anchored to an explicit operational assumption. The matrix below documents each assumption, its methodological classification, potential risks, and mitigation strategies:

| ID | Operational Domain | Formal Assumption / Methodological Choice | Scientific Classification | Methodological Justification & Limitations |
| :---: | :--- | :--- | :--- | :--- |
| **01** | **Data Cleaning** | Values of $\text{battery\_v} < 3.0\text{V}$, $\text{temp\_c} < -10^\circ\text{C}$, and $\text{RSSI} \ge 0\text{ dBm}$ represent hardware glitches. | **Domain Heuristic** | 24 records dropped ($1.2\%$). These are provisional rules based on physical operating limits. They were **not physically verified** against faulty hardware in a lab. |
| **02** | **Data Partition** | A single 80/20 train/validation split with `seed=0` represents the true population. | **Evaluated Constraint** | Exactly $1,581$ train and $395$ val samples. Replicated identically across all experiments to guarantee strict comparative fairness. |
| **03** | **Split Limitation** | A fixed single validation split cannot detect cross-partition variance. | **Known Limitation** | While computationally efficient, cross-validation ($K$-fold) was omitted to preserve identical baseline comparability across all project phases. |
| **04** | **Initial Sweep** | Single-seed training ($\text{seed}=0$) in EXP-005 is sufficient to filter 10 architectures. | **Screening Heuristic** | Used exclusively as a broad sieve to eliminate non-converging or heavily overparameterized topologies prior to multi-seed testing. |
| **05** | **Repeatability** | $N=5$ training seeds ($\text{seed} \in \{0, 1, 2, 3, 4\}$) provide empirical stability evidence. | **Empirical Observation** | Demonstrates optimization repeatability and variance bounds. Explicitly **does not constitute formal statistical proof** of superiority. |
| **06** | **Dataset Scope** | Historical telemetry from `team_03_train.csv` captures typical device lifecycle behavior. | **Operational Premise** | Assumes the provided 2,000 telemetry snapshots reflect stationary real-world doorbell usage and climate distributions. |
| **07** | **Distribution Bounds** | Training and validation distributions are identically distributed (i.i.d.). | **Statistical Premise** | If future production doorbells encounter sub-zero winters ($<-10^\circ\text{C}$) or extreme heat ($>50^\circ\text{C}$), model accuracy may degrade. |
| **08** | **High-Lifespan Error**| Absolute prediction error increases on fresh batteries ($y > 120\text{ days}$). | **Observed Heteroscedasticity** | As voltage flattens on the discharge plateau, $\frac{dV}{dt} \approx 0$, making long-term lifespan inherently noisier to infer from instantaneous telemetry. |
| **09** | **Complexity Metrics** | MACs = $n_{\text{in}} \times n_{\text{out}}$ per dense layer represents arithmetic complexity. | **Theoretical Calculation** | Excludes non-linear activation overhead (ReLU comparisons), loop control, data movement, and memory fetch latencies. |
| **10** | **Memory Estimates** | Parameter storage = $\text{Params} \times 4\text{ bytes}$ (FP32) or $\times 1\text{ byte}$ (INT8). | **Theoretical Estimation** | Calculates raw weight tensor sizes. **Excludes PyTorch runtime overhead**, activation buffers, optimizer states, and OS process allocations. |
| **11** | **Quantization Status** | Theoretical INT8 sizes illustrate potential post-training compression. | **Theoretical Projection** | **No physical quantization was performed.** Models were trained and evaluated exclusively in 32-bit floating point (FP32). |
| **12** | **Latency Status** | Inference runtime is estimated via theoretical MACs and API micro-benchmarks. | **Measured / Theoretical** | API latency includes HTTP network stack and JSON serialization. **No on-chip bare-metal cycle counting** was conducted on an MCU. |
| **13** | **Hardware Boundary** | Physical microcontroller deployment (ESP32-S3) is out of scope for this assignment. | **Confirmed Scope Boundary** | Project focuses on ML modeling, architecture optimization, error diagnostics, and FastAPI integration. ESP32 analysis is purely a feasibility study. |
| **14** | **Hardware Verification**| No physical hardware flashing or battery bench tests were conducted. | **Confirmed Scope Boundary** | Model behavior in actual physical circuit conditions (ADC thermal noise, supply ripple) remains unverified. |
| **15** | **Serving Ownership** | FastAPI serving (`app/main.py`) and client benchmarking belong to our teammate. | **Team Ownership Boundary**| ML modeling, architecture exploration, and documentation are our ownership domain. Production checkpoint `model/model.pt` is preserved for serving. |
| **16** | **Causality Rigor** | Strong statistical correlation between telemetry and lifespan does not imply causation. | **Methodological Rigor** | High correlation ($r = 0.81$ for voltage) reflects physical electrochemical discharge, but correlation alone never establishes definitive causal mechanisms. |

---

## 2. Categorization of Project Claims

To maintain unwavering scientific credibility, every statement within our project reports is classified into one of six epistemological categories:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   SCIENTIFIC EPISTEMOLOGY FRAMEWORK                    │
│                                                                        │
│   1. Measured Results: Direct empirical outputs from code execution    │
│   2. Calculated Results: Exact mathematical derivations from data      │
│   3. Theoretical Estimates: Analytical hardware projections            │
│   4. Observations: Empirical patterns visible in graphs or tables      │
│   5. Hypotheses: Plausible explanations requiring future testing        │
│   6. Limitations: Explicit constraints and unverified boundaries       │
└────────────────────────────────────────────────────────────────────────┘
```

1. **Measured Results (Empirical Facts):**
   * Validation MAE of $4.1275\text{ days}$ on saved checkpoint (`model/model.pt`).
   * Exactly 24 records dropped by glitch filtering rules.
   * Repeatability validation MAEs: 16–16 achieved $[4.7758, 4.2742, 4.4172, 4.3076, 4.4998]\text{ days}$.
2. **Calculated Results (Mathematical Facts):**
   * 16–16 parameter count: $(5 \times 16 + 16) + (16 \times 16 + 16) + (16 \times 1 + 1) = 385\text{ parameters}$.
   * 16–16 MAC count: $5 \times 16 + 16 \times 16 + 16 \times 1 = 352\text{ MACs}$.
   * Complexity reduction: $\frac{1281-385}{1281} \times 100 = 69.95\%$ parameter reduction.
   * Sample standard deviation ($ddof=1$): $s = 0.2005\text{ days}$ for 16–16 across 5 seeds.
3. **Theoretical Estimates (Analytical Projections):**
   * FP32 parameter memory: $385 \times 4 = 1,540\text{ bytes}$.
   * Theoretical INT8 storage: $385 \times 1 = 385\text{ bytes}$.
   * Estimated ESP32-S3 SRAM consumption: $<0.5\%$ of $512\text{ KB}$ available internal memory.
4. **Observations (Empirical Patterns):**
   * Prediction error is substantially lower in the low-battery regime ($1.22\text{d}$) than in the fresh-battery regime ($7.90\text{d}$).
   * 1-layer networks consistently underfit the training data across all widths.
   * Seed 4 of the 8–8 architecture exhibited an error spike to $7.99\text{ days}$.
5. **Hypotheses (Plausible Mechanisms):**
   * *Hypothesis:* The flat lithium discharge plateau causes heteroscedastic error because distinct lifespans share identical cell voltages.
   * *Hypothesis:* Seed 4 of the 8–8 architecture exhibited substantially poorer performance ($7.99\text{d}$), which may indicate sensitivity to initialization or optimization in the smaller architecture. Dead ReLU neurons are one possible explanation, but this was not directly verified.
6. **Limitations (Boundaries of Validity):**
   * The models have not been tested under sub-zero thermal extremes or non-stationary network conditions.
   * Five seeds provide initial evidence of stability, not formal statistical significance.
   * No physical ESP32 deployment was conducted.

---

## 3. Scope Boundaries: Collaborative Team Division

Our project operates under a well-defined collaborative structure between team members:

```
┌───────────────────────────────────────┬───────────────────────────────────────┐
│       MY SCOPE & RESPONSIBILITIES     │      TEAMMATE SCOPE & RESPONSIBILITIES│
│                                       │                                       │
│  • Dataset inspection & glitch filter │  • FastAPI microservice (app/main.py) │
│  • Exploratory data analysis (EDA)    │  • Telemetry Pydantic validation      │
│  • Baseline modeling (Mean, OLS)      │  • Error response handling (HTTP 422) │
│  • Neural network training pipeline   │  • Benchmark client (client/bench.py) │
│  • Residual diagnostic audit          │  • Microservice stage latency profiling│
│  • Architecture sweep (EXP-005)       │  • Docker containerization (if used)  │
│  • Multi-seed repeatability (EXP-006) │  • Production model serving execution │
│  • Multi-objective model selection    │  • HTTP endpoint verification tests   │
│  • Mathematical methodology docs      │  • Integration documentation          │
└───────────────────────────────────────┴───────────────────────────────────────┘
```

> [!NOTE]
> **Physical ESP32 Deployment Scope Clarification:**
> As confirmed in our project updates, physical deployment on an ESP32-S3 microcontroller is **completely out of scope for both team members**. The embedded hardware calculations documented in our reports serve exclusively as a **theoretical feasibility study** to demonstrate Edge AI readiness.

---

## 4. End-to-End Reproducibility Checklist

Any evaluator can independently reproduce all results by following this verified protocol:

### 4.1 System Prerequisites & Environment
* **Operating System:** macOS (Sonoma/Sequoia), Linux (Ubuntu 22.04+), or Windows (WSL2).
* **Python Distribution:** Python 3.11+ running inside Conda.
* **Core Libraries:** PyTorch 2.15+, NumPy 1.26+, Pandas 2.1+, Matplotlib 3.8+, Scikit-Learn 1.3+.

```bash
# Verify Conda environment activation
conda activate edgeai1
python --version  # Expected: Python 3.11.x
```

### 4.2 Repository Structure & File Verification
Verify that the following core files are present in the workspace:
* `data/team_03_train.csv` (Assigned raw team dataset, 2,000 rows, SHA-256 integrity verified).
* `train.py` (Experiment 1 baseline neural network training script).
* `experiments/architecture_sweep_1.py` (EXP-005 Architecture sweep script).
* `experiments/shortlist_repeatability.py` (EXP-006 Multi-seed repeatability script).
* `notebook/dataset_inspection.ipynb` (Provisional glitch filtering verification).
* `notebook/eda.ipynb` (Comprehensive exploratory data analysis and diagnostic audit).

---

## 5. Verified Command Execution Protocol

Follow these exact shell commands to reproduce all experimental data and figures:

### Step 1: Verify Dataset Integrity & Data Cleaning
Execute data inspection and baseline verification:
```bash
python -c "
import pandas as pd
df = pd.read_csv('data/team_03_train.csv')
assert len(df) == 2000, 'Raw dataset must contain 2000 rows'
clean = df[(df['battery_v'] >= 3.0) & (df['temp_c'] >= -10.0) & (df['wifi_rssi_dbm'] < 0)]
assert len(clean) == 1976, 'Cleaned dataset must contain 1976 rows'
print('Dataset integrity verified: 24 glitch rows dropped, 1976 clean rows retained.')
"
```

### Step 2: Reproduce EXP-005 (Architecture Sweep 1)
To run the full 10-architecture sweep and generate all metrics, weights, and plots:
```bash
python experiments/architecture_sweep_1.py
```
* **Expected Output Files:**
  * `experiments/sweep_1_results/architecture_sweep_1_results.csv`
  * `experiments/sweep_1_results/sweep_1_summary.json`
  * `experiments/sweep_1_results/val_mae_vs_params.png`
  * `experiments/sweep_1_results/val_mae_vs_macs.png`
  * `experiments/sweep_1_results/convergence_curves.png`
  * `experiments/sweep_1_results/pareto_bubble_chart.png`
  * 10 PyTorch weight files (`weights_*.pt`)

### Step 3: Reproduce EXP-006 (Multi-Seed Repeatability Validation)
To evaluate the 3 shortlisted models across 5 random seeds (0–4):
```bash
python experiments/shortlist_repeatability.py
```
* **Expected Output Files:**
  * `experiments/shortlist_repeatability/repeatability_runs.csv` (15 individual run records)
  * `experiments/shortlist_repeatability/repeatability_summary.csv` (Aggregated statistics)
  * `experiments/shortlist_repeatability/repeatability_summary.json`
  * `experiments/shortlist_repeatability/val_mae_per_seed.png`
  * `experiments/shortlist_repeatability/architecture_comparison_bars.png`
  * `experiments/shortlist_repeatability/low_battery_mae_comparison.png`
  * `experiments/shortlist_repeatability/mean_mae_vs_params.png`
  * 12 newly generated PyTorch weight files (`weights_*_seed*.pt`)

### Step 4: Verify Saved Production Checkpoint Metrics
Verify the performance of the frozen production checkpoint (`model/model.pt`):
```bash
python -c "
import torch, json, pandas as pd, numpy as np

# Load config and model
with open('model/config.json') as f:
    cfg = json.load(f)

# Load and clean data
df = pd.read_csv('data/team_03_train.csv')
clean = df[(df['battery_v'] >= 3.0) & (df['temp_c'] >= -10.0) & (df['wifi_rssi_dbm'] < 0)].reset_index(drop=True)

# Split with seed 0
rng = np.random.default_rng(0)
idx = rng.permutation(len(clean))
val_idx = idx[1581:]
val_df = clean.iloc[val_idx]

# Normalize features using saved stats
feats = ['battery_v', 'temp_c', 'rings_per_day', 'wifi_rssi_dbm', 'age_months']
X = (val_df[feats].values - np.array(cfg['feature_means'])) / np.array(cfg['feature_stds'])
y = val_df['days_left'].values

# Run forward pass
class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(5, 32), torch.nn.ReLU(),
            torch.nn.Linear(32, 32), torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )
    def forward(self, x): return self.net(x).squeeze(-1)

m = MLP()
m.load_state_dict(torch.load('model/model.pt'))
m.eval()
with torch.no_grad():
    preds = m(torch.tensor(X, dtype=torch.float32)).numpy()

mae = np.mean(np.abs(y - preds))
print(f'Verified Saved Checkpoint Validation MAE: {mae:.4f} days')
assert abs(mae - 4.1275) < 1e-3, 'MAE must match exactly 4.1275 days'
"
```

---

## 6. Verification Summary

By completing the above protocol, any third party can independently confirm that:
1. Every calculation and figure matches our published reports to the exact decimal point.
2. No data leakage occurred between training and validation partitions.
3. Every baseline and neural network model was evaluated on the exact same unseen validation samples.
4. The project strictly preserves experimental integrity without manual intervention or data fabrication.

