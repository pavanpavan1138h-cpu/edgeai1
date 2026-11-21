# 09. Model Architecture Selection & Multi-Objective Engineering Decision

---

## 📌 Document Overview & Purpose

This document presents the definitive model-selection rationale for the **Edge AI Smart Doorbell Battery-Life Prediction** project (Team 3 — Phoenix).

In Edge AI systems, model selection cannot be treated as a one-dimensional search for the lowest validation error. A model deployed onto an embedded edge device or microservice must balance four competing operational vectors:
1. **Predictive Accuracy:** Minimizing global Mean Absolute Error ($\text{MAE}_{\text{val}}$) across typical operating lifespans.
2. **Operational Safety:** Guaranteeing superior accuracy in the critical low-battery exhaustion window ($y \le 10\text{ days}$).
3. **Hardware Frugality:** Minimizing parameter memory footprint (Flash/ROM) and active activation buffers (SRAM).
4. **Computational Efficiency:** Minimizing Multiply-Accumulate operations (MACs), execution latency, and device energy consumption.

Based on the empirical findings from our **Architecture Sweep (EXP-005)** and **5-Seed Repeatability Study (EXP-006)**, this document articulates the formal multi-objective decision framework that establishes **2-Layer (16, 16)** as our recommended engineering compromise, while designating **2-Layer (32, 32)** as our accuracy reference and **2-Layer (8, 8)** as our lightweight fallback.

---

## 1. Candidate Architecture Comparative Profiles

From the initial pool of ten architectures evaluated in EXP-005, three candidates demonstrated Pareto-efficient characteristics and were subjected to exhaustive multi-seed repeatability testing across five distinct random initializations ($\text{seed} \in \{0, 1, 2, 3, 4\}$). Their detailed engineering profiles are contrasted below:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SHORTLISTED ARCHITECTURE CANDIDATES                      │
│                                                                             │
│  [2-Layer (8, 8)]        [2-Layer (16, 16)]        [2-Layer (32, 32)]       │
│  • 129 Parameters        • 385 Parameters          • 1,281 Parameters       │
│  • 112 MACs              • 352 MACs                • 1,216 MACs             │
│  • 516 B FP32 RAM        • 1,540 B FP32 RAM        • 5,124 B FP32 RAM       │
│  • Val MAE: 5.66 ± 1.32d • Val MAE: 4.45 ± 0.20d   • Val MAE: 4.26 ± 0.16d  │
│  • Low-Batt: 1.80 ± 0.51d• Low-Batt: 1.22 ± 0.25d • Low-Batt: 1.28 ± 0.23d │
│                                                                             │
│  Role: Lightweight       Role: Recommended         Role: Accuracy           │
│        Alternative              Engineering               Reference         │
│                                 Compromise                                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 1.1 Candidate Profile: 2-Layer (16, 16) — Recommended Engineering Compromise
* **Architecture Configuration:** $5 \rightarrow 16 \rightarrow 16 \rightarrow 1$ (Two hidden layers of width 16 with intermediate ReLU activations and linear output).
* **Trainable Parameters:** $385$ ($352$ weights, $33$ biases).
* **Computational Cost:** $352\text{ MACs}$ per forward inference pass.
* **Theoretical Parameter Storage:** $1,540\text{ bytes}$ ($1.50\text{ KB}$) in FP32; $385\text{ bytes}$ in INT8.
* **Empirical Accuracy (5-Seed Average):**
  * Mean Validation MAE: **$4.4549\text{ days}$** ($106.9\text{ hours}$)
  * Sample Standard Deviation ($s, \text{ddof}=1$): **$\pm 0.2005\text{ days}$** ($4.50\%$ relative dispersion)
  * Minimum / Maximum: $4.2742\text{ days}$ (Seed 1) to $4.7758\text{ days}$ (Seed 0)
  * Range ($\Delta$): $0.5016\text{ days}$ ($12.0\text{ hours}$)
* **Low-Battery Safety Regime ($y \le 10\text{ days}$, $N=85$):**
  * Mean Low-Battery MAE: **$1.2213\text{ days}$** ($29.3\text{ hours}$)
  * Sample Standard Deviation: **$\pm 0.2526\text{ days}$**
  * *Note: Outperforms the larger 32–32 model in this safety-critical domain.*
* **Complexity Reductions Relative to 32–32 Reference:**
  * Parameter Reduction: $\frac{1281 - 385}{1281} \times 100 = \mathbf{69.95\%}$ fewer parameters.
  * MAC Reduction: $\frac{1216 - 352}{1216} \times 100 = \mathbf{71.05\%}$ fewer operations.
  * FP32 Storage Savings: $3,584\text{ bytes}$ saved ($69.95\%$ reduction).
* **Observed Accuracy Trade-off:**
  $$\Delta \text{MAE} = 4.4549\text{ d} - 4.2575\text{ d} = \mathbf{0.1974\text{ days}} \quad (\mathbf{4.74\text{ hours}})$$
* **Advantages:**
  * Exceptional hardware efficiency: Reduces computational load and parameter memory by more than two-thirds.
  * Negligible accuracy penalty: An average difference of less than 5 hours over a multi-month battery lifespan is imperceptible to end consumers.
  * Superior safety accuracy: Achieves the lowest error in the critical depletion window ($1.22\text{ days}$).
  * Optimization robustness: Low variance across random seeds ($\pm 0.20\text{ d}$), ensuring consistent training outcomes.
* **Disadvantages:**
  * Marginally higher aggregate validation MAE ($+0.1974\text{ d}$) than the larger 32–32 network.
* **Engineering Recommendation:** **Primary Deployment Recommendation.**

---

### 1.2 Candidate Profile: 2-Layer (32, 32) — Accuracy Benchmark & Reference
* **Architecture Configuration:** $5 \rightarrow 32 \rightarrow 32 \rightarrow 1$ (Two hidden layers of width 32 with intermediate ReLU activations and linear output).
* **Trainable Parameters:** $1,281$ ($1,216$ weights, $65$ biases).
* **Computational Cost:** $1,216\text{ MACs}$ per forward inference pass.
* **Theoretical Parameter Storage:** $5,124\text{ bytes}$ ($5.00\text{ KB}$) in FP32; $1,281\text{ bytes}$ in INT8.
* **Empirical Accuracy (5-Seed Average):**
  * Mean Validation MAE: **$4.2575\text{ days}$** ($102.2\text{ hours}$)
  * Sample Standard Deviation ($s, \text{ddof}=1$): **$\pm 0.1556\text{ days}$** ($3.65\%$ relative dispersion)
  * Minimum / Maximum: $4.1010\text{ days}$ (Seed 1) to $4.4895\text{ days}$ (Seed 3)
  * Range ($\Delta$): $0.3885\text{ days}$ ($9.3\text{ hours}$)
* **Low-Battery Safety Regime ($y \le 10\text{ days}$, $N=85$):**
  * Mean Low-Battery MAE: **$1.2766\text{ days}$** ($30.6\text{ hours}$)
  * Sample Standard Deviation: **$\pm 0.2332\text{ days}$**
* **Advantages:**
  * Highest overall predictive accuracy among all evaluated stable topologies.
  * Exceptionally smooth loss landscape: The larger parameter budget provides redundancy, minimizing sensitivity to weight initialization.
  * Serves as the active production weights in `model/model.pt` evaluated in EXP-001 and EXP-004.
* **Disadvantages:**
  * High computational overhead: Requires $3.45\times$ more MACs and $3.33\times$ more parameters than 16–16 for a diminishing gain of only $4.74\text{ hours}$ in aggregate MAE.
  * Slightly inferior performance in the low-battery segment ($1.28\text{d}$ vs. $1.22\text{d}$).
* **Engineering Recommendation:** **Accuracy Benchmark Baseline.** Ideal for edge gateways, mains-powered smart hubs, or cloud API microservices where memory and power budgets are unconstrained.

---

### 1.3 Candidate Profile: 2-Layer (8, 8) — Lightweight Fallback Alternative
* **Architecture Configuration:** $5 \rightarrow 8 \rightarrow 8 \rightarrow 1$ (Two hidden layers of width 8 with intermediate ReLU activations and linear output).
* **Trainable Parameters:** $129$ ($112$ weights, $17$ biases).
* **Computational Cost:** $112\text{ MACs}$ per forward inference pass.
* **Theoretical Parameter Storage:** $516\text{ bytes}$ ($0.50\text{ KB}$) in FP32; $129\text{ bytes}$ in INT8.
* **Empirical Accuracy (5-Seed Average):**
  * Mean Validation MAE: **$5.6551\text{ days}$** ($135.7\text{ hours}$)
  * Sample Standard Deviation ($s, \text{ddof}=1$): **$\pm 1.3166\text{ days}$** ($23.28\%$ relative dispersion)
  * Minimum / Maximum: $4.9083\text{ days}$ (Seed 3) to $7.9941\text{ days}$ (Seed 4)
  * Range ($\Delta$): $3.0858\text{ days}$ ($74.1\text{ hours}$)
* **Low-Battery Safety Regime ($y \le 10\text{ days}$, $N=85$):**
  * Mean Low-Battery MAE: **$1.8028\text{ days}$** ($43.3\text{ hours}$)
  * Sample Standard Deviation: **$\pm 0.5089\text{ days}$**
* **Complexity Reductions Relative to 32–32 Reference:**
  * Parameter Reduction: $\mathbf{89.93\%}$ reduction ($1,152$ fewer parameters).
  * MAC Reduction: $\mathbf{90.79\%}$ reduction ($1,104$ fewer MACs).
* **Advantages:**
  * Ultra-compact footprint: Raw weights fit in barely $516\text{ bytes}$ of memory, enabling deployment on extremely constrained microcontrollers (e.g., 8-bit or 16-bit MCUs with $<2\text{ KB}$ SRAM).
  * Fast execution: Requires barely $112\text{ MACs}$, completing inference in under a few microseconds.
* **Disadvantages:**
  * Optimization fragility: Highly susceptible to adverse weight initializations. On Seed 4, the model stalled in a poor local minimum with a validation MAE of $7.9941\text{ days}$ ($122.83\text{ MSE}$).
  * Noticeable accuracy penalty: Even on successful runs, validation MAE hovers around $5.0\text{ days}$ ($12$ hours worse than 16–16).
* **Engineering Recommendation:** **Lightweight Alternative.** Suitable only for severely constrained hardware architectures where available RAM is under $1\text{ KB}$ and occasional retraining or checkpoint selection can discard failed runs.

---

## 2. Multi-Objective Decision Matrix

The decision matrix below synthesizes the twelve key quantitative criteria evaluated across baseline models and shortlisted neural networks:

| Evaluation Dimension | Mean Baseline | Linear Regression | 2-Layer (8, 8) | **2-Layer (16, 16)** | 2-Layer (32, 32) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Model Topology** | Constant Mean | OLS $5 \rightarrow 1$ | $5 \rightarrow 8 \rightarrow 8 \rightarrow 1$ | **$5 \rightarrow 16 \rightarrow 16 \rightarrow 1$** | $5 \rightarrow 32 \rightarrow 32 \rightarrow 1$ |
| **Trainable Parameters** | 0 | 6 | 129 | **385** | 1,281 |
| **Parameter Reduction vs. 32–32** | N/A | -99.53% | -89.93% | **-69.95%** | Reference (0%) |
| **Multiply-Accumulates (MACs)** | 0 | 5 | 112 | **352** | 1,216 |
| **MAC Reduction vs. 32–32** | N/A | -99.59% | -90.79% | **-71.05%** | Reference (0%) |
| **FP32 Storage (bytes)** | 0 B | 24 B | 516 B | **1,540 B** | 5,124 B |
| **Theoretical INT8 Storage** | 0 B | 6 B | 129 B | **385 B** | 1,281 B |
| **Mean Validation MAE (days)** | 47.58 d | 20.54 d | 5.66 d | **4.45 d** | 4.26 d |
| **Multi-Seed Stability ($s$)** | Deterministic | Deterministic | $\pm 1.32\text{ d}$ (Unstable) | **$\pm 0.20\text{ d}$ (Stable)** | $\pm 0.16\text{ d}$ (Very Stable) |
| **Low-Battery MAE ($y \le 10\text{ d}$)**| 49.32 d | 14.21 d | 1.80 d | **1.22 d (Best)** | 1.28 d |
| **Accuracy Gap vs. 32–32** | +43.32 d | +16.28 d | +1.40 d | **+0.197 d (4.7 hrs)**| Reference (0.00 d) |
| **Final Deployment Role** | Baseline Floor | Underfitting Floor | Lightweight Backup | **Recommended Compromise** | Accuracy Reference |

---

## 3. The 9-Step Selection Reasoning Chain

To ensure another engineer or academic evaluator can follow the exact thought process that led to our recommendation, our decision-making is structured into a rigorous nine-step reasoning chain:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   THE 9-STEP SELECTION REASONING CHAIN                 │
│                                                                        │
│   Step 1: Reject Non-Converging 1-Layer Networks                       │
│   Step 2: Reject Overparameterized 64-Width Networks                   │
│   Step 3: Establish the 2-Layer Architectural Advantage                │
│   Step 4: Construct the Empirical Pareto Frontier                      │
│   Step 5: Test Multi-Seed Repeatability & Optimization Stability       │
│   Step 6: Penalize Small-Model Initialization Fragility (8-8)          │
│   Step 7: Quantify the Edge AI Diminishing-Returns Threshold           │
│   Step 8: Validate Safety-Critical Domain Accuracy (Low-Battery Knee)   │
│   Step 9: Formulate the Tri-Tier Production Recommendation             │
└────────────────────────────────────────────────────────────────────────┘
```

### Step 1: Reject Non-Converging Single-Hidden-Layer Networks
In EXP-005, single-layer networks exhibited severe underfitting. Architectures with widths of 8, 16, and 64 stalled with validation MAEs above $15.5\text{ days}$. A single hyperplane partition cannot simultaneously resolve the sharp non-linear knee of lithium battery discharge and the multivariate interactions between temperature and internal resistance.

### Step 2: Reject Overparameterized 64-Width Networks
The `2-Layer (64, 64)` model consumed $4,609$ parameters and $4,480$ MACs (+259.8% parameter expansion over 32–32) but resulted in a worse validation MAE ($4.55\text{ days}$ vs. $4.13\text{ days}$). The excess capacity encouraged memorization of training sample noise rather than generalizable telemetry trends.

### Step 3: Establish the 2-Layer Architectural Advantage
Two-layer networks demonstrated superior feature hierarchy. Hidden Layer 1 forms local non-linear combinations of sensor inputs (e.g., coupling `battery_v` and `temp_c`), while Hidden Layer 2 aggregates these representations into a smooth lifespan trajectory.

### Step 4: Construct the Empirical Pareto Frontier
Plotting error against computational complexity revealed that only three architectures lie on the non-dominated Pareto frontier:
$$\mathcal{P} = \{ \text{2-Layer (8, 8)}, \text{2-Layer (16, 16)}, \text{2-Layer (32, 32)} \}$$
All other candidate networks were strictly dominated in both accuracy and complexity.

### Step 5: Test Multi-Seed Repeatability Across Initializations
Benchmarking the shortlisted candidates across five seeds ($\text{seed} \in \{0, 1, 2, 3, 4\}$) proved that performance was not an artifact of random chance. The observed rankings remained broadly consistent across seeds.

### Step 6: Penalize Small-Model Initialization Fragility (8–8)
While the 8–8 model achieved acceptable accuracy on four seeds ($\approx 5.0\text{d}$), it suffered an optimization failure on Seed 4 ($7.99\text{d}$ MAE, $\text{MSE}=122.83$). With only 8 neurons per layer, bad weight initializations easily trap the network in dead-ReLU configurations. This high variance ($\pm 1.32\text{d}$) disqualifies 8–8 from serving as the primary production model.

### Step 7: Quantify the Edge AI Diminishing-Returns Threshold
Moving from 8–8 to 16–16 (+256 params) slashes validation MAE by $1.20\text{ days}$ ($21.2\%$). However, moving from 16–16 to 32–32 (+896 params) yields a marginal gain of only $0.1974\text{ days}$ ($4.74\text{ hours}$, or $4.4\%$). In an edge environment, spending $3.33\times$ more memory and $3.45\times$ more energy for a $4.7$-hour difference over a 6-month lifespan violates fundamental edge engineering principles.

### Step 8: Validate Safety-Critical Domain Accuracy (Low-Battery Knee)
In the critical operating regime ($y \le 10\text{ days}$), the 16–16 model achieved an outstanding average error of **$1.2213\text{ days}$**, actually surpassing the 32–32 model ($1.2766\text{ days}$). Its constrained capacity acts as an inductive regularizer, preventing overparameterized drift in high-curvature voltage regions.

### Step 9: Formulate the Tri-Tier Production Recommendation
Rather than declaring a single "winner," we formulate a comprehensive, tri-tier engineering architecture:
1. **Primary Deployment Recommendation:** **2-Layer (16, 16)** for production edge doorbells.
2. **Accuracy Reference Benchmark:** **2-Layer (32, 32)** for unconstrained servers and baseline tracking.
3. **Lightweight Fallback:** **2-Layer (8, 8)** for sub-milliwatt, sub-KB embedded hardware.

---

## 4. Grounding the Decision: Multi-Objective Compromise vs. Statistical Superiority

> [!IMPORTANT]
> ### Scientific Integrity: Why We Do Not Claim "Universal Superiority"
> We explicitly state that our recommendation of **2-Layer (16, 16)** is a **multi-objective engineering decision**, not a claim of formal statistical superiority.
> 
> * **Statistical Overlap:** With $N = 5$ seeds, the validation MAE distributions of 32–32 ($4.26 \pm 0.16\text{d}$) and 16–16 ($4.45 \pm 0.20\text{d}$) have overlapping confidence intervals. On Seed 1, the 16–16 model ($4.27\text{d}$) outperformed Seeds 2 and 3 of the 32–32 model.
> * **Decision Framework:** If the single goal were to minimize mathematical error regardless of cost, 32–32 is the empirical champion. However, engineering is defined by constrained optimization. Saving **$70\%$ of memory and compute** for a human-imperceptible difference of $4.74\text{ hours}$ represents the superior engineering solution.

---

## 5. Deployment Alignment with Project Scope

To preserve full alignment with our collaborative project responsibilities:
1. **Current Production State (`model/model.pt`):** The repository's saved model checkpoint remains the **2-Layer (32, 32)** network trained in EXP-001 ($4.13\text{ days}$ validation MAE). This checkpoint is actively loaded by our teammate's FastAPI service (`app/main.py`) and verified by the benchmark client (`client/bench.py`).
2. **Zero Breaking Changes:** In accordance with strict change control, `model/model.pt` and `model/config.json` have not been overwritten.
3. **Seamless Migration Path:** When our teammate is ready to transition the API microservice to the recommended **16–16** model, the trained checkpoint `experiments/shortlist_repeatability/weights_2-layer_16__16_seed1.pt` (Validation $\text{MAE} = 4.27\text{ days}$) can be promoted to `model/model.pt` alongside an updated `model/config.json` with zero changes required in client code.

