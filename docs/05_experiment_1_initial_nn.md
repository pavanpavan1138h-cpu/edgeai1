# 05. Experiment 1 Report: Baseline Neural Network ($5 \rightarrow 32 \rightarrow 32 \rightarrow 1$)

---

## 1. Experiment Overview & Objective

* **Experiment Identifier:** Mission 2 — Experiment 1
* **Objective:** Implement, train, and evaluate a baseline two-hidden-layer neural network ($5 \rightarrow 32 \rightarrow 32 \rightarrow 1$ with ReLU) against the established Mean and Linear Regression baselines on the Team 3 dataset.
* **Evaluation Metric:** Mean Absolute Error (MAE), measured in physical days.
* **Optimization Objective:** Mean Squared Error (MSE) loss.

---

## 2. Dataset & Partitioning

* **Original Dataset:** `data/team_03_train.csv` (2,000 observations)
* **Provisionally Filtered Dataset:** 1,976 observations (24 corrupt sensor rows removed: voltage dropouts, temp faults, unconnected Wi-Fi).
* **Train / Validation Split:** 80% training / 20% validation using `np.random.default_rng(0)`:
  * **Training Set:** **1,581 records**
  * **Validation Set:** **395 records**
* **Input Feature Standardisation:** Transformed using training-only statistics ($\mu, \sigma$).
* **Target Feature:** `days_left` kept unnormalized in days.

---

## 3. Network Architecture

The network follows the project's official `BatteryNet` module:

$$x \in \mathbb{R}^5 \xrightarrow{\text{Linear}} h_1 \in \mathbb{R}^{32} \xrightarrow{\text{ReLU}} h_1 \xrightarrow{\text{Linear}} h_2 \in \mathbb{R}^{32} \xrightarrow{\text{ReLU}} h_2 \xrightarrow{\text{Linear}} \hat{y} \in \mathbb{R}^1$$

1. **Input Layer (5 dimensions):** Receives normalized vectors `[battery_v, temp_c, rings_per_day, wifi_rssi_dbm, age_months]`.
2. **Hidden Layer 1 (32 neurons):** Fully connected `Linear(5, 32)` followed by non-linear `ReLU` activation.
3. **Hidden Layer 2 (32 neurons):** Fully connected `Linear(32, 32)` followed by non-linear `ReLU` activation.
4. **Output Layer (1 neuron):** Fully connected `Linear(32, 1)` with no activation (linear output) to predict continuous remaining days.

---

## 4. Training Hyperparameters

All training was executed in the conda environment `edgeai1` using PyTorch with a fixed random seed:

* **Random Seed:** `torch.manual_seed(0)` (ensuring full determinism)
* **Loss Function:** Mean Squared Error (MSE)
* **Optimizer:** Adam
* **Learning Rate ($\alpha$):** $0.003$ ($3 \times 10^{-3}$)
* **Batch Size ($B$):** 128 samples per mini-batch
* **Total Epochs:** 1,500 epochs
* **Early Stopping:** None applied (full 1,500 epochs trained to record complete convergence behavior).

---

## 5. Training Results & Observed Epoch Progression

| Epoch Checkpoint | Training Loss (MSE) | Training MAE (days) | Validation Loss (MSE) | Validation MAE (days) |
| :---: | :---: | :---: | :---: | :---: |
| **0** | $7,523.03$ | $64.57$ | $7,668.75$ | $66.32$ |
| **300** | $32.95$ | $3.83$ | $36.85$ | $4.17$ |
| **600** | $32.62$ | $3.80$ | $38.51$ | $4.27$ |
| **900** | $32.25$ | $3.78$ | $38.45$ | $4.28$ |
| **1200** | $29.96$ | $3.58$ | $36.80$ | $4.13$ |
| **1499 (Final)** | **$30.29$** | **$3.60$** | **$36.24$** | **$4.13$** |

### Key Checkpoint Values:
* **Final Training MAE:** **$3.60\text{ days}$** (MSE: $30.29$)
* **Final Validation MAE:** **$4.13\text{ days}$** (MSE: $36.24$)
* **Best Validation MAE Observed:** **$3.96\text{ days}$** at **Epoch 979** (MSE: $34.51$)

### 🚨 Critical Note on Checkpoint Selection:
In the starter training pipeline (`train.py`), the model weights saved to `model/model.pt` and documented in `model/config.json` represent the **final training state (Epoch 1499)** rather than the historically best validation checkpoint (Epoch 979). 
* Current saved model performance: **$4.13\text{ days}$ MAE**.
* Potential best checkpoint performance: **$3.96\text{ days}$ MAE**.
* *Open Investigation:* Whether the saved artifact should be updated to strictly represent the best validation checkpoint will be resolved during the upcoming diagnostic audit.

---

## 6. Baseline Performance Comparison

| Model | Model Complexity | Validation MAE | Validation MSE | Error Reduction vs Baseline 1 | Error Reduction vs Linear Reg |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline 1: Predict Mean** | 0 parameters | $47.58\text{ days}$ | $3,354.21$ | $0.0\%$ (Reference) | — |
| **Baseline 2: Linear Regression** | 6 parameters | $20.54\text{ days}$ | $792.14$ | $56.8\%$ | $0.0\%$ (Reference) |
| **Neural Network ($5 \rightarrow 32 \rightarrow 32 \rightarrow 1$)** | **1,281 parameters** | **$4.13\text{ days}$** | **$36.24$** | **$91.3\%$** | **$79.9\%$** |

### Mathematical Error Reduction:
$$\text{Improvement over Linear Regression} = \frac{20.54 - 4.13}{20.54} \times 100\% = \mathbf{79.9\%}$$
$$\text{Improvement over Mean Baseline} = \frac{47.58 - 4.13}{47.58} \times 100\% = \mathbf{91.3\%}$$

---

## 7. Learning Curve Analysis & Overfitting Assessment

The training trajectory in `notebook/eda.ipynb` displays two clear phases:
1. **Rapid Descent (Epochs 0 to 200):** Validation MAE drops sharply from $66.32\text{ days}$ to below $5.0\text{ days}$ within the first 150 epochs.
2. **Stable Asymptote (Epochs 200 to 1,500):** The model plateaus smoothly around $3.60\text{ days}$ training MAE and $4.13\text{ days}$ validation MAE.

### Generalization Assessment:
* **Train-Validation Gap:** The final gap between Training MAE ($3.60\text{ days}$) and Validation MAE ($4.13\text{ days}$) is only **$0.53\text{ days}$**.
* **Loss Behavior:** The validation loss does not turn upward during the 1,500 epochs, indicating that a network with two hidden layers of 32 neurons does not suffer from overt memorization on this dataset.
* *Scientific Caution:* While the small gap between training and validation error suggests healthy generalization, a small gap alone does not conclusively prove the total absence of regional overfitting (e.g., overfitting specific sub-regions of the input space). This requires the residual diagnostic audit outlined in `docs/06_diagnostics.md`.

---

## 8. Model Complexity & Hardware Operations

For an edge microcontroller such as the ESP32-S3, computational complexity is governed by Multiply-Accumulate operations (MACs) and parameter storage.

### Layer-by-Layer Breakdown:

| Layer Index & Type | Dimension ($a \rightarrow b$) | Weights ($a \times b$) | Biases ($b$) | Total Parameters | MACs per Prediction | Float32 Memory | Int8 Memory |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Layer 1: Linear** | $5 \rightarrow 32$ | 160 | 32 | 192 | 160 | 768 B | 192 B |
| **Layer 2: Linear** | $32 \rightarrow 32$ | 1,024 | 32 | 1,056 | 1,024 | 4,224 B | 1,056 B |
| **Layer 3: Linear** | $32 \rightarrow 1$ | 32 | 1 | 33 | 32 | 132 B | 33 B |
| **TOTALS** | — | **1,216** | **65** | **1,281** | **1,216** | **5,124 B** | **1,281 B** |

*(ReLU activations require zero trainable parameters and simple threshold comparisons).*

### Storage Metrics:
* **Raw Parameter Footprint (Float32):** $1,281 \times 4\text{ bytes} = \mathbf{5,124\text{ bytes}}\ (\approx 5.00\text{ KB})$.
* **Raw Parameter Footprint (Int8 Quantized):** $1,281 \times 1\text{ byte} = \mathbf{1,281\text{ bytes}}\ (\approx 1.25\text{ KB})$.
* **Serialized Model File on Disk (`model/model.pt`):** **$7,869\text{ bytes}$** ($\approx 7.68\text{ KB}$).
  * *Why is the file larger than 5,124 bytes?* PyTorch `.pt` files are compressed ZIP archives containing serialization headers, tensor type descriptors, string parameter names (`net.0.weight`, etc.), and structural metadata. During embedded deployment (e.g. C array or TFLite Micro), only the raw weight arrays (5,124 bytes) are stored.

---

## 9. Theoretical Edge AI Feasibility on ESP32-S3

> [!NOTE]
> **Theoretical Evaluation Context:** Physical deployment onto an ESP32-S3 microcontroller is **out of scope** for this assignment. The specifications below provide theoretical grounding to evaluate whether our neural network architecture is appropriately compact for real-world edge IoT constraints.

The project specification considers the **Espressif ESP32-S3** microcontroller archetype:
* **Internal SRAM:** $\approx 512\text{ KB}$ (estimated free application SRAM budget: $\approx 300\text{ KB}$).
* **External PSRAM:** $8\text{ MB}$
* **External Flash:** $16\text{ MB}$

### Feasibility Analysis:
1. **Weight Budget:** The raw model weights ($5.0\text{ KB}$ in float32, or $1.25\text{ KB}$ in int8) occupy less than **$1.7\%$** of the estimated $300\text{ KB}$ free internal SRAM.
2. **Computational Budget:** Performing $1,216\text{ MACs}$ on an ESP32-S3 dual-core Xtensa LX7 processor running at $240\text{ MHz}$ requires only a fraction of a millisecond.

### Theoretical Realities & Embedded Caveats:
* **Weight Size $\neq$ Total RAM Usage:** In a physical embedded deployment, memory must also accommodate intermediate activation buffers, the C/C++ runtime inference engine, the task stack, and networking buffers (Wi-Fi/TLS buffers can consume $40\text{ KB} - 80\text{ KB}$).
* **Quantization:** Although float32 weights fit easily, int8 quantization can be evaluated to measure accuracy-vs-memory trade-offs.
* **Scope Boundary:** While these calculations demonstrate theoretical viability, physical microcontroller flashing and firmware testing are not required deliverables; validation is performed via our FastAPI serving microservice and client benchmarking.

---

## 10. Engineering Interpretation

The neural network decisively outperforms both baseline models, cutting prediction error from $20.54\text{ days}$ down to $4.13\text{ days}$.

### Why Does the Network Win?
* **Piecewise Linear Non-Linearity:** Two hidden layers of 32 ReLU neurons can model non-linear curved surfaces, allowing the network to capture the steep drop in battery life at the low-voltage knee.
* **Feature Coupling:** The network learns non-linear interactions between high chime rates and degraded Wi-Fi signal strength without requiring manual polynomial feature engineering.

### Scientific Caution:
While the numerical performance is strong, **we do not claim that the model has perfectly learned true physical electrochemistry**. A flexible neural network can achieve low validation error simply by interpolating well within dense data regions. Confirming whether the model behaves physically soundly across the entire operational space requires the rigorous error and residual diagnostics detailed in the next document.

