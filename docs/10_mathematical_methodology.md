# 10. Mathematical & Computational Methodology

---

## 📌 Document Overview & Purpose

This document provides the authoritative mathematical and computational formulations utilized throughout the **Edge AI Smart Doorbell Battery-Life Prediction** project (Team 3 — Phoenix). 

Every error metric, statistical descriptor, complexity formula, Multiply-Accumulate (MAC) calculation, memory estimation, and percentage reduction used in this repository is formally defined below with theoretical derivations and worked numerical examples. This guarantees that every numerical figure presented in our reports is independently verifiable by any evaluator or researcher.

---

## 1. Machine Learning Performance & Error Metrics

### 1.1 Mean Absolute Error (MAE)
Mean Absolute Error measures the average magnitude of prediction errors in the original physical units of the target variable ($days$):

$$\boxed{\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} \left| y_i - \hat{y}_i \right|}$$

Where:
* $y_i \in \mathbb{R}^+$: Ground-truth remaining battery lifespan for sample $i$ (measured in days).
* $\hat{y}_i \in \mathbb{R}$: Model prediction for sample $i$ (measured in days).
* $\left| y_i - \hat{y}_i \right|$: Absolute error for sample $i$.
* $N$: Total number of evaluated observations ($N = 395$ in validation).

**Engineering Interpretation:** MAE provides a direct, linearly proportional representation of operational error. An MAE of $4.13\text{ days}$ indicates that, across all operating regimes, predictions deviate from actual remaining lifespan by an average of $4.13\text{ days}$.

---

### 1.2 Mean Squared Error (MSE)
Mean Squared Error computes the average of the squared prediction errors:

$$\boxed{\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} \left( y_i - \hat{y}_i \right)^2}$$

**Engineering Interpretation:** By squaring residuals, MSE penalizes large errors quadratically. It serves as the smooth, differentiable objective loss function during PyTorch backpropagation (`nn.functional.mse_loss`). Its physical units are $\text{days}^2$.

---

### 1.3 Root Mean Squared Error (RMSE)
Root Mean Squared Error is the square root of the Mean Squared Error:

$$\boxed{\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} \left( y_i - \hat{y}_i \right)^2}}$$

**Engineering Interpretation:** RMSE restores the unit of measurement to physical **days** while retaining disproportionate sensitivity to large outlier errors. When $\text{RMSE} \gg \text{MAE}$, it indicates the presence of a heavy right-tail of large errors.
* *Example (Reference Model 32–32, Seed 0):* $\text{MSE} = 36.2441 \implies \text{RMSE} = \sqrt{36.2441} = \mathbf{6.0203\text{ days}}$ (compared to $\text{MAE} = 4.1275\text{ days}$).

---

### 1.4 Median Absolute Error (MedAE)
The Median Absolute Error calculates the 50th percentile of the sorted absolute error array:

$$\boxed{\text{MedAE} = \operatorname{median}\left( \left| y_1 - \hat{y}_1 \right|, \left| y_2 - \hat{y}_2 \right|, \dots, \left| y_N - \hat{y}_N \right| \right)}$$

**Engineering Interpretation:** MedAE is completely robust to extreme tail outliers. It describes the typical operational error experienced by the median user.
* *Worked Example (Reference Model 32–32, Seed 0):* Across 395 validation samples, sorting absolute residuals reveals that the middle value is $\mathbf{2.4197\text{ days}}$. Thus, exactly $50\%$ of all predictions have an error smaller than $2.42\text{ days}$, proving that the mean MAE ($4.13\text{d}$) is pulled upward by high-lifespan outliers.

---

### 1.5 Mean Signed Error (Model Bias)
Mean Signed Error measures the systematic directional tendency of model predictions:

$$\boxed{\text{Bias} = \frac{1}{N} \sum_{i=1}^{N} \left( \hat{y}_i - y_i \right)}$$

* **Our Implemented Convention:** $\hat{y}_i - y_i$ (Prediction minus Actual).
* **Positive Bias ($\text{Bias} > 0$):** The model systematically over-predicts remaining battery life (tends to overestimate days).
* **Negative Bias ($\text{Bias} < 0$):** The model systematically under-predicts remaining battery life (tends to underestimate days).
* *Example (Reference Model 32–32, Seed 0):* $\text{Bias} = \mathbf{-1.2172\text{ days}}$, indicating a mild conservative underprediction tendency across the full distribution.

---

### 1.6 Low-Battery Regime MAE ($0 - 10\text{ Days}$)
To evaluate model reliability where battery replacement warnings are critical, we define the low-battery subset strictly using **actual ground-truth lifespan**:

$$\text{Low-Battery Subset} = \left\{ i \in \{1, \dots, N\} \mid 0.0 \le y_i \le 10.0\text{ days} \right\}$$

$$\boxed{\text{MAE}_{\text{low}} = \frac{1}{N_{\text{low}}} \sum_{i \in \text{Low-Battery}} \left| y_i - \hat{y}_i \right|}$$

* **Validation Sample Count:** Exactly **$N_{\text{low}} = 85\text{ samples}$** ($21.52\%$ of the 395 validation samples).
* **Evaluation Principle:** Ground truth $y_i$ is used to partition the slice (not prediction $\hat{y}_i$) to evaluate performance on genuinely depleted hardware.
* *Worked Example (16–16, Seed 1):* Sum of absolute errors across the 85 samples = $80.41\text{ days} \implies \text{MAE}_{\text{low}} = \frac{80.41}{85} = \mathbf{0.9460\text{ days}}$ (sub-day accuracy in the alert regime).

---

### 1.7 Sample Standard Deviation Across Seeds ($s$)
To measure variability across training seeds $s \in \{0, 1, 2, 3, 4\}$, the repeatability experiment script (`experiments/shortlist_repeatability.py`) computes the **unbiased sample standard deviation**:

$$\boxed{s = \sqrt{\frac{1}{K - 1} \sum_{k=1}^{K} \left( m_k - \bar{m} \right)^2}}$$

Where:
* $K = 5$ (Number of seeds evaluated).
* $m_k$: Metric value (e.g. Validation MAE) for seed $k$.
* $\bar{m} = \frac{1}{K} \sum_{k=1}^K m_k$: Sample mean across the 5 seeds.
* **Bessel's Correction ($K - 1 = 4$, `ddof=1`):** Verified directly in code at line 266:
  `np.std(val_maes, ddof=1)`. Sample standard deviation is strictly used, not population standard deviation (`ddof=0`).

#### Worked Example: Sample Standard Deviation for `2-Layer (16, 16)`
Across seeds 0 through 4, validation MAEs are:
$$m = [4.7758, 4.2742, 4.4172, 4.3076, 4.4998]$$

1. **Calculate Sample Mean ($\bar{m}$):**
   $$\bar{m} = \frac{4.7758 + 4.2742 + 4.4172 + 4.3076 + 4.4998}{5} = \frac{22.2746}{5} = \mathbf{4.45492 \approx 4.4549\text{ days}}$$
2. **Calculate Squared Deviations $(m_k - \bar{m})^2$:**
   * $(4.7758 - 4.45492)^2 = (+0.32088)^2 = 0.102964$
   * $(4.2742 - 4.45492)^2 = (-0.18072)^2 = 0.032660$
   * $(4.4172 - 4.45492)^2 = (-0.03772)^2 = 0.001423$
   * $(4.3076 - 4.45492)^2 = (-0.14732)^2 = 0.021703$
   * $(4.4998 - 4.45492)^2 = (+0.04488)^2 = 0.002014$
   * Sum of squared deviations: $\sum = 0.160764$
3. **Apply Bessel's Correction ($K - 1 = 4$):**
   $$s^2 = \frac{0.160764}{4} = 0.040191$$
   $$s = \sqrt{0.040191} = \mathbf{0.200477 \approx 0.2005\text{ days}}$$
This matches the reported value in `repeatability_summary.csv` to 4 decimal places.

---

## 2. Trainable Parameter Formulations

Our neural network architectures consist of fully connected (dense) linear layers with intermediate ReLU activations:

For a dense linear layer with $n_{\text{in}}$ inputs and $n_{\text{out}}$ outputs:
$$\boxed{P_{\text{layer}} = n_{\text{in}} \cdot n_{\text{out}} + n_{\text{out}}}$$

Where:
* $n_{\text{in}} \cdot n_{\text{out}}$: Trainable weight parameters connecting every input neuron to every output neuron.
* $n_{\text{out}}$: Trainable bias parameters (one scalar bias per output neuron).
* ReLU activation layers contain zero trainable parameters ($P_{\text{ReLU}} = 0$).

### 2.1 Complete Layer-by-Layer Derivations for All 10 Architectures

The table below details the exact weight and bias derivations for every candidate architecture evaluated in Sweep 1 (EXP-005):

| Architecture | Layer 1 ($5 \rightarrow w_1$) | Layer 2 ($w_1 \rightarrow w_2$) | Output Layer ($w \rightarrow 1$) | Total Parameter Derivation | Total Trainable Parameters ($P_{\text{total}}$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`1-Layer (4)`** | $5 \times 4 + 4 = 24$ | N/A | $4 \times 1 + 1 = 5$ | $24 + 5$ | **29** |
| **`1-Layer (8)`** | $5 \times 8 + 8 = 48$ | N/A | $8 \times 1 + 1 = 9$ | $48 + 9$ | **57** |
| **`1-Layer (16)`** | $5 \times 16 + 16 = 96$ | N/A | $16 \times 1 + 1 = 17$ | $96 + 17$ | **113** |
| **`1-Layer (32)`** | $5 \times 32 + 32 = 192$ | N/A | $32 \times 1 + 1 = 33$ | $192 + 33$ | **225** |
| **`1-Layer (64)`** | $5 \times 64 + 64 = 384$ | N/A | $64 \times 1 + 1 = 65$ | $384 + 65$ | **449** |
| **`2-Layer (4, 4)`** | $5 \times 4 + 4 = 24$ | $4 \times 4 + 4 = 20$ | $4 \times 1 + 1 = 5$ | $24 + 20 + 5$ | **49** |
| **`2-Layer (8, 8)`** | $5 \times 8 + 8 = 48$ | $8 \times 8 + 8 = 72$ | $8 \times 1 + 1 = 9$ | $48 + 72 + 9$ | **129** |
| **`2-Layer (16, 16)`**| $5 \times 16 + 16 = 96$ | $16 \times 16 + 16 = 272$ | $16 \times 1 + 1 = 17$ | $96 + 272 + 17$ | **385** |
| **`2-Layer (32, 32)`**| $5 \times 32 + 32 = 192$| $32 \times 32 + 32 = 1,056$| $32 \times 1 + 1 = 33$ | $192 + 1,056 + 33$ | **1,281** |
| **`2-Layer (64, 64)`**| $5 \times 64 + 64 = 384$| $64 \times 64 + 64 = 4,160$| $64 \times 1 + 1 = 65$ | $384 + 4,160 + 65$ | **4,609** |

#### Worked Example: Selected Architecture `2-Layer (16, 16)` ($5 \rightarrow 16 \rightarrow 16 \rightarrow 1$)
* **Layer 1 (Input $\rightarrow$ Hidden 1):**
  $$\text{Weights} = 5 \times 16 = 80, \quad \text{Biases} = 16 \implies P_1 = 80 + 16 = \mathbf{96}$$
* **Layer 2 (Hidden 1 $\rightarrow$ Hidden 2):**
  $$\text{Weights} = 16 \times 16 = 256, \quad \text{Biases} = 16 \implies P_2 = 256 + 16 = \mathbf{272}$$
* **Layer 3 (Hidden 2 $\rightarrow$ Output):**
  $$\text{Weights} = 16 \times 1 = 16, \quad \text{Biases} = 1 \implies P_3 = 16 + 1 = \mathbf{17}$$
* **Total Parameters:**
  $$P_{\text{total}} = P_1 + P_2 + P_3 = 96 + 272 + 17 = \mathbf{385\text{ parameters}}$$

---

## 3. Multiply-Accumulate (MAC) Operations Formulations

A Multiply-Accumulate (MAC) operation computes:
$$a \leftarrow a + (b \times c)$$
In fully connected layers, computing the linear combination for each output neuron requires $n_{\text{in}}$ multiplications and $n_{\text{in}} - 1$ additions (accumulations). In standardized deep learning complexity accounting, the number of input-weight multiplications defines the MAC count:

$$\boxed{\text{MACs}_{\text{layer}} = n_{\text{in}} \cdot n_{\text{out}}}$$

### 3.1 Scope & Counting Convention
* **Included:** All scalar multiplications between input feature activations and weight parameters across all linear layers during one single forward inference pass.
* **Excluded:** 
  1. Bias additions (counted as standalone element-wise additions, not MACs).
  2. Non-linear activation functions (ReLU element-wise thresholding $\max(0, z)$).
  3. Preprocessing feature standardizations ($Z = (X - \mu) / \sigma$).
* **Critical Distinction:** Theoretical MAC counts represent computational complexity, **not measured wall-clock execution latency**. Memory bus bandwidth, cache misses, and pipeline stalls govern hardware execution time.

### 3.2 Complete Layer-by-Layer MAC Derivations for All 10 Architectures

| Architecture | Layer 1 ($5 \rightarrow w_1$) | Layer 2 ($w_1 \rightarrow w_2$) | Output Layer ($w \rightarrow 1$) | Total MAC Derivation | Total Inference MACs |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`1-Layer (4)`** | $5 \times 4 = 20$ | N/A | $4 \times 1 = 4$ | $20 + 4$ | **24** |
| **`1-Layer (8)`** | $5 \times 8 = 40$ | N/A | $8 \times 1 = 8$ | $40 + 8$ | **48** |
| **`1-Layer (16)`** | $5 \times 16 = 80$ | N/A | $16 \times 1 = 16$ | $80 + 16$ | **96** |
| **`1-Layer (32)`** | $5 \times 32 = 160$ | N/A | $32 \times 1 = 32$ | $160 + 32$ | **192** |
| **`1-Layer (64)`** | $5 \times 64 = 320$ | N/A | $64 \times 1 = 64$ | $320 + 64$ | **384** |
| **`2-Layer (4, 4)`** | $5 \times 4 = 20$ | $4 \times 4 = 16$ | $4 \times 1 = 4$ | $20 + 16 + 4$ | **40** |
| **`2-Layer (8, 8)`** | $5 \times 8 = 40$ | $8 \times 8 = 64$ | $8 \times 1 = 8$ | $40 + 64 + 8$ | **112** |
| **`2-Layer (16, 16)`**| $5 \times 16 = 80$ | $16 \times 16 = 256$ | $16 \times 1 = 16$ | $80 + 256 + 16$ | **352** |
| **`2-Layer (32, 32)`**| $5 \times 32 = 160$ | $32 \times 32 = 1,024$| $32 \times 1 = 32$ | $160 + 1,024 + 32$ | **1,216** |
| **`2-Layer (64, 64)`**| $5 \times 64 = 320$ | $64 \times 64 = 4,096$| $64 \times 1 = 64$ | $320 + 4,096 + 64$ | **4,480** |

---

## 4. Parameter Memory Footprint Estimations

### 4.1 32-Bit Floating Point (FP32) Storage
Standard PyTorch neural network tensors store weights and biases as IEEE 754 single-precision floating point numbers ($4\text{ bytes per parameter}$):

$$\boxed{\text{RAM}_{\text{FP32}} = P_{\text{total}} \times 4\text{ bytes}}$$

* *Worked Example (16–16):* $385\text{ parameters} \times 4\text{ bytes} = \mathbf{1,540\text{ bytes}} \approx \mathbf{1.50\text{ KB}}$.
* *Worked Example (32–32 Reference):* $1,281\text{ parameters} \times 4\text{ bytes} = \mathbf{5,124\text{ bytes}} \approx \mathbf{5.00\text{ KB}}$.

### 4.2 Theoretical 8-Bit Integer (INT8) Storage
Under post-training quantization (PTQ) or quantization-aware training (QAT), parameters are mapped to 8-bit signed integers ($1\text{ byte per parameter}$):

$$\boxed{\text{RAM}_{\text{INT8}} = P_{\text{total}} \times 1\text{ byte}}$$

* *Worked Example (16–16):* $385\text{ parameters} \times 1\text{ byte} = \mathbf{385\text{ bytes}} \approx \mathbf{0.38\text{ KB}}$.
* *Worked Example (32–32 Reference):* $1,281\text{ parameters} \times 1\text{ byte} = \mathbf{1,281\text{ bytes}} \approx \mathbf{1.25\text{ KB}}$.
* **Methodological Notice:** These INT8 values are **theoretical arithmetic estimates** for edge feasibility analysis. **No physical quantization was executed** on model checkpoints in this project.

### 4.3 Memory Categorization & Scope Boundaries
To avoid misleading hardware claims, we explicitly distinguish raw parameter storage from total deployment memory:

| Memory Component | `2-Layer (16, 16)` | `2-Layer (32, 32)` | Description & Role |
| :--- | :---: | :---: | :--- |
| **Raw FP32 Parameter Storage** | **$1,540\text{ bytes}$** | **$5,124\text{ bytes}$** | Theoretical memory required for weight and bias tensors only. |
| **Theoretical INT8 Parameter Storage**| **$385\text{ bytes}$** | **$1,281\text{ bytes}$** | Theoretical quantized parameter storage (no quantization performed). |
| **Serialized Disk Checkpoint (`.pt`)** | $\approx 4,873\text{ bytes}$ | **$7,869\text{ bytes}$** | PyTorch ZIP container containing tensor metadata, pickle headers, and variable names. |
| **Peak Activation Memory** | $16 \times 4 = 64\text{ B}$ | $32 \times 4 = 128\text{ B}$ | Temporary RAM to store hidden layer forward activations. |
| **Python / Framework Runtime** | $\approx 40 - 60\text{ MB}$ | $\approx 40 - 60\text{ MB}$ | PyTorch C++ backend, memory allocators, and Python interpreter. |
| **Microcontroller Application Budget** | $1.5\text{ KB} \ll 300\text{ KB}$ | $5.1\text{ KB} \ll 300\text{ KB}$ | Available free internal SRAM on typical IoT chips (e.g. ESP32-S3). |

---

## 5. Percentage Complexity Reduction & Increase Formulations

When comparing a candidate architecture against an established reference model, percentage change is calculated as:

$$\boxed{\text{Reduction (\%)} = \frac{\text{Reference} - \text{Candidate}}{\text{Reference}} \times 100}$$

$$\boxed{\text{Increase (\%)} = \frac{\text{Candidate} - \text{Reference}}{\text{Reference}} \times 100}$$

### 5.1 Worked Numerical Derivations: Selected 16–16 vs. Reference 32–32

Using the established baseline reference model (`2-Layer (32, 32)`: $P = 1,281$, $\text{MACs} = 1,216$):

1. **Parameter Reduction:**
   $$\text{Reduction} = \frac{1,281 - 385}{1,281} \times 100 = \frac{896}{1,281} \times 100 = 69.945355 \dots \% \implies \mathbf{69.95\%}$$
2. **Computational (MAC) Reduction:**
   $$\text{Reduction} = \frac{1,216 - 352}{1,216} \times 100 = \frac{864}{1,216} \times 100 = 71.052631 \dots \% \implies \mathbf{71.05\%}$$
3. **FP32 Storage Memory Reduction:**
   $$\text{Reduction} = \frac{5,124\text{ B} - 1,540\text{ B}}{5,124\text{ B}} \times 100 = \frac{3,584}{5,124} \times 100 = 69.945355 \dots \% \implies \mathbf{69.95\%}$$
4. **Accuracy Trade-Off (Validation MAE across 5 Seeds):**
   * Reference 32–32 Mean MAE: $4.2575\text{ days}$
   * Candidate 16–16 Mean MAE: $4.4549\text{ days}$
   * Absolute Difference: $4.4549 - 4.2575 = \mathbf{0.1974\text{ days}}$
   * Difference in Hours: $0.1974\text{ days} \times 24\text{ hours/day} = \mathbf{4.7376\text{ hours}} \approx \mathbf{4.74\text{ hours}}$

*Conclusion:* The 16–16 architecture reduces parameters by $\approx 70\%$ and compute by $\approx 71\%$ at the cost of only $\approx 4.74\text{ hours}$ in mean validation MAE.

---

### 5.2 Complete Complexity Reduction / Increase Table Across All 10 Architectures

The table below benchmarks all 10 candidate architectures evaluated in Sweep 1 against the reference `2-Layer (32, 32)` model:

| Architecture | Trainable Parameters | Inference MACs | Parameter Reduction vs. Ref | MAC Reduction vs. Ref | Final Val MAE (Seed 0) | Pareto Optimal? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`1-Layer (4)`** | 29 | 24 | **$97.74\%$ reduction** | **$98.03\%$ reduction** | $10.53\text{ d}$ | **YES** |
| **`1-Layer (8)`** | 57 | 48 | **$95.55\%$ reduction** | **$96.05\%$ reduction** | $15.62\text{ d}$ | NO |
| **`1-Layer (16)`** | 113 | 96 | **$91.18\%$ reduction** | **$92.11\%$ reduction** | $15.53\text{ d}$ | NO |
| **`1-Layer (32)`** | 225 | 192 | **$82.44\%$ reduction** | **$84.21\%$ reduction** | $7.22\text{ d}$ | NO |
| **`1-Layer (64)`** | 449 | 384 | **$64.95\%$ reduction** | **$68.42\%$ reduction** | $15.52\text{ d}$ | NO |
| **`2-Layer (4, 4)`** | 49 | 40 | **$96.17\%$ reduction** | **$96.71\%$ reduction** | $8.08\text{ d}$ | **YES** |
| **`2-Layer (8, 8)`** | 129 | 112 | **$89.93\%$ reduction** | **$90.79\%$ reduction** | $5.08\text{ d}$ | **YES** |
| **`2-Layer (16, 16)`**| 385 | 352 | **$69.95\%$ reduction** | **$71.05\%$ reduction** | $4.78\text{ d}$ | **YES** |
| **`2-Layer (32, 32)`**| **1,281** | **1,216** | **REFERENCE ($0.00\%$)** | **REFERENCE ($0.00\%$)** | **$4.13\text{ d}$** | **YES (Accuracy Ref)** |
| **`2-Layer (64, 64)`**| 4,609 | 4,480 | **$259.80\%$ INCREASE** | **$268.42\%$ INCREASE** | $4.55\text{ d}$ | NO (Overfit) |

*Derivation for 64–64 Increase:*
$$\text{Param Increase} = \frac{4,609 - 1,281}{1,281} \times 100 = \frac{3,328}{1,281} \times 100 = \mathbf{259.80\%}$$
$$\text{MAC Increase} = \frac{4,480 - 1,216}{1,216} \times 100 = \frac{3,264}{1,216} \times 100 = \mathbf{268.42\%}$$
This proves that expanding to 64 units per layer inflates computational complexity by over $2.6\times$ while degrading generalization accuracy from $4.13\text{d}$ to $4.55\text{d}$.

