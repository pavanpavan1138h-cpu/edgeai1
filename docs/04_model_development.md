# 04. Neural Network Development & Preprocessing Methodology

---

## 1. Why a Neural Network for Battery-Life Prediction?

While classical linear models are lightweight and easy to interpret, our Exploratory Data Analysis established that:
1. Battery discharge curves exhibit a pronounced **non-linear knee** below $3.6\text{ V}$.
2. Telemetry features interact non-linearly (e.g., the combined effect of high daily rings and poor Wi-Fi reception accelerates battery drain far more rapidly than either factor alone).

A **Multilayer Perceptron (MLP)** neural network provides a flexible non-linear function approximator capable of learning complex piecewise-curved relationships without requiring manual feature engineering of polynomial cross-terms.

---

## 2. Core Concepts: How Multilayer Perceptrons Work

For students and evaluators learning these foundations, the network is built upon the following building blocks:

```
Inputs (x)          Hidden Layer 1 (h1)       Hidden Layer 2 (h2)       Output (y)
[battery_v]     ──► ( Neuron 1 )        ──►   ( Neuron 1 )        ──►
[temp_c]        ──► ( Neuron 2 )        ──►   ( Neuron 2 )        ──►  [ days_left ]
[rings_per_day] ──► (    ...   )        ──►   (    ...   )        ──►   (1 neuron,
[wifi_rssi_dbm] ──► ( Neuron 32 )       ──►   ( Neuron 32 )       ──►    linear)
[age_months]
  (5 inputs)          (32 neurons, ReLU)        (32 neurons, ReLU)
```

1. **Neuron (Perceptron):** The fundamental unit of computation. Each neuron receives inputs, multiplies them by learnable **weights**, adds a learnable **bias**, and produces a weighted sum:
   $$z = \sum_{i=1}^n w_i x_i + b = \mathbf{w}^T \mathbf{x} + b$$
2. **Weights ($w$):** Parameters representing the relative importance and directional effect (positive or negative) of each input on that neuron.
3. **Biases ($b$):** Learnable scalar offsets that allow the activation threshold to shift independently of the input magnitudes.
4. **Activation Function — Rectified Linear Unit (ReLU):**
   $$\text{ReLU}(z) = \max(0, z)$$
   * *Why ReLU?* If we connected linear layers together without non-linear activations, the entire network would mathematically collapse into a single large linear regression model. 
   * ReLU introduces non-linearity by turning off negative activations ($0$) while passing positive activations linearly ($z$). When dozens of ReLU neurons combine, they form a **piecewise linear approximation** capable of bending around sharp discharge curves and knees.
5. **Linear Output Layer:** The final layer contains a single neuron with **no activation function** (linear activation: $y = z$). This allows the model to output continuous positive real numbers representing remaining days.
6. **Forward Propagation:** The process of feeding an input vector through the layers sequentially to compute the model's prediction $\hat{y}$.
7. **Loss Function — Mean Squared Error (MSE):**
   $$\mathcal{L}_{\text{MSE}} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2$$
   * MSE is chosen as the optimization objective because it is smooth, continuously differentiable, and heavily penalizes large outlier errors, guiding gradient updates effectively.
8. **Evaluation Metric — Mean Absolute Error (MAE):**
   $$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$$
   * While MSE is ideal for training gradients, MAE is the primary human-interpretable metric for evaluators: an MAE of $4.13$ directly means the model is off by an average of $4.13\text{ days}$.
9. **Backpropagation:** Using the calculus chain rule, the algorithm computes the partial derivative of the loss with respect to every weight and bias in the network ($\frac{\partial \mathcal{L}}{\partial w}$, $\frac{\partial \mathcal{L}}{\partial b}$).
10. **Adam Optimizer (Adaptive Moment Estimation):** An advanced variant of stochastic gradient descent that tracks both the first moment (running average of gradients) and second moment (uncentered variance of gradients). It automatically adapts the step size for each parameter, ensuring stable convergence.
11. **Hyperparameters:**
    * **Learning Rate ($\alpha = 0.003$):** The step size taken along the gradient direction during optimization.
    * **Batch Size ($B = 128$):** The number of training samples processed before updating weights. Mini-batches balance vectorization speed with gradient noise that helps escape local minima.
    * **Epochs ($1,500$):** The number of complete passes through the entire training dataset.
12. **Generalization vs. Overfitting:**
    * **Generalization:** The model's ability to accurately predict unseen validation samples.
    * **Overfitting:** When a model memorizes idiosyncrasies or noise in the training set, causing training error to plummet while validation error rises.

---

## 3. Data Preprocessing & Input Standardization

A major requirement in training deep learning models is proper feature scaling.

### A. The Scale Mismatch Problem
In raw telemetry, sensor features exist on radically different scales:
* `battery_v` spans $\approx 0.85\text{ V}$ ($3.35\text{ V} - 4.20\text{ V}$).
* `wifi_rssi_dbm` spans $\approx 43\text{ dBm}$ ($-88\text{ dBm} - -45\text{ dBm}$).
* `temp_c` spans $\approx 38^\circ\text{C}$ ($8^\circ\text{C} - 46^\circ\text{C}$).
* `age_months` spans $36\text{ months}$ ($0 - 36$).

Without normalization, the loss landscape becomes an elongated, steep valley. The weights for `wifi_rssi_dbm` receive large gradients, while the weights for `battery_v` (the most vital feature!) receive tiny updates, causing training to stall or require thousands of unnecessary iterations.

### B. Standardization Formula
We apply standard score (Z-score) normalization:
$$z = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
Where $\mu_{\text{train}}$ is the mean and $\sigma_{\text{train}}$ is the standard deviation.

### C. Why Normalization Stats Come ONLY from the Training Set
> **🚨 Critical Machine Learning Rule:**
> Normalization statistics ($\mu, \sigma$) must be computed **strictly on the 1,581 training rows**. 
> * **No Data Leakage:** Using validation or whole-dataset statistics leaks distribution boundaries of unseen data into the model.
> * **Production Simulation:** When the model is deployed on an ESP32-S3 microcontroller or FastAPI server, it receives single incoming sensor readings one at a time. It cannot calculate a batch mean on future inputs; it must standardize using the fixed pre-computed training $\mu$ and $\sigma$.
> * **Validation Transformation:** The validation set ($395\text{ rows}$) is transformed using the **exact same training $\mu$ and $\sigma$**.

### D. Verified Training Normalization Parameters
Calculated on the 1,581 training rows (`random_seed = 0`):

| Feature Name | Training Mean ($\mu$) | Training Std Dev ($\sigma$) |
| :--- | :---: | :---: |
| `battery_v` | $3.782831\text{ V}$ | $0.244777\text{ V}$ |
| `temp_c` | $27.987982^\circ\text{C}$ | $6.852437^\circ\text{C}$ |
| `rings_per_day` | $7.872233$ | $5.482431$ |
| `wifi_rssi_dbm` | $-66.229599\text{ dBm}$ | $12.483250\text{ dBm}$ |
| `age_months` | $18.014547\text{ months}$ | $10.814246\text{ months}$ |

### E. Why Target `days_left` is NOT Normalized
The target `days_left` is kept unnormalized in its native physical unit of **days**:
1. Preserves direct physical interpretability of the loss function.
2. Eliminates the need for inverse-scaling operations during edge deployment.
3. Matches the project's standard baseline evaluation pipeline.

