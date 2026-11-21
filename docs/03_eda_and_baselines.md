# 03. Exploratory Data Analysis (EDA) & Baseline Benchmarks

---

## 1. Motivation: Why Perform Exploratory Data Analysis?

Before designing or training neural network architectures, machine learning engineering requires exploring how features behave, how they relate to the target variable, and whether linear assumptions hold. 

The primary goals of our EDA were:
1. **Understand feature distributions:** Detect skewness, multimodal behavior, and variances across inputs.
2. **Analyze target characteristics:** Understand the distribution of `days_left` to evaluate the difficulty of the regression task.
3. **Discover non-linearities:** Identify which sensor features have curved or piecewise relationships with battery life that linear models cannot capture.
4. **Establish classical baselines:** Measure baseline error metrics against which all neural network improvements can be quantitatively evaluated.

---

## 2. Feature Distributions & Observations

On the provisionally cleaned dataset (1,976 records), univariate distributions were evaluated using histograms and Kernel Density Estimation (KDE):

* **`battery_v` (3.35 V to 4.20 V):** Positively skewed towards higher operational voltages ($3.7\text{ V} - 4.1\text{ V}$), with a sharp decrease in observation frequency as voltage drops towards the $3.35\text{ V}$ cutoff.
* **`temp_c` (8.0 °C to 46.0 °C):** Displays a classic Gaussian (Normal) bell curve centered symmetrically around $\mu \approx 28.1^\circ\text{C}$ and $\sigma \approx 6.8^\circ\text{C}$, matching real-world seasonal ambient temperatures.
* **`rings_per_day` (0 to 38 rings):** Strongly right-skewed (Poisson-like). The majority of days experience modest chime activity ($4 - 10$ rings), with a sparse long tail of high-traffic days.
* **`wifi_rssi_dbm` (-88 dBm to -45 dBm):** Roughly uniform distribution spanning fringe coverage ($-88\text{ dBm}$) to strong near-access-point signal strength ($-45\text{ dBm}$).
* **`age_months` (0 to 36 months):** Uniform distribution from brand-new devices ($0$ months) to three-year-old installations ($36$ months).

---

## 3. In-Depth Target Variable Analysis (`days_left`)

The target variable `days_left` exhibits distinct statistical characteristics:

| Metric | Clean Target Value |
| :--- | :---: |
| **Total Valid Records** | 1,976 |
| **Mean ($\mu$)** | **$65.93\text{ days}$** |
| **Median (50th Percentile)** | **$55.25\text{ days}$** |
| **Standard Deviation ($\sigma$)** | **$57.88\text{ days}$** |
| **Minimum** | **$0.10\text{ days}$** |
| **First Quartile (Q1, 25%)** | **$12.75\text{ days}$** |
| **Third Quartile (Q3, 75%)** | **$105.03\text{ days}$** |
| **Maximum** | **$317.10\text{ days}$** |
| **Skewness** | **$+1.173$** (Positively / Right-Skewed) |

### Machine Learning Implications:
* **Asymmetric Error Penalties:** Because over $50\%$ of the observations represent devices with fewer than 55 days remaining, the model encounters many more low-lifespan examples during training.
* **The Long Tail:** Devices with $> 200\text{ days}$ left are comparatively sparse. Regression models evaluated with Mean Squared Error (MSE) can suffer large squared penalties on rare high-lifespan points if they underpredict the upper tail.

---

## 4. Feature vs. Target Relationships: Identifying Non-Linearities

We generated five bivariate scatter plots evaluating `days_left` against each input feature, comparing a linear trendline against a degree-3 non-linear polynomial:

```
               [ days_left ]
                    ▲
                    │        .---. (Steady discharge plateau: 3.7V - 4.1V)
                    │       /
                    │      /
                    │     /
                    │    /
                    │   /
                    │  |  <--- The "Discharge Knee" (~3.5V - 3.6V)
                    │  |       (Battery life plummets vertically)
                    └──┴────────────────────────► battery_v
                      3.3V  3.5V  3.7V  3.9V  4.2V
```

### Analysis of Relationships:
1. **`battery_v` vs `days_left` $\rightarrow$ STRONGLY NON-LINEAR:**
   * This is the single most critical relationship in the system.
   * Lithium-ion batteries do not discharge along a straight diagonal line. They feature a relatively flat operational plateau between $3.7\text{ V}$ and $4.1\text{ V}$, followed by a sharp **"discharge knee"** around $3.5\text{ V} - 3.6\text{ V}$ where voltage rapidly drops towards cutoff.
   * In this knee region, battery capacity plummets to near zero over a very narrow voltage window. A straight linear line cannot bend at this knee and fundamentally miscalculates remaining life near depletion.
2. **`rings_per_day` vs `days_left` $\rightarrow$ INVERSE NON-LINEAR ($\sim 1/x$):**
   * Battery lifespan is physically governed by $\text{Lifespan} = \frac{\text{Stored Energy}}{\text{Daily Consumption Rate}}$. Since daily consumption scales with activity count, the resulting lifespan follows an inverse relationship.
3. **`wifi_rssi_dbm` vs `days_left` $\rightarrow$ NON-LINEAR RF SCALING:**
   * At poor Wi-Fi reception (e.g. $-85\text{ dBm}$), power amplifiers draw disproportionately higher radio current and initiate packet retransmissions, accelerating battery drain compared to solid reception ($-50\text{ dBm}$).
4. **`temp_c` & `age_months` vs `days_left`:**
   * Both exhibit steady downward trends representing ambient self-discharge and irreversible battery capacity fade over time.

---

## 5. Correlation Analysis & Its Limitations

We computed the Pearson correlation matrix on the clean dataset:

| Feature Name | Pearson Correlation ($r$) with `days_left` | Interpretation |
| :--- | :---: | :--- |
| **`battery_v`** | **$+0.730$** | Strong positive linear association |
| **`rings_per_day`** | **$-0.264$** | Moderate negative association |
| **`age_months`** | **$-0.218$** | Moderate negative association |
| **`wifi_rssi_dbm`** | **$+0.198$** | Weak-to-moderate positive association (closer to 0 is stronger) |
| **`temp_c`** | **$-0.065$** | Weak negative linear association |

### Critical Technical Caveats on Correlation:
* **Correlation $\neq$ Causation:** A mathematical correlation coefficient indicates that two variables co-vary, but does not prove direct physical cause without experimental control.
* **Pearson Measures ONLY Linear Association:** If a relationship is curved or non-linear (like $y = x^2$ or a sigmoidal battery discharge knee), Pearson's $r$ underestimates the true predictive relationship. 
* **Compound Interactions:** While `temp_c` has an individual correlation of only $-0.065$, temperature interacts strongly with internal battery resistance and Wi-Fi transmission efficiency. Neural networks can capture these multi-variable interactions where linear correlation metrics see weak individual signal.

---

## 6. Baseline Models: Benchmarking Reference Points

In machine learning, training a complex neural network without comparing it to simple baselines is poor engineering practice. Baselines answer: *Is deep learning actually adding value over simple arithmetic or classical statistics?*

Both baselines were evaluated on an **80/20 train/validation split** with `random_seed = 0` (1,581 training rows, 395 validation rows).

### Baseline 1: Predict the Mean
* **Concept:** Always predict the scalar mean of the training target ($\bar{y}_{\text{train}} = 65.98\text{ days}$) regardless of what the sensor readings report.
* **Validation MAE:** **$47.58\text{ days}$**
* **Validation MSE:** **$3,354.21$**
* **Limitation:** Represents the floor of model performance. It is completely blind to device voltage, temperature, or activity.

### Baseline 2: Linear Regression via Ordinary Least Squares
* **Concept:** Fit a linear hyperplane $y = w^T X + b$ using `np.linalg.lstsq` on the augmented matrix $[X, 1]$:
  $$\min_w \| X_{\text{train}} w - y_{\text{train}} \|_2^2$$
* **Fitted Linear Equation:**
  $$\text{days\_left} = 181.9049 \cdot \text{battery\_v} - 0.2486 \cdot \text{temp\_c} - 2.5595 \cdot \text{rings\_per\_day} + 1.5008 \cdot \text{wifi\_rssi\_dbm} - 1.0167 \cdot \text{age\_months} - 477.7202$$
* **Validation MAE:** **$20.54\text{ days}$**
* **Validation MSE:** **$792.14$**
* **Limitation & Failure Modes:**
  1. *Fails at Low Voltages ($< 3.6\text{ V}$):* Overestimates remaining days because it cannot bend downward at the sharp discharge knee (sometimes predicting positive days when the battery is dead, or predicting negative days).
  2. *Fails at High Voltages ($> 4.1\text{ V}$):* Underestimates the longevity of fresh, fully charged cells.
  3. *Zero Feature Interaction:* Assumes the effect of rings is completely independent of Wi-Fi signal strength, missing the multiplicative compounding power drain.

These baselines establish our performance thresholds: any viable neural network must decisively beat **$20.54\text{ days}$ MAE**.

