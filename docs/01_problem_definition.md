# 01. Problem Definition & Machine Learning Formulation

---

## 1. Problem Context: The Smart Doorbell Battery Problem

Smart home IoT devices, such as connected video doorbells, operate primarily on rechargeable Lithium-Ion battery cells. In current commercial firmware, battery status is typically communicated through coarse battery percentage indicators or abrupt notifications warning the homeowner that the battery is "Low". 

For an end-user, "Low" is ambiguous and unhelpful:
* Does "Low" mean the doorbell will shut down in two hours, or does it have three days of operation left?
* If an unexpected cold snap arrives or the homeowner receives multiple package deliveries in an afternoon, when will the camera actually die?

The smart doorbell company wants to replace coarse status warnings with an accurate, actionable prediction: **exactly how many days of operation remain before the battery depletes (`days_left`)?**

---

## 2. Why Battery-Life Prediction Matters

1. **User Experience & Peace of Mind:** Homeowners can plan recharges proactively (e.g., charging over the weekend) rather than experiencing sudden security camera outages.
2. **Dynamic Energy Management:** If the device accurately knows its remaining operational lifespan, it can dynamically throttle power-hungry subsystems (such as Wi-Fi transmission power or video bitrate) to extend critical uptime.
3. **Hardware Health Monitoring:** Comparing predicted battery life against historical discharge rates enables early detection of degraded battery cells or firmware bugs causing excessive background power draw.

---

## 3. The Edge AI Motivation

Traditional IoT architectures transmit raw sensor telemetry directly to centralized cloud servers, where large machine learning models compute inferences and return predictions back to the user app. 

In real-world smart doorbell deployments, this cloud-centric approach encounters severe physical limitations:
* **Connectivity Failures:** When local Wi-Fi drops or the internet service provider fails, a cloud-dependent doorbell loses all ability to compute battery health.
* **Latency & Radio Energy Cost:** Powering up a high-power Wi-Fi radio transceiver (e.g., ESP32-S3 Wi-Fi drawing $150\text{ mA} - 250\text{ mA}$) to transmit multiple telemetry payloads to a remote cloud API drains significant battery power.
* **Cloud Infrastructure Cost:** Serving millions of active doorbells with frequent cloud API calls generates continuous hosting, compute, and bandwidth expenses.
* **Privacy:** Users prefer telemetry and operational data to remain local rather than continuously streamed to third-party servers.

**The Edge AI Goal & Scope:** Train a compact, highly accurate neural network that is small enough to fit within the constrained static RAM ($\approx 300\text{ KB}$) of an ultra-low-power microcontroller (such as the Espressif ESP32-S3). In this assignment, our required implementation scope focuses on dataset analysis, baseline models, neural network experiments, architecture comparison, model selection, necessary API endpoint integration (`app/main.py`), and API inference testing (`client/bench.py`). Physical deployment onto physical ESP32-S3 hardware is out of scope; embedded hardware constraints are evaluated theoretically to motivate and justify model compactness.

---

## 4. Machine Learning Formulation

We formulate the battery-life prediction problem as a **Supervised Tabular Regression** task:

* **Supervised Learning:** The model is trained on historical paired observations $(x_i, y_i)$, where $x_i \in \mathbb{R}^5$ represents the sensor reading vector and $y_i \in \mathbb{R}^+$ represents the ground-truth remaining battery lifespan.
* **Regression:** The target variable is continuous, measured in days, spanning from $0.1\text{ days}$ up to $\approx 317\text{ days}$.

### Input Features ($x \in \mathbb{R}^5$)
The system measures five environmental and operational telemetry features:

| Feature Name | Description | Measurement Unit | Expected Hardware Range |
| :--- | :--- | :---: | :---: |
| `battery_v` | Battery terminal voltage | Volts ($V$) | $3.0\text{ V} - 4.25\text{ V}$ |
| `temp_c` | Ambient outdoor temperature | Celsius ($^\circ C$) | $-10^\circ\text{C} - 50^\circ\text{C}$ |
| `rings_per_day` | Number of doorbell chime / ring events per day | Integer count | $0 - 50$ rings |
| `wifi_rssi_dbm` | Wi-Fi Received Signal Strength Indicator | $dBm$ (negative) | $-95\text{ dBm} - -20\text{ dBm}$ |
| `age_months` | Age of the installed battery cell | Months | $0 - 36\text{ months}$ |

### Target Variable ($y$)
* **`days_left`:** The continuous quantity of remaining operational days before battery voltage drops to cutoff and the doorbell powers down.

---

## 5. Model Training vs. Model Deployment (Inference)

A foundational concept in Edge AI is the distinction between **training** and **deployment**:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      OFFLINE MODEL TRAINING                             │
│  Environment: High-performance workstation / cloud CPU/GPU              │
│  • Processes full datasets (2,000 observations in batches of 128)        │
│  • Computes loss, backpropagation, and weight gradient updates (Adam)    │
│  • Iterates over 1,500 epochs to minimize Mean Squared Error (MSE)       │
│  • Memory usage: Hundreds of megabytes (PyTorch computational graph)    │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │ Exports weights (model.pt)
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      DEPLOYED INFERENCE (EDGE / API)                    │
│  Environment: Lightweight FastAPI Server / Embedded Edge Device         │
│  • Evaluates a single input vector: x = [3.86, 28.0, 5, -66, 22]         │
│  • Executes forward pass only (no backprop, no gradients, no optimizer) │
│  • Highly constrained resource budget: < 300 KB SRAM, < 10 ms latency   │
│  • Model weights must be compact, frozen, and efficient                 │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Objectives & Expected Outcomes

1. **Mission 1: Data Understanding & Baselines:**
   * Explore the Team 3 dataset (`data/team_03_train.csv`) and isolate corrupt sensor telemetry.
   * Establish baseline performance using two classical techniques: constant mean prediction and linear regression via Ordinary Least Squares (`np.linalg.lstsq`).
2. **Mission 2: Neural Network Engineering:**
   * Implement, standardize, and train a Multilayer Perceptron ($5 \rightarrow 32 \rightarrow 32 \rightarrow 1$ with ReLU).
   * Evaluate learning dynamics, diagnose overfitting, and calculate exact parameters, Multiply-Accumulates (MACs), and memory footprint.
3. **Mission 3: API Integration & Testing:**
   * Deploy the model via a FastAPI microservice implementing `/health`, `/model_info`, `/predict`, and `/predict_batch`.
   * Enforce Pydantic input validation ranges so corrupt sensor readings return **HTTP 422**.
   * Profile request latency across stages (`parse`, `normalise`, `inference`, `build`) using `client/bench.py`.
4. **Mission 4: Theoretical Edge Feasibility Analysis:**
   * Analyze whether the model theoretically fits inside an Espressif ESP32-S3 microcontroller ($\approx 300\text{ KB}$ available internal SRAM).
   * Evaluate trade-offs between on-device edge execution versus cloud API inference (physical hardware flashing is out of scope).

