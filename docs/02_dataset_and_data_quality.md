# 02. Dataset & Data Quality Audit

---

## 1. Dataset Overview

The dataset assigned to **Team 3 (Phoenix)** is stored at:
```
data/team_03_train.csv
```

### Dimensions & Schema
* **Total Observations:** 2,000 rows
* **Total Columns:** 6 columns (5 input features + 1 target variable)
* **File Format:** Comma-Separated Values (CSV), UTF-8 encoded
* **Raw File Integrity:** All inspections were performed in a read-only manner; the original CSV file remains **100% untouched and preserved**.

| Column Name | Role | Data Type | Physical Meaning | Unit |
| :--- | :--- | :---: | :--- | :---: |
| `battery_v` | Input Feature | `float64` | Battery terminal voltage | Volts ($V$) |
| `temp_c` | Input Feature | `float64` | Ambient operating temperature | Celsius ($^\circ C$) |
| `rings_per_day` | Input Feature | `int64` | Doorbell chime / activity count per day | Integer count |
| `wifi_rssi_dbm` | Input Feature | `int64` | Wi-Fi Received Signal Strength Indicator | $dBm$ (negative) |
| `age_months` | Input Feature | `int64` | Battery service age | Months |
| `days_left` | **Target Variable** | `float64` | Remaining operational battery life | Days |

---

## 2. Data Integrity Checks

### A. Missing-Value Audit (`df.isnull().sum()`)
We evaluated the dataset for null, NaN, or unpopulated fields:

| Column | Missing Count | Percentage (%) |
| :--- | :---: | :---: |
| `battery_v` | 0 | 0.00% |
| `temp_c` | 0 | 0.00% |
| `rings_per_day` | 0 | 0.00% |
| `wifi_rssi_dbm` | 0 | 0.00% |
| `age_months` | 0 | 0.00% |
| `days_left` | 0 | 0.00% |

**Audit Outcome:** There are **zero missing values** across all columns. No mathematical imputation (mean/median substitution) was required.

### B. Duplicate Record Audit (`df.duplicated().sum()`)
We searched for exact duplicate rows where every feature and the target variable are identical.
* **Result:** `0` duplicate rows.
* Every one of the 2,000 records represents a unique telemetry observation.

---

## 3. Raw Descriptive Statistics & Hardware Domain Sanity Check

```python
df.describe()
```

| Metric | `battery_v` (V) | `temp_c` (°C) | `rings_per_day` | `wifi_rssi_dbm` (dBm) | `age_months` | `days_left` (days) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Count** | 2000 | 2000 | 2000 | 2000 | 2000 | 2000 |
| **Mean** | 3.7667 | 27.7696 | 8.0830 | -64.7170 | 18.0670 | 65.7989 |
| **Std Dev** | 0.3514 | 10.3134 | 5.5683 | 13.5670 | 10.7888 | 57.9199 |
| **Min** | **0.0000** | **-127.0000** | 0.0000 | -88.0000 | 0.0000 | 0.1000 |
| **25% (Q1)** | 3.5715 | 23.5000 | 4.0000 | -76.0000 | 9.0000 | 12.5000 |
| **50% (Median)**| 3.7845 | 28.2500 | 7.0000 | -65.0000 | 18.0000 | 54.8000 |
| **75% (Q3)** | 4.0000 | 32.8000 | 11.0000 | -54.0000 | 27.0000 | 104.5750 |
| **Max** | 4.2000 | 46.0000 | 38.0000 | **0.0000** | 36.0000 | 317.1000 |

### Domain Sanity Discrepancies
When comparing the minimum and maximum boundaries against physical hardware realities, three significant anomalies were flagged:
1. **`battery_v` Min = 0.0 V:** A single-cell Lithium-ion battery reaches hardware low-voltage cutoff around $3.0\text{ V} - 3.2\text{ V}$. An active doorbell measuring and transmitting Wi-Fi telemetry **cannot operate at 0.0 V**.
2. **`temp_c` Min = -127.0 °C:** $-127.0^\circ\text{C}$ is physically unrealistic for outdoor residential environments. In embedded systems, digital temperature sensors (e.g., Dallas DS18B20 1-Wire sensors) return $-127^\circ\text{C}$ (`0x80`/`0x81`) as a hardware communication error code when data lines disconnect or pull-ups fail.
3. **`wifi_rssi_dbm` Max = 0 dBm:** Wi-Fi RSSI is measured in logarithmic negative decibels relative to one milliwatt ($-90\text{ dBm}$ to $-30\text{ dBm}$). An RSSI of $0\text{ dBm}$ represents $1\text{ mW}$ directly on the antenna, which in real consumer firmware indicates an unassociated Wi-Fi stack returning a default uninitialized value.

---

## 4. Provisional Sensor-Glitch Investigation

To prevent training corrupted telemetry patterns into our neural network, we audited the suspicious values:

### A. Glitch Breakdown

| Anomaly Type | Filtering Condition | Rows Detected | Characteristics | Interpretation |
| :--- | :--- | :---: | :--- | :--- |
| **Voltage Dropout** | `battery_v < 3.0` | **9 rows** | Every detected row has `battery_v == 0.0` | ADC conversion dropout / brownout read |
| **Temperature Disconnect** | `temp_c < -10.0` | **5 rows** | Every detected row has `temp_c == -127.0` | Digital bus communication fault code |
| **Wi-Fi Null State** | `wifi_rssi_dbm >= 0` | **10 rows** | Every detected row has `wifi_rssi_dbm == 0` | Unconnected / unassociated network driver default |
| **Total Distinct Glitches**| **Combined OR** | **24 rows** | No overlapping rows (all disjoint) | **1.2% of raw records** |

* **Clean Records Retained:** **1,976 rows** ($98.8\%$ of dataset).

### B. Important Scientific & Engineering Disclaimer
> **⚠️ Note on Provisional Assumptions:**
> While the three filtering rules are physically well-justified by electronics and embedded systems domain knowledge, **we do not claim these are definitively proven hardware faults**. 
> Without access to the raw analog sensor datasheets or the internal data-generation logger source code, these rules remain **provisional domain-based assumptions**. In an industrial production setting, these hypotheses would be formally verified with the hardware engineering team.

### C. Why Real Extremes Were Kept
We explicitly distinguished between **physically impossible readings** and **unusual but valid observations**:
* **High Ring Counts (`rings_per_day = 38`):** While 38 rings is well above the median (7 rings), it represents a valid high-activity day (e.g., a party or holiday delivery surge). It was **retained**.
* **Low Battery Lifespan (`days_left = 0.1`):** A battery with a fraction of a day remaining represents a valid, critical operational state near shutdown. It was **retained**.
* **Zero Ring Counts (`rings_per_day = 0`):** Vacant homes or quiet days legitimately have zero rings. It was **retained**.

---

## 5. Glitch Filtering Implementation

The filtering logic used across `notebook/dataset_inspection.ipynb`, `notebook/eda.ipynb`, and `train.py` is:

```python
# Combined provisional sensor-glitch filter
bad_mask = (df['battery_v'] < 3.0) | (df['temp_c'] < -10.0) | (df['wifi_rssi_dbm'] >= 0)

# Filter for modeling only; original raw dataframe remains preserved
df_clean = df[~bad_mask].copy().reset_index(drop=True)
```

The resulting clean dataset of 1,976 records forms the foundation for all subsequent Exploratory Data Analysis and Model Training experiments.

