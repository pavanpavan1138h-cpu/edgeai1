# 08. API Endpoint Integration, Inference Testing & Theoretical Edge Feasibility

---

## 📌 Document Status: SPECIFICATION & EXPERIMENTAL PROTOCOL

> [!IMPORTANT]
> ### Project Scope Boundary & Implementation Clarification
> **Physical deployment onto an ESP32-S3 microcontroller is NOT required for this assignment.**
> 
> * **Required Implementation Scope:**
>   1. Serving the trained PyTorch model via a lightweight **FastAPI microservice** (`app/main.py`).
>   2. Rejecting corrupt sensor telemetry at the API boundary with **HTTP 422 (Unprocessable Entity)** via Pydantic schema validation.
>   3. Performing API inference testing and stage latency profiling (`t["parse"]`, `t["normalise"]`, `t["inference"]`, `t["build"]`, and server total) using the client benchmark script (`client/bench.py`).
>   4. Comprehensive evaluation of model efficiency (trainable parameters, Multiply-Accumulate operations / MACs, and memory footprint in KB) to support model architecture selection.
> * **Out of Scope (Unless explicitly required later):**
>   * Physical ESP32-S3 microcontroller flashing, wiring, and hardware integration.
>   * Embedded C/C++ firmware development and FreeRTOS task integration.
>   * Physical hardware inference testing or on-chip memory/power profiling.
>   * Hardware-specific model conversion (e.g., ESP-DL, TFLite Micro C byte arrays).
>   * Hardware quantization or optimization strictly for physical embedded deployment.
> * **Role of ESP32-S3 Discussion:** The hardware specifications and calculations documented in Section 4 are retained as a **theoretical Edge AI feasibility study** to ground model sizing in realistic IoT hardware constraints and justify why compact neural networks are essential.

---

## 1. Serving Architecture: The FastAPI Microservice

To serve our trained model for application and testing purposes, the project provides a lightweight, modular web service implemented in `app/main.py` using **FastAPI** and **Uvicorn**:

```
[ Doorbell Telemetry / Benchmark Client ]
                 │
                 │ HTTP POST /predict (JSON Payload)
                 ▼
┌─────────────────────────────────────────────────────────────┐
│                 FASTAPI SERVING PIPELINE                    │
│                                                             │
│  Stage 1: [ Parse & Validate ]   ──► Pydantic Reading       │
│                                      (Returns HTTP 422 if   │
│                                       glitch detected)      │
│  Stage 2: [ Normalise ]          ──► (x - MU) / SD          │
│  Stage 3: [ PyTorch Inference ]  ──► net(x) (1 CPU thread)  │
│  Stage 4: [ Build Response ]     ──► {"days_left": ...,     │
│                                       "timings_ms": {...}}  │
└─────────────────────────────────────────────────────────────┘
```

### Endpoints Specification:

1. **`GET /health`:** Returns operational status, team identifier, parameter count, and validation MAE:
   ```json
   {
     "status": "ok",
     "team": 3,
     "params": 385,
     "val_mae": 4.27
   }
   ```
2. **`GET /model_info`:** Computes dynamic per-layer statistics directly from the loaded model (`net.net`):
   ```json
   {
     "layers": [
       {"name": "linear1", "in": 5, "out": 16, "params": 96, "bytes_f32": 384, "macs": 80},
       {"name": "linear2", "in": 16, "out": 16, "params": 272, "bytes_f32": 1088, "macs": 256},
       {"name": "linear3", "in": 16, "out": 1, "params": 17, "bytes_f32": 68, "macs": 16}
     ],
     "total_params": 385,
     "total_bytes_f32": 1540,
     "total_bytes_int8": 385,
     "total_macs": 352,
     "fits_300kb_sram_f32": true,
     "fits_300kb_sram_int8": true
   }
   ```
3. **`POST /predict`:** Accepts one JSON telemetry reading and returns predicted `days_left` with stage timings.
4. **`POST /predict_batch`:** Accepts up to 10,000 readings and computes vectorized batch predictions.

---

## 2. Pydantic Input Validation: Refusing Corrupt Telemetry (HTTP 422)

Smart doorbell telemetry frequently encounters corrupted sensor readings identical to the glitches identified during Data Inspection. Rather than silently accepting corrupted data and returning erroneous predictions, the API rejects invalid inputs at the boundary with **HTTP 422 (Unprocessable Entity)**.

### Implemented Validation Rules (`app/main.py`):
* `battery_v`: `Field(ge=3.0, le=4.3)` (Rejects $0.0\text{ V}$ dropouts and dangerous over-voltages).
* `temp_c`: `Field(ge=-25.0, le=60.0)` (Rejects $-127.0\text{ ^\circ C}$ sensor disconnect bus errors).
* `rings_per_day`: `Field(ge=0.0, le=100.0)` (Rejects negative or physically impossible ring frequencies).
* `wifi_rssi_dbm`: `Field(ge=-100.0, lt=0.0)` (Rejects default $0\text{ dBm}$ unassociated values).
* `age_months`: `Field(ge=0.0, le=60.0)` (Rejects negative device ages).

### Production Safeguard for Model Outputs:
As established in the Diagnostic Audit ([`docs/06_diagnostics.md`](06_diagnostics.md)), raw linear output layers may produce tiny negative predictions when batteries are near exhaustion. `app/main.py` explicitly enforces non-negativity before constructing the JSON response:
```python
prediction = float(np.maximum(raw_output, 0.0))
```

---

## 3. Request Latency Profiling Protocol

To analyze request efficiency and answer the engineering question (*"Where does the time go in a web request?"*), the API profiles execution latency across four distinct internal stages:
* **`t["parse"]`:** Parsing raw JSON payload into structured Pydantic objects.
* **`t["normalise"]`:** Vectorized standardisation $(X - \mu) / \sigma$ using NumPy.
* **`t["inference"]`:** Single forward pass through PyTorch `net(x)`.
* **`t["build"]`:** Formatting output dictionary and serializing JSON response.
* **`t["server_total"]`:** Total internal server-side processing duration.

### Containerized API Run

The runtime container installs only the packages needed to serve the API and copies the active model/configuration into the image. From the repository root, with Docker running:

```bash
docker compose up --build -d
docker compose ps
curl http://localhost:8000/health
curl http://localhost:8000/model_info
```

Open `http://localhost:8000/docs` to try the endpoints. Stop the service with `docker compose down`. The image build downloads Python dependencies, so the build environment needs access to the configured package indexes. The Codespaces devcontainer enables Docker-in-Docker; rebuild the Codespace after changing its Docker feature.

### External Benchmarking with `client/bench.py`:
When running API inference tests, we benchmark the running service using the client script:
```bash
python client/bench.py --url http://localhost:8000 --n 200
```
This script measures:
* First request cold-start latency.
* 50th (p50), 95th (p95), and 99th (p99) percentile round-trip latency.
* Network and framework overhead:
  $$\text{Latency}_{\text{network}} = \text{Round Trip Time} - \text{server\_total}$$

### Current Local Benchmark Results (Active 16-16 Checkpoint)

The active 16-16 checkpoint was benchmarked locally on 2026-11-21 using 200 `/predict` requests:

* Cold first request: 16.4 ms round trip
* Warm p50 round trip: 2.194 ms
* Warm p95 round trip: 3.735 ms
* Warm p99 round trip: 5.174 ms
* Warm p50 server total: 0.257 ms
* Warm p50 inference stage: 0.173 ms
* Full measurements: [`benchmark_results_16x16_selected_model.csv`](../benchmark_results_16x16_selected_model.csv)

### Historical Local Benchmark Results (32-32 Reference Model)

The earlier 32-32 reference checkpoint benchmark is retained for comparison:

* Cold first request: 20.1 ms round trip
* Warm p50 round trip: 2.634 ms
* Warm p95 round trip: 3.789 ms
* Warm p99 round trip: 4.507 ms
* Warm p50 server total: 0.332 ms
* Warm p50 inference stage: 0.229 ms
* Full measurements: [`benchmark_results_32x32_reference_model.csv`](../benchmark_results_32x32_reference_model.csv)

---

## 4. Theoretical Edge Hardware Feasibility: ESP32-S3 Analysis (Conceptual Study)

> [!NOTE]
> **Theoretical Evaluation Only:** Physical deployment, C++ firmware flashing, and hardware wiring on an ESP32-S3 development board are **out of scope** for this assignment. The calculations below are retained as a theoretical feasibility study to demonstrate that our selected neural network is appropriately sized for realistic edge IoT hardware.

The target edge microcontroller archetype considered in this course is the **Espressif ESP32-S3**:
* **Processor:** Dual-core Xtensa LX7 @ $240\text{ MHz}$
* **Internal SRAM:** $\approx 512\text{ KB}$ (estimated free application SRAM budget: $\approx 300\text{ KB}$)
* **External PSRAM:** $8\text{ MB}$ (accessible over SPI/octal-SPI, slower than internal SRAM)
* **Flash Memory:** $16\text{ MB}$

### Theoretical Sizing Analysis (Selected Model: 385 parameters):
| Metric | Float32 Representation | Int8 Quantized Representation | ESP32-S3 Budget | Fits in SRAM? |
| :--- | :---: | :---: | :---: | :---: |
| **Model Weight Storage** | $1,540\text{ bytes}\ (\approx 1.50\text{ KB})$ | $385\text{ bytes}\ (\approx 0.38\text{ KB})$ | $300\text{ KB}$ | **YES ($0.5\%$ of SRAM)** |
| **Peak Activation Memory** | $16 \times 4 = 64\text{ bytes}$ | $16 \times 1 = 16\text{ bytes}$ | $300\text{ KB}$ | **YES ($< 0.1\%$ of SRAM)** |
| **Inference MACs** | $352\text{ MACs}$ | $352\text{ MACs}$ | $\sim 480\text{ MIPS}$ | **YES ($< 0.1\text{ ms}$)** |

### Theoretical Edge AI Takeaways:
1. **Comfortable Memory Margin:** The selected model requires only $1.54\text{ KB}$ for weights and $64\text{ bytes}$ for activation buffers, leaving over $99\%$ of the stated SRAM budget for networking buffers, RTOS tasks, and application logic.
2. **Minimal Computational Load:** At $352\text{ MACs}$, computing one prediction consumes fewer than $10,000$ clock cycles on a $240\text{ MHz}$ core, requiring microjoules of energy.

---

## 5. Architectural Trade-Off Analysis: On-Device Edge Inference vs. Cloud Serving (Comparative Evaluation)

To demonstrate foundational understanding of Edge AI system design, the table below compares hypothetical on-device embedded inference against our implemented API service:

| Dimension | On-Device Microcontroller Execution (Theoretical) | Cloud / Central API Server Execution (Implemented) |
| :--- | :--- | :--- |
| **Inference Latency** | **Instantaneous ($< 1\text{ ms}$)** | High & Variable ($50\text{ ms} - 500\text{ ms}$ round trip) |
| **Wi-Fi Down Resilience** | **100% Operational** (calculates battery locally) | Complete Failure (requires network connection) |
| **Battery Energy Cost** | **Microjoules** (CPU active for a few cycles) | **Millijoules** (powering Wi-Fi radio to transmit) |
| **Cloud Hosting Cost** | **Zero server cost** | Ongoing monthly server / API gateway bills |
| **User Privacy** | Telemetry never leaves the device | Telemetry continuously logged on remote servers |
| **Model Updatability** | Requires firmware OTA update | Trivially updated in central server backend |
| **Implementation Scope** | Out of scope for this assignment | **Implemented & Verified in `app/main.py`** |
