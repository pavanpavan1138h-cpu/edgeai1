# Edge AI Smart Doorbell Battery-Life Prediction

Team 3 — Phoenix | AI255TBD Edge AI, Module 1

This repository contains the data analysis, model experiments, selected PyTorch checkpoint, FastAPI inference service, Docker setup, tests, and technical reports for predicting a smart doorbell's remaining battery life from sensor telemetry.

> **Current served model:** 2-layer MLP `5 → 16 → 16 → 1`, seed 1. Its seed-specific validation MAE is **4.27 days**. The five-seed experiment mean is **4.4549 ± 0.2005 days**. The 32–32 model remains an experiment/reference, not the active checkpoint.

## Team

| Team member | Name | USN |
|---|---|---|
| 1 | Pavan R | 1RV24CI081
| 2 | Shreyank D K | 1RV24CI119
| 3 | Adarsh K R | 1RV24CI009
| 4 | Ankitha A | 1RV25CI402

Team name: **Phoenix**.

## Project summary

The model estimates `days_left` from five telemetry inputs:

| Feature | Meaning | API validation range |
|---|---|---:|
| `battery_v` | Battery voltage (V) | 3.0–4.3 |
| `temp_c` | Temperature (°C) | -25–60 |
| `rings_per_day` | Daily ring count | 0–100 |
| `wifi_rssi_dbm` | Wi-Fi signal strength (dBm) | -100–less than 0 |
| `age_months` | Battery/device age (months) | 0–60 |

The training dataset is `data/team_03_train.csv` (2,000 records). The project identified and provisionally filtered 24 sensor-glitch rows for modeling, leaving 1,976 records. The original CSV is retained. The clean data uses a deterministic 80/20 split (1,581 training and 395 validation records); feature normalization statistics are computed from training rows only.

The filtering thresholds are project assumptions based on suspected sensor error codes, not a claim that all readings outside those rules are physically impossible. API input ranges are deliberately broader than the clean training-data ranges.

## Results at a glance

### Prediction accuracy

| Model / experiment | Validation MAE | Notes |
|---|---:|---|
| Predict validation-set training mean | 47.58 days | Baseline |
| Ordinary least-squares linear regression | 20.54 days | Baseline |
| 32–32 MLP, seed 0 | 4.13 days | Historical single-run reference |
| 16–16 MLP, seed 1 | 4.2742 days | **Current served checkpoint** |
| 16–16 MLP, five-seed mean | 4.4549 ± 0.2005 days | Repeatability summary |
| 32–32 MLP, five-seed mean | 4.2575 ± 0.1556 days | Larger accuracy reference |

The team selected 16–16 as a size/accuracy compromise: **385 parameters**, **352 MACs**, **1,540 FP32 weight bytes**, compared with 1,281 parameters and 1,216 MACs for 32–32. The reported INT8 footprint is a theoretical one-byte-per-parameter estimate; the served checkpoint is not actually quantized.

The selected model's five-seed mean low-battery MAE (validation records with actual `days_left` ≤10) is **1.2213 days**. See the repeatability report for the full methods and per-seed results.

### API latency benchmarks

The checked-in benchmark CSVs were collected locally on Windows using 200 sequential `/predict` requests. Percentiles below exclude the first (cold) request; they are illustrative measurements from this machine, not Codespaces or production guarantees.

| Served architecture during run | Cold first request | Warm round-trip p50 / p95 / p99 | Warm server-total p50 | Warm inference p50 | Results |
|---|---:|---:|---:|---:|---|
| 16–16 selected checkpoint | 16.415 ms | 2.194 / 3.735 / 5.174 ms | 0.257 ms | 0.173 ms | [`benchmark_results_16x16_selected_model.csv`](benchmark_results_16x16_selected_model.csv) |
| 32–32 historical reference | 20.083 ms | 2.634 / 3.789 / 4.507 ms | 0.332 ms | 0.229 ms | [`benchmark_results_32x32_reference_model.csv`](benchmark_results_32x32_reference_model.csv) |

Latency varies with host load, network, and runtime. Do not treat this local benchmark as a direct comparison of model accuracy or a hardware performance claim.

## Repository guide

| Path | Purpose |
|---|---|
| `app/model.py` | Shared PyTorch `BatteryNet` definition and feature order |
| `app/main.py` | FastAPI application, checkpoint loading, validation, inference, and model metadata |
| `app.py` | Convenience launcher for the API |
| `model/model.pt` | Active 16–16 seed-1 state dictionary |
| `model/config.json` | Active architecture, feature order, normalization statistics, and checkpoint MAE |
| `train.py` | Data filtering, baselines, training, and model export |
| `tests/test_app.py` | API behavior tests |
| `client/bench.py` | HTTP latency benchmark client |
| `experiments/architecture_sweep_1.py` | Ten-architecture sweep |
| `experiments/shortlist_repeatability.py` | Three architectures across five seeds |
| `experiments/sweep_1_results/` | Architecture sweep metrics, plots, and weights |
| `experiments/shortlist_repeatability/` | Per-seed metrics, plots, and checkpoints |
| `data/team_03_train.csv` | Team 3 raw telemetry dataset |
| `notebook/` | Dataset inspection and EDA notebooks |
| `docs/` | Detailed experiment, methodology, selection, and deployment reports |
| `Dockerfile`, `compose.yaml` | Container image and Compose service definition |
| `requirements.txt` | Development, training, testing, and API dependencies |
| `requirements-runtime.txt` | API-only container dependencies |
| `.devcontainer/devcontainer.json` | GitHub Codespaces configuration, including Docker-in-Docker |

## Run locally with Python

Python 3.11 is recommended. Run commands from the repository root.

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest tests -q
python app.py
```

### Linux, macOS, or GitHub Codespaces

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest tests -q
python app.py
```

The test suite should report **7 passed**. The server listens on port 8000. In Codespaces, open port **8000** in the Ports tab.

## Run with Docker Compose

Docker Desktop or Docker-in-Docker must be available and running. Building the image requires network access to the Python package indexes, including the PyTorch CPU wheel index.

```bash
docker compose up --build -d
docker compose ps
docker compose logs -f battery-api
```

The service is published on port 8000. Stop it with:

```bash
docker compose down
```

For Codespaces, the devcontainer has a Docker-in-Docker feature configured. Rebuild the Codespace after changing devcontainer settings, then check `docker info` before running Compose.

## API usage

Interactive OpenAPI documentation: `http://localhost:8000/docs`

| Method and path | Description |
|---|---|
| `GET /health` | Service status, team, active model parameter count, checkpoint validation MAE |
| `GET /model_info` | Dynamic per-linear-layer parameters, FP32 bytes, MACs, totals, theoretical SRAM-fit flags |
| `POST /predict` | Predict from one telemetry reading |
| `POST /predict_batch` | Predict from 1–10,000 readings |

Example request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"battery_v":3.8,"temp_c":28,"rings_per_day":10,"wifi_rssi_dbm":-65,"age_months":12}'
```

The response contains a non-negative `days_left` value and `timings_ms` fields: `parse`, `normalise`, `inference`, `build`, and `server_total`. Out-of-range fields, missing fields, and unexpected fields are rejected with HTTP 422 rather than scored as valid readings.

For a batch request, wrap the readings in a `rows` array:

```json
{
  "rows": [
    {
      "battery_v": 3.8,
      "temp_c": 28,
      "rings_per_day": 10,
      "wifi_rssi_dbm": -65,
      "age_months": 12
    }
  ]
}
```

`/model_info` reports theoretical FP32 and INT8 parameter-storage estimates. INT8 size is not evidence that quantization or embedded INT8 inference has been implemented.

## Run a benchmark

Keep the API running, then use a separate terminal:

```bash
python client/bench.py --url http://localhost:8000 --n 200
```

The client reports cold first-request latency, warm p50/p95/p99 round-trip and server stages, and network/framework overhead. It writes a new `bench_<timestamp>.csv` in the current directory. For Codespaces, use the forwarded port URL if running the client outside the Codespace; label those results separately from the checked-in local CSVs.

## Retrain or reproduce experiments

The active `model/model.pt` is the repeatability experiment's **16–16 seed-1 checkpoint**. `train.py` defaults to seed 0 and overwrites `model/model.pt` and `model/config.json`; retraining with the command below therefore replaces the active seed-1 checkpoint with a newly trained seed-0 checkpoint and its own validation metric:

```bash
python train.py --team 3 --hidden 16,16 --epochs 1500
```

To reproduce the published architecture and repeatability experiments:

```bash
python experiments/architecture_sweep_1.py
python experiments/shortlist_repeatability.py
```

These scripts write experiment artifacts beneath `experiments/`. Review the output and back up the active model before running any script that overwrites production model files.

## Scope and limitations

- The API and Docker container provide a testable software deployment; this is not physical microcontroller deployment.
- ESP32-S3 SRAM fit, MAC counts, and INT8 weight size are theoretical calculations. Firmware, hardware flashing, measured on-device timing, and actual quantization are not included.
- Dataset glitch rules are provisional assumptions and do not establish ground truth sensor failure labels.
- Validation metrics are based on the documented fixed split; they do not guarantee performance on future devices or unseen operating conditions.
- Five training seeds provide a repeatability check, not formal proof of statistical significance.
- The project returns a point estimate, not a calibrated uncertainty interval.

## Detailed documentation

Start with [`docs/README.md`](docs/README.md) for the documentation index. Key reports:

- [`docs/01_problem_definition.md`](docs/01_problem_definition.md) — task definition and scope
- [`docs/02_dataset_and_data_quality.md`](docs/02_dataset_and_data_quality.md) — data schema, quality, and filtering
- [`docs/03_eda_and_baselines.md`](docs/03_eda_and_baselines.md) — EDA and baseline metrics
- [`docs/05_experiment_1_initial_nn.md`](docs/05_experiment_1_initial_nn.md) — historical 32–32 reference experiment
- [`docs/06_diagnostics.md`](docs/06_diagnostics.md) — residual and error diagnostics
- [`docs/07_architecture_experiments.md`](docs/07_architecture_experiments.md) — architecture sweep
- [`docs/08_repeatability.md`](docs/08_repeatability.md) — five-seed results
- [`docs/09_model_selection.md`](docs/09_model_selection.md) — rationale for selecting 16–16
- [`docs/10_mathematical_methodology.md`](docs/10_mathematical_methodology.md) — parameter, MAC, and metric derivations
- [`docs/11_assumptions_and_reproducibility.md`](docs/11_assumptions_and_reproducibility.md) — assumptions and reproduction notes
- [`docs/teammate_edge_deployment.md`](docs/teammate_edge_deployment.md) — API, container, benchmark, and theoretical edge analysis
