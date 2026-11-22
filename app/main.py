"""
app/main.py -- STARTER battery API. Look for TODO.

    uvicorn app.main:app --host 0.0.0.0 --port 8000

GET  /health          -> {"status": "ok", "team": 3, "params": 385, "val_mae": 4.27}
GET  /model_info      -> the per-layer table: params, bytes, MACs, and whether it fits the doorbell
POST /predict         one reading  -> {"days_left": 52.1, "timings_ms": {...}}
POST /predict_batch   {"rows": [reading, ...]} -> {"days_left": [...], "timings_ms": {...}}
A reading that is out of range gets HTTP 422 with the reason, never a guess.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, Request, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from torch import nn

from app.model import FEATURES, BatteryNet

ROOT = Path(__file__).resolve().parent.parent
with (ROOT / "model" / "config.json").open("r", encoding="utf-8") as config_file:
    cfg = json.load(config_file)
net = BatteryNet(tuple(cfg["hidden"]))
net.load_state_dict(torch.load(ROOT / "model" / "model.pt", map_location="cpu", weights_only=True))
net.eval()
torch.set_num_threads(1)
MU, SD = np.array(cfg["mu"], np.float32), np.array(cfg["sd"], np.float32)
N_PARAMS = sum(p.numel() for p in net.parameters())


class Reading(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "battery_v": 3.85,
                "temp_c": 25.0,
                "rings_per_day": 15,
                "wifi_rssi_dbm": -65,
                "age_months": 12
            }
        }
    )

    battery_v: float = Field(ge=3.0, le=4.3)
    temp_c: float = Field(ge=-25.0, le=60.0)
    rings_per_day: float = Field(ge=0.0, le=100.0)
    wifi_rssi_dbm: float = Field(ge=-100.0, lt=0.0)
    age_months: float = Field(ge=0.0, le=60.0)


class Batch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rows: list[Reading] = Field(min_length=1, max_length=10000)


app = FastAPI(title="Doorbell battery API")
ms = lambda t0: round((time.perf_counter() - t0) * 1000, 3)


@app.get("/model_info")
def model_info():
    layers = []
    linear_index = 0
    total_params = 0
    total_macs = 0

    for module in net.net:
        if not isinstance(module, nn.Linear):
            continue
        linear_index += 1
        params = module.in_features * module.out_features + module.out_features
        macs = module.in_features * module.out_features
        layers.append({
            "name": f"linear{linear_index}",
            "in": module.in_features,
            "out": module.out_features,
            "params": params,
            "bytes_f32": params * 4,
            "macs": macs,
        })
        total_params += params
        total_macs += macs

    total_bytes_f32 = total_params * 4
    total_bytes_int8 = total_params
    sram_budget = 300 * 1024
    return {
        "layers": layers,
        "total_params": total_params,
        "total_bytes_f32": total_bytes_f32,
        "total_bytes_int8": total_bytes_int8,
        "total_macs": total_macs,
        "fits_300kb_sram_f32": total_bytes_f32 <= sram_budget,
        "fits_300kb_sram_int8": total_bytes_int8 <= sram_budget,
    }


@app.get("/health")
def health():
    return {"status": "ok", "team": cfg["team"], "params": N_PARAMS, "val_mae": cfg["val_mae"]}


async def run(request: Request, schema):
    t_all = time.perf_counter(); t = {}
    t0 = time.perf_counter()
    try:
        data = schema.model_validate_json(await request.body())
    except ValidationError as e:
        return JSONResponse(status_code=422, content={"error": "bad reading", "detail": json.loads(e.json())})
    t["parse"] = ms(t0)
    rows = data.rows if isinstance(data, Batch) else [data]

    t0 = time.perf_counter()
    x = (np.array([[getattr(r, f) for f in FEATURES] for r in rows], np.float32) - MU) / SD
    t["normalise"] = ms(t0)

    t0 = time.perf_counter()
    with torch.no_grad():
        y = net(torch.from_numpy(x)).numpy()
    t["inference"] = ms(t0)

    t0 = time.perf_counter()
    days = [round(float(v), 2) for v in np.maximum(y, 0)]
    body = {"days_left": days if isinstance(data, Batch) else days[0]}
    body["timings_ms"] = t
    t["build"] = ms(t0)
    t["server_total"] = ms(t_all)
    body["timings_ms"] = t
    return body


@app.post("/predict")
async def predict(
    request: Request,
    reading: Reading = Body(
        ...,
        examples={
            "realistic": {
                "summary": "Realistic doorbell reading",
                "value": {
                    "battery_v": 3.85,
                    "temp_c": 25.0,
                    "rings_per_day": 15,
                    "wifi_rssi_dbm": -65,
                    "age_months": 12
                }
            }
        }
    )
):
    return await run(request, Reading)


@app.post("/predict_batch")
async def predict_batch(request: Request, batch: Batch):
    return await run(request, Batch)
