from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


VALID_READING = {
    "battery_v": 3.8,
    "temp_c": 28.0,
    "rings_per_day": 10,
    "wifi_rssi_dbm": -65,
    "age_months": 12,
}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["team"] == 3
    assert body["params"] == 385
    assert body["val_mae"] == 4.27


def test_model_info_is_calculated_from_network():
    response = client.get("/model_info")
    assert response.status_code == 200
    body = response.json()
    assert [layer["name"] for layer in body["layers"]] == ["linear1", "linear2", "linear3"]
    assert body["total_params"] == 385
    assert body["total_bytes_f32"] == 1540
    assert body["total_bytes_int8"] == 385
    assert body["total_macs"] == 352
    assert [layer["in"] for layer in body["layers"]] == [5, 16, 16]
    assert [layer["out"] for layer in body["layers"]] == [16, 16, 1]
    assert body["fits_300kb_sram_f32"] is True
    assert body["fits_300kb_sram_int8"] is True


def test_predict_returns_prediction_and_stage_timings():
    response = client.post("/predict", json=VALID_READING)
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["days_left"], (int, float))
    assert body["days_left"] >= 0
    assert set(body["timings_ms"]) >= {
        "parse",
        "normalise",
        "inference",
        "build",
        "server_total",
    }


def test_openapi_documents_predict_request_body():
    response = client.get("/openapi.json")

    assert response.status_code == 200
    data = response.json()
    request_body = data["paths"]["/predict"]["post"]["requestBody"]
    json_content = request_body["content"]["application/json"]
    schema = json_content["schema"]
    if "$ref" in schema:
        ref_path = schema["$ref"].lstrip("#/").split("/")
        schema = data
        for part in ref_path:
            schema = schema[part]
    assert schema["properties"].keys() == VALID_READING.keys()
    assert json_content["example"]["battery_v"] == 3.85


def test_predict_rejects_glitch_reading():
    invalid = dict(VALID_READING, battery_v=0.0)
    response = client.post("/predict", json=invalid)
    assert response.status_code == 422


def test_predict_rejects_unknown_field():
    invalid = dict(VALID_READING, unknown=123)
    response = client.post("/predict", json=invalid)
    assert response.status_code == 422


def test_batch_prediction():
    response = client.post("/predict_batch", json={"rows": [VALID_READING, VALID_READING]})
    assert response.status_code == 200
    body = response.json()
    assert len(body["days_left"]) == 2
    assert all(value >= 0 for value in body["days_left"])


def test_empty_batch_is_rejected():
    response = client.post("/predict_batch", json={"rows": []})
    assert response.status_code == 422
