"""
experiments/compare_400_single_vs_batch.py -- Benchmark comparison:
400 individual /predict requests vs 1 /predict_batch request containing 400 readings.
"""
import argparse
import json
import time

import numpy as np
import requests


def generate_reading(rng):
    return {
        "battery_v": round(float(rng.uniform(3.4, 4.2)), 3),
        "temp_c": round(float(rng.uniform(10.0, 45.0)), 1),
        "rings_per_day": int(rng.integers(0, 30)),
        "wifi_rssi_dbm": int(rng.integers(-88, -45)),
        "age_months": int(rng.integers(0, 37)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    url = args.url.rstrip("/")

    # Check health
    health = requests.get(f"{url}/health", timeout=10).json()
    print(f"Connected to API: {health}")

    rng = np.random.default_rng(42)
    readings = [generate_reading(rng) for _ in range(400)]

    # Warmup request
    warmup_res = requests.post(f"{url}/predict", json=readings[0], timeout=10)
    assert warmup_res.status_code == 200

    print("\n" + "=" * 60)
    print("EXPERIMENT: 400 INDIVIDUAL /predict REQUESTS")
    print("=" * 60)

    session = requests.Session()
    individual_rows = []
    individual_preds = []

    t_single_start = time.perf_counter()
    for i, r in enumerate(readings):
        t0 = time.perf_counter()
        resp = session.post(f"{url}/predict", json=r, timeout=30)
        rt = (time.perf_counter() - t0) * 1000.0
        assert resp.status_code == 200, f"Request {i} failed: {resp.status_code}"
        data = resp.json()
        individual_preds.append(data["days_left"])
        tm = data.get("timings_ms", {})
        row = {"i": i, "round_trip": rt, **tm}
        if "server_total" in tm:
            row["network_and_framework"] = rt - tm["server_total"]
        individual_rows.append(row)
    t_single_total = (time.perf_counter() - t_single_start) * 1000.0

    print(f"Total wall-clock time for 400 individual requests: {t_single_total:.2f} ms ({t_single_total/1000:.3f} s)")
    print(f"First request round-trip (cold in loop): {individual_rows[0]['round_trip']:.3f} ms")

    keys = ["round_trip", "parse", "normalise", "inference", "build", "server_total", "network_and_framework"]
    print(f"\n{'Metric / Stage':26s} {'p50 ms':>10} {'p95 ms':>10} {'p99 ms':>10} {'Mean ms':>10} {'Sum ms':>10}")
    print("-" * 80)
    for k in keys:
        vals = np.array([row[k] for row in individual_rows if k in row])
        if vals.size:
            print(f"{k:26s} {np.percentile(vals, 50):10.3f} {np.percentile(vals, 95):10.3f} {np.percentile(vals, 99):10.3f} {np.mean(vals):10.3f} {np.sum(vals):10.3f}")

    print("\n" + "=" * 60)
    print("EXPERIMENT: 1 /predict_batch REQUEST (400 READINGS)")
    print("=" * 60)

    t_batch_start = time.perf_counter()
    batch_resp = session.post(f"{url}/predict_batch", json={"rows": readings}, timeout=30)
    t_batch_total = (time.perf_counter() - t_batch_start) * 1000.0
    assert batch_resp.status_code == 200, f"Batch request failed: {batch_resp.status_code}"
    batch_data = batch_resp.json()
    batch_preds = batch_data["days_left"]
    batch_timings = batch_data.get("timings_ms", {})

    print(f"Total wall-clock round-trip for 1 batch request: {t_batch_total:.2f} ms ({t_batch_total/1000:.3f} s)")
    print("\nBatch Server Stage Timings:")
    for k, v in batch_timings.items():
        print(f"  {k:20s}: {v:8.3f} ms")
    if "server_total" in batch_timings:
        net_framework = t_batch_total - batch_timings["server_total"]
        print(f"  {'network_and_framework':20s}: {net_framework:8.3f} ms")

    # Verify consistency
    assert len(batch_preds) == 400
    assert len(individual_preds) == 400
    diffs = [abs(b - p) for b, p in zip(batch_preds, individual_preds)]
    exact_matches = sum(1 for d in diffs if d == 0.0)
    max_diff = max(diffs)
    assert max_diff <= 0.01, f"Excessive prediction mismatch: {max_diff}"
    print(f"\nPrediction Consistency Check:")
    print(f"  Exact matches: {exact_matches}/400 ({exact_matches/400*100:.1f}%)")
    print(f"  Max absolute difference after round(v, 2): {max_diff:.2f} days (attributable to GEMM vs GEMV float summation rounding at boundary)")

    # Comparison summary
    speedup = t_single_total / t_batch_total
    time_saved = t_single_total - t_batch_total
    single_sum_inference = sum(row.get("inference", 0) for row in individual_rows)
    batch_inference = batch_timings.get("inference", 0)
    single_sum_network = sum(row.get("network_and_framework", 0) for row in individual_rows)
    batch_network = t_batch_total - batch_timings.get("server_total", 0)

    print("\n" + "=" * 60)
    print("COMPARATIVE EFFICIENCY ANALYSIS (400 Readings)")
    print("=" * 60)
    print(f"Wall-Clock Total:       400 Single: {t_single_total:8.2f} ms | 1 Batch: {t_batch_total:8.2f} ms | Speedup: {speedup:6.1f}x (Saved {time_saved:.1f} ms)")
    print(f"Per-Reading Latency:    400 Single: {t_single_total/400:8.3f} ms | 1 Batch: {t_batch_total/400:8.3f} ms | Reduction: {((t_single_total-t_batch_total)/t_single_total)*100:.1f}%")
    print(f"PyTorch Inference Time: 400 Single: {single_sum_inference:8.2f} ms | 1 Batch: {batch_inference:8.2f} ms | Speedup: {single_sum_inference/batch_inference:6.1f}x")
    print(f"Network & HTTP Overhead:400 Single: {single_sum_network:8.2f} ms | 1 Batch: {batch_network:8.2f} ms | Reduction: {((single_sum_network-batch_network)/single_sum_network)*100:.1f}%")


if __name__ == "__main__":
    main()
