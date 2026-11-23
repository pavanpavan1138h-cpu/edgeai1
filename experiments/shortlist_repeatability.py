"""
shortlist_repeatability.py -- Repeatability and multi-seed validation for shortlisted models.
Evaluates the three shortlisted architectures from EXP-005 across 5 training random seeds (0, 1, 2, 3, 4).
Varies training randomness (initialization, data shuffle) while keeping dataset, cleaning,
partitioning (seed 0), and preprocessing strictly fixed.

Shortlisted Architectures:
1. 2-Layer (8, 8)   : 129 params, 112 MACs (Ultra-lightweight TinyML candidate)
2. 2-Layer (16, 16) : 385 params, 352 MACs (Balanced candidate)
3. 2-Layer (32, 32) : 1,281 params, 1,216 MACs (Reference accuracy champion)
"""
import os
import sys
import time
import json
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.model import FEATURES, BatteryNet

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "shortlist_repeatability")
SWEEP_1_DIR = os.path.join(os.path.dirname(__file__), "sweep_1_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def compute_model_complexity(model):
    """Computes exact trainable parameters and MACs."""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_macs = sum(m.in_features * m.out_features for m in model.net if isinstance(m, nn.Linear))
    return total_params, total_macs

def load_data_and_verify_split():
    """Loads dataset, filters glitches, and generates deterministic seed-0 split."""
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "team_03_train.csv")
    df = pd.read_csv(data_path)
    
    # Provisional glitch filter
    bad = (df.battery_v < 3.0) | (df.temp_c < -10.0) | (df.wifi_rssi_dbm >= 0)
    df_clean = df[~bad].reset_index(drop=True)
    assert len(df_clean) == 1976, f"Expected 1976 clean rows, got {len(df_clean)}"
    
    # Exact 80/20 split from train.py using default_rng(0)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(df_clean))
    n_val = len(df_clean) // 5  # exactly 395
    val_df, tr_df = df_clean.iloc[idx[:n_val]], df_clean.iloc[idx[n_val:]]
    
    X_tr = tr_df[FEATURES].to_numpy(np.float32)
    y_tr = tr_df.days_left.to_numpy(np.float32)
    X_val = val_df[FEATURES].to_numpy(np.float32)
    y_val = val_df.days_left.to_numpy(np.float32)
    
    # Training-only standardization
    mu, sd = X_tr.mean(axis=0), X_tr.std(axis=0)
    sd[sd == 0] = 1.0
    
    X_tr_norm = (X_tr - mu) / sd
    X_val_norm = (X_val - mu) / sd
    
    # Low-battery mask (actual days_left between 0 and 10 days)
    low_mask = (y_val >= 0.0) & (y_val <= 10.0)
    print(f"Data ready: {len(X_tr)} train rows, {len(X_val)} val rows, {low_mask.sum()} low-battery (0-10d) val samples.")
    assert low_mask.sum() == 85, f"Expected 85 low-battery validation samples, got {low_mask.sum()}"
    
    return X_tr_norm, y_tr, X_val_norm, y_val, low_mask

def get_seed0_results(arch_name, hidden, slug, X_tr, y_tr, X_val, y_val, low_mask):
    """Loads and verifies seed 0 results directly from Sweep 1 artifacts."""
    weight_path = os.path.join(SWEEP_1_DIR, slug)
    net = BatteryNet(hidden)
    net.load_state_dict(torch.load(weight_path))
    net.eval()
    
    with torch.no_grad():
        tr_preds = net(torch.tensor(X_tr)).numpy()
        val_preds = net(torch.tensor(X_val)).numpy()
        
    final_tr_mae = float(np.mean(np.abs(tr_preds - y_tr)))
    final_tr_mse = float(np.mean((tr_preds - y_tr)**2))
    final_val_mae = float(np.mean(np.abs(val_preds - y_val)))
    final_val_mse = float(np.mean((val_preds - y_val)**2))
    final_val_rmse = float(np.sqrt(final_val_mse))
    med_ae = float(np.median(np.abs(val_preds - y_val)))
    mean_signed_error = float(np.mean(val_preds - y_val))
    low_bat_mae = float(np.mean(np.abs(val_preds[low_mask] - y_val[low_mask])))
    
    # Sweep 1 recorded metadata
    sweep1_csv = pd.read_csv(os.path.join(SWEEP_1_DIR, "architecture_sweep_1_results.csv"))
    arch_row = sweep1_csv[sweep1_csv["architecture"] == arch_name].iloc[0]
    
    # Save a reference copy of weights in isolated repeatability directory
    dest_path = os.path.join(OUTPUT_DIR, f"weights_{slug.replace('.pt', '')}_seed0.pt")
    torch.save(net.state_dict(), dest_path)
    
    params, macs = compute_model_complexity(net)
    
    return {
        "architecture": arch_name,
        "seed": 0,
        "trainable_params": params,
        "macs": macs,
        "final_train_mae": round(final_tr_mae, 4),
        "final_train_mse": round(final_tr_mse, 4),
        "final_val_mae": round(final_val_mae, 4),
        "final_val_mse": round(final_val_mse, 4),
        "final_val_rmse": round(final_val_rmse, 4),
        "median_abs_error": round(med_ae, 4),
        "val_mean_signed_error": round(mean_signed_error, 4),
        "low_battery_val_mae": round(low_bat_mae, 4),
        "low_battery_samples": int(low_mask.sum()),
        "training_duration_sec": round(float(arch_row["training_duration_sec"]), 2),
        "final_epoch": 1499,
        "best_val_mae": round(float(arch_row["best_val_mae"]), 4),
        "best_epoch": int(arch_row["best_epoch"]),
        "convergence_notes": "Stable convergence (Reused from EXP-005)"
    }

def train_and_evaluate(arch_name, hidden, seed, X_tr, y_tr, X_val, y_val, low_mask):
    """Trains a model with a specified seed and records full metrics."""
    # Seed all relevant random number generators
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    Xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.float32)
    Xvt = torch.tensor(X_val, dtype=torch.float32)
    
    model = BatteryNet(hidden)
    params, macs = compute_model_complexity(model)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    epochs = 1500
    batch_size = 128
    
    best_val_mae = float("inf")
    best_epoch = -1
    start_time = time.time()
    
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), batch_size):
            b = perm[i:i + batch_size]
            pred = model(Xt[b])
            loss = nn.functional.mse_loss(pred, yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            
        # Check validation performance every 10 epochs
        if ep % 10 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                v_preds = model(Xvt).numpy()
                v_mae = float(np.mean(np.abs(v_preds - y_val)))
                if v_mae < best_val_mae:
                    best_val_mae = v_mae
                    best_epoch = ep
                    
    duration = time.time() - start_time
    
    # Final evaluation at Epoch 1499
    model.eval()
    with torch.no_grad():
        final_tr_preds = model(Xt).numpy()
        final_val_preds = model(Xvt).numpy()
        
    final_tr_mae = float(np.mean(np.abs(final_tr_preds - y_tr)))
    final_tr_mse = float(np.mean((final_tr_preds - y_tr)**2))
    final_val_mae = float(np.mean(np.abs(final_val_preds - y_val)))
    final_val_mse = float(np.mean((final_val_preds - y_val)**2))
    final_val_rmse = float(np.sqrt(final_val_mse))
    med_ae = float(np.median(np.abs(final_val_preds - y_val)))
    mean_signed_error = float(np.mean(final_val_preds - y_val))
    low_bat_mae = float(np.mean(np.abs(final_val_preds[low_mask] - y_val[low_mask])))
    
    # Save weights inside isolated experiment directory
    slug = arch_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace(",", "_")
    weight_path = os.path.join(OUTPUT_DIR, f"weights_{slug}_seed{seed}.pt")
    torch.save(model.state_dict(), weight_path)
    
    conv_note = "Stable convergence"
    if np.isnan(final_val_mae) or np.isinf(final_val_mae):
        conv_note = "Diverged"
    elif final_val_mae > 25.0:
        conv_note = "Severe Underfitting"
        
    return {
        "architecture": arch_name,
        "seed": seed,
        "trainable_params": params,
        "macs": macs,
        "final_train_mae": round(final_tr_mae, 4),
        "final_train_mse": round(final_tr_mse, 4),
        "final_val_mae": round(final_val_mae, 4),
        "final_val_mse": round(final_val_mse, 4),
        "final_val_rmse": round(final_val_rmse, 4),
        "median_abs_error": round(med_ae, 4),
        "val_mean_signed_error": round(mean_signed_error, 4),
        "low_battery_val_mae": round(low_bat_mae, 4),
        "low_battery_samples": int(low_mask.sum()),
        "training_duration_sec": round(duration, 2),
        "final_epoch": epochs - 1,
        "best_val_mae": round(best_val_mae, 4),
        "best_epoch": best_epoch,
        "convergence_notes": conv_note
    }

def run_experiment():
    X_tr, y_tr, X_val, y_val, low_mask = load_data_and_verify_split()
    
    shortlist = [
        {"name": "2-Layer (8, 8)", "hidden": (8, 8), "slug": "weights_2-layer_8__8.pt"},
        {"name": "2-Layer (16, 16)", "hidden": (16, 16), "slug": "weights_2-layer_16__16.pt"},
        {"name": "2-Layer (32, 32)", "hidden": (32, 32), "slug": "weights_2-layer_32__32.pt"}
    ]
    
    runs = []
    
    print("\n" + "="*80)
    print("PHASE 7.2: ARCHITECTURE REPEATABILITY (3 ARCHITECTURES x 5 SEEDS = 15 RUNS)")
    print("="*80)
    
    for arch in shortlist:
        print(f"\nEvaluating Architecture: {arch['name']} (hidden={arch['hidden']})...")
        
        # 1. Ingest verified Seed 0 result
        print(f"  [Seed 0] Ingesting verified EXP-005 baseline result...")
        res0 = get_seed0_results(arch["name"], arch["hidden"], arch["slug"], X_tr, y_tr, X_val, y_val, low_mask)
        runs.append(res0)
        print(f"  --> Seed 0: Val MAE = {res0['final_val_mae']:.4f}d | Low-Bat MAE = {res0['low_battery_val_mae']:.4f}d | MedAE = {res0['median_abs_error']:.4f}d")
        
        # 2. Train remaining seeds 1, 2, 3, 4
        for seed in [1, 2, 3, 4]:
            print(f"  [Seed {seed}] Training 1,500 epochs (torch.manual_seed({seed}))...")
            res = train_and_evaluate(arch["name"], arch["hidden"], seed, X_tr, y_tr, X_val, y_val, low_mask)
            runs.append(res)
            print(f"  --> Seed {seed}: Val MAE = {res['final_val_mae']:.4f}d | Low-Bat MAE = {res['low_battery_val_mae']:.4f}d | MedAE = {res['median_abs_error']:.4f}d | Time = {res['training_duration_sec']:.2f}s")
            
    runs_df = pd.DataFrame(runs)
    
    # Save per-run CSV
    runs_csv_path = os.path.join(OUTPUT_DIR, "repeatability_runs.csv")
    runs_df.to_csv(runs_csv_path, index=False)
    print(f"\nSaved Per-Run CSV: {runs_csv_path}")
    
    # Calculate Aggregate Metrics
    summary = []
    for arch in shortlist:
        sub = runs_df[runs_df["architecture"] == arch["name"]]
        val_maes = sub["final_val_mae"].values
        low_maes = sub["low_battery_val_mae"].values
        
        summary.append({
            "architecture": arch["name"],
            "hidden_layers": 2,
            "trainable_params": int(sub["trainable_params"].iloc[0]),
            "macs": int(sub["macs"].iloc[0]),
            "seeds_evaluated": len(sub),
            "mean_val_mae": round(float(np.mean(val_maes)), 4),
            "std_val_mae": round(float(np.std(val_maes, ddof=1)), 4),
            "min_val_mae": round(float(np.min(val_maes)), 4),
            "max_val_mae": round(float(np.max(val_maes)), 4),
            "val_mae_range": round(float(np.max(val_maes) - np.min(val_maes)), 4),
            "mean_val_mse": round(float(np.mean(sub["final_val_mse"])), 4),
            "mean_val_rmse": round(float(np.mean(sub["final_val_rmse"])), 4),
            "mean_low_battery_mae": round(float(np.mean(low_maes)), 4),
            "std_low_battery_mae": round(float(np.std(low_maes, ddof=1)), 4),
            "mean_train_mae": round(float(np.mean(sub["final_train_mae"])), 4),
            "mean_signed_error": round(float(np.mean(sub["val_mean_signed_error"])), 4),
            "mean_duration_sec": round(float(np.mean(sub["training_duration_sec"])), 2)
        })
        
    summary_df = pd.DataFrame(summary)
    
    # Save aggregate summary CSV & JSON
    summary_csv_path = os.path.join(OUTPUT_DIR, "repeatability_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Saved Aggregate CSV: {summary_csv_path}")
    
    summary_json_path = os.path.join(OUTPUT_DIR, "repeatability_summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved Summary JSON: {summary_json_path}")
    
    # Generate Visualizations
    generate_plots(runs_df, summary_df)
    
    return runs_df, summary_df

def generate_plots(runs_df, summary_df):
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    palette = {"2-Layer (8, 8)": "#2ca02c", "2-Layer (16, 16)": "#1f77b4", "2-Layer (32, 32)": "#ff7f0e"}
    
    # Plot 1: Validation MAE by architecture and seed (Jitter/Strip + Mean + Error Bar)
    plt.figure(figsize=(9, 6), dpi=300)
    for i, row in summary_df.iterrows():
        arch = row["architecture"]
        sub = runs_df[runs_df["architecture"] == arch]
        x_base = i
        
        # Individual seed scatter points
        seeds = sub["seed"].values
        maes = sub["final_val_mae"].values
        plt.scatter([x_base] * len(maes), maes, color=palette[arch], s=80, alpha=0.7, edgecolors="black", zorder=4)
        for s, m in zip(seeds, maes):
            plt.annotate(f"s{s}: {m:.2f}d", (x_base, m), textcoords="offset points", xytext=(12, -3), fontsize=8.5)
            
        # Error bar: Mean ± 1 Std Dev
        plt.errorbar(x_base, row["mean_val_mae"], yerr=row["std_val_mae"], fmt="D", color="black",
                     ecolor="black", elinewidth=2, capsize=8, capthick=2, markersize=8, zorder=5,
                     label=f"{arch} Mean ({row['mean_val_mae']:.2f}d ± {row['std_val_mae']:.2f}d)")
                     
    plt.xticks(range(len(summary_df)), summary_df["architecture"], fontsize=11, fontweight="bold")
    plt.ylabel("Validation MAE (days)", fontsize=12, fontweight="bold")
    plt.title("Repeatability: Validation MAE Distribution Across 5 Training Seeds (0–4)", fontsize=13, fontweight="bold", pad=12)
    plt.ylim(3.5, 6.0)
    plt.legend(frameon=True, fontsize=9.5, loc="upper right")
    plt.tight_layout()
    p1_path = os.path.join(OUTPUT_DIR, "val_mae_per_seed.png")
    plt.savefig(p1_path)
    plt.close()
    print(f"Saved: {p1_path}")
    
    # Plot 2: Architecture comparison bar chart with variability bars
    plt.figure(figsize=(8, 5.5), dpi=300)
    bars = plt.bar(summary_df["architecture"], summary_df["mean_val_mae"],
                   yerr=summary_df["std_val_mae"], capsize=8,
                   color=[palette[a] for a in summary_df["architecture"]],
                   alpha=0.85, edgecolor="black", linewidth=1.2)
                   
    for bar, (_, row) in zip(bars, summary_df.iterrows()):
        y_val = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, y_val / 2,
                 f"Mean: {row['mean_val_mae']:.2f}d\n(±{row['std_val_mae']:.2f}d)\n[{row['min_val_mae']:.2f}d – {row['max_val_mae']:.2f}d]",
                 ha='center', va='center', color='white', fontsize=10, fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.6))
                 
    plt.ylabel("Mean Validation MAE (days)", fontsize=12, fontweight="bold")
    plt.title("Shortlist Comparison: Mean Validation MAE Across 5 Training Seeds", fontsize=13, fontweight="bold", pad=12)
    plt.ylim(0, 6.5)
    plt.tight_layout()
    p2_path = os.path.join(OUTPUT_DIR, "architecture_comparison_bars.png")
    plt.savefig(p2_path)
    plt.close()
    print(f"Saved: {p2_path}")
    
    # Plot 3: Mean MAE vs Parameter Count with Error Bars
    plt.figure(figsize=(9, 6), dpi=300)
    plt.errorbar(summary_df["trainable_params"], summary_df["mean_val_mae"],
                 yerr=summary_df["std_val_mae"], fmt="o-", color="#1f77b4",
                 ecolor="red", elinewidth=1.8, capsize=6, markersize=8, linewidth=1.8, zorder=4)
                 
    for _, row in summary_df.iterrows():
        plt.annotate(f"{row['architecture']}\n{row['mean_val_mae']:.2f}d ± {row['std_val_mae']:.2f}d\n({row['trainable_params']} params)",
                     (row["trainable_params"], row["mean_val_mae"]),
                     textcoords="offset points", xytext=(0, 15), ha="center", fontsize=9.5, fontweight="bold")
                     
    plt.xlabel("Trainable Parameter Count", fontsize=12, fontweight="bold")
    plt.ylabel("Mean Validation MAE (days)", fontsize=12, fontweight="bold")
    plt.title("Shortlist Pareto Trade-Off: Mean Validation MAE vs. Parameter Count", fontsize=13, fontweight="bold", pad=12)
    plt.ylim(3.5, 6.0)
    plt.tight_layout()
    p3_path = os.path.join(OUTPUT_DIR, "mean_mae_vs_params.png")
    plt.savefig(p3_path)
    plt.close()
    print(f"Saved: {p3_path}")
    
    # Plot 4: Low-Battery MAE comparison across architectures and seeds
    plt.figure(figsize=(9, 6), dpi=300)
    for i, row in summary_df.iterrows():
        arch = row["architecture"]
        sub = runs_df[runs_df["architecture"] == arch]
        x_base = i
        
        seeds = sub["seed"].values
        low_maes = sub["low_battery_val_mae"].values
        plt.scatter([x_base] * len(low_maes), low_maes, color=palette[arch], s=80, alpha=0.7, edgecolors="black", zorder=4)
        for s, m in zip(seeds, low_maes):
            plt.annotate(f"s{s}: {m:.2f}d", (x_base, m), textcoords="offset points", xytext=(12, -3), fontsize=8.5)
            
        plt.errorbar(x_base, row["mean_low_battery_mae"], yerr=row["std_low_battery_mae"], fmt="s", color="black",
                     ecolor="black", elinewidth=2, capsize=8, capthick=2, markersize=8, zorder=5,
                     label=f"{arch} Mean ({row['mean_low_battery_mae']:.2f}d ± {row['std_low_battery_mae']:.2f}d)")
                     
    plt.xticks(range(len(summary_df)), summary_df["architecture"], fontsize=11, fontweight="bold")
    plt.ylabel("Low-Battery Validation MAE (days)", fontsize=12, fontweight="bold")
    plt.title("Low-Battery Regime (0–10 Days Lifespan, N=85): MAE Across 5 Seeds", fontsize=13, fontweight="bold", pad=12)
    plt.ylim(0.8, 2.5)
    plt.legend(frameon=True, fontsize=9.5, loc="upper right")
    plt.tight_layout()
    p4_path = os.path.join(OUTPUT_DIR, "low_battery_mae_comparison.png")
    plt.savefig(p4_path)
    plt.close()
    print(f"Saved: {p4_path}")

if __name__ == "__main__":
    runs_df, summary_df = run_experiment()
    print("\n" + "="*80)
    print("PHASE 7.2 EXECUTION COMPLETE -- AGGREGATE SUMMARY:")
    print("="*80)
    print(summary_df[["architecture", "trainable_params", "macs", "mean_val_mae", "std_val_mae", "min_val_mae", "max_val_mae", "mean_low_battery_mae", "std_low_battery_mae"]].to_string())

