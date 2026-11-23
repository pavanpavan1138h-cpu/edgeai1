"""
architecture_sweep_1.py -- Systematic architecture sweep for Team 3 (Phoenix).
Evaluates 10 neural network architectures (5 single-hidden-layer and 5 two-hidden-layer)
using the exact Experiment 1 data split, preprocessing, and training controls.

Strict controls:
- Dataset: data/team_03_train.csv
- Filter: battery_v < 3.0 | temp_c < -10.0 | wifi_rssi_dbm >= 0 (24 rows dropped)
- Split: rng = np.random.default_rng(0), 80/20 train/val (1581 train / 395 val)
- Normalization: training-derived mu & sd only
- Optimizer: Adam, lr=3e-3, batch_size=128, epochs=1500, loss=MSE
- Seed: torch.manual_seed(0) before each architecture initialization
- Preserves reference model/model.pt and model/config.json untouched!
"""
import os
import sys
import time
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.model import FEATURES, BatteryNet

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "sweep_1_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def compute_model_complexity(model):
    """Computes exact trainable parameters and MACs for BatteryNet."""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_macs = 0
    for m in model.net:
        if isinstance(m, nn.Linear):
            total_macs += m.in_features * m.out_features
    return total_params, total_macs

def load_and_preprocess_data():
    """Loads raw dataset, applies provisional glitch filters, and performs 80/20 split."""
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "team_03_train.csv")
    df = pd.read_csv(data_path)
    
    # Provisional glitch filter
    bad = (df.battery_v < 3.0) | (df.temp_c < -10.0) | (df.wifi_rssi_dbm >= 0)
    print(f"Data Loaded: {len(df)} total rows, {bad.sum()} glitches dropped -> {len(df) - bad.sum()} clean rows.")
    df_clean = df[~bad].reset_index(drop=True)
    
    # Exact 80/20 split from train.py
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
    
    return X_tr_norm, y_tr, X_val_norm, y_val, mu, sd

def run_sweep():
    X_tr, y_tr, X_val, y_val, mu, sd = load_and_preprocess_data()
    
    # 10 Architectures to evaluate
    architectures = [
        # Single Hidden Layer
        {"name": "1-Layer (4)", "hidden": (4,), "layers": 1, "widths": "4"},
        {"name": "1-Layer (8)", "hidden": (8,), "layers": 1, "widths": "8"},
        {"name": "1-Layer (16)", "hidden": (16,), "layers": 1, "widths": "16"},
        {"name": "1-Layer (32)", "hidden": (32,), "layers": 1, "widths": "32"},
        {"name": "1-Layer (64)", "hidden": (64,), "layers": 1, "widths": "64"},
        # Two Hidden Layers
        {"name": "2-Layer (4, 4)", "hidden": (4, 4), "layers": 2, "widths": "4, 4"},
        {"name": "2-Layer (8, 8)", "hidden": (8, 8), "layers": 2, "widths": "8, 8"},
        {"name": "2-Layer (16, 16)", "hidden": (16, 16), "layers": 2, "widths": "16, 16"},
        {"name": "2-Layer (32, 32)", "hidden": (32, 32), "layers": 2, "widths": "32, 32"},  # Reference
        {"name": "2-Layer (64, 64)", "hidden": (64, 64), "layers": 2, "widths": "64, 64"},
    ]
    
    Xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.float32)
    Xvt = torch.tensor(X_val, dtype=torch.float32)
    
    results = []
    histories = {}
    
    print("\n" + "="*80)
    print("STARTING ARCHITECTURE SWEEP 1 (10 MODELS)")
    print("="*80)
    
    for idx, arch in enumerate(architectures, 1):
        print(f"\n[{idx}/10] Training {arch['name']} -- hidden={arch['hidden']}...")
        
        # Set exact reproducible seed before each model initialization
        torch.manual_seed(0)
        np.random.seed(0)
        
        model = BatteryNet(arch["hidden"])
        params, macs = compute_model_complexity(model)
        fp32_bytes = params * 4
        int8_bytes = params * 1
        
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        epochs = 1500
        batch_size = 128
        
        history = {
            "epoch": [],
            "train_loss": [],
            "train_mae": [],
            "val_mae": [],
            "val_mse": []
        }
        
        start_time = time.time()
        best_val_mae = float("inf")
        best_epoch = -1
        
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
                
            # Evaluate every 10 epochs and at epoch 0 and 1499
            if ep % 10 == 0 or ep == epochs - 1:
                model.eval()
                with torch.no_grad():
                    tr_preds = model(Xt).numpy()
                    val_preds = model(Xvt).numpy()
                    
                    tr_mae = float(np.mean(np.abs(tr_preds - y_tr)))
                    tr_mse = float(np.mean((tr_preds - y_tr)**2))
                    v_mae = float(np.mean(np.abs(val_preds - y_val)))
                    v_mse = float(np.mean((val_preds - y_val)**2))
                    
                    history["epoch"].append(ep)
                    history["train_loss"].append(tr_mse)
                    history["train_mae"].append(tr_mae)
                    history["val_mae"].append(v_mae)
                    history["val_mse"].append(v_mse)
                    
                    if v_mae < best_val_mae:
                        best_val_mae = v_mae
                        best_epoch = ep
                        
            if ep in [0, 300, 600, 900, 1200, 1499]:
                print(f"    Epoch {ep:4d} | Train MAE: {tr_mae:.2f}d | Val MAE: {v_mae:.2f}d | Loss: {tr_mse:.2f}")
                
        duration = time.time() - start_time
        
        # Final metrics evaluation at Epoch 1499
        model.eval()
        with torch.no_grad():
            final_tr_preds = model(Xt).numpy()
            final_val_preds = model(Xvt).numpy()
            
            final_tr_mae = float(np.mean(np.abs(final_tr_preds - y_tr)))
            final_tr_mse = float(np.mean((final_tr_preds - y_tr)**2))
            final_val_mae = float(np.mean(np.abs(final_val_preds - y_val)))
            final_val_mse = float(np.mean((final_val_preds - y_val)**2))
            final_val_rmse = float(np.sqrt(final_val_mse))
            
        print(f"--> Done: Final Val MAE: {final_val_mae:.4f}d (Best: {best_val_mae:.4f}d @ ep {best_epoch}) | Duration: {duration:.2f}s")
        
        # Save experiment model weights separately from production
        arch_slug = arch['name'].lower().replace(" ", "_").replace("(", "").replace(")", "").replace(",", "_")
        weight_path = os.path.join(OUTPUT_DIR, f"weights_{arch_slug}.pt")
        torch.save(model.state_dict(), weight_path)
        
        # Assess convergence / stability
        instability = "None (Stable convergence)"
        if np.isnan(final_val_mae) or np.isinf(final_val_mae):
            instability = "Diverged (NaN/Inf)"
        elif final_val_mae > 30.0:
            instability = "Severe Underfitting"
            
        results.append({
            "architecture": arch["name"],
            "hidden_layers": arch["layers"],
            "layer_widths": arch["widths"],
            "trainable_params": params,
            "macs": macs,
            "fp32_memory_bytes": fp32_bytes,
            "int8_memory_bytes": int8_bytes,
            "final_train_mae": round(final_tr_mae, 4),
            "final_train_mse": round(final_tr_mse, 4),
            "final_val_mae": round(final_val_mae, 4),
            "final_val_mse": round(final_val_mse, 4),
            "final_val_rmse": round(final_val_rmse, 4),
            "best_val_mae": round(best_val_mae, 4),
            "best_epoch": best_epoch,
            "final_epoch": epochs - 1,
            "training_duration_sec": round(duration, 2),
            "convergence_notes": instability
        })
        
        histories[arch["name"]] = history

    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Compute Pareto Optimality
    # Model i dominates model j if (Val_MAE_i <= Val_MAE_j and Params_i <= Params_j and MACs_i <= MACs_j) with at least one strict
    pareto_flags = []
    for i in range(len(results_df)):
        dominated = False
        row_i = results_df.iloc[i]
        for j in range(len(results_df)):
            if i == j:
                continue
            row_j = results_df.iloc[j]
            # Does row_j dominate row_i?
            less_or_equal = (row_j["final_val_mae"] <= row_i["final_val_mae"]) and \
                            (row_j["trainable_params"] <= row_i["trainable_params"]) and \
                            (row_j["macs"] <= row_i["macs"])
            strictly_less = (row_j["final_val_mae"] < row_i["final_val_mae"]) or \
                            (row_j["trainable_params"] < row_i["trainable_params"]) or \
                            (row_j["macs"] < row_i["macs"])
            if less_or_equal and strictly_less:
                dominated = True
                break
        pareto_flags.append(not dominated)
        
    results_df["pareto_optimal"] = pareto_flags
    
    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, "architecture_sweep_1_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV Results: {csv_path}")
    
    # Save JSON Summary
    summary_path = os.path.join(OUTPUT_DIR, "sweep_1_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved JSON Summary: {summary_path}")
    
    # Generate Visualizations
    generate_plots(results_df, histories)
    
    return results_df

def generate_plots(df, histories):
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # Plot 1: Validation MAE vs Parameters
    plt.figure(figsize=(10, 6), dpi=300)
    for layers in [1, 2]:
        sub = df[df["hidden_layers"] == layers]
        label = "Single Hidden Layer" if layers == 1 else "Two Hidden Layers"
        marker = "o" if layers == 1 else "s"
        color = "#1f77b4" if layers == 1 else "#ff7f0e"
        plt.plot(sub["trainable_params"], sub["final_val_mae"], linestyle="--", marker=marker,
                 color=color, label=label, markersize=8, linewidth=1.5)
        for _, row in sub.iterrows():
            plt.annotate(row["architecture"], (row["trainable_params"], row["final_val_mae"]),
                         textcoords="offset points", xytext=(5, 6), fontsize=9)
            
    # Highlight Pareto frontier
    pareto_df = df[df["pareto_optimal"]].sort_values("trainable_params")
    plt.plot(pareto_df["trainable_params"], pareto_df["final_val_mae"], color="#2ca02c",
             linestyle="-", linewidth=2.5, label="Pareto Frontier (Non-dominated)")
    plt.scatter(pareto_df["trainable_params"], pareto_df["final_val_mae"], color="#2ca02c",
                s=120, zorder=5, edgecolors="black", linewidths=1.5)
    
    plt.axhline(20.54, color="gray", linestyle=":", label="Linear Regression Baseline (20.54d)")
    plt.xlabel("Trainable Parameter Count", fontsize=12, fontweight="bold")
    plt.ylabel("Validation MAE (days)", fontsize=12, fontweight="bold")
    plt.title("Architecture Sweep 1: Validation MAE vs. Parameter Complexity", fontsize=14, fontweight="bold", pad=12)
    plt.legend(frameon=True, fontsize=10, loc="upper right")
    plt.tight_layout()
    p1_path = os.path.join(OUTPUT_DIR, "val_mae_vs_params.png")
    plt.savefig(p1_path)
    plt.close()
    print(f"Saved: {p1_path}")
    
    # Plot 2: Validation MAE vs MACs
    plt.figure(figsize=(10, 6), dpi=300)
    for layers in [1, 2]:
        sub = df[df["hidden_layers"] == layers]
        label = "Single Hidden Layer" if layers == 1 else "Two Hidden Layers"
        marker = "o" if layers == 1 else "s"
        color = "#1f77b4" if layers == 1 else "#ff7f0e"
        plt.plot(sub["macs"], sub["final_val_mae"], linestyle="--", marker=marker,
                 color=color, label=label, markersize=8, linewidth=1.5)
        for _, row in sub.iterrows():
            plt.annotate(row["architecture"], (row["macs"], row["final_val_mae"]),
                         textcoords="offset points", xytext=(5, 6), fontsize=9)
            
    pareto_macs = df[df["pareto_optimal"]].sort_values("macs")
    plt.plot(pareto_macs["macs"], pareto_macs["final_val_mae"], color="#2ca02c",
             linestyle="-", linewidth=2.5, label="Pareto Frontier (Non-dominated)")
    plt.scatter(pareto_macs["macs"], pareto_macs["final_val_mae"], color="#2ca02c",
                s=120, zorder=5, edgecolors="black", linewidths=1.5)
                
    plt.axhline(20.54, color="gray", linestyle=":", label="Linear Regression Baseline (20.54d)")
    plt.xlabel("Inference Multiply-Accumulates (MACs)", fontsize=12, fontweight="bold")
    plt.ylabel("Validation MAE (days)", fontsize=12, fontweight="bold")
    plt.title("Architecture Sweep 1: Validation MAE vs. Computational Operations (MACs)", fontsize=14, fontweight="bold", pad=12)
    plt.legend(frameon=True, fontsize=10, loc="upper right")
    plt.tight_layout()
    p2_path = os.path.join(OUTPUT_DIR, "val_mae_vs_macs.png")
    plt.savefig(p2_path)
    plt.close()
    print(f"Saved: {p2_path}")
    
    # Plot 3: Training and Validation Convergence Curves (10 subplots or 2 panels)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300, sharey=True)
    
    for name, hist in histories.items():
        if "1-Layer" in name:
            axes[0].plot(hist["epoch"], hist["val_mae"], label=name, linewidth=1.5)
        else:
            axes[1].plot(hist["epoch"], hist["val_mae"], label=name, linewidth=1.5)
            
    axes[0].set_title("Single Hidden Layer Convergence", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Validation MAE (days)", fontsize=11, fontweight="bold")
    axes[0].set_ylim(0, 25)
    axes[0].axhline(20.54, color="gray", linestyle=":", label="Linear Reg (20.54d)")
    axes[0].legend(frameon=True, fontsize=9)
    
    axes[1].set_title("Two Hidden Layers Convergence", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=11, fontweight="bold")
    axes[1].set_ylim(0, 25)
    axes[1].axhline(20.54, color="gray", linestyle=":", label="Linear Reg (20.54d)")
    axes[1].legend(frameon=True, fontsize=9)
    
    plt.suptitle("Architecture Sweep 1: Validation MAE Trajectory Across 1,500 Epochs", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    p3_path = os.path.join(OUTPUT_DIR, "convergence_curves.png")
    plt.savefig(p3_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p3_path}")
    
    # Plot 4: Pareto Frontier Multi-Metric Overview
    fig, ax = plt.subplots(figsize=(11, 7), dpi=300)
    scatter = ax.scatter(df["trainable_params"], df["final_val_mae"], 
                         s=df["macs"] * 0.4 + 40, c=df["hidden_layers"], 
                         cmap="coolwarm", alpha=0.85, edgecolors="black", linewidths=1.5)
    
    # Annotate Pareto points
    for _, row in df.iterrows():
        is_pareto = row["pareto_optimal"]
        prefix = "★ " if is_pareto else ""
        fontweight = "bold" if is_pareto else "normal"
        color = "darkgreen" if is_pareto else "black"
        ax.annotate(f"{prefix}{row['architecture']}\n({row['final_val_mae']:.2f}d, {row['trainable_params']}p)",
                    (row["trainable_params"], row["final_val_mae"]),
                    textcoords="offset points", xytext=(0, 10 if is_pareto else -16),
                    ha="center", fontsize=8.5, fontweight=fontweight, color=color)
        
    cbar = plt.colorbar(scatter, ax=ax, ticks=[1, 2])
    cbar.set_label("Number of Hidden Layers", fontsize=11, fontweight="bold")
    cbar.ax.set_yticklabels(["1 Hidden Layer", "2 Hidden Layers"])
    
    ax.set_xlabel("Trainable Parameters", fontsize=12, fontweight="bold")
    ax.set_ylabel("Final Validation MAE (days)", fontsize=12, fontweight="bold")
    ax.set_title("Pareto Frontier: Accuracy vs. Parameter Footprint (Bubble Size = MACs)", fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    p4_path = os.path.join(OUTPUT_DIR, "pareto_bubble_chart.png")
    plt.savefig(p4_path)
    plt.close()
    print(f"Saved: {p4_path}")

if __name__ == "__main__":
    df_res = run_sweep()
    print("\n" + "="*80)
    print("SWEEP 1 EXECUTION COMPLETE -- RESULTS PREVIEW:")
    print("="*80)
    print(df_res[["architecture", "trainable_params", "macs", "final_val_mae", "final_val_mse", "best_val_mae", "best_epoch", "pareto_optimal"]].to_string())

