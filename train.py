"""
train.py -- STARTER. Clean the data, fit baselines, train the network, save it for the server.

    python train.py --team 7 --hidden 32,32
Writes model/model.pt and model/config.json, which app/main.py loads.

Look for TODO. The training loop is given; the decisions are yours.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from app.model import FEATURES, BatteryNet

ap = argparse.ArgumentParser()
ap.add_argument("--team", type=int, required=True)
ap.add_argument("--hidden", default="32,32")
ap.add_argument("--epochs", type=int, default=1500)
a = ap.parse_args()
hidden = tuple(int(h) for h in a.hidden.split(","))

df = pd.read_csv(f"data/team_{a.team:02d}_train.csv")
# Mission 1: Sensor glitch filtering rule
# Drops: 1) voltage dropouts (battery_v < 3.0V), 2) temp bus disconnects (temp_c < -10°C), 3) wifi unconnected default (wifi_rssi_dbm >= 0)
bad = (df.battery_v < 3.0) | (df.temp_c < -10.0) | (df.wifi_rssi_dbm >= 0)
print(f"{len(df)} rows, {bad.sum()} glitch rows dropped ({len(df) - bad.sum()} clean rows remaining)")
df = df[~bad]

rng = np.random.default_rng(0)
idx = rng.permutation(len(df)); n_val = len(df) // 5
val, tr = df.iloc[idx[:n_val]], df.iloc[idx[n_val:]]
X, y = tr[FEATURES].to_numpy(np.float32), tr.days_left.to_numpy(np.float32)
Xv, yv = val[FEATURES].to_numpy(np.float32), val.days_left.to_numpy(np.float32)
# Mission 2: Normalisation (mu and sd computed from the TRAINING rows only to prevent data leakage)
mu, sd = X.mean(axis=0), X.std(axis=0)
sd[sd == 0] = 1.0

mae = lambda p, t: float(np.mean(np.abs(p - t)))
print(f"baseline (predict the mean): {mae(np.full_like(yv, y.mean()), yv):.2f} days")

# Mission 1: Baseline 2: Linear regression with np.linalg.lstsq on [X, 1]
A_tr = np.hstack([X, np.ones((len(X), 1), dtype=np.float32)])
A_val = np.hstack([Xv, np.ones((len(Xv), 1), dtype=np.float32)])
w_lin, _, _, _ = np.linalg.lstsq(A_tr, y, rcond=None)
print(f"baseline (linear regression): {mae(A_val @ w_lin, yv):.2f} days")

torch.manual_seed(0)
net = BatteryNet(hidden)
opt = torch.optim.Adam(net.parameters(), lr=3e-3)
Xt, yt = torch.tensor((X - mu) / sd), torch.tensor(y)
Xvt = torch.tensor((Xv - mu) / sd)
for ep in range(a.epochs):
    perm = torch.randperm(len(Xt))
    for i in range(0, len(Xt), 128):
        b = perm[i:i + 128]
        loss = torch.nn.functional.mse_loss(net(Xt[b]), yt[b])
        opt.zero_grad(); loss.backward(); opt.step()
    if ep % 300 == 0 or ep == a.epochs - 1:
        with torch.no_grad():
            print(f"  epoch {ep:4d}  val MAE {mae(net(Xvt).numpy(), yv):.2f} days")
with torch.no_grad():
    val_mae = mae(net(Xvt).numpy(), yv)
print(f"network {hidden}:            {val_mae:.2f} days")

os.makedirs("model", exist_ok=True)
torch.save(net.state_dict(), "model/model.pt")
json.dump(dict(team=a.team, hidden=list(hidden), mu=mu.tolist(), sd=sd.tolist(),
               features=FEATURES, val_mae=round(val_mae, 2)), open("model/config.json", "w"), indent=2)
print("saved model/model.pt and model/config.json")
