"""
contact_multiprobe_pool.npz(phi=-120,-60,0,60,120 5곳 전부 계산해둔 것)에서 프로브 조합을
여러 개 골라 각각 학습시켜서 비교. 조합마다 새로 물리계산 안 하고 이미 계산된 걸 인덱싱만
해서 쓰므로 빠름.
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, '..', '..', 'data', 'contact_scenarios')

d = np.load(os.path.join(DATA_DIR, 'contact_multiprobe_pool_robust.npz'))
B_free_pool, B_load_pool, y_all, y_cols = d['B_free'], d['B_load'], d['y'], list(d['y_columns'])
phi_pool = list(d['phi_probes'])
ycol = {c: i for i, c in enumerate(y_cols)}
print(f"풀 데이터: n={len(B_free_pool)}, 후보 phi={phi_pool}")

TARGETS = ['s', 'F_mag', 'Fx', 'Fy']
y = y_all[:, [ycol[t] for t in TARGETS]]

# search_optimal_probe_angles_multiLM.py가 찾은 L_M 전체 robust 최적조합
# (phi_pool 인덱스: 0=-150, 1=-90, 2=-30, 3=60)
COMBOS = {
    "2probe_robust (-150,-30)": [0, 2],
    "3probe_robust (-90,-30,60)": [1, 2, 3],
    "4probe_all (-150,-90,-30,60)": [0, 1, 2, 3],
}


class MultiProbeEstimator(nn.Module):
    def __init__(self, n_probes, n_targets):
        super().__init__()
        self.n_probes = n_probes
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Flatten(), nn.Linear(32 * 5 * 5, 64), nn.ReLU(),
        )
        self.regressor = nn.Sequential(
            nn.Linear(64 * n_probes, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(128, n_targets)
        )

    def forward(self, x):
        embeds = [self.encoder(x[:, p]) for p in range(self.n_probes)]
        return self.regressor(torch.cat(embeds, dim=1))


def train_and_eval(probe_idx, epochs=120, seed=0):
    Bf = B_free_pool[:, probe_idx]
    Bl = B_load_pool[:, probe_idx]
    delta = Bl - Bf
    n_probes = len(probe_idx)
    X = delta.reshape(len(delta), n_probes, 5, 5, 3).transpose(0, 1, 4, 2, 3)

    X_mean, X_std = X.mean(), X.std()
    X_norm = (X - X_mean) / X_std
    y_mean, y_std = y.mean(axis=0), y.std(axis=0)
    y_norm = (y - y_mean) / y_std

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X_norm))
    split = int(0.8 * len(idx))
    train_idx, val_idx = idx[:split], idx[split:]

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_norm[train_idx]).float(), torch.tensor(y_norm[train_idx]).float()),
        batch_size=64, shuffle=True)
    val_X = torch.tensor(X_norm[val_idx]).float()
    val_y_t = torch.tensor(y_norm[val_idx]).float()

    model = MultiProbeEstimator(n_probes, len(TARGETS))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        for bx, by in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        pred_norm = model(val_X).numpy()
    true_norm = val_y_t.numpy()
    pred = pred_norm * y_std + y_mean
    true = true_norm * y_std + y_mean

    r2 = {}
    mae = {}
    for i, t in enumerate(TARGETS):
        r2[t] = 1 - np.sum((pred[:, i] - true[:, i]) ** 2) / np.sum((true[:, i] - true[:, i].mean()) ** 2)
        mae[t] = np.mean(np.abs(pred[:, i] - true[:, i]))
    return r2, mae


results = {}
for name, probe_idx in COMBOS.items():
    t0 = time.time()
    phis = [phi_pool[i] for i in probe_idx]
    print(f"\n=== {name}  (phi={phis}) ===", flush=True)
    r2, mae = train_and_eval(probe_idx)
    dt = time.time() - t0
    results[name] = {"phis": phis, "r2": r2, "mae": mae, "wall_time_s": dt}
    print(f"  s: R2={r2['s']:.3f} MAE={mae['s']:.2f}mm | F_mag: R2={r2['F_mag']:.3f} MAE={mae['F_mag']*1000:.3f}mN | "
          f"Fx: R2={r2['Fx']:.3f} | Fy: R2={r2['Fy']:.3f}  ({dt:.0f}s)", flush=True)

print("\n\n=== 전체 비교 요약 ===")
print(f"{'조합':<28} {'s R2':>7} {'F_mag R2':>9} {'Fx R2':>7} {'Fy R2':>7} {'평균 R2':>8}")
import json
for name, r in results.items():
    avg_r2 = np.mean(list(r["r2"].values()))
    r["avg_r2"] = float(avg_r2)
    print(f"{name:<28} {r['r2']['s']:7.3f} {r['r2']['F_mag']:9.3f} {r['r2']['Fx']:7.3f} {r['r2']['Fy']:7.3f} {avg_r2:8.3f}")

out_path = os.path.join(DATA_DIR, "probe_combo_comparison.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\n저장: {out_path}")
