"""2026-08-26: _diag_lm_bfield_distinguishability.py로 B-field 자체는 L_M=0~12.5mm를
물리적으로 잘 구분한다는 게 확인됐음(가설 반박) - 그럼 대체모델(서로게이트)의 mom_ux/uy
예측이 L_M=0 근처에서 유독 나쁜지(경계/외삽 효과) 확인. _diag_surrogate_only.py와 동일한
해시기반 홀드아웃 + 앙상블 방식이지만 mom_* 타겟을 포함하고 L_M 구간별로 residual을 쪼갬."""
import hashlib
import json
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score

HERE = os.path.dirname(os.path.abspath(__file__))
FEA_DATA_DIR = os.path.join(HERE, "..", "..", "..", "data", "contact_scenarios", "fea")

FEATURES = ["L_M_mm", "phi_deg", "beta_deg", "contact_s_mm", "push_depth_mm"]
TARGETS = ["mom_ux_avg_mm", "mom_uy_avg_mm"]
DEFAULTS = {"L_M_mm": 50.0, "phi_deg": 60.0, "beta_deg": 0.0}

all_rows = []
for r in json.load(open(os.path.join(FEA_DATA_DIR, "fea_lm_phi_pos_matv2_all.json"), encoding="utf-8")):
    row = dict(DEFAULTS)
    row.update(r)
    if "mom_ux_avg_mm" not in row:
        continue  # 옛 frac 근사 행은 실측이 아니므로 이 진단에서 제외
    all_rows.append(row)

print(f"mom_* 실측값 있는 행: {len(all_rows)}개")


def is_holdout_row(r, frac=0.2):
    key = f"{r['L_M_mm']}_{r['phi_deg']}_{r['beta_deg']}_{r['contact_s_mm']}"
    h = int(hashlib.md5(key.encode()).hexdigest(), 16)
    return (h % 10000) < int(frac * 10000)


holdout_rows = [r for r in all_rows if is_holdout_row(r)]
fit_rows = [r for r in all_rows if not is_holdout_row(r)]
print(f"fit={len(fit_rows)}, holdout={len(holdout_rows)}")

X = np.array([[r[f] for f in FEATURES] for r in fit_rows])
y = np.array([[r[t] for t in TARGETS] for r in fit_rows])
X_mean, X_std = X.mean(axis=0), X.std(axis=0)
X_std[X_std < 1e-9] = 1.0
y_mean, y_std = y.mean(axis=0), y.std(axis=0)
y_std[y_std < 1e-9] = 1.0
Xn = (X - X_mean) / X_std
yn = (y - y_mean) / y_std

X_hold = np.array([[r[f] for f in FEATURES] for r in holdout_rows])
y_hold = np.array([[r[t] for t in TARGETS] for r in holdout_rows])
Xn_hold = (X_hold - X_mean) / X_std
lm_hold = X_hold[:, 0]


class SurrogateMLP(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, n_out),
        )

    def forward(self, x):
        return self.net(x)


def train_mlp(X_tr, y_tr, seed, epochs=2000, lr=1e-3, weight_decay=1e-4):
    torch.manual_seed(seed)
    model = SurrogateMLP(X_tr.shape[1], y_tr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    Xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.float32)
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(Xt), yt)
        loss.backward()
        opt.step()
    return model


N_ENSEMBLE = 10
models = [train_mlp(Xn, yn, seed=i) for i in range(N_ENSEMBLE)]

with torch.no_grad():
    preds_n = np.mean([m(torch.tensor(Xn_hold, dtype=torch.float32)).numpy() for m in models], axis=0)
preds_phys = preds_n * y_std + y_mean

print(f"\n=== 서로게이트 mom_* 단독, 실측 홀드아웃(n={len(holdout_rows)}) 전체 R^2 ===")
for i, t in enumerate(TARGETS):
    r2 = r2_score(y_hold[:, i], preds_phys[:, i])
    print(f"  {t}: R^2={r2:.3f}")

print("\n=== L_M 구간별 residual(|pred-true|) 비교 (경계/외삽 효과 확인) ===")
bins = [(0.0, 12.5), (12.5, 25.0), (25.0, 50.0), (50.0, 100.0)]
for lo, hi in bins:
    mask = (lm_hold >= lo) & (lm_hold < hi if hi < 100.0 else lm_hold <= hi)
    n = mask.sum()
    if n == 0:
        print(f"  L_M in [{lo},{hi}): n=0")
        continue
    err = np.abs(preds_phys[mask] - y_hold[mask])
    print(f"  L_M in [{lo},{hi}): n={n}, mom_ux MAE={err[:,0].mean():.3f}mm, mom_uy MAE={err[:,1].mean():.3f}mm")
