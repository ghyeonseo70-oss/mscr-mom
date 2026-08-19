"""서로게이트 변위(ux,uy) 정확도가 물리기반 피처 추가만으로 개선되는지 테스트.
K1(s<a1)/K_RIGID(a1<=s<a2)/K2(s>=a2) 경계(a1=L_M-4,a2=L_M+4)가 L_M에 따라 움직이는 걸
원래 피처(L_M, contact_s_mm 그대로)만으로는 MLP가 145개로 배우기 어려울 수 있다는 가설 -
s_rel=contact_s_mm-L_M(부호있는 MOM 중심 기준 상대위치)을 추가해서 같은 홀드아웃(시드42)으로
비교. 새 FEA를 전혀 안 쓰고 피처 엔지니어링만으로 되는지 보는 게 목적."""
import json
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score

HERE = os.path.dirname(os.path.abspath(__file__))
FEA_DATA_DIR = os.path.join(HERE, "..", "..", "..", "data", "contact_scenarios", "fea")

BASE_FEATURES = ["L_M_mm", "phi_deg", "beta_deg", "contact_s_mm", "push_depth_mm"]
TARGETS = ["tip_ux_avg_mm", "tip_uy_avg_mm", "tip_uz_avg_mm", "tip_theta_deg_board",
           "Fx_total_N", "Fy_total_N", "Fz_total_N", "F_mag_N"]
DEFAULTS = {"L_M_mm": 50.0, "phi_deg": 60.0, "beta_deg": 0.0}

all_rows = []
for r in json.load(open(os.path.join(FEA_DATA_DIR, "fea_lm_phi_pos_matv2_all.json"), encoding="utf-8")):
    row = dict(DEFAULTS)
    row.update(r)
    row["s_rel_mm"] = row["contact_s_mm"] - row["L_M_mm"]  # MOM 중심 기준 부호있는 상대위치
    row["a1_mm"] = row["L_M_mm"] - 4.0
    row["a2_mm"] = row["L_M_mm"] + 4.0
    row["seg_code"] = -1.0 if row["contact_s_mm"] < row["a1_mm"] else (0.0 if row["contact_s_mm"] < row["a2_mm"] else 1.0)
    all_rows.append(row)

rng_holdout = np.random.default_rng(42)
perm = rng_holdout.permutation(len(all_rows))
n_holdout = max(20, int(len(all_rows) * 0.2))
holdout_idx, fit_idx = perm[:n_holdout], perm[n_holdout:]
holdout_rows = [all_rows[i] for i in holdout_idx]
fit_rows = [all_rows[i] for i in fit_idx]
print(f"fit={len(fit_rows)}, holdout={len(holdout_rows)}")


class SurrogateMLP(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(),
                                  nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, n_out))

    def forward(self, x):
        return self.net(x)


def train_mlp(X_tr, y_tr, seed, epochs=2000, lr=1e-3, weight_decay=1e-4):
    torch.manual_seed(seed)
    model = SurrogateMLP(X_tr.shape[1], y_tr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    Xt, yt = torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32)
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(Xt), yt)
        loss.backward()
        opt.step()
    return model


def run(feature_set, label):
    X = np.array([[r[f] for f in feature_set] for r in fit_rows])
    y = np.array([[r[t] for t in TARGETS] for r in fit_rows])
    X_mean, X_std = X.mean(axis=0), X.std(axis=0)
    X_std[X_std < 1e-9] = 1.0
    y_mean, y_std = y.mean(axis=0), y.std(axis=0)
    y_std[y_std < 1e-9] = 1.0
    Xn, yn = (X - X_mean) / X_std, (y - y_mean) / y_std

    X_hold = np.array([[r[f] for f in feature_set] for r in holdout_rows])
    y_hold = np.array([[r[t] for t in TARGETS] for r in holdout_rows])
    Xn_hold = (X_hold - X_mean) / X_std

    models = [train_mlp(Xn, yn, seed=i) for i in range(10)]
    with torch.no_grad():
        preds_n = np.mean([m(torch.tensor(Xn_hold, dtype=torch.float32)).numpy() for m in models], axis=0)
    preds_phys = preds_n * y_std + y_mean

    print(f"\n=== {label} (피처: {feature_set}) ===")
    for i, t in enumerate(TARGETS):
        r2 = r2_score(y_hold[:, i], preds_phys[:, i])
        print(f"  {t}: R^2={r2:.3f}")


run(BASE_FEATURES, "기존 피처 그대로 (대조군)")
run(BASE_FEATURES + ["s_rel_mm"], "s_rel_mm(=s-L_M) 추가")
run(BASE_FEATURES + ["s_rel_mm", "seg_code"], "s_rel_mm + seg_code(구간 원핫 비슷하게) 추가")
