"""서로게이트(대체모델)만 따로 떼서 36개 실측 홀드아웃에 테스트 - B-field/CNN 파이프라인을
아예 안 거치고, "서로게이트가 145개로 학습해서 안 본 36개 FEA를 얼마나 잘 맞추는가"만 순수하게 봄.
train_segment_classifier_singleprobe_beta0180_4seg.py와 완전히 동일한 홀드아웃 분리(시드 42)."""
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
TARGETS = ["tip_ux_avg_mm", "tip_uy_avg_mm", "tip_uz_avg_mm", "tip_theta_deg_board",
           "Fx_total_N", "Fy_total_N", "Fz_total_N", "F_mag_N"]
DEFAULTS = {"L_M_mm": 50.0, "phi_deg": 60.0, "beta_deg": 0.0}

all_rows = []
for r in json.load(open(os.path.join(FEA_DATA_DIR, "fea_lm_phi_pos_matv2_all.json"), encoding="utf-8")):
    row = dict(DEFAULTS)
    row.update(r)
    all_rows.append(row)

# 2026-08-25: train_segment_classifier_singleprobe_beta0180_4seg.py와 동일한 해시기반
# 고정 홀드아웃 (데이터 개수가 바뀌어도 기존 행의 소속이 안 바뀜 - 상세 이유는 그 파일 참고)
def is_holdout_row(r, frac=0.2):
    key = f"{r['L_M_mm']}_{r['phi_deg']}_{r['beta_deg']}_{r['contact_s_mm']}"
    h = int(hashlib.md5(key.encode()).hexdigest(), 16)
    return (h % 10000) < int(frac * 10000)


holdout_idx = [i for i, r in enumerate(all_rows) if is_holdout_row(r)]
fit_idx = [i for i, r in enumerate(all_rows) if not is_holdout_row(r)]
holdout_rows = [all_rows[i] for i in holdout_idx]
fit_rows = [all_rows[i] for i in fit_idx]
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

print("\n=== 서로게이트 단독, 실측 홀드아웃(n=%d) R^2 (B-field/CNN 안 거침) ===" % len(holdout_rows))
for i, t in enumerate(TARGETS):
    r2 = r2_score(y_hold[:, i], preds_phys[:, i])
    print(f"  {t}: R^2={r2:.3f}")

print("\n(참고: 5-fold CV였던 원래 로그의 근사치 - fit set 181->145로 줄어서 약간 다를 수 있음)")
