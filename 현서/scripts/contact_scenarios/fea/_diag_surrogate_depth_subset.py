"""2026-09-21: 0.05mm 행이 대체모델(서로게이트) 학습을 오염시키는지 확인.

배경: 34번에서 0.05mm 데이터가 신뢰 구간 밖임을 확인했는데(힘의 67%가 0.001mN 미만,
메쉬 0.3mm 대비 관입이 1/6), 깊이 랜덤 샘플링으로 바꾼 뒤 돌린 드라이런에서 서로게이트의
Fy_total_N R^2가 0.649(655행 시절) -> 0.440으로 떨어짐. 데이터가 늘었는데 오히려 나빠진
것이라 0.05mm 오염이 의심됨.

방법: 같은 조건(5-fold, 10앙상블)에서 학습 데이터만 바꿔 비교.
  (A) 전체 857행
  (B) 0.05mm 제외 (=0.10/0.20만)
  (C) 0.10mm만 (깊이 축 없음, 예전 상태 재현)
(B)가 (A)보다 좋으면 0.05mm를 빼는 게 맞고, (B)가 (C)보다 좋으면 깊이 축 추가가 이득.
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "..", "data", "contact_scenarios", "fea",
                     "fea_lm_phi_pos_matv2_all.json")

FEATURES = ["L_M_mm", "phi_deg", "beta_deg", "contact_s_mm", "push_depth_mm"]
TARGETS = ["tip_ux_avg_mm", "tip_uy_avg_mm", "tip_theta_deg_board",
           "Fx_total_N", "Fy_total_N", "F_mag_N", "mom_theta_deg_board"]
N_ENSEMBLE = int(os.environ.get("N_ENSEMBLE", 5))   # 원본은 10, 비교용이라 5로 줄여 시간 절약
EPOCHS = int(os.environ.get("EPOCHS", 2000))


class SurrogateMLP(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, n_out))

    def forward(self, x):
        return self.net(x)


def train_mlp(X_tr, y_tr, seed):
    torch.manual_seed(seed)
    m = SurrogateMLP(X_tr.shape[1], y_tr.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    lf = nn.MSELoss()
    Xt, yt = torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32)
    for _ in range(EPOCHS):
        opt.zero_grad()
        lf(m(Xt), yt).backward()
        opt.step()
    return m


def evaluate(rows, label):
    rows = [r for r in rows if all(t in r for t in TARGETS)]
    X = np.array([[r[f] for f in FEATURES] for r in rows])
    y = np.array([[r[t] for t in TARGETS] for r in rows])
    X_mean, X_std = X.mean(axis=0), X.std(axis=0)
    X_std[X_std < 1e-9] = 1.0
    y_mean, y_std = y.mean(axis=0), y.std(axis=0)
    y_std[y_std < 1e-9] = 1.0
    Xn, yn = (X - X_mean) / X_std, (y - y_mean) / y_std

    preds = np.zeros_like(yn)
    for tr, va in KFold(n_splits=5, shuffle=True, random_state=0).split(Xn):
        models = [train_mlp(Xn[tr], yn[tr], seed=i) for i in range(N_ENSEMBLE)]
        with torch.no_grad():
            p = np.mean([m(torch.tensor(Xn[va], dtype=torch.float32)).numpy() for m in models], axis=0)
        preds[va] = p
    return {t: r2_score(yn[:, i], preds[:, i]) for i, t in enumerate(TARGETS)}, len(rows)


rows = [r for r in json.load(open(DATA, encoding="utf-8")) if all(t in r for t in TARGETS)]
def depth(r): return round(r.get("push_depth_mm", 0.1), 3)

# ⚠️ 28번 교훈: 학습셋마다 5-fold R^2를 재서 비교하면 셋마다 정답값 분산이 달라
# R^2가 서로 비교 불가능해짐. 그래서 **공통 테스트셋 하나**를 고정해놓고, 학습셋만 바꿔가며
# "같은 문제"를 풀게 해서 비교함. 테스트셋은 신뢰할 수 있는 깊이(0.10/0.20)에서만 뽑음
# (0.05mm는 애초에 정답 자체가 못 미더워서 채점 기준으로 부적합).
rng = np.random.default_rng(0)
reliable = [r for r in rows if depth(r) in (0.10, 0.20)]
idx = rng.permutation(len(reliable))
test_rows = [reliable[i] for i in idx[:int(0.25 * len(reliable))]]
test_keys = {(r["L_M_mm"], r["phi_deg"], r["beta_deg"], r["contact_s_mm"], depth(r)) for r in test_rows}
def in_test(r):
    return (r["L_M_mm"], r["phi_deg"], r["beta_deg"], r["contact_s_mm"], depth(r)) in test_keys

pool = [r for r in rows if not in_test(r)]
subsets = [
    ("(A) 0.05 포함", pool),
    ("(B) 0.05 제외", [r for r in pool if depth(r) != 0.05]),
    ("(C) 0.10만", [r for r in pool if depth(r) == 0.10]),
]
print(f"공통 테스트셋: {len(test_rows)}행 (0.10/0.20mm에서만 추출, 셋 다 동일하게 채점)\n")


def fit_and_score(train_rows, test_rows):
    X = np.array([[r[f] for f in FEATURES] for r in train_rows])
    y = np.array([[r[t] for t in TARGETS] for r in train_rows])
    X_mean, X_std = X.mean(axis=0), X.std(axis=0)
    X_std[X_std < 1e-9] = 1.0
    y_mean, y_std = y.mean(axis=0), y.std(axis=0)
    y_std[y_std < 1e-9] = 1.0
    Xn, yn = (X - X_mean) / X_std, (y - y_mean) / y_std
    models = [train_mlp(Xn, yn, seed=i) for i in range(N_ENSEMBLE)]

    Xte = (np.array([[r[f] for f in FEATURES] for r in test_rows]) - X_mean) / X_std
    yte = np.array([[r[t] for t in TARGETS] for r in test_rows])
    with torch.no_grad():
        p = np.mean([m(torch.tensor(Xte, dtype=torch.float32)).numpy() for m in models], axis=0)
    pred = p * y_std + y_mean
    # 공통 테스트셋이라 R^2도 비교 가능하지만, 28번 교훈대로 MAE도 같이 봄.
    # 힘(N)은 값이 너무 작아 mN으로 환산해야 MAE가 눈에 보임.
    scale = {t: (1000.0 if t.endswith("_N") else 1.0) for t in TARGETS}
    return {t: (r2_score(yte[:, i], pred[:, i]),
                np.abs(yte[:, i] - pred[:, i]).mean() * scale[t])
            for i, t in enumerate(TARGETS)}

results = {}
for label, sub in subsets:
    print(f"{label} 학습 중... (n={len(sub)})", flush=True)
    results[label] = fit_and_score(sub, test_rows)

print(f"\n=== 공통 테스트셋 {len(test_rows)}행 채점 (R² / MAE) ===")
print(f"{'타겟':22s}" + "".join(f"{lbl:>24s}" for lbl, _ in subsets))
for t in TARGETS:
    row = f"{t:22s}"
    best = max(results[lbl][t][0] for lbl, _ in subsets)
    for lbl, _ in subsets:
        r2, mae = results[lbl][t]
        mark = "*" if abs(r2 - best) < 1e-9 else " "
        row += f"{r2:>13.3f}{mark}/{mae:>8.4f} "
    print(row)
print(f"\n{'(학습 행 수)':22s}" + "".join(f"{len(sub):>24d}" for _, sub in subsets))
print("* = 해당 타겟 최고 R²  |  MAE 단위: 변위 mm / 각도 deg / 힘 mN")
