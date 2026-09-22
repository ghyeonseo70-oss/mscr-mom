"""2026-09-22(39번-b): 대체모델이 깊이 의존성을 실제 FEA와 같게 재현하는지 확인.

상황: depth_head가 합성-val에서는 R^2=0.642로 잘 배웠는데 실측 홀드아웃에서 -0.621로
무너짐(예측이 거의 상수 0.12mm). 합성/실측 각각은 깊이 정보를 갖고 있음이 39번에서
확인됐으므로(둘 다 RF 70%), 남은 설명은 **두 데이터의 깊이→신호 매핑이 서로 다르다**는 것.
합성 매핑을 배운 CNN이 실측에 그대로 적용하면 틀릴 수밖에 없음.

방법: 대체모델 학습에서 제외된 홀드아웃 행만 가지고, 같은 (L_M,phi,beta,s,depth)에 대해
  - 실제 FEA 변위 (정답)
  - 대체모델 예측 변위
를 **깊이별로** 비교. 대체모델이 깊이에 따른 변화폭을 압축/과장하면 그게 원인.
"""
import hashlib
import json
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold

HERE = os.path.dirname(os.path.abspath(__file__))
FEA = os.path.join(HERE, "..", "..", "..", "data", "contact_scenarios", "fea",
                    "fea_lm_phi_pos_matv2_all.json")

FEATURES = ["L_M_mm", "phi_deg", "beta_deg", "contact_s_mm", "push_depth_mm"]
TARGETS = ["tip_ux_avg_mm", "tip_uy_avg_mm", "tip_theta_deg_board"]
DEFAULTS = {"L_M_mm": 50.0, "phi_deg": 60.0, "beta_deg": 0.0}


def is_holdout_row(r, frac=0.2):
    key = f"{r['L_M_mm']}_{r['phi_deg']}_{r['beta_deg']}_{r['contact_s_mm']}"
    return (int(hashlib.md5(key.encode()).hexdigest(), 16) % 10000) < int(frac * 10000)


class MLP(nn.Module):
    def __init__(self, ni, no):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(ni, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(),
                                  nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, no))

    def forward(self, x):
        return self.net(x)


rows = []
for r in json.load(open(FEA, encoding="utf-8")):
    row = dict(DEFAULTS)
    row.update(r)
    rows.append(row)

fit = [r for r in rows if not is_holdout_row(r)]
hold = [r for r in rows if is_holdout_row(r)]
print(f"대체모델 학습 {len(fit)}행 / 홀드아웃 {len(hold)}행 (학습 때와 동일한 해시 분할)")

X = np.array([[r[f] for f in FEATURES] for r in fit])
y = np.array([[r[t] for t in TARGETS] for r in fit])
Xm, Xs = X.mean(0), X.std(0)
Xs[Xs < 1e-9] = 1.0
ym, ys = y.mean(0), y.std(0)
ys[ys < 1e-12] = 1.0

models = []
for seed in range(5):
    torch.manual_seed(seed)
    m = MLP(len(FEATURES), len(TARGETS))
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt = torch.tensor((X - Xm) / Xs, dtype=torch.float32)
    yt = torch.tensor((y - ym) / ys, dtype=torch.float32)
    for _ in range(2000):
        opt.zero_grad()
        nn.functional.mse_loss(m(Xt), yt).backward()
        opt.step()
    models.append(m)

Xh = np.array([[r[f] for f in FEATURES] for r in hold])
yh = np.array([[r[t] for t in TARGETS] for r in hold])
dh = np.array([round(r["push_depth_mm"], 3) for r in hold])
with torch.no_grad():
    p = np.mean([m(torch.tensor((Xh - Xm) / Xs, dtype=torch.float32)).numpy() for m in models], axis=0)
pred = p * ys + ym

mv_true = np.hypot(yh[:, 0], yh[:, 1])
mv_pred = np.hypot(pred[:, 0], pred[:, 1])
print(f"\n{'깊이':>7s} {'n':>4s} {'실제 팁이동 중앙값':>18s} {'대체모델 예측':>16s} {'비율(예측/실제)':>16s}")
for d in np.unique(dh):
    m = dh == d
    t, q = np.median(mv_true[m]), np.median(mv_pred[m])
    print(f"{d:7.2f} {int(m.sum()):4d} {t:16.4f}mm {q:14.4f}mm {q/max(t,1e-9):15.2f}")

print("\n[깊이 변화에 따른 '기울기' 재현도]")
base = np.unique(dh)[len(np.unique(dh)) // 2]
for d in np.unique(dh):
    if d == base:
        continue
    mt = np.median(mv_true[dh == d]) / max(np.median(mv_true[dh == base]), 1e-9)
    mp = np.median(mv_pred[dh == d]) / max(np.median(mv_pred[dh == base]), 1e-9)
    print(f"  {base:.2f}mm -> {d:.2f}mm : 실제 {mt:.2f}배, 대체모델 {mp:.2f}배"
          f"  {'<-- 압축됨' if abs(mp-1) < abs(mt-1)*0.7 else ''}")
print("\n대체모델 배율이 실제보다 1에 가까우면(압축) 깊이 차이를 뭉개고 있다는 뜻 -")
print("그 경우 합성 데이터의 깊이-신호 관계가 실제와 달라 CNN이 실측으로 전이되지 않음.")
