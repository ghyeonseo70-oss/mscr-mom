"""가설 검증: Fx_board(=로컬Fy_total_N)가 |phi|>=90에서 왜 계속 약한가(HIGH_PHI_WEIGHT=3배
줘도 R^2=0.343)? 두 가지 후보를 구분:
  (a) 신호 자체가 그 구간에서 원래 작다/노이즈가 크다(물리적 한계, 토크제로 특이점) - 그러면
      대체모델 단독(B-field/CNN 안 거침)도 그 구간에서 이미 약해야 함.
  (b) B-field 인코딩/CNN 쪽에서 생기는 문제 - 그러면 대체모델 단독은 그 구간에서도 괜찮은데
      CNN 파이프라인만 유독 나빠야 함.
대체모델 단독 홀드아웃 R^2를 |phi|>=90 vs <90으로 쪼개서 확인 + 두 힘 성분의 실제 크기
(표준편차/평균절대값)를 구간별로 비교."""
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
phi_hold = X_hold[:, FEATURES.index("phi_deg")]


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

high_phi = np.abs(phi_hold) >= 90
i_fy = TARGETS.index("Fy_total_N")  # =Fx_board
i_fx = TARGETS.index("Fx_total_N")  # =Fy_board

print("\n=== 대체모델 단독 실측 홀드아웃 R^2 (B-field/CNN 안 거침) ===")
for label, mask in [("전체", np.ones(len(phi_hold), dtype=bool)), ("|phi|<90", ~high_phi), ("|phi|>=90", high_phi)]:
    n = mask.sum()
    if n < 3:
        print(f"  [{label}] n={n} - 생략")
        continue
    r2_fy = r2_score(y_hold[mask, i_fy], preds_phys[mask, i_fy])
    r2_fx = r2_score(y_hold[mask, i_fx], preds_phys[mask, i_fx])
    print(f"  [{label}] n={n}  Fy_total_N(=Fx_board): R^2={r2_fy:.3f}   Fx_total_N(=Fy_board): R^2={r2_fx:.3f}")

# ---- 힘 크기(신호 자체 크기) 비교: 전체 데이터(fit+holdout) 기준 ----
all_X = np.array([[r[f] for f in FEATURES] for r in all_rows])
all_y = np.array([[r[t] for t in TARGETS] for r in all_rows])
all_phi = all_X[:, FEATURES.index("phi_deg")]
all_high = np.abs(all_phi) >= 90

print("\n=== 힘 신호 크기(전체 518개 기준, mN) - '그 구간에서 원래 힘이 작은가' 확인 ===")
for label, mask in [("|phi|<90", ~all_high), ("|phi|>=90", all_high)]:
    fy_vals = all_y[mask, i_fy] * 1000
    fx_vals = all_y[mask, i_fx] * 1000
    print(f"  [{label}] n={mask.sum()}  "
          f"Fy_total_N(=Fx_board): std={fy_vals.std():.4f}mN, mean|x|={np.abs(fy_vals).mean():.4f}mN | "
          f"Fx_total_N(=Fy_board): std={fx_vals.std():.4f}mN, mean|x|={np.abs(fx_vals).mean():.4f}mN")
