"""
FEA 스윕 결과(188개, 1~3단계 병합)로 "빠른 대체모델"을 학습.
입력: L_M, phi, beta(원주각), contact_s, push_depth (5개)
출력: 접촉으로 인한 팁 변위/회전 tip_ux, tip_uy, tip_uz, tip_theta_deg_board (4개)

이 모델이 검증되면, force_model.py의 버그(phi≠0에서 접촉힘 방향이 틀리는 문제) 없이
빠르게(밀리초 단위) 많은 시나리오를 생성해서 magpylib 자기장 계산 -> CNN 학습데이터로 이어갈 수 있음.

샘플이 188개뿐이라(5차원 입력 대비 적음) 5-fold 교차검증으로 일반화 성능을 확인하고,
MLP와 RandomForest 두 가지를 비교해서 실제로 학습이 되는지부터 정직하게 확인한다.
"""
import json
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "..", "..", "data", "contact_scenarios", "fea")
MODELS_DIR = os.path.join(HERE, "..", "..", "..", "models")

FEATURES = ["L_M_mm", "phi_deg", "beta_deg", "contact_s_mm", "push_depth_mm"]
TARGETS = ["tip_ux_avg_mm", "tip_uy_avg_mm", "tip_uz_avg_mm", "tip_theta_deg_board"]
DEFAULTS = {"L_M_mm": 50.0, "phi_deg": 60.0, "beta_deg": 0.0}


def load_rows():
    rows = []
    for fname in ["fea_bent_contact_sweep.json", "fea_geom_sweep_all.json", "fea_angle_sweep_all.json"]:
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            continue
        for r in json.load(open(path)):
            row = dict(DEFAULTS)
            row.update(r)
            rows.append(row)
    return rows


rows = load_rows()
print(f"전체 병합 샘플 수: {len(rows)}")

X = np.array([[r[f] for f in FEATURES] for r in rows])
y = np.array([[r[t] for t in TARGETS] for r in rows])

X_mean, X_std = X.mean(axis=0), X.std(axis=0)
X_std[X_std < 1e-9] = 1.0
y_mean, y_std = y.mean(axis=0), y.std(axis=0)
y_std[y_std < 1e-9] = 1.0
Xn = (X - X_mean) / X_std
yn = (y - y_mean) / y_std


class SurrogateMLP(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, n_out),
        )

    def forward(self, x):
        return self.net(x)


def train_mlp(X_tr, y_tr, X_val, epochs=800, lr=1e-3, weight_decay=1e-3):
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
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X_val, dtype=torch.float32)).numpy()
    return pred, model


# ── 5-fold 교차검증: MLP vs RandomForest, 타깃별 R^2 ──────────────────────
kf = KFold(n_splits=5, shuffle=True, random_state=0)
mlp_preds = np.zeros_like(yn)
rf_preds = np.zeros_like(yn)

for tr_idx, val_idx in kf.split(Xn):
    X_tr, X_val = Xn[tr_idx], Xn[val_idx]
    y_tr, y_val = yn[tr_idx], yn[val_idx]

    pred_mlp, _ = train_mlp(X_tr, y_tr, X_val)
    mlp_preds[val_idx] = pred_mlp

    rf = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=0)
    rf.fit(X_tr, y_tr)
    rf_preds[val_idx] = rf.predict(X_val)

print("\n=== 5-fold 교차검증 R^2 (타깃별) ===")
print(f"{'타깃':<22} {'MLP':>8} {'RandomForest':>14}")
for i, t in enumerate(TARGETS):
    r2_mlp = r2_score(yn[:, i], mlp_preds[:, i])
    r2_rf = r2_score(yn[:, i], rf_preds[:, i])
    print(f"{t:<22} {r2_mlp:>8.3f} {r2_rf:>14.3f}")

# ── 전체 데이터로 최종 모델 학습 (RandomForest 채택 - 표에서 더 나은 쪽으로 아래서 자동 결정) ──
r2_mlp_avg = np.mean([r2_score(yn[:, i], mlp_preds[:, i]) for i in range(len(TARGETS))])
r2_rf_avg = np.mean([r2_score(yn[:, i], rf_preds[:, i]) for i in range(len(TARGETS))])
print(f"\n평균 R^2: MLP={r2_mlp_avg:.3f}, RandomForest={r2_rf_avg:.3f}")

best = "rf" if r2_rf_avg >= r2_mlp_avg else "mlp"
print(f"-> 채택: {best}")

os.makedirs(MODELS_DIR, exist_ok=True)
if best == "rf":
    final_model = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=0)
    final_model.fit(Xn, yn)
    import pickle
    with open(os.path.join(MODELS_DIR, "fea_surrogate_rf.pkl"), "wb") as f:
        pickle.dump({"model": final_model, "X_mean": X_mean, "X_std": X_std,
                     "y_mean": y_mean, "y_std": y_std, "features": FEATURES, "targets": TARGETS}, f)
    print(f"저장: {os.path.join(MODELS_DIR, 'fea_surrogate_rf.pkl')}")
else:
    _, final_model = train_mlp(Xn, yn, Xn)
    torch.save({"state_dict": final_model.state_dict(), "X_mean": X_mean, "X_std": X_std,
                "y_mean": y_mean, "y_std": y_std, "features": FEATURES, "targets": TARGETS},
               os.path.join(MODELS_DIR, "fea_surrogate_mlp.pth"))
    print(f"저장: {os.path.join(MODELS_DIR, 'fea_surrogate_mlp.pth')}")
