"""다른 컴퓨터에서 채워준 FEA 공백(fea_beta_generalization_check.json, fea_lm10_gap_check.json,
갱신된 fea_lm_phi_pos_sweep_all.json - L_M=100 보강)이 대체모델 5-fold R^2를 실제로
개선하는지 확인. 기존 SOURCES 대비 새 파일 2개 추가한 것만 비교(전 15만개 재생성은 안 함)."""
import json
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

HERE = os.path.dirname(os.path.abspath(__file__))
FEA_DATA_DIR = os.path.join(HERE, "..", "..", "..", "data", "contact_scenarios", "fea")

FEATURES = ["L_M_mm", "phi_deg", "beta_deg", "contact_s_mm", "push_depth_mm"]
TARGETS = ["tip_ux_avg_mm", "tip_uy_avg_mm", "tip_uz_avg_mm", "tip_theta_deg_board",
           "Fx_total_N", "Fy_total_N", "Fz_total_N", "F_mag_N"]
DEFAULTS = {"L_M_mm": 50.0, "phi_deg": 60.0, "beta_deg": 0.0}
N_ENSEMBLE = 10

OLD_SOURCES = ["fea_lm_phi_pos_sweep_all.json", "fea_bent_contact_sweep.json",
               "fea_geom_sweep_all.json", "fea_angle_sweep_all.json"]
NEW_SOURCES = OLD_SOURCES + ["fea_beta_generalization_check.json", "fea_lm10_gap_check.json"]
NEWEST_SOURCES = NEW_SOURCES + ["fea_beta_fine_resolution.json"]


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


def predict_ensemble(models, X_val):
    preds = []
    for m in models:
        m.eval()
        with torch.no_grad():
            preds.append(m(torch.tensor(X_val, dtype=torch.float32)).numpy())
    return np.mean(preds, axis=0)


def load_rows(sources):
    all_rows = []
    for fname in sources:
        path = os.path.join(FEA_DATA_DIR, fname)
        for r in json.load(open(path)):
            row = dict(DEFAULTS)
            row.update(r)
            all_rows.append(row)
    return all_rows


def run_cv(sources, tag):
    all_rows = load_rows(sources)
    X = np.array([[r[f] for f in FEATURES] for r in all_rows])
    y = np.array([[r[t] for t in TARGETS] for r in all_rows])
    X_mean, X_std = X.mean(axis=0), X.std(axis=0)
    X_std[X_std < 1e-9] = 1.0
    y_mean, y_std = y.mean(axis=0), y.std(axis=0)
    y_std[y_std < 1e-9] = 1.0
    Xn = (X - X_mean) / X_std
    yn = (y - y_mean) / y_std

    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    preds = np.zeros_like(yn)
    for tr_idx, val_idx in kf.split(Xn):
        fold_models = [train_mlp(Xn[tr_idx], yn[tr_idx], seed=i) for i in range(N_ENSEMBLE)]
        preds[val_idx] = predict_ensemble(fold_models, Xn[val_idx])
    print(f"\n=== [{tag}] n={len(all_rows)}개, 대체모델({N_ENSEMBLE}-앙상블) 5-fold R^2 ===")
    for i, t in enumerate(TARGETS):
        print(f"  {t}: R^2={r2_score(yn[:, i], preds[:, i]):.3f}")


if __name__ == "__main__":
    run_cv(OLD_SOURCES, "기존 4개 소스")
    run_cv(NEW_SOURCES, "새 2개 소스 추가(45도 간격, 이전 결과 악화)")
    run_cv(NEWEST_SOURCES, "+beta 정밀 스윕(15도 간격, transition 근처)")
