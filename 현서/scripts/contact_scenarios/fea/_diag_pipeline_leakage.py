"""가설 검증: CNN이 "진짜 물리"가 아니라 "서로게이트의 (변위-힘) 내부 스토리"를 외웠는가?

같은 36개 실측 홀드아웃 지점에서 B-field를 두 가지 방식으로 만들어 같은(학습된) CNN에 넣어봄:
  경로A("real"): 실측 tip_ux,uy,theta로 B-field 생성 (PROJECT_STATUS.md 실측검증 블록과 동일)
  경로B("surrogate-style"): 서로게이트가 예측한 tip_ux,uy,theta로 B-field 생성 (학습 데이터 생성과 동일 방식)
CNN의 Fx 예측이 경로B에서는 서로게이트 자신의 Fx 예측과 가깝고, 경로A(실측 변위)에서는
실측 Fx와 안 맞는다면 - CNN이 진짜 힘-변위 물리가 아니라 서로게이트의 내부 매핑을 외웠다는 뜻.
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score

HERE = os.path.dirname(os.path.abspath(__file__))
FEA_DATA_DIR = os.path.join(HERE, "..", "..", "..", "data", "contact_scenarios", "fea")
MODELS_DIR = os.path.join(HERE, "..", "..", "..", "models")
HYUNSEO_DIR = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
REPO_ROOT = os.path.abspath(os.path.join(HYUNSEO_DIR, ".."))
FORCE_MODEL_DIR = os.path.join(REPO_ROOT, "scripts", "force_model")
sys.path.insert(0, FORCE_MODEL_DIR)
import force_model as fm
import magpylib as magpy
from scipy.spatial.transform import Rotation

FEATURES = ["L_M_mm", "phi_deg", "beta_deg", "contact_s_mm", "push_depth_mm"]
TARGETS = ["tip_ux_avg_mm", "tip_uy_avg_mm", "tip_uz_avg_mm", "tip_theta_deg_board",
           "Fx_total_N", "Fy_total_N", "Fz_total_N", "F_mag_N"]
DEFAULTS = {"L_M_mm": 50.0, "phi_deg": 60.0, "beta_deg": 0.0}
N_CLASSES = 4
BIN_WIDTH_MM = 20.0


def s_to_bin(s_mm):
    return min(N_CLASSES - 1, int(s_mm / BIN_WIDTH_MM))


all_rows = []
for r in json.load(open(os.path.join(FEA_DATA_DIR, "fea_lm_phi_pos_matv2_all.json"), encoding="utf-8")):
    row = dict(DEFAULTS)
    row.update(r)
    all_rows.append(row)

rng_holdout = np.random.default_rng(42)
perm = rng_holdout.permutation(len(all_rows))
n_holdout = max(20, int(len(all_rows) * 0.2))
holdout_idx, fit_idx = perm[:n_holdout], perm[n_holdout:]
holdout_rows = [all_rows[i] for i in holdout_idx]
fit_rows = [all_rows[i] for i in fit_idx]

X = np.array([[r[f] for f in FEATURES] for r in fit_rows])
y = np.array([[r[t] for t in TARGETS] for r in fit_rows])
X_mean, X_std = X.mean(axis=0), X.std(axis=0)
X_std[X_std < 1e-9] = 1.0
y_mean, y_std = y.mean(axis=0), y.std(axis=0)
y_std[y_std < 1e-9] = 1.0
Xn, yn = (X - X_mean) / X_std, (y - y_mean) / y_std


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


print("서로게이트 재학습 중(145개, 10-앙상블)...")
surrogates = [train_mlp(Xn, yn, seed=i) for i in range(10)]


def predict_surrogate(L_M, phi, beta, s, depth):
    x = np.array([[L_M, phi, beta, s, depth]])
    xn = (x - X_mean) / X_std
    xt = torch.tensor(xn, dtype=torch.float32)
    with torch.no_grad():
        pn = np.mean([m(xt).numpy()[0] for m in surrogates], axis=0)
    p = pn * y_std + y_mean
    return dict(zip(TARGETS, p))


class SingleProbeClassifier(nn.Module):
    def __init__(self, n_probes=1, n_classes=N_CLASSES, n_force=2, n_config=2):
        super().__init__()
        self.n_probes = n_probes
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Flatten(), nn.Linear(32 * 5 * 5, 64), nn.ReLU())
        self.trunk = nn.Sequential(nn.Linear(64 * n_probes, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3))
        self.seg_head = nn.Linear(128, n_classes)
        self.force_head = nn.Linear(128, n_force)
        self.s_head = nn.Linear(128, 1)
        self.config_head = nn.Linear(128, n_config)

    def forward(self, x):
        embeds = [self.encoder(x[:, p]) for p in range(self.n_probes)]
        h = self.trunk(torch.cat(embeds, dim=1))
        return self.seg_head(h), self.force_head(h), self.s_head(h).squeeze(-1), self.config_head(h)


ckpt = torch.load(os.path.join(MODELS_DIR, "position_segment_classifier_singleprobe_beta0180_4seg.pth"),
                   map_location="cpu", weights_only=False)
cnn = SingleProbeClassifier()
cnn.load_state_dict(ckpt["state_dict"])
cnn.eval()
X_mean2, X_std2 = ckpt["X_mean"], ckpt["X_std"]
f_mean, f_std = ckpt["f_mean"], ckpt["f_std"]

SENSOR_HEIGHT_MM = 15
sensor_positions = [(x, y, SENSOR_HEIGHT_MM) for y in np.linspace(180, 0, 5) for x in np.linspace(0, 180, 5)]
sensors = magpy.Collection([magpy.Sensor(position=pos) for pos in sensor_positions])
MAGNET_BR_TESLA = 0.4
main_magnet = magpy.magnet.Cylinder(polarization=(0, MAGNET_BR_TESLA, 0), dimension=(2, 2))
mom = magpy.magnet.Cylinder(polarization=(0, -MAGNET_BR_TESLA, 0), dimension=(1, 8))
mscr_robot = magpy.Collection(main_magnet, mom)


def compute_B(xLM_l, yLM_l, thLM, xL_l, yL_l, thL):
    xLM_b, yLM_b = fm.to_board_frame(xLM_l, yLM_l)
    xL_b, yL_b = fm.to_board_frame(xL_l, yL_l)
    mom.position = (float(xLM_b), float(yLM_b), 0)
    mom.orientation = Rotation.from_euler("z", -thLM, degrees=True)
    main_magnet.position = (float(xL_b), float(yL_b), 0)
    main_magnet.orientation = Rotation.from_euler("z", -thL, degrees=True)
    return magpy.getB(mscr_robot, sensors) * 1e6


def cnn_predict(img):
    xn = (img - X_mean2) / X_std2
    xt = torch.tensor(xn[None, None], dtype=torch.float32)
    with torch.no_grad():
        seg_logits, force_pred, s_pred, _ = cnn(xt)
    fx_fy = force_pred.numpy()[0] * f_std + f_mean
    return fx_fy  # [Fy_local, Fx_local] 순서 (fb와 동일)


real_fx, sur_fx_own, cnn_fx_realpath, cnn_fx_surpath = [], [], [], []
for r in holdout_rows:
    L_M, phi, beta, s, depth = r["L_M_mm"], r["phi_deg"], r["beta_deg"], r["contact_s_mm"], r["push_depth_mm"]
    try:
        r_free = fm.solve_shape(L_M=L_M, phi_deg=phi, loads=[])
    except Exception:
        continue
    xL_free, yL_free, thL_free = r_free["x_L"], r_free["y_L"], r_free["theta_L_deg"]
    xLM_free, yLM_free, thLM_free = r_free["x_LM"], r_free["y_LM"], r_free["theta_LM_deg"]
    B_free = compute_B(xLM_free, yLM_free, thLM_free, xL_free, yL_free, thL_free)
    frac = L_M / 100.0

    # 경로 A: 실측 변위
    d_xL, d_yL = r["tip_uy_avg_mm"], r["tip_ux_avg_mm"]
    d_thL = -r["tip_theta_deg_board"]
    d_xLM, d_yLM, d_thLM = d_xL * frac, d_yL * frac, d_thL * frac
    B_load_real = compute_B(xLM_free + d_xLM, yLM_free + d_yLM, thLM_free + d_thLM,
                             xL_free + d_xL, yL_free + d_yL, thL_free + d_thL)
    img_real = (B_load_real - B_free).reshape(5, 5, 3).transpose(2, 0, 1)
    fx_real_path = cnn_predict(img_real)[1]  # 열1 = Fx_total_N(로컬)

    # 경로 B: 서로게이트 예측 변위 (학습 데이터 생성과 동일 방식)
    pred = predict_surrogate(L_M, phi, beta, s, depth)
    d_xL_s, d_yL_s = pred["tip_uy_avg_mm"], pred["tip_ux_avg_mm"]
    d_thL_s = -pred["tip_theta_deg_board"]
    d_xLM_s, d_yLM_s, d_thLM_s = d_xL_s * frac, d_yL_s * frac, d_thL_s * frac
    B_load_sur = compute_B(xLM_free + d_xLM_s, yLM_free + d_yLM_s, thLM_free + d_thLM_s,
                            xL_free + d_xL_s, yL_free + d_yL_s, thL_free + d_thL_s)
    img_sur = (B_load_sur - B_free).reshape(5, 5, 3).transpose(2, 0, 1)
    fx_sur_path = cnn_predict(img_sur)[1]

    real_fx.append(r["Fx_total_N"])
    sur_fx_own.append(pred["Fx_total_N"])
    cnn_fx_realpath.append(fx_real_path)
    cnn_fx_surpath.append(fx_sur_path)

real_fx = np.array(real_fx)
sur_fx_own = np.array(sur_fx_own)
cnn_fx_realpath = np.array(cnn_fx_realpath)
cnn_fx_surpath = np.array(cnn_fx_surpath)

print(f"\n=== 진단 결과 (n={len(real_fx)}) ===")
print(f"[검산] CNN(실측B-field) vs 실측Fx:        R^2={r2_score(real_fx, cnn_fx_realpath):.3f}  (=PROJECT_STATUS의 0.009와 대조용)")
print(f"CNN(서로게이트B-field) vs 실측Fx:          R^2={r2_score(real_fx, cnn_fx_surpath):.3f}")
print(f"CNN(서로게이트B-field) vs 서로게이트 자체Fx: R^2={r2_score(sur_fx_own, cnn_fx_surpath):.3f}   <- 이게 높으면 'CNN=서로게이트 모방' 가설 지지")
print(f"CNN(실측B-field) vs CNN(서로게이트B-field): R^2={r2_score(cnn_fx_surpath, cnn_fx_realpath):.3f}  (같은 지점, B-field 생성경로만 다름 - 낮으면 CNN이 두 입력을 완전 다르게 취급한다는 뜻)")
print(f"서로게이트 자체Fx vs 실측Fx:                R^2={r2_score(real_fx, sur_fx_own):.3f}  (참고: 이전 진단과 동일해야 함)")
