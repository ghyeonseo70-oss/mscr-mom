"""가설: Fx_board(=Fy_total_N, 진동하는 어려운 성분)를 CNN이 직접 예측하는 대신, 이미 CNN이
B-field로부터 따로 예측하고 있는 s/phi/L_M(config_head)과, 이미 잘 예측하는 Fy_board(=Fx_total_N,
F_mag*cosθ에 해당)로부터 힘 크기(F_mag)를 역산한 뒤, Fx_board = -F_mag*sinθ로 순수 기하학
재구성하면 지금(직접 회귀, R^2=0.601)보다 나은지 확인.

중요: 재학습 없이, 이미 저장된 체크포인트의 예측값(s_pred, phi_pred, Fy_board_pred)만 갖고
계산 - "진짜 정답 s/phi"가 아니라 "모델이 실제로 낸(어느 정도 틀린) 예측"을 쓰는 현실적인
테스트. 지난번(_diag 등)에서 진짜 s/phi로 했을 때 R^2=0.834가 나왔던 건 낙관적 상한선이었고,
이번이 실제로 기대할 수 있는 숫자."""
import hashlib
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

DEFAULTS = {"L_M_mm": 50.0, "phi_deg": 60.0, "beta_deg": 0.0}
N_CLASSES = 4


def is_holdout_row(r, frac=0.2):
    key = f"{r['L_M_mm']}_{r['phi_deg']}_{r['beta_deg']}_{r['contact_s_mm']}"
    h = int(hashlib.md5(key.encode()).hexdigest(), 16)
    return (h % 10000) < int(frac * 10000)


all_rows = []
for r in json.load(open(os.path.join(FEA_DATA_DIR, "fea_lm_phi_pos_matv2_all.json"), encoding="utf-8")):
    row = dict(DEFAULTS)
    row.update(r)
    all_rows.append(row)

holdout_idx = [i for i, r in enumerate(all_rows) if is_holdout_row(r)]
real_holdout_rows = [all_rows[i] for i in holdout_idx]
print(f"실측 홀드아웃: n={len(real_holdout_rows)}")

# ---- 모델 로드 ----
ckpt = torch.load(os.path.join(MODELS_DIR, "position_segment_classifier_singleprobe_beta0180_4seg.pth"),
                   map_location="cpu", weights_only=False)


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
        self.lm_zero_head = nn.Linear(128, 1)  # 2026-08-27 추가 - state_dict 키 맞추기용

    def forward(self, x):
        embeds = [self.encoder(x[:, p]) for p in range(self.n_probes)]
        h = self.trunk(torch.cat(embeds, dim=1))
        return (self.seg_head(h), self.force_head(h), self.s_head(h).squeeze(-1), self.config_head(h),
                self.lm_zero_head(h).squeeze(-1))


cnn = SingleProbeClassifier()
cnn.load_state_dict(ckpt["state_dict"])
cnn.eval()
X_mean2, X_std2 = ckpt["X_mean"], ckpt["X_std"]
f_mean, f_std = ckpt["f_mean"], ckpt["f_std"]
s_mean, s_std = ckpt["s_mean"], ckpt["s_std"]
c_mean, c_std = ckpt["c_mean"], ckpt["c_std"]
force_names = ckpt["force_names"]  # ["Fx_board_N", "Fy_board_N"]
config_names = ckpt["config_names"]  # ["L_M_mm", "phi_deg"]

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


real_X, real_fy_total, real_fx_total, meta = [], [], [], []
for r in real_holdout_rows:
    L_M, phi = r["L_M_mm"], r["phi_deg"]
    try:
        r_free = fm.solve_shape(L_M=L_M, phi_deg=phi, loads=[])
    except Exception:
        continue
    d_xL_local, d_yL_local = r["tip_uy_avg_mm"], r["tip_ux_avg_mm"]
    d_thL = -r["tip_theta_deg_board"]
    frac = L_M / 100.0
    d_xLM_local, d_yLM_local, d_thLM = d_xL_local * frac, d_yL_local * frac, d_thL * frac
    xL_free, yL_free, thL_free = r_free["x_L"], r_free["y_L"], r_free["theta_L_deg"]
    xLM_free, yLM_free, thLM_free = r_free["x_LM"], r_free["y_LM"], r_free["theta_LM_deg"]
    B_free = compute_B(xLM_free, yLM_free, thLM_free, xL_free, yL_free, thL_free)
    B_load = compute_B(xLM_free + d_xLM_local, yLM_free + d_yLM_local, thLM_free + d_thLM,
                        xL_free + d_xL_local, yL_free + d_yL_local, thL_free + d_thL)
    real_X.append((B_load - B_free).reshape(5, 5, 3).transpose(2, 0, 1))
    real_fy_total.append(r["Fy_total_N"])  # = Fx_board (어려운 타겟, 정답)
    real_fx_total.append(r["Fx_total_N"])  # = Fy_board (쉬운 타겟, 정답)
    meta.append({"L_M_true": L_M, "phi_true": phi, "s_true": r["contact_s_mm"]})

real_X = np.array(real_X, dtype=np.float32)
real_X_norm = (real_X - X_mean2) / X_std2
real_fy_true = np.array(real_fy_total)  # Fx_board 정답
real_fx_true = np.array(real_fx_total)  # Fy_board 정답

with torch.no_grad():
    rX = torch.tensor(real_X_norm[:, None]).float()
    _, r_force_pred, r_s_pred, r_config_pred, _ = cnn(rX)
    force_phys = r_force_pred.numpy() * f_std + f_mean       # [Fx_board_pred, Fy_board_pred]
    s_phys = (r_s_pred.numpy() * s_std + s_mean)
    config_phys = r_config_pred.numpy() * c_std + c_mean      # [L_M_pred, phi_pred]

fx_board_pred_direct = force_phys[:, force_names.index("Fx_board_N")]  # 지금까지의 직접회귀 예측
fy_board_pred_direct = force_phys[:, force_names.index("Fy_board_N")]
lm_pred = config_phys[:, config_names.index("L_M_mm")]
phi_pred = config_phys[:, config_names.index("phi_deg")]

print(f"\n[참고] 지금까지의 직접회귀 R^2: Fx_board={r2_score(real_fy_true, fx_board_pred_direct):.3f}  "
      f"Fy_board={r2_score(real_fx_true, fy_board_pred_direct):.3f}")

# ---- 예측된 s/phi/L_M으로 theta(국소접선각) 계산 -> 기하학 재구성 ----
theta_cache = {}


def theta_at_s(L_M, phi, s):
    key = (round(L_M, 1), round(phi, 1))
    if key not in theta_cache:
        try:
            r = fm.solve_shape(L_M=max(L_M, 0.5), phi_deg=phi, loads=[], return_curve=True)
            theta_cache[key] = (r["curve_s_mm"], r["curve_theta_deg"])
        except Exception:
            theta_cache[key] = None
    cached = theta_cache[key]
    if cached is None:
        return None
    cs, cth = cached
    nearest_i = np.argmin(np.abs(cs - s))
    return cth[nearest_i]


fy_board_recon, fx_board_recon = [], []
valid = []
for i in range(len(real_X)):
    theta_pred = theta_at_s(lm_pred[i], phi_pred[i], s_phys[i])
    if theta_pred is None:
        valid.append(False)
        fy_board_recon.append(np.nan)
        fx_board_recon.append(np.nan)
        continue
    theta_rad = np.radians(theta_pred)
    # Fy_board_model(=Fx_total_N) 예측값을 F_mag*cos(theta)로 보고 F_mag 역산 -> Fx_board 재구성
    cos_t = np.cos(theta_rad)
    if abs(cos_t) < 0.05:  # 거의 90도 근처 - 나눗셈 불안정, 이 지점은 직접예측값 유지
        valid.append(False)
        fy_board_recon.append(fy_board_pred_direct[i])
        fx_board_recon.append(fx_board_pred_direct[i])
        continue
    f_mag_est = fy_board_pred_direct[i] / cos_t
    fx_board_recon.append(-f_mag_est * np.sin(theta_rad))
    fy_board_recon.append(fy_board_pred_direct[i])  # 그대로 유지(이미 좋음)
    valid.append(True)

fx_board_recon = np.array(fx_board_recon)
fy_board_recon = np.array(fy_board_recon)
valid = np.array(valid)

print(f"\n[기하학 재구성] 안정적으로 재구성된 케이스: {valid.sum()}/{len(valid)} (나머지는 cos(theta)~0이라 직접예측값 유지)")
print(f"Fx_board 재구성 R^2(전체): {r2_score(real_fy_true, fx_board_recon):.3f}")
print(f"Fx_board 재구성 R^2(안정 케이스만, n={valid.sum()}): "
      f"{r2_score(real_fy_true[valid], fx_board_recon[valid]):.3f}")
print(f"(비교) Fx_board 직접회귀 R^2(같은 안정 케이스만): "
      f"{r2_score(real_fy_true[valid], fx_board_pred_direct[valid]):.3f}")

mae_recon = np.abs(fx_board_recon[valid] - real_fy_true[valid]).mean() * 1000
mae_direct = np.abs(fx_board_pred_direct[valid] - real_fy_true[valid]).mean() * 1000
print(f"MAE: 재구성={mae_recon:.4f}mN vs 직접회귀={mae_direct:.4f}mN")
