"""L_M 실측 홀드아웃 예측 오차를 L_M 개별 값별로 쪼개서 확인 - 2026-08-26 발견:
전체 데이터 개수는 L_M값마다 비슷한데(60~75개), L_M=0mm만 유독 MAE가 크게(다른 값의
4배 이상) 나쁨. 가운데(25~62.5mm)는 다 좋고 양쪽 끝(0, 87.5mm)으로 갈수록 나빠지는
전형적 회귀모델의 "범위 경계 효과"인데, L_M=0은 그 경계효과를 훨씬 넘어서는 수준 -
K1(베이스~MOM) 구간 길이가 정확히 0이 되는 구조적 특이점이기 때문으로 추정.
재학습할 때마다 이 스크립트로 패턴이 재현되는지 확인할 것."""
import hashlib
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

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
real_holdout_rows = [r for r in all_rows if is_holdout_row(r)]

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
c_mean, c_std = ckpt["c_mean"], ckpt["c_std"]
config_names = ckpt["config_names"]

SENSOR_HEIGHT_MM = 15
sensor_positions = [(x, y, SENSOR_HEIGHT_MM) for y in np.linspace(180, 0, 5) for x in np.linspace(0, 180, 5)]
sensors = magpy.Collection([magpy.Sensor(position=pos) for pos in sensor_positions])
main_magnet = magpy.magnet.Cylinder(polarization=(0, 0.4, 0), dimension=(2, 2))
mom = magpy.magnet.Cylinder(polarization=(0, -0.4, 0), dimension=(1, 8))
mscr_robot = magpy.Collection(main_magnet, mom)


def compute_B(xLM_l, yLM_l, thLM, xL_l, yL_l, thL):
    xLM_b, yLM_b = fm.to_board_frame(xLM_l, yLM_l)
    xL_b, yL_b = fm.to_board_frame(xL_l, yL_l)
    mom.position = (float(xLM_b), float(yLM_b), 0)
    mom.orientation = Rotation.from_euler("z", -thLM, degrees=True)
    main_magnet.position = (float(xL_b), float(yL_b), 0)
    main_magnet.orientation = Rotation.from_euler("z", -thL, degrees=True)
    return magpy.getB(mscr_robot, sensors) * 1e6


real_X, real_c = [], []
for r in real_holdout_rows:
    L_M, phi = r["L_M_mm"], r["phi_deg"]
    try:
        r_free = fm.solve_shape(L_M=L_M, phi_deg=phi, loads=[])
    except Exception:
        continue
    d_xL_local, d_yL_local = r["tip_uy_avg_mm"], r["tip_ux_avg_mm"]
    d_thL = -r["tip_theta_deg_board"]
    if "mom_ux_avg_mm" in r:
        d_xLM_local, d_yLM_local, d_thLM = r["mom_uy_avg_mm"], r["mom_ux_avg_mm"], -r["mom_theta_deg_board"]
    else:
        frac = L_M / 100.0
        d_xLM_local, d_yLM_local, d_thLM = d_xL_local * frac, d_yL_local * frac, d_thL * frac
    xL_free, yL_free, thL_free = r_free["x_L"], r_free["y_L"], r_free["theta_L_deg"]
    xLM_free, yLM_free, thLM_free = r_free["x_LM"], r_free["y_LM"], r_free["theta_LM_deg"]
    B_free = compute_B(xLM_free, yLM_free, thLM_free, xL_free, yL_free, thL_free)
    B_load = compute_B(xLM_free + d_xLM_local, yLM_free + d_yLM_local, thLM_free + d_thLM,
                        xL_free + d_xL_local, yL_free + d_yL_local, thL_free + d_thL)
    real_X.append((B_load - B_free).reshape(5, 5, 3).transpose(2, 0, 1))
    real_c.append([L_M, phi])

real_X = np.array(real_X, dtype=np.float32)
real_c_arr = np.array(real_c, dtype=np.float32)
real_X_norm = (real_X - X_mean2) / X_std2
with torch.no_grad():
    rX = torch.tensor(real_X_norm[:, None]).float()
    _, _, _, r_config_pred, r_lm_zero_logit = cnn(rX)
    r_config_phys = r_config_pred.numpy() * c_std + c_mean
    r_lm_zero_pred = (r_lm_zero_logit.numpy() > 0)

lm_idx = config_names.index("L_M_mm")
lm_pred = r_config_phys[:, lm_idx]
lm_true = real_c_arr[:, lm_idx]

print(f"{'L_M':>6} {'n':>3} {'MAE':>8}   예측범위")
for lm_val in sorted(set(lm_true)):
    mask = lm_true == lm_val
    n = mask.sum()
    mae = np.abs(lm_pred[mask] - lm_true[mask]).mean()
    print(f"{lm_val:6.1f} {n:3d} {mae:8.2f}   [{lm_pred[mask].min():.1f}, {lm_pred[mask].max():.1f}]")

# 2026-08-27 추가: lm_zero_head(L_M<임계값 분류) 성능 + 하이브리드(분류+회귀) L_M
LM_ZERO_THRESHOLD_MM = ckpt.get("lm_zero_threshold_mm", 6.25)
true_zero = lm_true < LM_ZERO_THRESHOLD_MM
zero_acc = (r_lm_zero_pred == true_zero).mean()
hybrid = np.where(r_lm_zero_pred, 0.0, lm_pred)
mae_h = np.abs(hybrid - lm_true).mean()
print(f"\nlm_zero_head 분류 정확도={zero_acc*100:.1f}% (임계값 {LM_ZERO_THRESHOLD_MM}mm)")
print(f"하이브리드(분류+회귀) L_M 전체 MAE={mae_h:.2f}mm (순수 회귀 위 표 대비 개선 여부 확인)")

print(f"\n{'L_M':>6} {'n':>3} {'회귀MAE':>8} {'하이브리드MAE':>12}   (L_M별 세분화)")
for lm_val in sorted(set(lm_true)):
    mask = lm_true == lm_val
    n = mask.sum()
    mae_reg = np.abs(lm_pred[mask] - lm_true[mask]).mean()
    mae_hyb = np.abs(hybrid[mask] - lm_true[mask]).mean()
    print(f"{lm_val:6.1f} {n:3d} {mae_reg:8.2f} {mae_hyb:12.2f}")
