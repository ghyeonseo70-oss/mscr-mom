"""s(접촉위치) 예측 오차가 어디서 집중되는지 확인. 전체 R^2=0.899로 이미 괜찮지만, 더 올리려면
어디가 약한지부터 알아야 함 - Fx_board/L_M 때와 같은 방식(원인 먼저 확인 후 처방).

확인 축:
1. |phi|<90 vs |phi|>=90 (Fx_board가 나쁜 그 구간과 s도 겹치는지)
2. s값 구간별(0-20/20-40/40-60/60-80mm) - 옛 교훈: s가 클수록(팁 쪽) FEA 자체가 지렛대효과로
   신호가 약해져서 수렴 실패율이 높았음 -> 그쪽 실측 데이터가 상대적으로 부족/불안정할 가능성
3. 구간(segment) 경계 근처(± 3mm) vs 안쪽 - 분류 혼동이 s 회귀에도 전이되는지
4. L_M값별 - L_M=0 근처가 s 예측도 끌고 내려가는지(L_M=0는 이미 알려진 문제 구간)
"""
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
BIN_WIDTH_MM = 20.0
BOUNDARY_MARGIN_MM = 3.0


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
print(f"실측 홀드아웃: n={len(real_holdout_rows)}")

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
        self.lm_zero_head = nn.Linear(128, 1)

    def forward(self, x):
        embeds = [self.encoder(x[:, p]) for p in range(self.n_probes)]
        h = self.trunk(torch.cat(embeds, dim=1))
        return (self.seg_head(h), self.force_head(h), self.s_head(h).squeeze(-1), self.config_head(h),
                self.lm_zero_head(h).squeeze(-1))


cnn = SingleProbeClassifier()
cnn.load_state_dict(ckpt["state_dict"])
cnn.eval()
X_mean2, X_std2 = ckpt["X_mean"], ckpt["X_std"]
s_mean, s_std = ckpt["s_mean"], ckpt["s_std"]

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


real_X, meta = [], []
for r in real_holdout_rows:
    L_M, phi, s = r["L_M_mm"], r["phi_deg"], r["contact_s_mm"]
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
    seg_class = min(N_CLASSES - 1, int(s / BIN_WIDTH_MM))
    dist_to_boundary = min(s % BIN_WIDTH_MM, BIN_WIDTH_MM - (s % BIN_WIDTH_MM))
    meta.append({"s": s, "phi": phi, "L_M": L_M, "high_phi": abs(phi) >= 90,
                 "seg_class": seg_class, "near_boundary": dist_to_boundary < BOUNDARY_MARGIN_MM})

real_X = np.array(real_X, dtype=np.float32)
real_X_norm = (real_X - X_mean2) / X_std2
s_true = np.array([m["s"] for m in meta])

with torch.no_grad():
    rX = torch.tensor(real_X_norm[:, None]).float()
    _, _, r_s_pred, _, _ = cnn(rX)
    s_pred = r_s_pred.numpy() * s_std + s_mean

print(f"\nfree-shape 계산 성공: {len(real_X)}/{len(real_holdout_rows)}")


def r2(pred, true):
    ss_res = np.sum((pred - true) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def report(mask, label):
    n = mask.sum()
    if n < 3:
        print(f"  [{label}] n={n} - 너무 적어서 R^2 생략")
        return
    mae = np.mean(np.abs(s_pred[mask] - s_true[mask]))
    print(f"  [{label}] n={n}  R^2={r2(s_pred[mask], s_true[mask]):.3f}  MAE={mae:.2f}mm")


high_phi = np.array([m["high_phi"] for m in meta])
near_boundary = np.array([m["near_boundary"] for m in meta])
seg_class = np.array([m["seg_class"] for m in meta])
L_M_arr = np.array([m["L_M"] for m in meta])

print("\n=== 전체 ===")
report(np.ones(len(meta), dtype=bool), "전체")

print("\n=== phi 구간별 ===")
report(~high_phi, "|phi|<90")
report(high_phi, "|phi|>=90")

print("\n=== s 구간별(BIN_WIDTH_MM 기준 구간과 동일, 마지막 구간은 min() clamp로 상한 없음) ===")
s_max_actual = s_true.max()
for c in range(N_CLASSES):
    lo = c * BIN_WIDTH_MM
    hi = (c + 1) * BIN_WIDTH_MM if c < N_CLASSES - 1 else s_max_actual
    report(seg_class == c, f"s {lo:.0f}-{hi:.0f}mm (구간{c})")

print("\n=== 구간 경계(±3mm) 근처 vs 안쪽 ===")
report(near_boundary, "경계 근처")
report(~near_boundary, "구간 안쪽")

print("\n=== L_M값별 ===")
for lm_val in sorted(set(L_M_arr.tolist())):
    report(L_M_arr == lm_val, f"L_M={lm_val:.1f}mm")

print("\n=== 교차: L_M=0 & phi 구간 ===")
report((L_M_arr == 0) & ~high_phi, "L_M=0 & |phi|<90")
report((L_M_arr == 0) & high_phi, "L_M=0 & |phi|>=90")
report((L_M_arr != 0) & high_phi, "L_M!=0 & |phi|>=90")
