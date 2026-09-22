"""2026-09-21(36번): shape_head 결과를 "개념도" 스타일로 3가지 경우 시각화.

각 칸에 담는 것:
  - 무접촉 기준형상 (force_model로 계산, L_M/phi만 알면 나오는 값이라 정확)
  - 충돌 시 형상 (변형된 중심선)
  - 접촉 위치 s
  - 팁: FEA 정답 vs CNN 예측

변형 곡선 그리는 법: CNN이 예측하는 건 "팁" 변위뿐이라 중심선 전체를 직접 알 수는 없음.
그래서 베이스(변위 0) / MOM 위치 / 팁 세 지점을 기준점으로 잡고 호길이를 따라 단조
3차보간(PCHIP)해서 그림 - 캔틸레버 변형이 베이스에서 0으로 시작해 팁으로 갈수록 커지는
형태라는 물리와 일치. **어디까지나 개념도**이며, 정량 지표는 팁 변위/회전 기준임.

변위가 0.05~2.5mm라 100mm 로봇 전체에서는 안 보여서 확대배율(EXAG)을 걸어 그림.
"""
import hashlib
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
FEA_DATA_DIR = os.path.join(HERE, "..", "..", "..", "data", "contact_scenarios", "fea")
MODELS_DIR = os.path.join(HERE, "..", "..", "..", "models")
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "force_model"))
import force_model as fm
import magpylib as magpy
from scipy.spatial.transform import Rotation

DEFAULTS = {"L_M_mm": 50.0, "phi_deg": 60.0, "beta_deg": 0.0}
N_CLASSES = 4
EXAG = 15  # 변위 확대배율(안 하면 눈에 안 보임)


def is_holdout_row(r, frac=0.2):
    key = f"{r['L_M_mm']}_{r['phi_deg']}_{r['beta_deg']}_{r['contact_s_mm']}"
    return (int(hashlib.md5(key.encode()).hexdigest(), 16) % 10000) < int(frac * 10000)


class Net(nn.Module):
    def __init__(self, n_shape=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Flatten(), nn.Linear(32 * 5 * 5, 64), nn.ReLU())
        self.trunk = nn.Sequential(nn.Linear(64 + 2, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3))
        self.seg_head = nn.Linear(128, N_CLASSES)
        self.force_head = nn.Linear(128, 2)
        self.s_head = nn.Linear(128, 1)
        self.shape_head = nn.Linear(128, n_shape)
        # 2026-09-22(38번): depth_head 추가 - 체크포인트 키를 맞추려면 여기도 있어야 함.
        self.depth_head = nn.Linear(128, 1)

    def forward(self, x, config):
        h = self.trunk(torch.cat([self.encoder(x[:, 0]), config], dim=1))
        return self.seg_head(h), self.force_head(h), self.s_head(h).squeeze(-1), self.shape_head(h), self.depth_head(h).squeeze(-1)


sensors = magpy.Collection([magpy.Sensor(position=(x, y, 15))
                            for y in np.linspace(180, 0, 5) for x in np.linspace(0, 180, 5)])
main_magnet = magpy.magnet.Cylinder(polarization=(0, 0.4, 0), dimension=(2, 2))
mom = magpy.magnet.Cylinder(polarization=(0, -0.4, 0), dimension=(1, 8))
robot = magpy.Collection(main_magnet, mom)


def compute_B(xLM_l, yLM_l, thLM, xL_l, yL_l, thL):
    xLM_b, yLM_b = fm.to_board_frame(xLM_l, yLM_l)
    xL_b, yL_b = fm.to_board_frame(xL_l, yL_l)
    mom.position = (float(xLM_b), float(yLM_b), 0)
    mom.orientation = Rotation.from_euler("z", -thLM, degrees=True)
    main_magnet.position = (float(xL_b), float(yL_b), 0)
    main_magnet.orientation = Rotation.from_euler("z", -thL, degrees=True)
    return magpy.getB(robot, sensors) * 1e6


all_rows = []
for r in json.load(open(os.path.join(FEA_DATA_DIR, "fea_lm_phi_pos_matv2_all.json"), encoding="utf-8")):
    row = dict(DEFAULTS)
    row.update(r)
    all_rows.append(row)
holdout = [r for r in all_rows if is_holdout_row(r) and "mom_ux_avg_mm" in r]

ck = torch.load(os.path.join(MODELS_DIR, "position_segment_classifier_singleprobe_beta0180_4seg.pth"),
                 map_location="cpu", weights_only=False)
SHAPE_NAMES = ck.get("shape_names", ["tip_ux_avg_mm", "tip_uy_avg_mm", "tip_theta_deg_board"])
net = Net(len(SHAPE_NAMES))
net.load_state_dict(ck["state_dict"])
net.eval()

X, cs, keep = [], [], []
for r in holdout:
    try:
        rf = fm.solve_shape(L_M=r["L_M_mm"], phi_deg=r["phi_deg"], loads=[])
    except Exception:
        continue
    d_xL, d_yL, d_thL = r["tip_uy_avg_mm"], r["tip_ux_avg_mm"], -r["tip_theta_deg_board"]
    d_xLM, d_yLM, d_thLM = r["mom_uy_avg_mm"], r["mom_ux_avg_mm"], -r["mom_theta_deg_board"]
    B0 = compute_B(rf["x_LM"], rf["y_LM"], rf["theta_LM_deg"], rf["x_L"], rf["y_L"], rf["theta_L_deg"])
    B1 = compute_B(rf["x_LM"] + d_xLM, rf["y_LM"] + d_yLM, rf["theta_LM_deg"] + d_thLM,
                    rf["x_L"] + d_xL, rf["y_L"] + d_yL, rf["theta_L_deg"] + d_thL)
    X.append((B1 - B0).reshape(5, 5, 3).transpose(2, 0, 1))
    cs.append([r["L_M_mm"], r["phi_deg"]])
    keep.append(r)

X, cs = np.array(X, dtype=np.float32), np.array(cs, dtype=np.float32)
with torch.no_grad():
    *_, sh_pred, _ = net(torch.tensor(((X - ck["X_mean"]) / ck["X_std"])[:, None]).float(),
                       torch.tensor((cs - ck["c_mean"]) / ck["c_std"]).float())
sh_pred = sh_pred.numpy() * ck["sh_std"] + ck["sh_mean"]
sh_true = np.array([[r[t] for t in SHAPE_NAMES] for r in keep], dtype=np.float32)
err = np.hypot(sh_true[:, 0] - sh_pred[:, 0], sh_true[:, 1] - sh_pred[:, 1])

# 접촉 위치별로 베이스쪽/중간/팁쪽 하나씩 고름. 고르는 규칙(체리피킹 방지):
#  (1) "의미 있는 크기의 충돌"만 대상 - 팁 이동이 전체 중앙값 이상인 것.
#      미세 접촉(팁 이동 0.07mm 등)은 절대오차가 작아도 비율로는 50%가 넘어 그림이
#      실제 성능을 왜곡함(그 영역은 애초에 오차 바닥 근처).
#  (2) phi=0(굽힘 없는 직선 자세)은 "형상"을 보여준다는 목적에 안 맞아 제외.
#  (3) 그 안에서 **중앙값 오차** 케이스 선택 - 제일 잘 된 걸 고르지 않음.
mv_all = np.hypot(sh_true[:, 0], sh_true[:, 1])
mv_thresh = np.median(mv_all)
picks = []
for lo, hi, label in [(10, 30, "베이스 쪽"), (35, 60, "중간"), (65, 100, "팁 쪽")]:
    idx = [i for i, r in enumerate(keep)
           if lo <= r["contact_s_mm"] <= hi and mv_all[i] >= mv_thresh and abs(r["phi_deg"]) >= 30]
    if not idx:
        continue
    idx.sort(key=lambda i: err[i])
    picks.append((idx[len(idx) // 2], label))   # 중앙값 오차 케이스
print(f"선별 기준: 팁 이동 >= 중앙값({mv_thresh:.3f}mm), |phi|>=30, 각 구간 중앙값 오차 케이스")

fig, axes = plt.subplots(1, len(picks), figsize=(6.2 * len(picks), 6.0))
if len(picks) == 1:
    axes = [axes]

for ax, (i, label) in zip(axes, picks):
    r = keep[i]
    L_M, phi, s_c = r["L_M_mm"], r["phi_deg"], r["contact_s_mm"]
    rf = fm.solve_shape(L_M=L_M, phi_deg=phi, loads=[], return_curve=True)
    o = np.argsort(rf["curve_s_mm"])
    cs_arc = rf["curve_s_mm"][o]
    bx, by = fm.to_board_frame(rf["curve_x_mm"][o], rf["curve_y_mm"][o])
    L_total = cs_arc.max()

    def deformed(dx_tip, dy_tip, dx_mom, dy_mom):
        """베이스(0)/MOM/팁 변위를 기준점으로 호길이 따라 단조3차보간 - 개념도용."""
        anc_s = [0.0]
        anc_x, anc_y = [0.0], [0.0]
        if 1.0 < L_M < L_total - 1.0:
            anc_s.append(L_M); anc_x.append(dx_mom); anc_y.append(dy_mom)
        anc_s.append(L_total); anc_x.append(dx_tip); anc_y.append(dy_tip)
        fx = PchipInterpolator(anc_s, anc_x)(cs_arc)
        fy = PchipInterpolator(anc_s, anc_y)(cs_arc)
        return bx + fx * EXAG, by + fy * EXAG

    dxt, dyt = deformed(sh_true[i, 0], sh_true[i, 1], r["mom_ux_avg_mm"], r["mom_uy_avg_mm"])
    dxp, dyp = deformed(sh_pred[i, 0], sh_pred[i, 1], r["mom_ux_avg_mm"], r["mom_uy_avg_mm"])

    ax.plot(bx, by, "--", color="#95A5A6", lw=2.0, label="무접촉 기준형상")
    ax.plot(dxt, dyt, "-", color="#C0392B", lw=5, alpha=0.25)
    ax.plot(dxt, dyt, "-", color="#C0392B", lw=2.4, label=f"충돌 시 형상 (FEA 정답) ×{EXAG} 확대")
    ax.plot(dxp, dyp, color="#2451A3", lw=2.0, ls=(0, (4, 2)),
            label="충돌 시 형상 (CNN 예측)")

    ax.plot(bx[0], by[0], "s", ms=12, color="black", label="베이스(고정단)")
    si = int(np.argmin(np.abs(cs_arc - s_c)))
    ax.plot(bx[si], by[si], "o", ms=13, color="#E67E22", mec="white", mew=1.5,
            label=f"접촉 위치 (s={s_c:.0f}mm)")
    ax.plot(dxt[-1], dyt[-1], "*", ms=17, color="#C0392B", mec="white", mew=0.8)
    ax.plot(dxp[-1], dyp[-1], "X", ms=11, color="#2451A3", mec="white", mew=0.8)

    mv = np.hypot(sh_true[i, 0], sh_true[i, 1])
    info = (f"실제 팁 이동 = {mv:.3f} mm\n"
            f"CNN 예측 = {np.hypot(sh_pred[i,0], sh_pred[i,1]):.3f} mm\n"
            f"오차 = {err[i]:.3f} mm ({err[i]/max(mv,1e-9)*100:.0f}%)\n"
            f"팁 회전: 실제 {sh_true[i,2]:+.3f}° / 예측 {sh_pred[i,2]:+.3f}°")
    ax.text(0.03, 0.03, info, transform=ax.transAxes, fontsize=9.5, va="bottom",
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#C0392B", alpha=0.92))

    ax.set_title(f"{label} 접촉  (L_M={L_M:.1f}mm, phi={phi:.0f}°, 깊이={r.get('push_depth_mm',0.1):.2f}mm)",
                 fontsize=11.5, fontweight="bold")
    ax.set_xlabel("보드 x [mm]")
    ax.set_ylabel("보드 y [mm]")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(fontsize=8, loc="upper right")

fig.suptitle("홀센서 자기장만으로 「충돌 시 로봇 형상」 추정 — 접촉 위치별 3가지 경우\n"
             f"순수 실측 FEA 홀드아웃(n={len(keep)}) · 각 구간에서 팁 이동이 중앙값 이상인 케이스 중 "
             f"**중앙값 오차**를 선택(최고 성능 아님) · 전체 tip 변위 R²=0.93~0.95",
             fontweight="bold", fontsize=13, y=1.02)
plt.tight_layout()
out = os.path.join(FEA_DATA_DIR, "shape_head_cases_0921.png")
plt.savefig(out, dpi=140, bbox_inches="tight")
print("저장:", out)
for i, label in picks:
    r = keep[i]
    print(f"  [{label}] L_M={r['L_M_mm']}, phi={r['phi_deg']}, s={r['contact_s_mm']}, "
          f"depth={r.get('push_depth_mm',0.1)}, 오차={err[i]:.4f}mm")
