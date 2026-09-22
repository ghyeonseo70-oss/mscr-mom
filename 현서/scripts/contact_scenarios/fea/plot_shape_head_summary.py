"""2026-09-21(36번): shape_head 결과 요약 그림.

위 3칸  - 산점도: 예측 vs 실제 (tip_ux, tip_uy, tip_theta), 순수 실측 FEA 홀드아웃 기준.
아래 3칸 - 실제로 추정한 "형상"이 어떻게 생겼는지: 로봇 중심선(보드좌표계)과, 충돌로
           팁이 움직인 결과를 자유형상/FEA정답/CNN예측 세 가지로 겹쳐 그림.
           변위가 0.05~1mm 수준이라 전체 그림에서는 안 보여서 팁 주변을 확대해 같이 표시.
"""
import hashlib
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from sklearn.metrics import r2_score

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
FEA_DATA_DIR = os.path.join(HERE, "..", "..", "..", "data", "contact_scenarios", "fea")
MODELS_DIR = os.path.join(HERE, "..", "..", "..", "models")
HYUNSEO_DIR = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
REPO_ROOT = os.path.abspath(os.path.join(HYUNSEO_DIR, ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "force_model"))
import force_model as fm
import magpylib as magpy
from scipy.spatial.transform import Rotation

DEFAULTS = {"L_M_mm": 50.0, "phi_deg": 60.0, "beta_deg": 0.0}
N_CLASSES = 4


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
holdout = [r for r in all_rows if is_holdout_row(r)]

ck = torch.load(os.path.join(MODELS_DIR, "position_segment_classifier_singleprobe_beta0180_4seg.pth"),
                 map_location="cpu", weights_only=False)
SHAPE_NAMES = ck.get("shape_names", ["tip_ux_avg_mm", "tip_uy_avg_mm", "tip_theta_deg_board"])
net = Net(n_shape=len(SHAPE_NAMES))
net.load_state_dict(ck["state_dict"])
net.eval()

X, cs, sh_true, keep = [], [], [], []
for r in holdout:
    try:
        rf = fm.solve_shape(L_M=r["L_M_mm"], phi_deg=r["phi_deg"], loads=[])
    except Exception:
        continue
    d_xL, d_yL, d_thL = r["tip_uy_avg_mm"], r["tip_ux_avg_mm"], -r["tip_theta_deg_board"]
    if "mom_ux_avg_mm" in r:
        d_xLM, d_yLM, d_thLM = r["mom_uy_avg_mm"], r["mom_ux_avg_mm"], -r["mom_theta_deg_board"]
    else:
        fr = r["L_M_mm"] / 100.0
        d_xLM, d_yLM, d_thLM = d_xL * fr, d_yL * fr, d_thL * fr
    B0 = compute_B(rf["x_LM"], rf["y_LM"], rf["theta_LM_deg"], rf["x_L"], rf["y_L"], rf["theta_L_deg"])
    B1 = compute_B(rf["x_LM"] + d_xLM, rf["y_LM"] + d_yLM, rf["theta_LM_deg"] + d_thLM,
                    rf["x_L"] + d_xL, rf["y_L"] + d_yL, rf["theta_L_deg"] + d_thL)
    X.append((B1 - B0).reshape(5, 5, 3).transpose(2, 0, 1))
    cs.append([r["L_M_mm"], r["phi_deg"]])
    sh_true.append([r[t] for t in SHAPE_NAMES])
    keep.append(r)

X = np.array(X, dtype=np.float32)
cs = np.array(cs, dtype=np.float32)
sh_true = np.array(sh_true, dtype=np.float32)
with torch.no_grad():
    _, _, s_pred, sh_pred, _ = net(torch.tensor(((X - ck["X_mean"]) / ck["X_std"])[:, None]).float(),
                                 torch.tensor((cs - ck["c_mean"]) / ck["c_std"]).float())
sh_pred = sh_pred.numpy() * ck["sh_std"] + ck["sh_mean"]
s_pred = s_pred.numpy() * ck["s_std"] + ck["s_mean"]
print(f"홀드아웃 {len(keep)}행 평가")

# ---- 그림 ----
fig = plt.figure(figsize=(16.5, 10))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.15], hspace=0.32, wspace=0.28)

LABELS = [("tip_ux (팁 x변위)", "mm"), ("tip_uy (팁 y변위)", "mm"), ("tip_theta (팁 회전)", "deg")]
COLORS = ["#2451A3", "#27AE60", "#C0392B"]
for i, ((label, unit), color) in enumerate(zip(LABELS, COLORS)):
    ax = fig.add_subplot(gs[0, i])
    t, p = sh_true[:, i], sh_pred[:, i]
    lo, hi = min(t.min(), p.min()), max(t.max(), p.max())
    pad = (hi - lo) * 0.06
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "r--", lw=1.2, label="y=x (완벽예측)")
    ax.scatter(t, p, s=34, alpha=0.65, color=color, edgecolor="black", linewidth=0.3)
    ax.set_xlabel(f"실제 {label} [{unit}]  (FEA 정답)")
    ax.set_ylabel(f"CNN 예측 {label} [{unit}]")
    ax.set_title(f"{label}\nR²={r2_score(t, p):.3f},  MAE={np.abs(t-p).mean():.4f}{unit}",
                 fontweight="bold", fontsize=11.5)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, ls=":", alpha=0.4)

# 아래: 실제 형상 예시 3개 - 변위가 큰 순으로 골라 눈에 보이게
mag = np.hypot(sh_true[:, 0], sh_true[:, 1])
picks = np.argsort(-mag)[[0, len(mag) // 6, len(mag) // 3]]
for j, idx in enumerate(picks):
    r = keep[idx]
    ax = fig.add_subplot(gs[1, j])
    rf = fm.solve_shape(L_M=r["L_M_mm"], phi_deg=r["phi_deg"], loads=[], return_curve=True)
    order = np.argsort(rf["curve_s_mm"])
    cx, cy = rf["curve_x_mm"][order], rf["curve_y_mm"][order]
    bx, by = fm.to_board_frame(cx, cy)
    ax.plot(bx, by, "-", color="#7F8C8D", lw=3, alpha=0.85, label="로봇 중심선(무접촉)")

    # 접촉 지점
    si = np.argmin(np.abs(rf["curve_s_mm"][order] - r["contact_s_mm"]))
    ax.plot(bx[si], by[si], "o", ms=11, mfc="none", mec="#E67E22", mew=2.5,
            label=f"접촉 지점 s={r['contact_s_mm']:.0f}mm")

    # 팁: 자유 / FEA 정답 / CNN 예측 (변위가 작아 EXAG배 확대해 표시)
    EXAG = 20
    tipx, tipy = fm.to_board_frame(rf["x_L"], rf["y_L"])
    tx_t, ty_t = tipx + sh_true[idx, 0] * EXAG, tipy + sh_true[idx, 1] * EXAG
    tx_p, ty_p = tipx + sh_pred[idx, 0] * EXAG, tipy + sh_pred[idx, 1] * EXAG
    ax.plot(tipx, tipy, "o", ms=9, color="#7F8C8D", label="팁(무접촉)")
    ax.plot(tx_t, ty_t, "*", ms=19, color="#C0392B", label=f"팁(FEA 정답) ×{EXAG} 확대")
    ax.plot(tx_p, ty_p, "X", ms=12, color="#2451A3", label="팁(CNN 예측)")
    ax.annotate("", xy=(tx_t, ty_t), xytext=(tipx, tipy),
                arrowprops=dict(arrowstyle="->", color="#C0392B", lw=1.6, alpha=0.8))

    err = np.hypot(sh_true[idx, 0] - sh_pred[idx, 0], sh_true[idx, 1] - sh_pred[idx, 1])
    ax.set_title(f"L_M={r['L_M_mm']:.1f}mm, phi={r['phi_deg']:.0f}°, 깊이={r.get('push_depth_mm',0.1):.2f}mm\n"
                 f"실제 팁 이동 {np.hypot(*sh_true[idx,:2]):.3f}mm → 예측 오차 {err:.3f}mm",
                 fontsize=10.5, fontweight="bold")
    ax.set_xlabel("보드 x [mm]")
    ax.set_ylabel("보드 y [mm]")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(fontsize=7.5, loc="best")

fig.suptitle("2026-09-21 shape_head: 홀센서 자기장만으로 「충돌 시 로봇 형상」 추정\n"
             f"순수 실측 FEA 홀드아웃 n={len(keep)} (대체모델 안 거친 진짜 물리 기준)",
             fontweight="bold", fontsize=14.5, y=0.985)
out = os.path.join(FEA_DATA_DIR, "shape_head_summary_0921.png")
plt.savefig(out, dpi=140, bbox_inches="tight")
print("저장:", out)
