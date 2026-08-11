"""
real_deformation_3panel.png에 "모델이 실제로 뭘 예측했는지"를 겹쳐서 보여주는 버전.
실제형상(FEA 실측 팁변위 기반, 강체회전 재구성)은 그대로 두고, 학습된 멀티태스크 모델
(force_v2)을 이 3가지 케이스(L_M=50,phi=60,s=10/30/80mm)에 직접 돌려서
예측 구간(초록 X)과 예측 힘(초록 화살표)을 실제(빨강)와 나란히 표시한다.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold

plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
FEA_DATA_DIR = os.path.join(HERE, "..", "..", "..", "data", "contact_scenarios", "fea")
MODELS_DIR = os.path.join(HERE, "..", "..", "..", "models")
sys.path.insert(0, os.path.join(HERE, "..", "..", "force_model"))
import force_model as fm
import magpylib as magpy
from scipy.spatial.transform import Rotation

OUT_DIR = FEA_DATA_DIR
L_M, PHI = 50.0, 60.0
DEPTH = 0.10
BETA = 0.0
CASES = [
    {"s": 10.0, "Fx": 0.0002019720998922835, "Fy": 6.327430460460517e-05,
     "tip_ux": 0.8806970896739125, "tip_uy": 0.31480696141304343},
    {"s": 30.0, "Fx": 7.149830539204157e-06, "Fy": 3.4980837749775783e-06,
     "tip_ux": 0.2675779315217391, "tip_uy": 0.10061267668478255},
    {"s": 80.0, "Fx": 7.721699922367469e-07, "Fy": 5.7933859862208164e-08,
     "tip_ux": 0.1328898239130435, "tip_uy": 0.044425707282608695},
]
EXAGGERATE = 15
BIN_MID = [10.0, 30.0, 50.0, 70.0, 90.0]
BIN_WIDTH_MM = 20.0
N_CLASSES, N_PROBES = 5, 11
PHI_PROBES = [-150.0, -120.0, -90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0, 120.0, 150.0]


def s_to_bin(s):
    return min(N_CLASSES - 1, int(s / BIN_WIDTH_MM))


# ── 1) 대체모델 재학습 (기존 스크립트들과 동일) ──────────────────────────────
FEATURES = ["L_M_mm", "phi_deg", "beta_deg", "contact_s_mm", "push_depth_mm"]
TARGETS = ["tip_ux_avg_mm", "tip_uy_avg_mm", "tip_uz_avg_mm", "tip_theta_deg_board",
           "Fx_total_N", "Fy_total_N", "Fz_total_N", "F_mag_N"]
DEFAULTS = {"L_M_mm": 50.0, "phi_deg": 60.0, "beta_deg": 0.0}
SOURCES = ["fea_lm_phi_pos_sweep_all.json", "fea_bent_contact_sweep.json",
           "fea_geom_sweep_all.json", "fea_angle_sweep_all.json"]

print("대체모델 재학습 중...")
all_rows = []
for fname in SOURCES:
    for r in json.load(open(os.path.join(FEA_DATA_DIR, fname))):
        row = dict(DEFAULTS)
        row.update(r)
        all_rows.append(row)
X = np.array([[r[f] for f in FEATURES] for r in all_rows])
Y = np.array([[r[t] for t in TARGETS] for r in all_rows])
X_mean, X_std = X.mean(0), X.std(0)
X_std[X_std < 1e-9] = 1.0
Y_mean, Y_std = Y.mean(0), Y.std(0)
Y_std[Y_std < 1e-9] = 1.0
Xn, Yn = (X - X_mean) / X_std, (Y - Y_mean) / Y_std


class SurrogateMLP(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(),
                                  nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, n_out))

    def forward(self, x):
        return self.net(x)


def train_mlp(Xtr, ytr, seed, epochs=2000):
    torch.manual_seed(seed)
    m = SurrogateMLP(Xtr.shape[1], ytr.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    lf = nn.MSELoss()
    Xt, yt = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.float32)
    for _ in range(epochs):
        opt.zero_grad()
        l = lf(m(Xt), yt)
        l.backward()
        opt.step()
    return m


surrogates = [train_mlp(Xn, Yn, seed=i) for i in range(10)]
for m in surrogates:
    m.eval()
print("대체모델 완료")

# ── 2) 분류기(force_v2) 로드 ──────────────────────────────────────────────
class MultiProbeClassifier(nn.Module):
    def __init__(self, n_probes=N_PROBES, n_classes=N_CLASSES, n_force=2):
        super().__init__()
        self.n_probes = n_probes
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Flatten(), nn.Linear(32 * 5 * 5, 64), nn.ReLU(),
        )
        self.trunk = nn.Sequential(nn.Linear(64 * n_probes, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3))
        self.seg_head = nn.Linear(128, n_classes)
        self.force_head = nn.Linear(128, n_force)

    def forward(self, x):
        embeds = [self.encoder(x[:, p]) for p in range(self.n_probes)]
        h = self.trunk(torch.cat(embeds, dim=1))
        return self.seg_head(h), self.force_head(h)


ckpt = torch.load(os.path.join(MODELS_DIR, "position_segment_classifier_multiprobe_150k_11probe_force_v2.pth"),
                   map_location="cpu", weights_only=False)
clf = MultiProbeClassifier()
clf.load_state_dict(ckpt["state_dict"])
clf.eval()
CX_mean, CX_std = ckpt["X_mean"], ckpt["X_std"]
f_mean, f_std = ckpt["f_mean"], ckpt["f_std"]

SENSOR_HEIGHT_MM = 15
sensor_positions = [(x, y, SENSOR_HEIGHT_MM) for y in np.linspace(180, 0, 5) for x in np.linspace(0, 180, 5)]
sensors = magpy.Collection([magpy.Sensor(position=pos) for pos in sensor_positions])
MAGNET_BR_TESLA = 0.36
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


def predict_surrogate(L_M, phi, beta, s, depth):
    x = np.array([[L_M, phi, beta, s, depth]])
    xn = (x - X_mean) / X_std
    xt = torch.tensor(xn, dtype=torch.float32)
    with torch.no_grad():
        pn = np.mean([m(xt).numpy()[0] for m in surrogates], axis=0)
    return dict(zip(TARGETS, pn * Y_std + Y_mean))


print("각 케이스별 모델 예측 계산 중 (11프로브 스캔)...")
free_cache = {}
for case in CASES:
    s = case["s"]
    probes = np.zeros((N_PROBES, 3, 5, 5), dtype=np.float32)
    for pi, phi in enumerate(PHI_PROBES):
        key = (round(L_M, 1), phi)
        if key in free_cache:
            r_free_p = free_cache[key]
        else:
            r_free_p = fm.solve_shape(L_M=L_M, phi_deg=phi, loads=[])
            free_cache[key] = r_free_p
        pred = predict_surrogate(L_M, phi, BETA, s, DEPTH)
        d_xL, d_yL, d_thL = pred["tip_uy_avg_mm"], pred["tip_ux_avg_mm"], -pred["tip_theta_deg_board"]
        frac = L_M / 100.0
        xL_f, yL_f, thL_f = r_free_p["x_L"], r_free_p["y_L"], r_free_p["theta_L_deg"]
        xLM_f, yLM_f, thLM_f = r_free_p["x_LM"], r_free_p["y_LM"], r_free_p["theta_LM_deg"]
        B_free = compute_B(xLM_f, yLM_f, thLM_f, xL_f, yL_f, thL_f)
        B_load = compute_B(xLM_f + d_xL * frac, yLM_f + d_yL * frac, thLM_f + d_thL * frac,
                            xL_f + d_xL, yL_f + d_yL, thL_f + d_thL)
        probes[pi] = (B_load - B_free).reshape(5, 5, 3).transpose(2, 0, 1)
    Xn_ = (probes[None] - CX_mean) / CX_std
    with torch.no_grad():
        seg_logits, force_pred = clf(torch.tensor(Xn_, dtype=torch.float32))
        case["pred_bin"] = int(seg_logits.argmax(dim=1)[0])
        fp = force_pred.numpy()[0] * f_std + f_mean
        case["Fx_pred"], case["Fy_pred"] = float(fp[0]), float(fp[1])
    case["true_bin"] = s_to_bin(s)
    print(f"  s={s}mm: 실제구간={case['true_bin']}번 예측구간={case['pred_bin']}번 "
          f"({'O' if case['true_bin']==case['pred_bin'] else 'X'})")

# ── 3) 플롯 ───────────────────────────────────────────────────────────
r_free = fm.solve_shape(L_M=L_M, phi_deg=PHI, loads=[], return_curve=True)
order0 = np.argsort(r_free["curve_s_mm"])
cs0, cx0, cy0 = r_free["curve_s_mm"][order0], r_free["curve_x_mm"][order0], r_free["curve_y_mm"][order0]
cth0 = r_free["curve_theta_deg"][order0]

fig, axes = plt.subplots(1, 3, figsize=(17, 7.8))
colors = ["#2a78d6", "#eb6834", "#1baf7a"]
for panel, case in enumerate(CASES):
    ax = axes[panel]
    s_mm, Fx, Fy = case["s"], case["Fx"], case["Fy"]

    theta = np.radians(np.interp(s_mm, cs0, cth0))
    normal = np.array([-np.cos(theta), np.sin(theta)])
    push_ang = np.degrees(np.arctan2(Fy, Fx))

    d_x_local, d_y_local = case["tip_uy"], case["tip_ux"]
    pivot = np.array([np.interp(s_mm, cs0, cx0), np.interp(s_mm, cs0, cy0)])
    tip0 = np.array([cx0[-1], cy0[-1]])
    tip_target = tip0 + np.array([d_x_local, d_y_local]) * EXAGGERATE
    v0, v1 = tip0 - pivot, tip_target - pivot
    dtheta = np.arctan2(v1[1], v1[0]) - np.arctan2(v0[1], v0[0])
    R = np.array([[np.cos(dtheta), -np.sin(dtheta)], [np.sin(dtheta), np.cos(dtheta)]])
    mask_after = cs0 >= s_mm
    cx_ex, cy_ex = cx0.copy(), cy0.copy()
    rel = np.stack([cx0[mask_after] - pivot[0], cy0[mask_after] - pivot[1]], axis=1) @ R.T
    cx_ex[mask_after] = pivot[0] + rel[:, 0]
    cy_ex[mask_after] = pivot[1] + rel[:, 1]

    ax.plot(cx0, cy0, "--", color="#9aa5ab", linewidth=2, label="무접촉 기준형상", zorder=2)
    ax.plot(cx_ex, cy_ex, "-", color=colors[panel], linewidth=2.6,
            label=f"접촉 시 실제형상(FEA, {EXAGGERATE}배 확대)", zorder=3)
    ax.plot(0, 0, "ks", markersize=10, zorder=5, label="베이스(고정단)")

    xt, yt = np.interp(s_mm, cs0, cx0), np.interp(s_mm, cs0, cy0)
    ax.plot(xt, yt, "o", color="#e34948", markersize=12, zorder=7,
            markeredgecolor="white", markeredgewidth=1.4, label=f"실제 접촉 (s={s_mm:.0f}mm)")

    # 예측 구간(초록 X): 예측된 구간의 중앙값 위치에 표시
    s_pred = BIN_MID[case["pred_bin"]]
    xp, yp = np.interp(s_pred, cs0, cx0), np.interp(s_pred, cs0, cy0)
    ax.plot(xp, yp, "x", color="#1E7B34", markersize=16, mew=3.2, zorder=8,
            label=f"예측 구간 (~{s_pred:.0f}mm)")
    if case["true_bin"] != case["pred_bin"]:
        ax.plot([xt, xp], [yt, yp], ":", color="#444444", linewidth=1.2, zorder=4)

    # 실제 힘(빨강) vs 예측 힘(초록) 화살표 - 둘 다 실제 접촉위치에서 시작
    f_arrow_len = 16
    f_true_dir = np.array([Fx, Fy]) / (np.hypot(Fx, Fy) + 1e-15)
    ax.annotate("", xy=(xt + f_true_dir[0] * f_arrow_len, yt + f_true_dir[1] * f_arrow_len), xytext=(xt, yt),
                arrowprops=dict(arrowstyle="-|>", color="#c0392b", linewidth=2.6, mutation_scale=18), zorder=9)
    Fx_p, Fy_p = case["Fx_pred"], case["Fy_pred"]
    f_pred_dir = np.array([Fx_p, Fy_p]) / (np.hypot(Fx_p, Fy_p) + 1e-15)
    ax.annotate("", xy=(xt + f_pred_dir[0] * f_arrow_len, yt + f_pred_dir[1] * f_arrow_len), xytext=(xt, yt),
                arrowprops=dict(arrowstyle="-|>", color="#1E7B34", linewidth=2.6, mutation_scale=18,
                                 linestyle=(0, (3, 2))), zorder=9)
    ax.plot([], [], "-", color="#c0392b", linewidth=2.4, label=f"실제 힘 F={np.hypot(Fx,Fy)*1000:.3f}mN")
    ax.plot([], [], "--", color="#1E7B34", linewidth=2.4, label=f"예측 힘 F={np.hypot(Fx_p,Fy_p)*1000:.3f}mN")

    seg_ok = "O" if case["true_bin"] == case["pred_bin"] else "X"
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.set_xlabel("x (mm, 보드좌표)")
    ax.set_ylabel("y (mm, 보드좌표)")
    ax.set_title(f"s={s_mm:.0f}mm, {DEPTH:.2f}mm 누름 (미는방향 {push_ang:.0f}도) — 구간판정 [{seg_ok}]\n"
                 f"실제구간={case['true_bin']}번 vs 예측구간={case['pred_bin']}번",
                 fontweight="bold", fontsize=11)
    ax.legend(fontsize=7.5, loc="best")

fig.suptitle(f"실제 형상 vs 모델 예측 — 구간 판정 + 힘 예측 (같은 깊이 {DEPTH:.2f}mm)",
             fontweight="bold", fontsize=13, y=1.03)
out = os.path.join(OUT_DIR, "real_deformation_with_prediction.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"저장: {out}")
