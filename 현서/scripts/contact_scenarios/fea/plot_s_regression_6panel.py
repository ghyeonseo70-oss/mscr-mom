"""
보조회귀(연속값 s) 모델의 실제 vs 예측을 6개 사례로 시각화. 각 사례는 진짜 FEA 데이터
(다양한 L_M/phi/s)에서 뽑았고, 그 케이스 고유의 굽힘 형상을 기준으로 그린다.
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

plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
FEA_DATA_DIR = os.path.join(HERE, "..", "..", "..", "data", "contact_scenarios", "fea")
MODELS_DIR = os.path.join(HERE, "..", "..", "..", "models")
sys.path.insert(0, os.path.join(HERE, "..", "..", "force_model"))
import force_model as fm
import magpylib as magpy
from scipy.spatial.transform import Rotation

N_CLASSES, N_PROBES = 5, 11
PHI_PROBES = [-150.0, -120.0, -90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0, 120.0, 150.0]
BETA, DEPTH = 0.0, 0.10
EXAGGERATE = 15

CASES = [
    {"L_M": 25.0, "phi": 90.0, "s": 10.0, "tip_ux": 0.9085647211956516, "tip_uy": -0.20530933423913045,
     "Fx": 0.00022645589171414688, "Fy": 4.346029521463862e-05},
    {"L_M": 75.0, "phi": -90.0, "s": 20.0, "tip_ux": 0.4820352413043478, "tip_uy": -0.12925243641304351,
     "Fx": 2.7770789272289625e-05, "Fy": -4.619419118934957e-06},
    {"L_M": 50.0, "phi": 90.0, "s": 50.0, "tip_ux": 0.1783192880434781, "tip_uy": 0.016529407119565215,
     "Fx": 1.476871439112012e-06, "Fy": 1.5111201753607552e-06},
    {"L_M": 75.0, "phi": 120.0, "s": 70.0, "tip_ux": 0.12266518152173919, "tip_uy": 0.0009171716375,
     "Fx": 6.818232730221249e-07, "Fy": 1.5606851380840877e-07},
    {"L_M": 75.0, "phi": -30.0, "s": 80.0, "tip_ux": 0.06056653750000002, "tip_uy": -0.05830700744565213,
     "Fx": 2.0031792124758597e-07, "Fy": -3.483375871492832e-07},
    {"L_M": 0.0, "phi": -90.0, "s": 10.0, "tip_ux": 0.7397460130434782, "tip_uy": 0.6151870179347827,
     "Fx": 0.00020797964740611332, "Fy": 5.3499800779696293e-05},
]


# ── 대체모델 재학습 ──────────────────────────────────────────────────────
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


def predict_surrogate(L_M, phi, beta, s, depth):
    x = np.array([[L_M, phi, beta, s, depth]])
    xn = (x - X_mean) / X_std
    xt = torch.tensor(xn, dtype=torch.float32)
    with torch.no_grad():
        pn = np.mean([m(xt).numpy()[0] for m in surrogates], axis=0)
    return dict(zip(TARGETS, pn * Y_std + Y_mean))


# ── 보조회귀 분류기(auxreg) 로드 ──────────────────────────────────────────
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
        self.s_head = nn.Linear(128, 1)

    def forward(self, x):
        embeds = [self.encoder(x[:, p]) for p in range(self.n_probes)]
        h = self.trunk(torch.cat(embeds, dim=1))
        return self.seg_head(h), self.force_head(h), self.s_head(h).squeeze(-1)


ckpt = torch.load(os.path.join(MODELS_DIR, "position_segment_classifier_multiprobe_150k_11probe_auxreg.pth"),
                   map_location="cpu", weights_only=False)
clf = MultiProbeClassifier()
clf.load_state_dict(ckpt["state_dict"])
clf.eval()
CX_mean, CX_std = ckpt["X_mean"], ckpt["X_std"]
f_mean, f_std = ckpt["f_mean"], ckpt["f_std"]
s_mean, s_std = ckpt["s_mean"], ckpt["s_std"]

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


print("각 케이스별 모델 예측 계산 중 (11프로브 스캔)...")
free_cache = {}
for case in CASES:
    L_M, s = case["L_M"], case["s"]
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
        seg_logits, force_pred, s_pred_n = clf(torch.tensor(Xn_, dtype=torch.float32))
        case["pred_bin"] = int(seg_logits.argmax(dim=1)[0])
        case["s_pred"] = float(s_pred_n[0].numpy() * s_std + s_mean)
        fp = force_pred.numpy()[0] * f_std + f_mean
        case["Fx_pred"], case["Fy_pred"] = float(fp[0]), float(fp[1])
    print(f"  L_M={L_M},phi={case['phi']},s={s}mm -> 예측 s={case['s_pred']:.1f}mm "
          f"(오차 {abs(case['s_pred']-s):.1f}mm)")

# ── 플롯 (2행 3열) ────────────────────────────────────────────────────
# phi=0 케이스는 거의 일직선, phi=120 케이스는 크게 휘어서 case마다 곡선 범위가 크게 다름
# (equal aspect를 쓰면 패널 크기가 서로 안 맞아 겹치는 문제 생김) -> 모든 패널에 공통 축범위 사용
COMMON_XLIM = (-8, 105)
COMMON_YLIM = (-45, 45)

fig, axes = plt.subplots(2, 3, figsize=(19, 13.5))
colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
for panel, case in enumerate(CASES):
    ax = axes[panel // 3][panel % 3]
    L_M, PHI, s_mm = case["L_M"], case["phi"], case["s"]
    Fx, Fy = case["Fx"], case["Fy"]

    r_free = fm.solve_shape(L_M=L_M, phi_deg=PHI, loads=[], return_curve=True)
    order0 = np.argsort(r_free["curve_s_mm"])
    cs0, cx0, cy0 = r_free["curve_s_mm"][order0], r_free["curve_x_mm"][order0], r_free["curve_y_mm"][order0]
    cth0 = r_free["curve_theta_deg"][order0]

    # 화살표는 "그 지점 표면의 법선"(임의) 대신, 실제로 곡선을 휘게 만든 바로 그 데이터
    # (팁 실측 변위 tip_ux,tip_uy)의 방향을 그대로 써서 화살표와 변형 모양이 항상 일치하게 함.
    d_x_local0, d_y_local0 = case["tip_uy"], case["tip_ux"]
    disp_dir = np.array([d_x_local0, d_y_local0])
    disp_dir = disp_dir / (np.linalg.norm(disp_dir) + 1e-15)
    push_ang = np.degrees(np.arctan2(disp_dir[1], disp_dir[0]))

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

    ax.plot(cx0, cy0, "--", color="#6b7580", linewidth=1.8, label="무접촉 기준형상", zorder=6)
    ax.plot(cx_ex, cy_ex, "-", color=colors[panel], linewidth=2.4,
            label=f"접촉 시 실제형상({EXAGGERATE}배 확대)", zorder=3)
    ax.plot(0, 0, "ks", markersize=9, zorder=5, label="베이스(고정단)")

    xt, yt = np.interp(s_mm, cs0, cx0), np.interp(s_mm, cs0, cy0)
    ax.plot(xt, yt, "o", color="#e34948", markersize=12, zorder=7,
            markeredgecolor="white", markeredgewidth=1.3, label=f"실제 s={s_mm:.0f}mm")

    s_pred = np.clip(case["s_pred"], cs0.min(), cs0.max())
    xp, yp = np.interp(s_pred, cs0, cx0), np.interp(s_pred, cs0, cy0)
    ax.plot(xp, yp, "x", color="#1E7B34", markersize=15, mew=3, zorder=8,
            label=f"예측 s={case['s_pred']:.1f}mm")
    ax.plot([xt, xp], [yt, yp], ":", color="#444444", linewidth=1.2, zorder=4)

    arrow_len = 14
    ax.annotate("", xy=(xt + disp_dir[0] * arrow_len, yt + disp_dir[1] * arrow_len), xytext=(xt, yt),
                arrowprops=dict(arrowstyle="-|>", color="#c0392b", linewidth=2.2, mutation_scale=16), zorder=9)

    err = abs(case["s_pred"] - s_mm)
    ax.set_xlim(*COMMON_XLIM)
    ax.set_ylim(*COMMON_YLIM)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.set_xlabel("x (mm, 로컬)", fontsize=9)
    ax.set_ylabel("y (mm, 로컬)", fontsize=9)
    ax.set_title(f"L_M={L_M:.0f}mm, phi={PHI:.0f}도 | 실제 s={s_mm:.0f}mm vs 예측 s={case['s_pred']:.1f}mm\n"
                 f"오차 {err:.1f}mm (변위 방향 {push_ang:.0f}도)",
                 fontweight="bold", fontsize=10)
    ax.legend(fontsize=7, loc="lower right")

fig.suptitle("연속값 s(접촉 정확한 위치) 예측 — 실제 vs 예측, 6개 사례 (보조회귀 모델)",
             fontweight="bold", fontsize=15, y=0.995)
fig.subplots_adjust(hspace=0.45, wspace=0.3)
out = os.path.join(FEA_DATA_DIR, "s_regression_6panel.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"저장: {out}")
