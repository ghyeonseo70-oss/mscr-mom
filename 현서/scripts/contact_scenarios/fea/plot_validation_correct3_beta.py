"""
정답 3개만 골라서 실제 vs 예측 시각화 + 케이스별 beta(원주방향 접촉각) 값 표시.
저장된 150k npz에는 케이스별 L_M/phi/beta가 안 남아있어서(공간절약), 여기선 학습때와 동일한
방식(대체모델 재학습 -> 소규모 합성)으로 beta를 추적하며 새로 몇백개 뽑아 그중 정답만 고른다.
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

FEATURES = ["L_M_mm", "phi_deg", "beta_deg", "contact_s_mm", "push_depth_mm"]
TARGETS = ["tip_ux_avg_mm", "tip_uy_avg_mm", "tip_uz_avg_mm", "tip_theta_deg_board",
           "Fx_total_N", "Fy_total_N", "Fz_total_N", "F_mag_N"]
DEFAULTS = {"L_M_mm": 50.0, "phi_deg": 60.0, "beta_deg": 0.0}
PHI_PROBES = [-150.0, -120.0, -90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0, 120.0, 150.0]
N_PROBES = len(PHI_PROBES)
N_CLASSES = 5
BIN_WIDTH_MM = 20.0
BIN_MID = [10.0, 30.0, 50.0, 70.0, 90.0]
FIXED_DEPTH = 0.10


def s_to_bin(s):
    return min(N_CLASSES - 1, int(s / BIN_WIDTH_MM))


class SurrogateMLP(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(),
                                  nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, n_out))

    def forward(self, x):
        return self.net(x)


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


print("대체모델 재학습 중...")
SOURCES = ["fea_lm_phi_pos_sweep_all.json", "fea_bent_contact_sweep.json",
           "fea_geom_sweep_all.json", "fea_angle_sweep_all.json"]
all_rows = []
for fname in SOURCES:
    for r in json.load(open(os.path.join(FEA_DATA_DIR, fname))):
        row = dict(DEFAULTS)
        row.update(r)
        all_rows.append(row)
X = np.array([[r[f] for f in FEATURES] for r in all_rows])
y = np.array([[r[t] for t in TARGETS] for r in all_rows])
X_mean, X_std = X.mean(0), X.std(0)
X_std[X_std < 1e-9] = 1.0
y_mean, y_std = y.mean(0), y.std(0)
y_std[y_std < 1e-9] = 1.0
Xn, yn = (X - X_mean) / X_std, (y - y_mean) / y_std


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


surrogates = [train_mlp(Xn, yn, seed=i) for i in range(10)]
for m in surrogates:
    m.eval()
print("대체모델 완료")

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
    return dict(zip(TARGETS, pn * y_std + y_mean))


ckpt = torch.load(os.path.join(MODELS_DIR, "position_segment_classifier_multiprobe_150k_11probe_force_v2.pth"),
                   map_location="cpu", weights_only=False)
clf = MultiProbeClassifier()
clf.load_state_dict(ckpt["state_dict"])
clf.eval()
CX_mean, CX_std = ckpt["X_mean"], ckpt["X_std"]
f_mean, f_std = ckpt["f_mean"], ckpt["f_std"]

print("검증용 샘플 생성 중 (beta 추적)...")
rng = np.random.default_rng(7)
free_cache = {}
found = []
attempts = 0
while len(found) < 3 and attempts < 400:
    attempts += 1
    L_M = rng.uniform(0, 100)
    s = rng.uniform(10, 100)
    beta = rng.uniform(0, 360)
    probes = np.zeros((N_PROBES, 3, 5, 5), dtype=np.float32)
    ok = True
    fx_l, fy_l = [], []
    for pi, phi in enumerate(PHI_PROBES):
        key = (round(L_M, 1), phi)
        if key in free_cache:
            r_free = free_cache[key]
        else:
            try:
                r_free = fm.solve_shape(L_M=L_M, phi_deg=phi, loads=[])
            except Exception:
                ok = False
                break
            free_cache[key] = r_free
        pred = predict_surrogate(L_M, phi, beta, s, FIXED_DEPTH)
        d_xL, d_yL, d_thL = pred["tip_uy_avg_mm"], pred["tip_ux_avg_mm"], -pred["tip_theta_deg_board"]
        frac = L_M / 100.0
        xL_f, yL_f, thL_f = r_free["x_L"], r_free["y_L"], r_free["theta_L_deg"]
        xLM_f, yLM_f, thLM_f = r_free["x_LM"], r_free["y_LM"], r_free["theta_LM_deg"]
        B_free = compute_B(xLM_f, yLM_f, thLM_f, xL_f, yL_f, thL_f)
        B_load = compute_B(xLM_f + d_xL * frac, yLM_f + d_yL * frac, thLM_f + d_thL * frac,
                            xL_f + d_xL, yL_f + d_yL, thL_f + d_thL)
        probes[pi] = (B_load - B_free).reshape(5, 5, 3).transpose(2, 0, 1)
        fx_l.append(pred["Fy_total_N"])
        fy_l.append(pred["Fx_total_N"])
    if not ok:
        continue
    Xn_ = (probes[None] - CX_mean) / CX_std
    with torch.no_grad():
        seg_logits, force_pred = clf(torch.tensor(Xn_, dtype=torch.float32))
        seg_pred = int(seg_logits.argmax(dim=1)[0])
        force_pred_phys = force_pred.numpy()[0] * f_std + f_mean
    true_bin = s_to_bin(s)
    if seg_pred == true_bin:  # 정답만 채택
        found.append({"L_M": L_M, "s": s, "beta": beta, "true_bin": true_bin, "pred_bin": seg_pred,
                       "Fx_true": np.mean(fx_l), "Fy_true": np.mean(fy_l),
                       "Fx_pred": force_pred_phys[0], "Fy_pred": force_pred_phys[1]})

print(f"{attempts}번 시도해서 정답 {len(found)}개 확보")

# ── 플롯 ──────────────────────────────────────────────────────────────
BIN_EDGES = [0, 20, 40, 60, 80, 100]
BIN_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
BIN_LABELS = ["0-20mm", "20-40mm", "40-60mm", "60-80mm", "80-100mm"]

fig, axes = plt.subplots(1, 3, figsize=(17, 7.3))
for panel, case in enumerate(found):
    ax = axes[panel]
    r = fm.solve_shape(L_M=case["L_M"], phi_deg=60.0, loads=[], return_curve=True)
    order = np.argsort(r["curve_s_mm"])
    cs, cx, cy = r["curve_s_mm"][order], r["curve_x_mm"][order], r["curve_y_mm"][order]
    cth = r["curve_theta_deg"][order]

    ax.plot(cx, cy, "--", color="#bbbbbb", linewidth=1.5, label="무접촉 기준형상", zorder=2)
    Fx_t, Fy_t = case["Fx_true"], case["Fy_true"]
    fmag_t = np.hypot(Fx_t, Fy_t)
    f_dir = np.array([Fx_t, Fy_t]) / (fmag_t + 1e-12)
    disp_scale = 25.0
    cx_def = cx + f_dir[0] * disp_scale * (cs / cs.max()) ** 2
    cy_def = cy + f_dir[1] * disp_scale * (cs / cs.max()) ** 2
    for b in range(5):
        lo, hi = BIN_EDGES[b], BIN_EDGES[b + 1]
        mask = (cs >= lo) & (cs <= hi)
        ax.plot(cx_def[mask], cy_def[mask], "-", color=BIN_COLORS[b], linewidth=10,
                solid_capstyle="round", zorder=2.5)
        if panel == 0:
            ax.plot([], [], "-", color=BIN_COLORS[b], linewidth=7, label=f"{b}번({BIN_LABELS[b]})")
    ax.plot(0, 0, "ks", markersize=10, zorder=5, label="베이스(고정단)" if panel == 0 else None)

    s_mark = BIN_MID[case["true_bin"]]
    xt, yt = np.interp(s_mark, cs, cx_def), np.interp(s_mark, cs, cy_def)
    ax.plot(xt, yt, "o", color="#e34948", markersize=13, zorder=7,
            markeredgecolor="white", markeredgewidth=1.5, label="실제=예측 구간")
    ax.plot(xt, yt, "x", color="#1E7B34", markersize=15, mew=3, zorder=8)

    # beta 방향 화살표: push_dir = cos(beta)*in_plane_normal + sin(beta)*binormal(z축)
    # (make_bent_contact_scene.py의 _point_and_normal_at_s와 동일한 정의)
    theta = np.radians(np.interp(s_mark, cs, cth))
    in_plane_normal = np.array([-np.cos(theta), np.sin(theta)])  # x,y 평면 안의 법선
    beta_rad = np.radians(case["beta"])
    push_xy = np.cos(beta_rad) * in_plane_normal  # 화면(x,y) 평면에 보이는 성분
    push_z = np.sin(beta_rad)  # 화면에 수직(지면 안/밖) 성분, -1~1
    beta_arrow_len = 14.0
    ax.annotate("", xy=(xt + push_xy[0] * beta_arrow_len, yt + push_xy[1] * beta_arrow_len),
                xytext=(xt, yt),
                arrowprops=dict(arrowstyle="-|>", color="#4a3aa7", linewidth=2.6, mutation_scale=20), zorder=9)
    if abs(push_z) > 0.3:  # 지면 방향 성분이 무시 못할 크기면 안/밖 표시 추가
        sym = "⊙" if push_z > 0 else "⊗"  # ⊙(바깥쪽/독자쪽) / ⊗(안쪽/화면속)
        ax.text(xt + push_xy[0] * beta_arrow_len * 0.55, yt + push_xy[1] * beta_arrow_len * 0.55 + 4,
                sym, fontsize=15, color="#4a3aa7", ha="center", va="center", fontweight="bold", zorder=10)
    ax.plot([], [], "-", color="#4a3aa7", linewidth=2.4, marker=">",
            label=f"누른 방향(beta={case['beta']:.0f}도)")

    Fx_p, Fy_p = case["Fx_pred"], case["Fy_pred"]
    y_lo = min(cy.min(), cy_def.min()) - 5
    y_hi = max(cy.max(), cy_def.max(), yt) + 20
    ax.set_ylim(y_lo, y_hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"사례 {panel+1} [정답]: L_M={case['L_M']:.0f}mm, beta={case['beta']:.0f}도\n"
                 f"구간={case['true_bin']}번(s={case['s']:.0f}mm) | "
                 f"실제F=({Fx_t*1000:.2f},{Fy_t*1000:.2f})mN vs 예측F=({Fx_p*1000:.2f},{Fy_p*1000:.2f})mN",
                 fontweight="bold", fontsize=10)
    ax.set_xlabel("x (mm, 로컬)")
    ax.set_ylabel("y (mm, 로컬)")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(fontsize=7.8, loc="best")

fig.suptitle("멀티프로브 모델 검증 — 정답 사례 3개 (beta 값 표시)", fontweight="bold", fontsize=14, y=1.04)
out = os.path.join(FEA_DATA_DIR, "validation_correct3_beta.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"저장: {out}")
