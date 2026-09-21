"""최신 체크포인트의 실측 홀드아웃(n=99, 해시기반 고정분할) 예측 vs 실제 산점도 - 세션
결론 요약용.

2026-09-14 갱신: L_M/phi가 더 이상 예측 대상이 아니라 known input으로 바뀐 구조
(SingleProbeClassifier.forward(x, config))에 맞춰 갱신. phi/L_M은 입력=정답이라
산점도로 그릴 의미가 없어져서 빠지고, 진짜 예측값인 s/Fx_board/Fy_board 3개만 남음."""
import hashlib
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

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
print(f"실측 홀드아웃: n={len(real_holdout_rows)}")

ckpt = torch.load(os.path.join(MODELS_DIR, "position_segment_classifier_singleprobe_beta0180_4seg.pth"),
                   map_location="cpu", weights_only=False)


class SingleProbeClassifier(nn.Module):
    """2026-09-14: config_head/lm_zero_head 제거, L_M/phi(config)를 trunk 입력으로 직접 받음
    (train_segment_classifier_singleprobe_beta0180_4seg.py와 동일 구조 - state_dict 키를
    맞추려면 반드시 같이 수정할 것)."""
    def __init__(self, n_probes=1, n_classes=N_CLASSES, n_force=2, n_config_in=2):
        super().__init__()
        self.n_probes = n_probes
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Flatten(), nn.Linear(32 * 5 * 5, 64), nn.ReLU())
        self.trunk = nn.Sequential(nn.Linear(64 * n_probes + n_config_in, 128),
                                    nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3))
        self.seg_head = nn.Linear(128, n_classes)
        self.force_head = nn.Linear(128, n_force)
        self.s_head = nn.Linear(128, 1)
        # 2026-09-21(36번): shape_head 추가 - 체크포인트 키를 맞추려면 여기도 있어야 함.
        self.shape_head = nn.Linear(128, 3)

    def forward(self, x, config):
        embeds = [self.encoder(x[:, p]) for p in range(self.n_probes)]
        h = self.trunk(torch.cat(embeds + [config], dim=1))
        return self.seg_head(h), self.force_head(h), self.s_head(h).squeeze(-1), self.shape_head(h)


cnn = SingleProbeClassifier()
cnn.load_state_dict(ckpt["state_dict"])
cnn.eval()
X_mean2, X_std2 = ckpt["X_mean"], ckpt["X_std"]
f_mean, f_std = ckpt["f_mean"], ckpt["f_std"]
s_mean, s_std = ckpt["s_mean"], ckpt["s_std"]
c_mean, c_std = ckpt["c_mean"], ckpt["c_std"]
force_names = ckpt["force_names"]
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


real_X, real_s, real_fx_board, real_fy_board, real_c = [], [], [], [], []
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
    real_s.append(r["contact_s_mm"])
    real_fx_board.append(r["Fy_total_N"])  # Fx_board 정답
    real_fy_board.append(r["Fx_total_N"])  # Fy_board 정답
    real_c.append([L_M, phi])

real_X = np.array(real_X, dtype=np.float32)
real_X_norm = (real_X - X_mean2) / X_std2
real_s = np.array(real_s)
real_fx_board = np.array(real_fx_board)
real_fy_board = np.array(real_fy_board)
real_c = np.array(real_c, dtype=np.float32)

# L_M/phi는 known input이라 정규화해서 config로 그대로 넣음(예측 아님).
real_c_norm = (real_c - c_mean) / c_std
with torch.no_grad():
    rX = torch.tensor(real_X_norm[:, None]).float()
    rC = torch.tensor(real_c_norm).float()
    _, r_force_pred, r_s_pred, _ = cnn(rX, rC)
    force_phys = r_force_pred.numpy() * f_std + f_mean
    s_phys = r_s_pred.numpy() * s_std + s_mean

fx_board_pred = force_phys[:, force_names.index("Fx_board_N")]
fy_board_pred = force_phys[:, force_names.index("Fy_board_N")]
phi_true = real_c[:, 1]
high_phi = np.abs(phi_true) >= 90  # 이 구간이 지금 계속 추적 중인 약한 축(25/26/27번 참고)

panels = [
    ("s (접촉위치, mm)", real_s, s_phys, "#2451A3"),
    ("Fx_board (mN)", real_fx_board * 1000, fx_board_pred * 1000, "#8E44AD"),
    ("Fy_board (mN)", real_fy_board * 1000, fy_board_pred * 1000, "#D68910"),
]

fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
for ax, (name, true_v, pred_v, color) in zip(axes, panels):
    r2 = r2_score(true_v, pred_v)
    lo, hi = min(true_v.min(), pred_v.min()), max(true_v.max(), pred_v.max())
    pad = (hi - lo) * 0.05
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "r--", linewidth=1.2, label="y=x(완벽예측)")
    if name.startswith("Fx_board"):
        # |phi|>=90(약한 구간)을 x 마커로 따로 표시 - 27번 재시도 스윕 효과를 볼 때 이 부분만
        # 추적하면 됨.
        r2_hp = r2_score(true_v[high_phi], pred_v[high_phi]) if high_phi.sum() >= 3 else float("nan")
        ax.scatter(true_v[~high_phi], pred_v[~high_phi], s=45, alpha=0.7, color=color,
                   edgecolor="black", linewidth=0.3, label="|phi|<90")
        ax.scatter(true_v[high_phi], pred_v[high_phi], s=55, alpha=0.85, color="black", marker="x",
                   label=f"|phi|>=90 (R²={r2_hp:.3f})")
    else:
        ax.scatter(true_v, pred_v, s=45, alpha=0.7, color=color, edgecolor="black", linewidth=0.3)
    ax.set_xlabel(f"실제 {name}")
    ax.set_ylabel(f"예측 {name}")
    ax.set_title(f"{name}\nR²={r2:.3f} (n={len(true_v)})", fontweight="bold", fontsize=12)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.4)

fig.suptitle("2026-09-14 최신: 순수 실측 FEA 홀드아웃(n=99) 예측 vs 실제\n"
             "(L_M/phi known input 구조 - phi/L_M은 입력이라 산점도 대상 아님, s/Fx_board/Fy_board만 진짜 예측)",
             fontweight="bold", fontsize=13, y=1.06)
plt.tight_layout()
out_path = os.path.join(HERE, "..", "..", "..", "data", "contact_scenarios", "final_scatter_summary_knowninput_0914.png")
plt.savefig(out_path, dpi=140, bbox_inches="tight")
print("저장:", out_path)
