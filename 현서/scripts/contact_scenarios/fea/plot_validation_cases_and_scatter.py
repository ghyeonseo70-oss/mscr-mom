"""
plot_nn2_results.py 스타일 그대로: 실제(빨간 원) vs 예측(초록 X)을 카테터 곡선 위에 비교하는
3-패널 그림 + 예측 vs 실제 산점도. 학습된 멀티태스크 모델(force_v2)의 진짜 검증셋 예측을 사용.
"""
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
sys.path.insert(0, os.path.join(HERE, "..", "..", "force_model"))
import force_model as fm

DATA_DIR = os.path.join(HERE, "..", "..", "..", "data", "contact_scenarios", "fea")
MODELS_DIR = os.path.join(HERE, "..", "..", "..", "models")
OUT_DIR = DATA_DIR

N_CLASSES = 5
N_PROBES = 11
BIN_WIDTH_MM = 20.0
BIN_MID = [10.0, 30.0, 50.0, 70.0, 90.0]  # 구간 중앙값(연속값 s가 따로 없어서 근사 위치로 사용)


class MultiProbeClassifier(nn.Module):
    def __init__(self, n_probes=N_PROBES, n_classes=N_CLASSES, n_force=2):
        super().__init__()
        self.n_probes = n_probes
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Flatten(), nn.Linear(32 * 5 * 5, 64), nn.ReLU(),
        )
        self.trunk = nn.Sequential(
            nn.Linear(64 * n_probes, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
        )
        self.seg_head = nn.Linear(128, n_classes)
        self.force_head = nn.Linear(128, n_force)

    def forward(self, x):
        embeds = [self.encoder(x[:, p]) for p in range(self.n_probes)]
        h = self.trunk(torch.cat(embeds, dim=1))
        return self.seg_head(h), self.force_head(h)


print("데이터/모델 로드 중...")
d = np.load(os.path.join(DATA_DIR, "segment_bfield_multiprobe_150k_11probe_force.npz"))
X_all, y_all, f_all = d["X"], d["y"], d["f"]

ckpt = torch.load(os.path.join(MODELS_DIR, "position_segment_classifier_multiprobe_150k_11probe_force_v2.pth"),
                   map_location="cpu", weights_only=False)
model = MultiProbeClassifier()
model.load_state_dict(ckpt["state_dict"])
model.eval()
X_mean, X_std = ckpt["X_mean"], ckpt["X_std"]
f_mean, f_std = ckpt["f_mean"], ckpt["f_std"]

rng = np.random.default_rng(0)
idx = rng.permutation(len(X_all))
val_idx = idx[int(0.9 * len(idx)):]  # 저장 당시와 동일한 split 규칙(파일 순서는 이미 섞여있었음-근사)

# ── 검증셋 전체(계산량 위해 5000개 서브샘플) 예측 ──────────────────────────
sub_idx = rng.choice(val_idx, size=min(5000, len(val_idx)), replace=False)
Xs = (X_all[sub_idx] - X_mean) / X_std
with torch.no_grad():
    seg_logits, force_pred = model(torch.tensor(Xs, dtype=torch.float32))
    seg_pred = seg_logits.argmax(dim=1).numpy()
    force_pred_phys = force_pred.numpy() * f_std + f_mean
y_true_sub = y_all[sub_idx]
f_true_sub = f_all[sub_idx]
fmag_pred = np.sqrt(force_pred_phys[:, 0] ** 2 + force_pred_phys[:, 1] ** 2)
fmag_true = np.sqrt(f_true_sub[:, 0] ** 2 + f_true_sub[:, 1] ** 2)

# ── 1) 3-패널 실제 vs 예측 (plot_nn2_results.py 스타일) ─────────────────
L_M, PHI = 50.0, 60.0
r = fm.solve_shape(L_M=L_M, phi_deg=PHI, loads=[], return_curve=True)
order = np.argsort(r["curve_s_mm"])
cs, cx, cy = r["curve_s_mm"][order], r["curve_x_mm"][order], r["curve_y_mm"][order]

BIN_EDGES = [0, 20, 40, 60, 80, 100]
BIN_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
BIN_LABELS = ["0-20mm", "20-40mm", "40-60mm", "60-80mm", "80-100mm"]

wrong_mask = y_true_sub != seg_pred
wrong_pool = np.where(wrong_mask)[0]
right_pool = np.where(~wrong_mask)[0]
# 원본 예시처럼 틀린 사례도 섞여서 오차(화살표)가 보이도록 최소 1~2개는 오답에서 뽑음
pick = np.concatenate([
    rng.choice(wrong_pool, size=min(2, len(wrong_pool)), replace=False),
    rng.choice(right_pool, size=1, replace=False),
])
rng.shuffle(pick)
fig, axes = plt.subplots(1, 3, figsize=(17, 7.3))
for panel, i in enumerate(pick):
    ax = axes[panel]
    # 무접촉 기준형상(점선) - 5구간 색깔 없이 회색
    ax.plot(cx, cy, "--", color="#bbbbbb", linewidth=1.5, label="무접촉 기준형상", zorder=2)

    # 접촉시 형상: 힘 방향으로 팁을 살짝 밀어서 그림(정확한 변위 값은 저장 안 돼있어 방향만 실제,
    # 크기는 보기 좋게 과장한 개념도 - 변형이 있다는 걸 보여주기 위한 목적)
    Fx_t, Fy_t = f_true_sub[i, 0], f_true_sub[i, 1]
    fmag_t = np.hypot(Fx_t, Fy_t)
    f_dir = np.array([Fx_t, Fy_t]) / (fmag_t + 1e-12)
    disp_scale = 25.0  # mm, 시각화용 과장 배율
    cx_def = cx + f_dir[0] * disp_scale * (cs / cs.max()) ** 2
    cy_def = cy + f_dir[1] * disp_scale * (cs / cs.max()) ** 2

    # 5구간을 색깔 띠로 표시(접촉시 형상 위에)
    for b in range(5):
        lo, hi = BIN_EDGES[b], BIN_EDGES[b + 1]
        mask = (cs >= lo) & (cs <= hi)
        ax.plot(cx_def[mask], cy_def[mask], "-", color=BIN_COLORS[b], linewidth=10,
                solid_capstyle="round", zorder=2.5)
        if panel == 0:
            ax.plot([], [], "-", color=BIN_COLORS[b], linewidth=7, label=f"{b}번({BIN_LABELS[b]})")
    ax.plot(0, 0, "ks", markersize=10, zorder=5, label="베이스(고정단)" if panel == 0 else None)

    s_true = BIN_MID[y_true_sub[i]]
    s_pred = BIN_MID[seg_pred[i]]
    xt, yt = np.interp(s_true, cs, cx_def), np.interp(s_true, cs, cy_def)
    xp, yp = np.interp(s_pred, cs, cx_def), np.interp(s_pred, cs, cy_def)
    ax.plot(xt, yt, "o", color="#e34948", markersize=13, zorder=7,
            markeredgecolor="white", markeredgewidth=1.5, label=f"실제 구간 (~{s_true:.0f}mm)")
    ax.plot(xp, yp, "x", color="#1E7B34", markersize=15, mew=3, zorder=7,
            label=f"예측 구간 (~{s_pred:.0f}mm)")
    if s_true != s_pred:
        ax.plot([xt, xp], [yt, yp], ":", color="#444444", linewidth=1.3, zorder=4)

    Fx_p, Fy_p = force_pred_phys[i, 0], force_pred_phys[i, 1]
    y_lo = min(cy.min(), cy_def.min()) - 5
    y_hi = max(cy.max(), cy_def.max(), yt, yp) + 15
    ax.set_ylim(y_lo, y_hi)
    ax.set_aspect("equal", adjustable="box")
    seg_ok = "O" if y_true_sub[i] == seg_pred[i] else "X"
    ax.set_title(f"사례 {panel+1}: 실제구간={y_true_sub[i]}번 vs 예측구간={seg_pred[i]}번 [{seg_ok}]\n"
                 f"실제 F=({Fx_t*1000:.2f},{Fy_t*1000:.2f})mN vs 예측=({Fx_p*1000:.2f},{Fy_p*1000:.2f})mN",
                 fontweight="bold", fontsize=10.5)
    ax.set_xlabel("x (mm, 로컬)")
    ax.set_ylabel("y (mm, 로컬)")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(fontsize=7.8, loc="best")

fig.suptitle("멀티프로브 모델 검증 사례 3개 — 실제 vs 예측 (구간 + 힘, 노이즈 제거 버전)",
             fontweight="bold", fontsize=14, y=1.03)
out1 = os.path.join(OUT_DIR, "validation_cases_3panel.png")
plt.savefig(out1, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"저장: {out1}")

# ── 2) 산점도: 예측 vs 실제 (Fx, Fy, F_mag) ─────────────────────────────
def scatter_r2(ax, true_v, pred_v, name, unit_scale=1000, unit="mN", color="#2a78d6"):
    t, p = true_v * unit_scale, pred_v * unit_scale
    ss_res = np.sum((p - t) ** 2)
    ss_tot = np.sum((t - t.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    ax.scatter(t, p, s=8, alpha=0.25, color=color, edgecolors="none")
    lo, hi = min(t.min(), p.min()), max(t.max(), p.max())
    ax.plot([lo, hi], [lo, hi], "--", color="#e34948", linewidth=1.5, label="y=x (완벽예측)")
    ax.set_xlabel(f"실제 {name} ({unit})")
    ax.set_ylabel(f"예측 {name} ({unit})")
    ax.set_title(f"{name}: R²={r2:.3f} (n={len(t)})", fontweight="bold", fontsize=11)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(fontsize=8, loc="upper left")


fig, axes = plt.subplots(1, 3, figsize=(16, 5.3))
scatter_r2(axes[0], f_true_sub[:, 0], force_pred_phys[:, 0], "Fx (보드좌표)")
scatter_r2(axes[1], f_true_sub[:, 1], force_pred_phys[:, 1], "Fy (보드좌표)", color="#eb6834")
scatter_r2(axes[2], fmag_true, fmag_pred, "F_mag (유도)", color="#1baf7a")
fig.suptitle("예측 vs 실제 산점도 (힘, 검증셋 5000개 샘플)", fontweight="bold", fontsize=13, y=1.04)
out2 = os.path.join(OUT_DIR, "scatter_force.png")
plt.savefig(out2, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"저장: {out2}")

# 구간(카테고리) 산점도 대신 confusion matrix 스타일이 맞지만, "산점도"로 요청하셨으니
# 실제구간 vs 예측구간(정수, 지터 추가)도 하나 더 그려줌
fig, ax = plt.subplots(figsize=(6, 6))
jitter = rng.normal(0, 0.12, size=len(sub_idx))
ax.scatter(y_true_sub + jitter, seg_pred + rng.normal(0, 0.12, size=len(sub_idx)),
           s=8, alpha=0.15, color="#2a78d6", edgecolors="none")
ax.plot([-0.5, 4.5], [-0.5, 4.5], "--", color="#e34948", linewidth=1.5, label="y=x (완벽예측)")
acc = (y_true_sub == seg_pred).mean()
ax.set_xlabel("실제 구간 번호")
ax.set_ylabel("예측 구간 번호")
ax.set_xticks(range(5))
ax.set_yticks(range(5))
ax.set_title(f"구간 예측 산점도 (지터 추가, 정확도={acc*100:.1f}%, n={len(sub_idx)})", fontweight="bold", fontsize=11)
ax.set_aspect("equal", adjustable="box")
ax.grid(True, linestyle=":", alpha=0.4)
ax.legend(fontsize=8, loc="upper left")
out3 = os.path.join(OUT_DIR, "scatter_segment.png")
plt.savefig(out3, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"저장: {out3}")
