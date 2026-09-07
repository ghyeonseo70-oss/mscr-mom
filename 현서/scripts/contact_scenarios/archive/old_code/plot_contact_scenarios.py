"""
contact_force_sweep.py가 만든 데이터셋(data/contact_force_scenarios.npz)을 시각화.

(A) 접촉힘 크기 vs 팁 위치 이탈량 산점도 (색=접촉위치)
(B) 무외력 vs 접촉시 형상 예시 몇 개 (직접 재계산, 곡선 전체)
(C) 접촉위치 x 힘크기 별 평균 이탈량 히트맵 — 어디서 눌러야 형상이 많이/적게 바뀌는지
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'force_model'))
import force_model as fm

# 순차형(sequential) 단일색상 계열 컬러맵: 저채도 연한색 -> 진한 청록. 등급을 눈으로 자연스럽게 읽히게.
SEQ_CMAP = LinearSegmentedColormap.from_list('seq', ['#dbe9f6', '#5b9bd5', '#1c4e80'])

d = np.load(os.path.join(HERE, '..', '..', 'data', 'contact_scenarios', 'contact_force_scenarios.npz'), allow_pickle=True)
data, cols = d["data"], list(d["columns"])
col = {c: i for i, c in enumerate(cols)}

F_mag = data[:, col["F_mag"]] * 1000  # N -> mN
s = data[:, col["s"]]
dx = data[:, col["xL_load"]] - data[:, col["xL_free"]]
dy = data[:, col["yL_load"]] - data[:, col["yL_free"]]
deviation = np.hypot(dx, dy)

fig = plt.figure(figsize=(15, 5.5))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.35)

# ── (A) F_mag vs 이탈량, 색=접촉위치 s ──────────────────────────────
ax1 = fig.add_subplot(gs[0])
sc = ax1.scatter(F_mag, deviation, c=s, cmap=SEQ_CMAP, s=14, alpha=0.75, edgecolors="none")
cbar = fig.colorbar(sc, ax=ax1, label="접촉위치 s (mm, 베이스로부터)")
ax1.set_xlabel("접촉힘 크기 |F| (mN)")
ax1.set_ylabel("팁 위치 이탈량 (mm)")
ax1.set_title("(A) 힘이 클수록 형상이 더 벗어남", fontweight="bold", fontsize=11)
ax1.grid(True, linestyle=":", alpha=0.5)

# ── (B) 예시 곡선 몇 개: 무외력 vs 접촉시 ──────────────────────────────
ax2 = fig.add_subplot(gs[1])
examples = [
    dict(L_M=50, phi_deg=60, s=30, F=0.005, ang=90, label="약한 힘, 베이스쪽"),
    dict(L_M=50, phi_deg=60, s=30, F=0.02, ang=90, label="강한 힘, 베이스쪽"),
    dict(L_M=50, phi_deg=60, s=85, F=0.01, ang=180, label="중간 힘, 팁쪽"),
]
colors = ["#5b9bd5", "#e34948", "#2e7d32"]
r_free = fm.solve_shape(L_M=50, phi_deg=60, loads=[], return_curve=True)
ax2.plot(r_free["curve_x_mm"], r_free["curve_y_mm"], color="gray", linewidth=2.5,
         linestyle="--", label="무외력 기준", zorder=5)
for ex, c in zip(examples, colors):
    Fx, Fy = ex["F"] * np.cos(np.radians(ex["ang"])), ex["F"] * np.sin(np.radians(ex["ang"]))
    r = fm.solve_shape(L_M=ex["L_M"], phi_deg=ex["phi_deg"], loads=[
        {"type": "point", "s": ex["s"], "Fx": Fx, "Fy": Fy}
    ], return_curve=True, theta_L_hint_deg=r_free["theta_L_deg"])
    ax2.plot(r["curve_x_mm"], r["curve_y_mm"], color=c, linewidth=2, label=ex["label"])
    ax2.plot(r["x_L"], r["y_L"], "o", color=c, markersize=6)
ax2.plot(0, 0, "ks", markersize=8)
ax2.set_xlabel("x (mm, 로컬)")
ax2.set_ylabel("y (mm, 로컬)")
ax2.set_title("(B) 접촉 예시: 형상이 실제로 이렇게 바뀜", fontweight="bold", fontsize=11)
ax2.set_aspect("equal")
ax2.grid(True, linestyle=":", alpha=0.5)
ax2.legend(fontsize=7.5, loc="lower right")

# ── (C) 접촉위치 x 힘크기 히트맵(평균 이탈량) ──────────────────────────────
ax3 = fig.add_subplot(gs[2])
N_BIN = 10
s_edges = np.linspace(fm.L * 0, fm.L, N_BIN + 1)
f_edges = np.linspace(F_mag.min(), F_mag.max(), N_BIN + 1)
grid = np.full((N_BIN, N_BIN), np.nan)
si = np.clip(np.digitize(s, s_edges) - 1, 0, N_BIN - 1)
fi = np.clip(np.digitize(F_mag, f_edges) - 1, 0, N_BIN - 1)
for i in range(N_BIN):
    for j in range(N_BIN):
        mask = (si == i) & (fi == j)
        if mask.sum() > 2:
            grid[j, i] = deviation[mask].mean()

masked = np.ma.masked_invalid(grid)
cmap2 = SEQ_CMAP.copy()
cmap2.set_bad(color="#f0efec")
im = ax3.imshow(masked, origin="lower", aspect="auto", cmap=cmap2,
                 extent=[s_edges[0], s_edges[-1], f_edges[0], f_edges[-1]])
fig.colorbar(im, ax=ax3, label="평균 이탈량 (mm)")
ax3.set_xlabel("접촉위치 s (mm)")
ax3.set_ylabel("접촉힘 크기 |F| (mN)")
ax3.set_title("(C) 어디를 누르면 더 잘 드러나는가", fontweight="bold", fontsize=11)

fig.suptitle("접촉(외력) 시나리오 스윕 결과 — data/contact_force_scenarios.npz (n=%d)" % len(data),
             fontweight="bold", fontsize=13, y=1.03)

out_path = os.path.join(HERE, "..", "..", "data", "contact_scenarios", "contact_scenarios_overview.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"저장: {out_path}")
