"""
같은 위치(s_t)에 크기가 다른 힘을 눌렀을 때 굽힘 곡선이 어떻게 달라지는지 시각화.
연속법(continuation): 힘을 잘게 나눠 늘려가며, 매 스텝 직전 해를 다음 스텝의 초기 추측값으로
넘겨서 같은 물리적 가지(branch)를 계속 추적 -> 다른(불안정한) 해로 튀는 문제 방지.
"""
import numpy as np
import matplotlib.pyplot as plt
import force_model as fm

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

L_M = 50        # MOM 위치 (mm)
PHI_DEG = 60    # 외부자기장 방향
S_T = 70        # 누르는 위치 (mm, 고정)
F_MAX_MN = 40
N_STEPS = 81    # 0~40mN을 81 스텝(0.5mN 간격)으로 잘게 나눠 연속법 추적

SNAPSHOT_LEVELS_MN = [0, 2, 5, 10, 20, 40]  # 그래프에 곡선으로 표시할 지점들

# ── 연속법으로 0~F_MAX까지 추적 ──────────────────────────────
force_levels = np.linspace(0, F_MAX_MN, N_STEPS) / 1000.0  # N
theta_hint = None
track = []  # (F_mN, x_L, y_L, x_LM, y_LM, theta_L_deg, curve or None)

for f_n in force_levels:
    f_mn = f_n * 1000
    loads = [] if f_n == 0 else [{"type": "point", "s": S_T, "Fx": 0, "Fy": -f_n}]
    want_curve = any(abs(f_mn - lvl) < 1e-6 for lvl in SNAPSHOT_LEVELS_MN)
    r = fm.solve_shape(L_M=L_M, phi_deg=PHI_DEG, loads=loads, return_curve=want_curve,
                        theta_L_hint_deg=theta_hint)
    theta_hint = r["theta_L_deg"]  # 다음 스텝의 초기 추측값으로 사용 (연속법 핵심)
    x_l, y_l = fm.to_board_frame(r["x_L"], r["y_L"])
    x_lm, y_lm = fm.to_board_frame(r["x_LM"], r["y_LM"])
    if want_curve:
        cx, cy = fm.to_board_frame(r["curve_x_mm"], r["curve_y_mm"])
    else:
        cx = cy = None
    track.append((f_mn, float(x_l), float(y_l), float(x_lm), float(y_lm), r["theta_L_deg"], (cx, cy)))

# ── 그래프 1: 스냅샷 곡선들 (팁=main 자석, 세모=MOM) ──────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
ax = axes[0]
colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(SNAPSHOT_LEVELS_MN)))

for (f_mn, x_l, y_l, x_lm, y_lm, th, curve), c in zip(
        [t for t in track if any(abs(t[0] - lvl) < 1e-6 for lvl in SNAPSHOT_LEVELS_MN)], colors):
    is_zero = f_mn == 0
    cx, cy = curve
    ax.plot(cx, cy, color=("black" if is_zero else c), linewidth=(2.5 if is_zero else 2),
            linestyle=("--" if is_zero else "-"),
            label=("힘 없음 (기준)" if is_zero else f"F={f_mn:.0f}mN"))
    ax.plot(x_l, y_l, "o", color=("black" if is_zero else c), markersize=8)  # 팁(main)
    ax.plot(x_lm, y_lm, "^", color=("black" if is_zero else c), markersize=9,
            markeredgecolor="black", markeredgewidth=0.6)  # MOM

ax.plot(fm.BOARD_BASE_X, 0, "ks", markersize=10)
ax.plot([], [], "o", color="gray", markersize=8, label="팁/main (●)")
ax.plot([], [], "^", color="gray", markersize=9, markeredgecolor="black", label="MOM (▲)")
ax.set_xlabel("x (mm) — 보드 좌표"); ax.set_ylabel("y (mm) — 보드 좌표")
ax.set_title(f"힘 크기별 굽힘 곡선 [보드 좌표계] (연속법 적용)\ns_t={S_T}mm 고정", fontweight="bold")
ax.set_xlim(0, 180); ax.set_ylim(0, 180)
ax.set_aspect("equal")
ax.grid(True, linestyle=":", alpha=0.5)
ax.legend(loc="upper right", fontsize=8.5)

# ── 그래프 2: 힘 크기에 따른 팁·MOM 위치 변화(연속 궤적) ──────────────────────────────
ax2 = axes[1]
fs = [t[0] for t in track]
xls = [t[1] for t in track]
yls = [t[2] for t in track]
xlms = [t[3] for t in track]
ylms = [t[4] for t in track]
sc = ax2.scatter(xls, yls, c=fs, cmap="viridis", s=18, marker="o", label="팁(main) 궤적")
sc2 = ax2.scatter(xlms, ylms, c=fs, cmap="viridis", s=18, marker="^", label="MOM 궤적")
ax2.plot(fm.BOARD_BASE_X, 0, "ks", markersize=10, zorder=5, label="베이스")
cbar = fig.colorbar(sc, ax=ax2)
cbar.set_label("힘 크기 F (mN)")
ax2.set_xlabel("x (mm) — 보드 좌표"); ax2.set_ylabel("y (mm) — 보드 좌표")
ax2.set_title("힘을 0->40mN까지 늘릴 때 [보드 좌표계]\n팁·MOM 위치의 연속적인 이동 궤적", fontweight="bold")
ax2.set_aspect("equal")
ax2.grid(True, linestyle=":", alpha=0.5)
ax2.legend(loc="best", fontsize=9)

plt.tight_layout()
out_path = "../../data/force_model/force_magnitude_comparison.png"
plt.savefig(out_path, dpi=150)
print(f"저장: {out_path}")

print("\n힘 크기별 팁/MOM 위치 (연속법):")
for f_mn, x_l, y_l, x_lm, y_lm, th, _ in track:
    if any(abs(f_mn - lvl) < 1e-6 for lvl in SNAPSHOT_LEVELS_MN):
        print(f"  F={f_mn:5.1f}mN -> tip=({x_l:6.1f}, {y_l:6.1f})  MOM=({x_lm:6.1f}, {y_lm:6.1f})  theta_L={th:7.1f}deg")
