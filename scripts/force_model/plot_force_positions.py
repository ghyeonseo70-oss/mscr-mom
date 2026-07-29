"""
같은 크기의 힘을 팔의 여러 위치(s_t)에 눌렀을 때 굽힘 곡선이 어떻게 달라지는지 시각화.
좌표는 홀센서 보드 전역좌표계(베이스=(90,0), 0~180범위)로 변환해서 표시.
"""
import numpy as np
import matplotlib.pyplot as plt
import force_model as fm

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

L_M = 50       # MOM 위치 (mm)
PHI_DEG = 60   # 외부자기장 방향
FORCE_N = 0.01 # 누르는 힘 크기 (N) = 10 mN
PRESS_POSITIONS = [10, 30, 50, 70, 90]  # 눌리는 위치들 (mm, arc length)

colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(PRESS_POSITIONS)))

fig, ax = plt.subplots(figsize=(7.5, 7.5))

# 기준선: 힘 없을 때
r0 = fm.solve_shape(L_M=L_M, phi_deg=PHI_DEG, loads=[], return_curve=True)
cx0, cy0 = fm.to_board_frame(r0["curve_x_mm"], r0["curve_y_mm"])
xl0, yl0 = fm.to_board_frame(r0["x_L"], r0["y_L"])
xlm0, ylm0 = fm.to_board_frame(r0["x_LM"], r0["y_LM"])
ax.plot(cx0, cy0, color="black", linewidth=2.5, linestyle="--", label="힘 없음 (기준)", zorder=5)
ax.plot(xl0, yl0, "ko", markersize=9, zorder=6, label="팁(main) - 힘없음")
ax.plot(xlm0, ylm0, "k^", markersize=10, zorder=6, label="MOM - 힘없음")

for s_t, c in zip(PRESS_POSITIONS, colors):
    # 누르는 방향: 팔의 접선에 수직에 가깝게, 대략 -y(로컬) 방향으로 누른다고 가정
    r = fm.solve_shape(L_M=L_M, phi_deg=PHI_DEG,
                        loads=[{"type": "point", "s": s_t, "Fx": 0, "Fy": -FORCE_N}],
                        return_curve=True)
    cx, cy = fm.to_board_frame(r["curve_x_mm"], r["curve_y_mm"])
    xl, yl = fm.to_board_frame(r["x_L"], r["y_L"])
    xlm, ylm = fm.to_board_frame(r["x_LM"], r["y_LM"])
    ax.plot(cx, cy, color=c, linewidth=2, label=f"s_t={s_t}mm 누름")
    ax.plot(xl, yl, "o", color=c, markersize=7)  # 팁(main)
    ax.plot(xlm, ylm, "^", color=c, markersize=9, markeredgecolor="black", markeredgewidth=0.7)  # MOM

    # 누르는 지점 표시 (기준 곡선 위, 대략적 위치 참고용)
    idx = np.argmin(np.abs(r0["curve_s_mm"] - s_t))
    ax.plot(cx0[idx], cy0[idx], "x", color=c, markersize=10, markeredgewidth=2)

ax.plot(fm.BOARD_BASE_X, 0, "ks", markersize=10, label="베이스 (고정, x=90)")
ax.plot([], [], "^", color="gray", markersize=9, markeredgecolor="black", label="MOM (▲ = 자석 위치)")
ax.plot([], [], "o", color="gray", markersize=7, label="팁/main (● = 자석 위치)")
ax.set_xlabel("x (mm) — 홀센서 보드 좌표")
ax.set_ylabel("y (mm) — 홀센서 보드 좌표")
ax.set_title(f"누르는 위치(s_t)에 따른 굽힘 곡선 변화 [보드 좌표계]\n(L_M={L_M}mm, φ={PHI_DEG}°, F={FORCE_N*1000:.0f}mN)",
             fontweight="bold")
ax.set_xlim(0, 180)
ax.set_ylim(0, 180)
ax.set_aspect("equal")
ax.grid(True, linestyle=":", alpha=0.5)
ax.legend(loc="upper right", fontsize=8.5)

plt.tight_layout()
out_path = "../../data/force_model/force_position_comparison.png"
plt.savefig(out_path, dpi=150)
print(f"저장: {out_path}")
