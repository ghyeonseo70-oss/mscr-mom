"""
MOM과 main 자석의 실제 극성 방향(S->N)을 화살표로 그려서
정말로 S극끼리 마주보는지(역배치) 눈으로 확인하는 진단용 플롯.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import force_model as fm

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

L_M = 50
PHI_DEG = 60

r = fm.solve_shape(L_M=L_M, phi_deg=PHI_DEG, loads=[], return_curve=True)
cx, cy = fm.to_board_frame(r["curve_x_mm"], r["curve_y_mm"])
xlm, ylm = fm.to_board_frame(r["x_LM"], r["y_LM"])
xl, yl = fm.to_board_frame(r["x_L"], r["y_L"])

theta_LM = np.radians(r["theta_LM_deg"])  # 로컬 접선각 (전방 방향)
theta_L = np.radians(r["theta_L_deg"])

# MOM(M1) 모멘트 방향 = theta_LM + pi (후방, S극은 그 반대인 전방)
mom_moment_dir = theta_LM + np.pi
mom_S_dir = theta_LM  # S극이 향한 방향(전방, main 쪽)

# main(M2) 모멘트 방향 = theta_L (전방, S극은 그 반대인 후방)
main_moment_dir = theta_L
main_S_dir = theta_L + np.pi  # S극이 향한 방향(후방, MOM 쪽)

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(cx, cy, color="black", linewidth=2, alpha=0.4, label="굽힘 곡선")
ax.plot(fm.BOARD_BASE_X, 0, "ks", markersize=10, label="베이스")

ARROW_LEN = 12  # 시각화용 화살표 길이(mm), 실제 자석 크기와 무관


def draw_magnet(ax, x, y, moment_dir, color, label_prefix):
    # 모멘트 벡터(S->N) 방향으로 막대를 그림: S끝 = -moment_dir 쪽, N끝 = +moment_dir 쪽
    dx, dy = np.cos(moment_dir), np.sin(moment_dir)
    x_S = x - dx * ARROW_LEN / 2
    y_S = y - dy * ARROW_LEN / 2
    x_N = x + dx * ARROW_LEN / 2
    y_N = y + dy * ARROW_LEN / 2
    ax.plot([x_S, x_N], [y_S, y_N], color=color, linewidth=5, solid_capstyle="round", zorder=4)
    ax.plot(x_S, y_S, "o", color="white", markersize=13, markeredgecolor=color, markeredgewidth=2.5, zorder=5)
    ax.plot(x_N, y_N, "o", color=color, markersize=13, zorder=5)
    ax.text(x_S, y_S, "S", ha="center", va="center", fontsize=9, fontweight="bold", color=color, zorder=6)
    ax.text(x_N, y_N, "N", ha="center", va="center", fontsize=9, fontweight="bold", color="white", zorder=6)
    return x_S, y_S, x_N, y_N


# 보드 좌표계에서는 to_board_frame이 (x_board=90+y_local, y_board=x_local) 이므로
# 로컬 각도 -> 보드 각도 변환도 동일 매핑 적용: 보드기준 각도 = local각도를 90도 회전한 것과 동치
# (x_board가 local_y에, y_board가 local_x에 대응하므로 각도도 90도 만큼 축이 바뀜)
def local_angle_to_board(theta_local):
    return theta_local - np.pi / 2  # local (cos,sin) -> board frame으로 축 스왑 시 -90도 회전


mom_moment_board = local_angle_to_board(mom_moment_dir)
main_moment_board = local_angle_to_board(main_moment_dir)

xS1, yS1, xN1, yN1 = draw_magnet(ax, xlm, ylm, mom_moment_board, "#2a78d6", "MOM")
xS2, yS2, xN2, yN2 = draw_magnet(ax, xl, yl, main_moment_board, "#e34948", "main")

ax.plot([], [], color="#2a78d6", linewidth=5, label="MOM (파랑)")
ax.plot([], [], color="#e34948", linewidth=5, label="main (빨강)")
ax.plot([], [], "o", color="white", markeredgecolor="black", markersize=10, label="S극")
ax.plot([], [], "o", color="black", markersize=10, label="N극")

ax.set_xlabel("x (mm) — 보드 좌표")
ax.set_ylabel("y (mm) — 보드 좌표")
ax.set_title(f"MOM·main 자석의 실제 극성 방향 (L_M={L_M}mm, φ={PHI_DEG}°)\nMOM의 S극과 main의 S극이 서로를 향해야 정상(역배치)",
             fontweight="bold")
ax.set_xlim(60, 170)
ax.set_ylim(-10, 90)
ax.set_aspect("equal")
ax.grid(True, linestyle=":", alpha=0.5)
ax.legend(loc="upper left", fontsize=9)

plt.tight_layout()
out_path = "../../data/force_model/polarity_check.png"
plt.savefig(out_path, dpi=150)
print(f"저장: {out_path}")

# S극끼리 마주보는지 수치로도 확인: MOM의 S극점과 main의 S극점 사이 거리가
# 두 자석 중심 사이 거리보다 짧아야 함(서로 가까이 마주본다는 뜻)
center_dist = np.hypot(xl - xlm, yl - ylm)
s_to_s_dist = np.hypot(xS2 - xS1, yS2 - yS1)
print(f"\nMOM-main 중심간 거리: {center_dist:.2f}mm")
print(f"MOM S극 - main S극 거리: {s_to_s_dist:.2f}mm  (중심간 거리보다 짧으면 S극끼리 마주보는 것)")
print(f"-> {'S극끼리 마주봄 (정상)' if s_to_s_dist < center_dist else '마주보지 않음 (문제)'}")
