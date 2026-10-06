"""F_mag≈0.05mN이 실제로 얼마나 "약하게 미는" 힘인지 카테터 형상 위에서 보여주는 그림.
L_M=50mm, phi=-120deg, s=15mm 실측 FEA 지점(F_mag=0.0486mN) 기준."""
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle

import force_model as fm

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

L_M, PHI, S_TARGET = 50.0, -120.0, 15.0
# 실측 FEA 값 (fea_lm_phi_pos_matv2_all.json, L_M=50,phi=-120,beta=0,s=15)
FMAG_MN = 0.0486
NORMAL = np.array([-0.9834412087572784, 0.18122745078498254])
BALL_CENTER = np.array([89.97155585899037, 15.173562046245172])
BALL_R = 0.4
TIP_UX, TIP_UY = 0.6561674728260869, 0.0332908066847826  # mm, 보드좌표계
TIP_SHIFT_MAG = math.hypot(TIP_UX, TIP_UY)
EXAGGERATION = 10  # 시각화용 확대 배율

r = fm.solve_shape(L_M=L_M, phi_deg=PHI, loads=[], return_curve=True)
cs, cx, cy = r["curve_s_mm"], r["curve_x_mm"], r["curve_y_mm"]
sort_idx = np.argsort(cs)
cs_sorted = cs[sort_idx]
bx, by = fm.to_board_frame(cx[sort_idx], cy[sort_idx])
tip_x0, tip_y0 = fm.to_board_frame(r["x_L"], r["y_L"])

nearest_i = np.argmin(np.abs(cs - S_TARGET))
contact_x0, contact_y0 = fm.to_board_frame(cx[nearest_i], cy[nearest_i])

fig, ax = plt.subplots(figsize=(10, 10))

ax.plot(bx, by, color="#2451A3", linewidth=3, label=f"카테터 형상 (L_M={L_M:.0f}mm, phi={PHI:.0f}°)", zorder=3)
ax.plot(bx[0], by[0], "ks", markersize=11, zorder=5, label="베이스(고정단)")

# 볼 인덴터(접촉 지점) - 실제 크기 그대로
ball = Circle(BALL_CENTER, BALL_R, facecolor="#95A5A6", edgecolor="black", linewidth=1, zorder=4, alpha=0.9)
ax.add_patch(ball)
ax.annotate(f"접촉점 s={S_TARGET:.0f}mm", BALL_CENTER, textcoords="offset points",
            xytext=(-70, 25), fontsize=11, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color="gray", linewidth=0.8))

# 미는 힘 화살표 (실제 크기는 아주 작아서 - 방향만 표시, 크기는 라벨로)
push_len = 6
push_start = BALL_CENTER + NORMAL * (BALL_R + 1.0)
ax.add_patch(FancyArrowPatch(tuple(push_start), tuple(push_start - NORMAL * push_len),
                              arrowstyle="-|>", mutation_scale=22, color="#C0392B", linewidth=2.5, zorder=6))
ax.annotate(f"미는 힘\nF_mag = {FMAG_MN:.3f}mN\n(= {FMAG_MN*1000:.1f}uN, 0.1mm 눌렀을 때)",
            tuple(push_start - NORMAL * push_len), textcoords="offset points",
            xytext=(15, -35), fontsize=11, color="#C0392B", fontweight="bold")

# 팁(자유단) 변위 - 실측값 그대로(왼쪽 작은 점) + 확대(오른쪽 화살표)
ax.plot(tip_x0, tip_y0, "^", color="#555", markersize=11, zorder=5, label="팁(자유단, 무접촉 기준)")
tip_shifted = np.array([tip_x0, tip_y0]) + np.array([TIP_UX, TIP_UY]) * EXAGGERATION
ax.add_patch(FancyArrowPatch((tip_x0, tip_y0), tuple(tip_shifted),
                              arrowstyle="-|>", mutation_scale=20, color="#27AE60", linewidth=2.5, zorder=6))
ax.annotate(f"팁 변위 = {TIP_SHIFT_MAG:.3f}mm\n(그림엔 {EXAGGERATION}배 확대해서 표시)",
            tuple(tip_shifted), textcoords="offset points", xytext=(10, -22),
            fontsize=11, color="#27AE60", fontweight="bold")
ax.set_ylim(top=by.max() + 12)

ax.set_xlabel("x (mm, 보드좌표)")
ax.set_ylabel("y (mm, 보드좌표)")
ax.set_title(f"F_mag=0.05mN(={FMAG_MN*1000:.1f}uN)이 실제로 얼마나 약한 힘인가\n"
             f"-> 0.1mm를 눌러도 팁은 {TIP_SHIFT_MAG:.3f}mm(1mm도 안 되게)밖에 안 움직임",
             fontweight="bold", fontsize=13)
ax.legend(loc="lower left", fontsize=10)
ax.set_aspect("equal")
ax.grid(True, linestyle=":", alpha=0.4)

plt.tight_layout()
out_path = "../../data/force_model/force_magnitude_example_0p05mN.png"
plt.savefig(out_path, dpi=150)
print(f"저장: {out_path}")
print(f"팁 변위 크기(실제, 확대 안 함): {TIP_SHIFT_MAG:.4f}mm = {TIP_SHIFT_MAG*1000:.1f}um")
