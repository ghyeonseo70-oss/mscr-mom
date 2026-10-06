"""push_depth=0.20mm(방금 메쉬 세밀화+min_inc 수정으로 수렴 성공한 케이스)가 실제로 얼마나
미는 건지 카테터 형상 위에서 보여주는 그림. L_M=12.5mm, phi=-120deg, s=30mm 기준
(F_mag=0.0126mN, _test_bigdepth_fix.py 실행 결과)."""
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle

import force_model as fm

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

L_M, PHI, S_TARGET, DEPTH = 12.5, -120.0, 30.0, 0.20
# 실측 FEA 값 (2026-10-06, _test_bigdepth_fix.py: 메쉬 세밀화+min_inc 수정 후 수렴 성공)
FMAG_MN = 0.0126
NORMAL = np.array([-0.9994336846170133, -0.0336498150494231])
BALL_CENTER = np.array([94.54477322, 29.14031665])
BALL_R = 0.4
TIP_UX, TIP_UY = 0.45341, 0.39119  # mm, 보드좌표계(run_contact.py TIP shift 출력)
TIP_SHIFT_MAG = math.hypot(TIP_UX, TIP_UY)
TIP_ROT_DEG = -0.490
EXAGGERATION = 1  # 확대 없이 실제 크기 그대로

r = fm.solve_shape(L_M=L_M, phi_deg=PHI, loads=[], return_curve=True)
cs, cx, cy = r["curve_s_mm"], r["curve_x_mm"], r["curve_y_mm"]
sort_idx = np.argsort(cs)
bx, by = fm.to_board_frame(cx[sort_idx], cy[sort_idx])
tip_x0, tip_y0 = fm.to_board_frame(r["x_L"], r["y_L"])

nearest_i = np.argmin(np.abs(cs - S_TARGET))
contact_x0, contact_y0 = fm.to_board_frame(cx[nearest_i], cy[nearest_i])

fig, ax = plt.subplots(figsize=(10, 10))

ax.plot(bx, by, color="#2451A3", linewidth=3, label=f"카테터 형상 (L_M={L_M:.1f}mm, phi={PHI:.0f}°)", zorder=3)
ax.plot(bx[0], by[0], "ks", markersize=11, zorder=5, label="베이스(고정단)")

# 볼 인덴터(접촉 지점) - 실제 크기 그대로
ball = Circle(BALL_CENTER, BALL_R, facecolor="#95A5A6", edgecolor="black", linewidth=1, zorder=4, alpha=0.9)
ax.add_patch(ball)
ax.annotate(f"접촉점 s={S_TARGET:.0f}mm\n압입깊이={DEPTH:.2f}mm", BALL_CENTER, textcoords="offset points",
            xytext=(-85, 20), fontsize=11, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color="gray", linewidth=0.8))

# 미는 힘 화살표
push_len = 6
push_start = BALL_CENTER + NORMAL * (BALL_R + 1.0)
ax.add_patch(FancyArrowPatch(tuple(push_start), tuple(push_start - NORMAL * push_len),
                              arrowstyle="-|>", mutation_scale=22, color="#C0392B", linewidth=2.5, zorder=6))
ax.annotate(f"미는 힘\nF_mag = {FMAG_MN:.4f}mN\n(= {FMAG_MN*1000:.1f}uN, {DEPTH:.2f}mm 눌렀을 때)",
            tuple(push_start - NORMAL * push_len), textcoords="offset points",
            xytext=(15, -45), fontsize=11, color="#C0392B", fontweight="bold")

# 팁(자유단) 변위 - 실측값 그대로(작은 점) + 확대(화살표)
ax.plot(tip_x0, tip_y0, "^", color="#555", markersize=11, zorder=5, label="팁(자유단, 무접촉 기준)")
tip_shifted = np.array([tip_x0, tip_y0]) + np.array([TIP_UX, TIP_UY]) * EXAGGERATION
ax.add_patch(FancyArrowPatch((tip_x0, tip_y0), tuple(tip_shifted),
                              arrowstyle="-|>", mutation_scale=20, color="#27AE60", linewidth=2.5, zorder=6))
exag_note = "실제 크기 그대로, 확대 없음" if EXAGGERATION == 1 else f"그림엔 {EXAGGERATION}배 확대해서 표시"
ax.annotate(f"팁 변위 = {TIP_SHIFT_MAG:.3f}mm, 회전 {TIP_ROT_DEG:.2f}°\n({exag_note})",
            tuple(tip_shifted), textcoords="offset points", xytext=(10, -26),
            fontsize=11, color="#27AE60", fontweight="bold")
ax.set_ylim(top=by.max() + 12)

ax.set_xlabel("x (mm, 보드좌표)")
ax.set_ylabel("y (mm, 보드좌표)")
ax.set_title(f"push_depth={DEPTH:.2f}mm(방금 수렴 성공한 더 깊은 압입)로 밀면 어떻게 되나\n"
             f"-> F_mag={FMAG_MN*1000:.1f}uN, 팁은 {TIP_SHIFT_MAG:.3f}mm/{abs(TIP_ROT_DEG):.2f}° 움직임",
             fontweight="bold", fontsize=13)
ax.legend(loc="lower left", fontsize=10)
ax.set_aspect("equal")
ax.grid(True, linestyle=":", alpha=0.4)

plt.tight_layout()
out_path = "../../data/force_model/force_magnitude_example_depth0p20mm.png"
plt.savefig(out_path, dpi=150)
print(f"저장: {out_path}")
print(f"팁 변위 크기(실제, 확대 안 함): {TIP_SHIFT_MAG:.4f}mm = {TIP_SHIFT_MAG*1000:.1f}um, 회전={TIP_ROT_DEG:.3f}deg")
