"""push_depth=0.30mm(아직 실제 FEA로 검증 안 됨 - 테스트 진행 중)를 밀면 대략 어느 정도
변할지 "추정"해서 보여주는 그림. L_M=12.5mm, phi=-120deg, s=30mm 기준.

**주의: 이건 진짜 FEA 결과가 아니라 추정치임.** 0.20mm 실측 결과(F_mag=0.0123mN,
팁변위=0.598mm, 팁회전=0.489deg, _test_ball_size.py)에 "깊이-변형 거의 선형(깊이 2배시
변형 2.2배, PROJECT_STATUS.md 34번에서 확인된 관계)"을 적용해서 0.30mm(1.5배)로
외삽한 값(배율 1.586배) - 실제 FEA 결과 나오면 이 스크립트의 숫자를 교체할 것."""
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle

import force_model as fm

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

L_M, PHI, S_TARGET, DEPTH = 12.5, -120.0, 30.0, 0.30
# 추정치(ESTIMATE) - 2026-10-06, 0.20mm 실측값(F=0.0123mN, 팁변위=0.598mm/0.489deg,
# ball_r=0.4 케이스)에 깊이-변형 스케일링(배율 1.586, 지수 1.138)을 적용해서 외삽.
IS_ESTIMATE = True
SCALE = 1.586
FMAG_MN = 0.0123 * SCALE
NORMAL = np.array([-0.9994336846170133, -0.0336498150494231])
BALL_CENTER = np.array([94.54477322, 29.14031665])
BALL_R = 0.4
TIP_UX, TIP_UY = 0.45341 * SCALE, 0.39119 * SCALE  # mm, 보드좌표계 - 추정(0.20mm 실측값 외삽)
TIP_SHIFT_MAG = math.hypot(TIP_UX, TIP_UY)
TIP_ROT_DEG = -0.489 * SCALE
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

# 미는 힘 화살표 (추정치라 점선 + 주황색으로 확정 결과와 구분)
push_len = 6
push_start = BALL_CENTER + NORMAL * (BALL_R + 1.0)
ax.add_patch(FancyArrowPatch(tuple(push_start), tuple(push_start - NORMAL * push_len),
                              arrowstyle="-|>", mutation_scale=22, color="#E67E22", linewidth=2.5,
                              linestyle="--", zorder=6))
ax.annotate(f"미는 힘 (추정)\nF_mag 약  {FMAG_MN:.4f}mN\n(= {FMAG_MN*1000:.1f}uN, {DEPTH:.2f}mm 눌렀을 때)",
            tuple(push_start - NORMAL * push_len), textcoords="offset points",
            xytext=(15, -45), fontsize=11, color="#E67E22", fontweight="bold")

# 팁(자유단) 변위 - 추정값(점선 화살표로 확정 결과와 구분)
ax.plot(tip_x0, tip_y0, "^", color="#555", markersize=11, zorder=5, label="팁(자유단, 무접촉 기준)")
tip_shifted = np.array([tip_x0, tip_y0]) + np.array([TIP_UX, TIP_UY]) * EXAGGERATION
ax.add_patch(FancyArrowPatch((tip_x0, tip_y0), tuple(tip_shifted),
                              arrowstyle="-|>", mutation_scale=20, color="#E67E22", linewidth=2.5,
                              linestyle="--", zorder=6))
exag_note = "실제 크기 그대로, 확대 없음" if EXAGGERATION == 1 else f"그림엔 {EXAGGERATION}배 확대해서 표시"
ax.annotate(f"팁 변위 (추정) 약  {TIP_SHIFT_MAG:.3f}mm, 회전 {abs(TIP_ROT_DEG):.2f}°\n({exag_note})",
            tuple(tip_shifted), textcoords="offset points", xytext=(10, -26),
            fontsize=11, color="#E67E22", fontweight="bold")
ax.set_ylim(top=by.max() + 12)

ax.set_xlabel("x (mm, 보드좌표)")
ax.set_ylabel("y (mm, 보드좌표)")
ax.set_title(f"[추정치, 실제 FEA 검증 전] push_depth={DEPTH:.2f}mm로 밀면 대략 어느 정도일까\n"
             f"-> 0.20mm 실측값을 깊이-변형 스케일링으로 외삽: F_mag약 {FMAG_MN*1000:.1f}uN, "
             f"팁 약 {TIP_SHIFT_MAG:.3f}mm/{abs(TIP_ROT_DEG):.2f}°",
             fontweight="bold", fontsize=13, color="#7a4a10")
ax.legend(loc="lower left", fontsize=10)
ax.set_aspect("equal")
ax.grid(True, linestyle=":", alpha=0.4)

plt.tight_layout()
out_path = "../../data/force_model/force_magnitude_example_depth0p30mm_ESTIMATE.png"
plt.savefig(out_path, dpi=150)
print(f"저장: {out_path}")
print(f"[추정치, 실제 결과 아님] 팁 변위: {TIP_SHIFT_MAG:.4f}mm, 회전={TIP_ROT_DEG:.3f}deg")
