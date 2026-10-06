"""phi=30도 예시로 "Fx_total_N=F*cos(theta), Fy_total_N=-F*sin(theta)" 기하학적 분해를
그림으로 설명. L_M=50mm, s=20mm 실측 FEA 지점(F_mag=0.0225mN) 기준."""
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Arc

import force_model as fm

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

L_M, PHI = 50.0, 30.0
S_TARGET = 20.0
# 실측 FEA 값 (fea_lm_phi_pos_matv2_all.json, L_M=50,phi=30,beta=0,s=20)
FX_REAL_MN, FY_REAL_MN, FMAG_MN = 0.020364, -0.009186, 0.022455

r = fm.solve_shape(L_M=L_M, phi_deg=PHI, loads=[], return_curve=True)
cs, cx, cy, cth = r["curve_s_mm"], r["curve_x_mm"], r["curve_y_mm"], r["curve_theta_deg"]
# curve_s_mm이 여러 구간(K1/K_RIGID/K2) shooting 결과를 이어붙인 거라 전역적으로 단조증가가
# 아님(2026-08-26 확인) - np.interp는 정렬을 가정하므로 그대로 쓰면 엉뚱한 값이 나옴.
# 정렬 후 보간(전체 곡선 그리기용)과 별개로, 목표 s에 가장 가까운 점을 직접 찾아 theta를 구함.
sort_idx = np.argsort(cs)
cs_sorted, cx_sorted, cy_sorted = cs[sort_idx], cx[sort_idx], cy[sort_idx]
bx, by = fm.to_board_frame(cx_sorted, cy_sorted)

nearest_i = np.argmin(np.abs(cs - S_TARGET))
x0, y0 = fm.to_board_frame(cx[nearest_i], cy[nearest_i])
theta_local = cth[nearest_i]
theta_rad = math.radians(theta_local)
tangent = np.array([math.sin(theta_rad), math.cos(theta_rad)])   # 보드 접선
normal = np.array([-math.cos(theta_rad), math.sin(theta_rad)])   # 보드 법선(beta=0)

fig, axes = plt.subplots(1, 2, figsize=(15, 7), gridspec_kw={"width_ratios": [1.3, 1]})
ax = axes[0]

# 튜브 곡선
ax.plot(bx, by, color="#2451A3", linewidth=3, label=f"튜브 형상 (L_M={L_M:.0f}mm, phi={PHI:.0f}°)")
ax.plot(bx[0], by[0], "ks", markersize=10, label="베이스(고정단)")
ax.plot(bx[-1], by[-1], "^", color="#555", markersize=10, label="팁(자유단)")
ax.plot(x0, y0, "o", color="black", markersize=9, zorder=5)
ax.annotate(f"접촉점 s={S_TARGET:.0f}mm", (x0, y0), textcoords="offset points",
            xytext=(-95, 12), fontsize=12, fontweight="bold")

# 접선 방향 (참고용, 얇은 회색 점선)
tan_len = 18
ax.plot([x0 - tangent[0]*tan_len*0.3, x0 + tangent[0]*tan_len],
        [y0 - tangent[1]*tan_len*0.3, y0 + tangent[1]*tan_len],
        "--", color="gray", linewidth=1)
ax.annotate("접선 방향", (x0 + tangent[0]*tan_len, y0 + tangent[1]*tan_len),
            fontsize=10, color="gray")

# 법선(접촉/힘) 방향 화살표 - 실제 F_mag 크기에 비례한 길이로(가독성 위해 확대)
scale = 1800  # mN -> mm 화살표 길이 배율(시각화용, 실제 비율 유지하며 확대)
force_vec = np.array([FX_REAL_MN, FY_REAL_MN]) * scale
ax.add_patch(FancyArrowPatch((x0, y0), (x0 + force_vec[0], y0 + force_vec[1]),
                              arrowstyle="-|>", mutation_scale=25, color="#C0392B", linewidth=3, zorder=6))
ax.annotate(f"반력 F (크기={FMAG_MN:.4f}mN)", (x0 + force_vec[0], y0 + force_vec[1]),
            textcoords="offset points", xytext=(10, -18), fontsize=12, color="#C0392B", fontweight="bold")

# Fx, Fy 성분(투영) - 점선 박스
ax.plot([x0, x0 + force_vec[0]], [y0, y0], ":", color="#1F77B4", linewidth=2.5)
ax.plot([x0 + force_vec[0], x0 + force_vec[0]], [y0, y0 + force_vec[1]], ":", color="#1F77B4", linewidth=2.5)
ax.annotate(f"Fx_total_N = F·cosθ = {FX_REAL_MN:.4f}mN", (x0 + force_vec[0]/2, y0 + 3),
            fontsize=11, color="#1F77B4", ha="center", fontweight="bold")
ax.annotate(f"Fy_total_N =\n-F·sinθ = {FY_REAL_MN:.4f}mN", (x0 + force_vec[0] + 4, y0 + force_vec[1]/2),
            fontsize=11, color="#1F77B4", fontweight="bold")

# theta 각도 호 표시(접선-보드y축 사이)
arc = Arc((x0, y0), 22, 22, angle=0, theta1=90 - theta_local, theta2=90, color="#27AE60", linewidth=2)
ax.add_patch(arc)
ax.annotate(f"θ={theta_local:.1f}°", (x0 + 8, y0 + 10), fontsize=12, color="#27AE60", fontweight="bold")

ax.set_xlabel("x (mm, 보드좌표)")
ax.set_ylabel("y (mm, 보드좌표)")
ax.set_title(f"phi={PHI:.0f}° 예시: s={S_TARGET:.0f}mm 접촉점에서 힘 벡터의 기하학적 분해", fontweight="bold", fontsize=13)
ax.legend(loc="upper left", fontsize=10)
ax.set_aspect("equal")
ax.grid(True, linestyle=":", alpha=0.4)

# ---- 오른쪽: 단위원으로 cos/sin 개념 설명 ----
ax2 = axes[1]
circle = plt.Circle((0, 0), 1, fill=False, color="gray", linewidth=1.5)
ax2.add_patch(circle)
ax2.axhline(0, color="black", linewidth=0.8)
ax2.axvline(0, color="black", linewidth=0.8)
px, py = math.cos(theta_rad), math.sin(theta_rad)
ax2.add_patch(FancyArrowPatch((0, 0), (px, py), arrowstyle="-|>", mutation_scale=22, color="#C0392B", linewidth=2.5))
ax2.plot([px, px], [0, py], ":", color="#1F77B4", linewidth=2)
ax2.plot([0, px], [0, 0], ":", color="#1F77B4", linewidth=2)
ax2.annotate(f"cos(θ)={px:.2f}\n(항상 -90°~90°에서 양수\n→ Fx_total_N이 늘 한쪽 부호)",
             (px/2, -0.28), fontsize=10.5, color="#1F77B4", ha="center", fontweight="bold")
ax2.annotate(f"sin(θ)={py:.2f}\n(θ가 0을 지나면 부호 반전\n→ Fy_total_N이 진동)",
             (px + 0.08, py/2), fontsize=10.5, color="#1F77B4", fontweight="bold")
arc2 = Arc((0, 0), 0.6, 0.6, angle=0, theta1=0, theta2=theta_local, color="#27AE60", linewidth=2)
ax2.add_patch(arc2)
ax2.annotate(f"θ={theta_local:.1f}°", (0.35, 0.12), fontsize=11, color="#27AE60", fontweight="bold")
ax2.set_xlim(-1.4, 1.6)
ax2.set_ylim(-1.4, 1.4)
ax2.set_aspect("equal")
ax2.set_title("단위원으로 본 cos/sin 관계", fontweight="bold", fontsize=13)
ax2.set_xticks([]); ax2.set_yticks([])

plt.tight_layout()
out_path = "../../data/force_model/fx_fy_geometry_example_phi30.png"
plt.savefig(out_path, dpi=150)
print(f"저장: {out_path}")
print(f"theta_local(s={S_TARGET})={theta_local:.2f}deg, normal(board)={normal}")
