"""
곡선을 구간별(segment1=K1, MOM 강체구간, segment2=K2)로 색을 나눠서
정말로 논문 식대로 서로 다른 곡률을 갖는지 명확하게 보여주는 진단용 플롯.
"""
import numpy as np
import matplotlib.pyplot as plt
import force_model as fm

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

L_M = 50
PHI_DEG = 60

r = fm.solve_shape(L_M=L_M, phi_deg=PHI_DEG, loads=[], return_curve=True)
s = r["curve_s_mm"]
theta = r["curve_theta_deg"]
x, y = fm.to_board_frame(r["curve_x_mm"], r["curve_y_mm"])

# 세그먼트 이음매에서 s가 중복되는 지점 제거 (미분 시 0으로 나누기 방지)
keep = np.concatenate([[True], np.abs(np.diff(s)) > 1e-9])
s, theta, x, y = s[keep], theta[keep], x[keep], y[keep]

a1 = L_M - fm.H_M / 2
a2 = L_M + fm.H_M / 2

seg1 = s < a1
rigid = (s >= a1) & (s <= a2)
seg2 = s > a2

fig, axes = plt.subplots(1, 2, figsize=(14, 7))

# 왼쪽: 전체 곡선, 구간별 색 (실제 스케일)
ax = axes[0]
ax.plot(x[seg1], y[seg1], color="#2a78d6", linewidth=3, label=f"구간1 (K1={fm.K1:.2e}, s=0~{a1:.0f}mm)")
ax.plot(x[rigid], y[rigid], color="#e34948", linewidth=4, label=f"MOM 강체 (s={a1:.0f}~{a2:.0f}mm)")
ax.plot(x[seg2], y[seg2], color="#1baf7a", linewidth=3, label=f"구간2 (K2={fm.K2:.2e}, s={a2:.0f}~100mm)")
xlm, ylm = fm.to_board_frame(r["x_LM"], r["y_LM"])
xl, yl = fm.to_board_frame(r["x_L"], r["y_L"])
ax.plot(xlm, ylm, "k^", markersize=12, zorder=6, label="MOM 중심")
ax.plot(xl, yl, "ko", markersize=12, zorder=6, label="팁(main)")
ax.plot(fm.BOARD_BASE_X, 0, "ks", markersize=10, label="베이스")
ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
ax.set_title(f"실제 스케일 (L_M={L_M}mm, φ={PHI_DEG}°)\nK1/K2={fm.K1/fm.K2:.2f}배 (논문 Fig.3(e)-(h) 디지털화 피팅값)", fontweight="bold")
ax.set_aspect("equal")
ax.grid(True, linestyle=":", alpha=0.5)
ax.legend(loc="upper left", fontsize=8)

# 오른쪽: 곡률(dθ/ds)을 s에 대해 직접 그림 -> 구간별로 정말 다른지 수치로 확인
ax2 = axes[1]
theta_rad = np.radians(theta)
dtheta_ds = np.gradient(theta_rad, s)
ax2.plot(s[seg1], dtheta_ds[seg1], color="#2a78d6", linewidth=2.5, label="구간1 곡률 (dθ/ds)")
ax2.plot(s[rigid], dtheta_ds[rigid], color="#e34948", linewidth=3, label="MOM 강체 (곡률=0 이어야함)")
ax2.plot(s[seg2], dtheta_ds[seg2], color="#1baf7a", linewidth=2.5, label="구간2 곡률 (dθ/ds)")
ax2.axhline(0, color="gray", linewidth=0.8)
ax2.set_xlabel("s (mm, arc length)")
ax2.set_ylabel("곡률 dθ/ds (rad/mm)")
ax2.set_title("구간별 곡률 — 이게 실제로 다른 값(계단모양)이어야\n논문 식(5)-(6)대로 계산된 것", fontweight="bold")
ax2.grid(True, linestyle=":", alpha=0.5)
ax2.legend(loc="best", fontsize=9)

plt.tight_layout()
out_path = "../../data/force_model/segments_detail.png"
plt.savefig(out_path, dpi=150)
print(f"저장: {out_path}")

print(f"\n구간1 평균 곡률: {np.mean(dtheta_ds[seg1]):.5f} rad/mm")
print(f"MOM 강체 평균 곡률: {np.mean(dtheta_ds[rigid]):.5f} rad/mm (0에 가까워야 함)")
print(f"구간2 평균 곡률: {np.mean(dtheta_ds[seg2]):.5f} rad/mm")
print(f"K1/K2 비율: {fm.K1/fm.K2:.3f}")
