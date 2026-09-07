"""
개념 설명용 그림 3장: (1) 5구간 분류, (2) beta(원주각) 개념, (3) 힘(Fx,Fy) 측정 개념.
plot_nn2_results.py의 카테터 곡선 그리는 스타일(무접촉 기준형상/접촉시 카테터 형상 등)을
그대로 재사용. 실제 데이터 기준: L_M=50mm, phi=60도 (스윕에서 가장 많이 쓰인 대표 조합).
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "force_model"))
import force_model as fm

OUT_DIR = os.path.join(HERE, "..", "..", "..", "data", "contact_scenarios", "fea")

L_M, PHI = 50.0, 60.0
r = fm.solve_shape(L_M=L_M, phi_deg=PHI, loads=[], return_curve=True)
order = np.argsort(r["curve_s_mm"])
cs, cx, cy = r["curve_s_mm"][order], r["curve_x_mm"][order], r["curve_y_mm"][order]


def base_plot():
    fig, ax = plt.subplots(figsize=(7, 6.5))
    ax.set_xlabel("x (mm, 로컬)")
    ax.set_ylabel("y (mm, 로컬)")
    ax.plot(0, 0, "ks", markersize=10, zorder=5, label="베이스(고정단)")
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", alpha=0.5)
    return fig, ax


# ── 1) 5구간 분류 개념도 ──────────────────────────────────────────────
fig, ax = base_plot()
BIN_EDGES = [0, 20, 40, 60, 80, 100]
BIN_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
BIN_LABELS = ["0-20mm", "20-40mm", "40-60mm", "60-80mm", "80-100mm"]
for i in range(5):
    lo, hi = BIN_EDGES[i], BIN_EDGES[i + 1]
    mask = (cs >= lo) & (cs <= hi)
    ax.plot(cx[mask], cy[mask], "-", color=BIN_COLORS[i], linewidth=11,
            solid_capstyle="round", zorder=2.5)
    ax.plot([], [], "-", color=BIN_COLORS[i], linewidth=8, label=f"{i}번 구간 ({BIN_LABELS[i]})")
ax.set_title("카테터 접촉위치 5구간 분류\n(자기장 변화만 보고 5개 중 어느 구간에 접촉했는지 인지)",
              fontweight="bold", fontsize=12)
ax.legend(fontsize=9, loc="best")
out1 = os.path.join(OUT_DIR, "concept_5segments.png")
plt.savefig(out1, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"저장: {out1}")

# ── 2) beta(원주방향 접촉각) 개념도 ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), gridspec_kw={"width_ratios": [1.3, 1]})
ax = axes[0]
ax.plot(0, 0, "ks", markersize=10, zorder=5, label="베이스(고정단)")
ax.plot(cx, cy, "-", color="#aacbe8", linewidth=11, solid_capstyle="round", zorder=2.5)
ax.plot(cx, cy, "-", color="#2a78d6", linewidth=2.2, solid_capstyle="round", zorder=3,
        label="카테터 중심선")
s_mark = 50.0
xt = np.interp(s_mark, cs, cx)
yt = np.interp(s_mark, cs, cy)
ax.plot(xt, yt, "o", color="#e34948", markersize=14, zorder=7,
        markeredgecolor="white", markeredgewidth=1.5, label=f"접촉 위치 (s={s_mark:.0f}mm)")
ax.annotate("여기서 잘라보면\n(오른쪽 단면도) →", xy=(xt, yt), xytext=(xt - 8, yt + 12),
            fontsize=10, ha="center", color="#444444",
            arrowprops=dict(arrowstyle="-|>", color="#444444", linewidth=1.5))
ax.set_xlabel("x (mm, 로컬)")
ax.set_ylabel("y (mm, 로컬)")
ax.set_aspect("equal")
ax.grid(True, linestyle=":", alpha=0.5)
ax.legend(fontsize=9, loc="best")
ax.set_title("카테터 옆면 (접촉위치 s=50mm 지점 표시)", fontweight="bold", fontsize=11)

ax2 = axes[1]
theta = np.linspace(0, 2 * np.pi, 200)
ax2.plot(np.cos(theta), np.sin(theta), "-", color="#2a78d6", linewidth=14, solid_capstyle="round")
beta_defs = [(0, "0도\n(바깥쪽)"), (90, "90도\n(위)"), (180, "180도\n(안쪽)"), (270, "270도\n(아래)")]
for beta_deg, label in beta_defs:
    rad = np.radians(beta_deg)
    x0, y0 = np.cos(rad), np.sin(rad)
    x1, y1 = 1.55 * np.cos(rad), 1.55 * np.sin(rad)
    ax2.annotate("", xy=(x1, y1), xytext=(x0, y0),
                 arrowprops=dict(arrowstyle="-|>", color="#e34948", linewidth=2.2))
    ax2.text(1.85 * np.cos(rad), 1.85 * np.sin(rad), label, ha="center", va="center",
              fontsize=10, fontweight="bold")
ax2.set_xlim(-2.3, 2.3)
ax2.set_ylim(-2.3, 2.3)
ax2.set_aspect("equal")
ax2.axis("off")
ax2.set_title("s=50mm 지점 단면\nbeta = 원주방향 접촉각\n(둥근 카테터 표면을 어느 방향에서 눌렀는지)",
               fontweight="bold", fontsize=11)
fig.suptitle("beta(원주방향 접촉각) 개념도 — 접촉위치(s)와는 별개로,\n"
             "같은 s라도 단면 둘레 어느 지점을 눌렀는지가 beta", fontweight="bold", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.88])
out2 = os.path.join(OUT_DIR, "concept_beta.png")
plt.savefig(out2, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"저장: {out2}")

# ── 3) 힘(Fx,Fy) 측정 개념도 ──────────────────────────────────────────
# 실제 FEA 데이터(L_M=50,phi=60,s=10mm) 기준: tip 변위로부터 접촉시 형상 근사
s_force = 10.0
tip_ux, tip_uy, tip_theta = 0.8807, 0.3148, -0.6065  # fea_lm_phi_pos_sweep_all.json 실측값
Fx_local, Fy_local = 0.000202, 0.0000633  # Fx_total_N, Fy_total_N (실측)
# 국소->보드좌표 축교환 규칙(이 코드베이스 전체에서 쓰는 것과 동일)
d_xL_local, d_yL_local = tip_uy, tip_ux
xL_load = np.interp(cs.max(), cs, cx) + d_xL_local
yL_load = np.interp(cs.max(), cs, cy) + d_yL_local
Fx_board, Fy_board = Fy_local, Fx_local
F_mag = np.hypot(Fx_board, Fy_board)
F_ang = np.degrees(np.arctan2(Fy_board, Fx_board))

fig, ax = base_plot()
ax.plot(cx, cy, "--", color="#bbbbbb", linewidth=1.5, label="무접촉 기준형상", zorder=2)
# 접촉시 형상(팁만 실측 변위로 근사 이동 - 개념도용 단순화, 중간구간은 기준형상 그대로 표시)
cx_load = cx.copy()
cy_load = cy.copy()
cx_load[-1] += d_xL_local
cy_load[-1] += d_yL_local
ax.plot(cx_load, cy_load, "-", color="#aacbe8", linewidth=11, solid_capstyle="round", zorder=2.5)
ax.plot(cx_load, cy_load, "-", color="#2a78d6", linewidth=2.2, solid_capstyle="round", zorder=3,
        label="접촉시 카테터 형상(팁 변위, 개념도)")
xt = np.interp(s_force, cs, cx)
yt = np.interp(s_force, cs, cy)
ax.plot(xt, yt, "o", color="#e34948", markersize=13, zorder=7,
        markeredgecolor="white", markeredgewidth=1.5, label=f"접촉 위치 (s={s_force:.0f}mm)")
y_lo, y_hi = cy.min() - 5, max(cy.max(), yt) + 15  # 화살표가 위로 뻗어도 잘리지 않게 여유 확보
ax.set_ylim(y_lo, y_hi)
ax.set_aspect("equal", adjustable="box")

f_dir = np.array([Fx_board, Fy_board]) / F_mag
arrow_len = 15.0
ax.annotate("", xy=(xt + f_dir[0] * arrow_len, yt + f_dir[1] * arrow_len), xytext=(xt, yt),
            arrowprops=dict(arrowstyle="-|>", color="#B23A32", linewidth=3, mutation_scale=25), zorder=8)
# 텍스트박스는 데이터좌표가 아니라 축 비율(axes fraction)로 고정 배치해서 화면 밖으로 안 나가게 함
ax.text(0.03, 0.03,
        f"F = (Fx={Fx_board*1000:.3f}, Fy={Fy_board*1000:.3f}) mN\n"
        f"크기(F_mag) = {F_mag*1000:.3f} mN\n각도 = {F_ang:.0f}도",
        transform=ax.transAxes, fontsize=9.5, ha="left", va="bottom", color="#B23A32",
        fontweight="bold", bbox=dict(boxstyle="round", facecolor="white", edgecolor="#B23A32", alpha=0.9))
ax.set_title(f"힘(Fx,Fy) 측정 개념도 — 접촉으로 생긴 반력을\n보드좌표계 Fx,Fy 벡터로 예측 (s={s_force:.0f}mm 예시)",
             fontweight="bold", fontsize=12)
ax.legend(fontsize=9, loc="upper right")
out3 = os.path.join(OUT_DIR, "concept_force.png")
plt.savefig(out3, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"저장: {out3}")
