"""EN_FACTOR(K1=EN*K2)이 1배(균일 강성)일 때와 2배(현재 설정)일 때 자유단 형상이
얼마나 달라지는지 직접 비교. K1/K2 자체 값은 안 바꾸고 EN_FACTOR만 임시로 오버라이드해서
같은 L_M/phi 조합에 대해 두 형상을 겹쳐 그림."""
import numpy as np
import matplotlib.pyplot as plt
import force_model as fm

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

LM_RATIOS = [0.25, 0.5, 0.75]
PHI_LIST = [30, 60, 90, 120, 150]
PHI_COLORS = {30: "#D95319", 60: "#EDB120", 90: "#7E2F8E", 120: "#77AC30", 150: "#4DBEEE"}


def solve_with_en(L_M, phi_deg, en_factor, hint=None):
    """K1만 EN_FACTOR*K2로 임시 오버라이드해서 푼다(K2, M1, M2 등 나머지는 그대로)."""
    orig_K1 = fm.K1
    fm.K1 = en_factor * fm.K2
    try:
        return fm.solve_shape(L_M=L_M, phi_deg=phi_deg, loads=[], return_curve=True,
                               theta_L_hint_deg=hint)
    finally:
        fm.K1 = orig_K1


fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

for ax, ratio in zip(axes, LM_RATIOS):
    L_M = ratio * fm.L
    for phi in PHI_LIST:
        c = PHI_COLORS[phi]
        r1 = solve_with_en(L_M, phi, en_factor=1.0)
        r2 = solve_with_en(L_M, phi, en_factor=2.0, hint=r1["theta_L_deg"])
        ax.plot(r1["curve_x_mm"] / 10, r1["curve_y_mm"] / 10, color=c, linewidth=1.6,
                 linestyle="--", alpha=0.55)
        ax.plot(r2["curve_x_mm"] / 10, r2["curve_y_mm"] / 10, color=c, linewidth=2.0)
    ax.plot(0, 0, "ks", markersize=7)
    ax.set_title(f"L_M/L = {ratio}", fontweight="bold")
    ax.set_xlabel("x (cm)")
    ax.set_xlim(-3, 10)
    ax.set_ylim(-9, 9)
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.axhline(0, color="gray", linewidth=0.6)

axes[0].set_ylabel("y (cm)")

from matplotlib.lines import Line2D
style_handles = [
    Line2D([0], [0], color="black", lw=1.6, linestyle="--", alpha=0.55, label="EN=1 (K1=K2, 균일)"),
    Line2D([0], [0], color="black", lw=2.0, label="EN=2 (K1=2×K2, 현재 설정)"),
]
color_handles = [Line2D([0], [0], color=PHI_COLORS[p], lw=2, label=f"φ=±{p}°") for p in PHI_LIST]
fig.legend(handles=style_handles + color_handles, loc="lower center", ncol=7,
           bbox_to_anchor=(0.5, -0.06), fontsize=10)

fig.suptitle("K1=K2(점선, EN=1) vs K1=2×K2(실선, EN=2) 자유단 형상 비교", fontweight="bold", fontsize=14)
plt.tight_layout(rect=[0, 0.06, 1, 1])
out_path = "../../data/force_model/en_factor_compare.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"저장: {out_path}")
