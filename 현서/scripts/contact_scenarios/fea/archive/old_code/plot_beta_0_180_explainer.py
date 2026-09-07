"""beta=0도/180도가 정확히 어떤 물리적 위치인지 보여주는 설명 그림.
카테터 단면(원)에서 원주 방향 각도로서의 beta와, 굽힘 평면과의 관계를 시각화.
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "..", "..", "..", "data", "contact_scenarios", "fea")

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

GREEN = "#1baf7a"
RED = "#e34948"
BLUE = "#2a78d6"
GRAY = "#8a8f98"
INK = "#0b0b0b"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

# ── Panel A: 단면(원) 위의 beta 각도 ──────────────────────────
ax1.set_aspect("equal")
R = 1.0
circle = Circle((0, 0), R, facecolor="#cfe0f3", edgecolor=BLUE, linewidth=2, zorder=1)
ax1.add_patch(circle)

# 굽힘 평면(수직선)
ax1.plot([0, 0], [-1.9, 1.9], color=GRAY, linestyle="--", linewidth=1.5, zorder=2)
ax1.text(-1.55, 1.75, "굽힘 평면\n(phi가 정하는 방향)", fontsize=9.5, color=GRAY, va="center", ha="center")

angles = {0: ("beta=0°", GREEN, "위에서 밈 (평면 안, 사용)"),
          90: ("beta=90°", RED, "옆에서 밈\n(평면 밖, 제외됨)"),
          180: ("beta=180°", GREEN, "아래서 밈 (평면 안, 사용)"),
          270: ("beta=270°", RED, "옆에서 밈\n(평면 밖, 제외됨)")}

for deg, (label, color, desc) in angles.items():
    rad = np.radians(90 - deg)  # 0도=위쪽(12시)에서 시작, 시계방향
    x, y = np.cos(rad), np.sin(rad)
    arrow = FancyArrowPatch((x * 1.55, y * 1.55), (x * R * 1.02, y * R * 1.02),
                             arrowstyle="-|>", mutation_scale=22, linewidth=2.8,
                             color=color, zorder=5)
    ax1.add_patch(arrow)
    if deg in (0, 180):
        ly_label = y * 2.15
        ly_desc = y * 2.15 + (0.28 if deg == 0 else -0.28)
        ax1.text(0.75, ly_label, label, fontsize=13, fontweight="bold", color=color, ha="center", va="center", zorder=6)
        ax1.text(0.75, ly_desc, desc, fontsize=8.5, color=color, ha="center",
                  va="bottom" if deg == 0 else "top", zorder=6)
    else:
        lx = x * 2.05
        ax1.text(lx, 0.2, label, fontsize=13, fontweight="bold", color=color,
                  ha="right" if deg == 270 else "left", va="center", zorder=6)
        ax1.text(lx, -0.15, desc, fontsize=8.5, color=color,
                  ha="right" if deg == 270 else "left", va="center", linespacing=1.4, zorder=6)

ax1.text(0, 0, "카테터\n단면", fontsize=11, ha="center", va="center", color=BLUE, fontweight="bold")
ax1.set_xlim(-2.6, 2.6)
ax1.set_ylim(-2.5, 2.5)
ax1.axis("off")
ax1.set_title("(A) 단면에서 본 beta — 원주 방향 각도", fontsize=13, fontweight="bold", pad=15)

# ── Panel B: 옆에서 본 굽힘 평면 + 반대방향 힘 ──────────────────────────
ax2.set_aspect("equal")
s = np.linspace(0, 100, 200)
curve_x = s
curve_y = 25 * np.sin(s / 100 * np.pi * 0.9)
ax2.plot(curve_x, curve_y, color=BLUE, linewidth=6, solid_capstyle="round", zorder=1,
         alpha=0.35, label="카테터(굽힘 평면 안)")
ax2.plot(curve_x, curve_y, color=BLUE, linewidth=2, zorder=2)
ax2.plot(0, 0, marker="s", color="black", markersize=10, zorder=6)

idx = int(len(s) * 0.5)
px, py = curve_x[idx], curve_y[idx]
dx_ = curve_x[idx + 1] - curve_x[idx - 1]
dy_ = curve_y[idx + 1] - curve_y[idx - 1]
norm = np.hypot(dx_, dy_)
nx, ny = -dy_ / norm, dx_ / norm  # 법선(위/아래)

ax2.annotate("", xy=(px + nx * 8, py + ny * 8), xytext=(px + nx * 25, py + ny * 25),
             arrowprops=dict(arrowstyle="-|>", color=GREEN, linewidth=3, mutation_scale=22))
ax2.text(px + nx * 30, py + ny * 30, "beta=0°\n(위에서 밈)", fontsize=12, fontweight="bold",
          color=GREEN, ha="center", va="center")

ax2.annotate("", xy=(px - nx * 8, py - ny * 8), xytext=(px - nx * 25, py - ny * 25),
             arrowprops=dict(arrowstyle="-|>", color=GREEN, linewidth=3, mutation_scale=22))
ax2.text(px - nx * 30, py - ny * 30, "beta=180°\n(아래서 밈)", fontsize=12, fontweight="bold",
          color=GREEN, ha="center", va="center")

# 화면 바깥(제외된 90/270)을 점선 화살표로 암시
ax2.annotate("", xy=(px + 3, py + 3), xytext=(px + 3, py + 3),
             arrowprops=dict(arrowstyle="-|>", color=RED, linewidth=2, linestyle=":", mutation_scale=18))
ax2.text(px, py - 45, "beta=90°/270°(화면 안-밖 방향, 제외됨)\n— 튜브 축 방향, 이 화면에 안 보임",
          fontsize=9.5, color=RED, ha="center", va="center", style="italic")

ax2.set_xlim(-15, 115)
ax2.set_ylim(-55, 60)
ax2.axis("off")
ax2.set_title("(B) 옆에서 본 굽힘 평면 — beta=0/180은 정반대 방향으로 밈", fontsize=13, fontweight="bold", pad=15)

fig.suptitle("beta=0° / 180°란 — 굽힘 평면 '안'에서 서로 반대쪽을 미는 것", fontsize=15, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "beta_0_180_explainer.png"), dpi=150, bbox_inches="tight")
print("저장 완료")
