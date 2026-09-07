"""오늘 시도한 접촉추정 CNN 3가지 버전(노이즈 있음/노이즈 없음/2단계 NN)의 최종 R^2를
한 그림으로 비교. "결국 뭐가 제일 나은가"를 한눈에 보여주는 것이 목적.
"""
import os

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Noto Sans CJK KR'
plt.rcParams['axes.unicode_minus'] = False

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "..", "..", "data", "contact_scenarios", "fea")

INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
GRAY_FAIL = "#c3c2b7"   # 노이즈 있음(실패, 폐기된 베이스라인)
BLUE = "#2a78d6"        # 노이즈 없음 direct CNN
ORANGE = "#eb6834"      # 2단계 이상적 상한선
AQUA = "#1baf7a"        # 2단계 엔드투엔드(실전, exposure bias 있음)
YELLOW = "#eda100"      # 2단계 엔드투엔드(NN2를 NN1 실제 예측값으로 재학습 - 현재 최선)

TARGETS = ["s (위치)", "F_mag", "Fx", "Fy"]
series = {
    "노이즈 있음 (초기, 폐기)": ([-0.304, -0.286, -0.268, -0.265], GRAY_FAIL),
    "노이즈 없음 - direct CNN": ([0.417, 0.686, 0.898, 0.905], BLUE),
    "2단계 NN - 이상적 상한선": ([0.682, 0.866, 0.957, 0.953], ORANGE),
    "2단계 NN - 엔드투엔드(exposure bias 있음)": ([0.257, 0.615, 0.864, 0.883], AQUA),
    "2단계 NN - 엔드투엔드(수정판, 현재 최선)": ([0.400, 0.690, 0.904, 0.912], YELLOW),
}

n_series = len(series)
n_targets = len(TARGETS)
x = np.arange(n_targets) * 1.3
width = 0.16

fig, ax = plt.subplots(figsize=(13.5, 7))

legend_handles = []
for i, (name, (values, color)) in enumerate(series.items()):
    offset = (i - (n_series - 1) / 2) * width
    bars = ax.bar(x + offset, values, width, color=color, label=name,
                  edgecolor="white", linewidth=0.8, zorder=3)
    legend_handles.append(bars)
    for b, v in zip(bars, values):
        va = "bottom" if v >= 0 else "top"
        dy = 0.02 if v >= 0 else -0.02
        ax.text(b.get_x() + b.get_width() / 2, v + dy, f"{v:.2f}", ha="center", va=va,
                fontsize=9, color=color, fontweight="normal", family="DejaVu Sans")

ax.axhline(0, color=INK, linewidth=1.2, zorder=2)
ax.set_xticks(x)
ax.set_xticklabels(TARGETS, fontsize=11)
ax.set_ylabel("R² (검증셋)", fontsize=11)
ax.set_ylim(-0.45, 1.05)
ax.grid(True, linestyle=":", color=GRID, alpha=0.8, axis="y", zorder=0)
ax.tick_params(colors=MUTED)
for spine in ax.spines.values():
    spine.set_color(GRID)
fig.suptitle("접촉추정 CNN — 오늘 시도한 5가지 버전 최종 성능 비교",
             fontweight="bold", fontsize=15, color=INK, y=0.99)
fig.text(0.5, 0.90,
         "노이즈가 신호와 거의 1:1(주범)이라 초기엔 전부 실패(R²<0) → 노이즈 제거로 대폭 개선 →\n"
         "2단계 NN 엔드투엔드는 처음엔 exposure bias로 direct CNN보다 나빴는데, NN2를 NN1의 실제 예측값으로 재학습하니 전 타깃에서 direct CNN 이상으로 개선",
         ha="center", fontsize=9.5, color=MUTED, linespacing=1.6)
fig.legend(handles=[h[0] for h in legend_handles], labels=list(series.keys()),
           loc="upper center", bbox_to_anchor=(0.5, 0.83), ncol=3, fontsize=9, frameon=False)

fig.tight_layout(rect=[0, 0, 1, 0.78])
out_path = os.path.join(DATA_DIR, "cnn_results_comparison.png")
fig.savefig(out_path, dpi=150, facecolor="#fcfcfb", bbox_inches="tight")
print(f"저장: {out_path}")
