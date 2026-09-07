"""오늘(2026-08-13) 최종 결과 요약 그림 2장 - 3프로브(자연스러운 움직임 재활용) 15만개
본실행 결과 포함. 랩미팅 pptx 2장짜리에 쓸 사진.
"""
import os

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "..", "..", "..", "data", "contact_scenarios", "fea")

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

GRAY = "#8a8f98"
BLUE = "#2a78d6"
GREEN = "#27AE60"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"

# ── Figure 1: 전체 비교 (최종 15만개 결과 반영) ──────────────────────────
CATS = ["구간(4개) 분류\n(balanced acc)", "연속값 위치 s\n(R²)", "힘 Fx\n(R²)", "힘 Fy\n(R²)"]
ONE_PROBE_CNN = [71.7, 76.6, 94.7, 95.6]
GBM = [76.0, 78.8, 71.0, 70.4]
THREE_PROBE_FINAL = [84.3, 90.1, 95.4, 96.0]
REF_11PROBE = [86.0, 95.4, 90.0, 88.0]

fig, ax = plt.subplots(figsize=(13, 7.2))
x = np.arange(len(CATS))
w = 0.24
b1 = ax.bar(x - w, ONE_PROBE_CNN, width=w, color=GRAY, label="1-probe (CNN)")
b2 = ax.bar(x, GBM, width=w, color=BLUE, label="1-probe (그래디언트부스팅)")
b3 = ax.bar(x + w, THREE_PROBE_FINAL, width=w, color=GREEN, label="3-probe (자연스러운 움직임 재활용, 15만개)")

for i, ref in enumerate(REF_11PROBE):
    ax.plot([x[i] - w * 1.6, x[i] + w * 1.6], [ref, ref], color="#e34948", linestyle="--", linewidth=1.6, zorder=5)

for bars in (b1, b2, b3):
    for rect in bars:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}", (rect.get_x() + rect.get_width() / 2, h), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=11, fontweight="bold", color=INK)

ax.plot([], [], color="#e34948", linestyle="--", linewidth=1.6, label="11-probe(능동탐색, 기각) 참고치")
ax.set_ylim(0, 105)
ax.set_xticks(x)
ax.set_xticklabels(CATS, fontsize=12)
ax.set_ylabel("balanced accuracy(%) 또는 R²×100", fontsize=12, color=MUTED)
ax.grid(True, axis="y", color=GRID, linewidth=1, zorder=0)
ax.set_axisbelow(True)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color("#c3c2b7")
ax.tick_params(colors=MUTED)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=11, frameon=False)
ax.set_title("3-probe(자연스러운 움직임 재활용, 15만개 본실행) — 11-probe 수준에 근접",
             fontsize=16, fontweight="bold", color=INK, pad=14)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "handoff_naturalmotion_final_comparison.png"), dpi=150, bbox_inches="tight")
print("저장 1완료")
plt.close(fig)

# ── Figure 2: 규모 확장 효과 + 최종 스코어카드 ──────────────────────────
fig2 = plt.figure(figsize=(13, 7.2))
gs = fig2.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.35)

ax1 = fig2.add_subplot(gs[0])
scales = ["1-probe\n(15만개)", "3-probe\n(6만개, 예비)", "3-probe\n(15만개, 최종)"]
seg_vals = [71.7, 79.5, 84.3]
s_vals = [76.6, 87.1, 90.1]
xg = np.arange(len(scales))
wg = 0.32
ax1.bar(xg - wg / 2, seg_vals, width=wg, color=BLUE, label="구간분류 balanced acc(%)")
ax1.bar(xg + wg / 2, s_vals, width=wg, color=GREEN, label="s R²×100")
for i in range(len(scales)):
    ax1.annotate(f"{seg_vals[i]:.1f}", (xg[i] - wg / 2, seg_vals[i]), textcoords="offset points",
                 xytext=(0, 4), ha="center", fontsize=10.5, fontweight="bold")
    ax1.annotate(f"{s_vals[i]:.1f}", (xg[i] + wg / 2, s_vals[i]), textcoords="offset points",
                 xytext=(0, 4), ha="center", fontsize=10.5, fontweight="bold")
ax1.set_xticks(xg)
ax1.set_xticklabels(scales, fontsize=11.5)
ax1.set_ylim(0, 100)
ax1.set_title("규모를 6만→15만개로 늘리자 추가 개선", fontsize=13.5, fontweight="bold")
ax1.grid(True, axis="y", color=GRID, linewidth=1, zorder=0)
ax1.set_axisbelow(True)
for spine in ["top", "right"]:
    ax1.spines[spine].set_visible(False)
ax1.legend(loc="lower center", bbox_to_anchor=(0.5, -0.24), ncol=1, fontsize=10.5, frameon=False)

ax2 = fig2.add_subplot(gs[1])
ax2.axis("off")
ax2.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax2.transAxes, facecolor="#f2f9f5",
                             edgecolor=GREEN, linewidth=1.5))
lines = [
    ("최종 모델 (3-probe, 15만개)", None, True),
    ("구간(4개) 분류", "84.3%  (balanced acc)", False),
    ("연속값 위치 s", "R²=0.901, MAE=4.60mm", False),
    ("힘 Fx (보드좌표)", "R²=0.954", False),
    ("힘 Fy (보드좌표)", "R²=0.960", False),
    ("L_M (슬랙 보정)", "R²=0.987, MAE=2.29mm", False),
]
y0 = 0.88
for label, val, is_title in lines:
    if is_title:
        ax2.text(0.06, y0, label, fontsize=15, fontweight="bold", color=INK, transform=ax2.transAxes)
        y0 -= 0.16
    else:
        ax2.text(0.08, y0, label, fontsize=11.5, color="#333", transform=ax2.transAxes)
        ax2.text(0.94, y0, val, fontsize=11.5, fontweight="bold", color=GREEN, ha="right", transform=ax2.transAxes)
        y0 -= 0.135
ax2.text(0.06, 0.07, "\"로봇을 일부러 재구성\"이 아니라 원래 하던 동작 중\n"
                      "자연스럽게 지나가는 φ 3개를 버퍼링 — 추가 동작 없이 정보만 재사용",
          fontsize=10.5, color="#555", transform=ax2.transAxes, va="bottom", linespacing=1.5)

fig2.suptitle("최종 결과 스코어카드 (2026-08-13)", fontsize=16, fontweight="bold", y=1.00)
fig2.tight_layout()
fig2.savefig(os.path.join(OUT_DIR, "handoff_naturalmotion_final_scorecard.png"), dpi=150, bbox_inches="tight")
print("저장 2완료")
