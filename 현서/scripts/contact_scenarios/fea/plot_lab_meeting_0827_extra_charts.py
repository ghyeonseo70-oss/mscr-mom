"""8월 랩미팅용 보조 막대그래프 4개 생성 - PROJECT_STATUS.md 20/21/22번에 이미 기록된
숫자를 그대로 시각화(재계산 아님, 문서화된 값 그대로 사용).
1) phi_breakdown_fx_bar.png   - Fx_board R^2: |phi|<90 vs |phi|>=90
2) high_phi_weight_attempts_bar.png - HIGH_PHI_WEIGHT 3.0/1.5 시도했지만 |phi|>=90 Fx_board 그대로
3) s_error_by_bin_bar.png     - s 오차(MAE) s값 구간별
4) lm_zero_hybrid_bar.png     - L_M=0 순수회귀 vs 하이브리드 MAE, 시드 42/43
"""
import os
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

GREEN = "#27AE60"
RED = "#C0392B"
GRAY = "#95A5A6"
ORANGE = "#E67E22"

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "..", "..", "..", "data", "contact_scenarios")
os.makedirs(OUT_DIR, exist_ok=True)


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print("저장:", path)


# ---- 1) phi 구간별 Fx_board R^2 ----
fig, ax = plt.subplots(figsize=(6, 5))
labels = ["|phi| < 90°\n(쉬운 구간)", "|phi| >= 90°\n(문제 구간)"]
values = [0.782, 0.355]
colors = [GREEN, RED]
bars = ax.bar(labels, values, color=colors, width=0.55)
for b, v in zip(bars, values):
    ax.annotate(f"R²={v:.3f}", (b.get_x() + b.get_width() / 2, v), textcoords="offset points",
                xytext=(0, 6), ha="center", fontsize=14, fontweight="bold")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("Fx_board R²")
ax.set_ylim(0, 1.0)
ax.set_title("Fx_board는 |phi|>=90 구간에서만 유독 나쁨\n(2026-08-27, HIGH_PHI_WEIGHT=1.5 재학습 기준)",
             fontweight="bold", fontsize=13)
ax.grid(axis="y", linestyle=":", alpha=0.5)
save(fig, "phi_breakdown_fx_bar.png")

# ---- 2) HIGH_PHI_WEIGHT 시도 - 개선 안 됨 ----
fig, ax = plt.subplots(figsize=(6, 5))
labels = ["가중치 3.0", "가중치 1.5\n(재튜닝)"]
values = [0.343, 0.355]
bars = ax.bar(labels, values, color=[GRAY, GRAY], width=0.5)
for b, v in zip(bars, values):
    ax.annotate(f"R²={v:.3f}", (b.get_x() + b.get_width() / 2, v), textcoords="offset points",
                xytext=(0, 6), ha="center", fontsize=14, fontweight="bold")
ax.set_ylabel("|phi|>=90 구간 Fx_board R²")
ax.set_ylim(0, 1.0)
ax.set_title("HIGH_PHI_WEIGHT를 3.0->1.5로 낮춰도 목표 구간은 그대로\n"
             "(0.343 -> 0.355, 사실상 무변화) - 이 레버는 폐기",
             fontweight="bold", fontsize=13, color=RED)
ax.grid(axis="y", linestyle=":", alpha=0.5)
save(fig, "high_phi_weight_attempts_bar.png")

# ---- 3) s 오차 구간별 ----
fig, ax = plt.subplots(figsize=(7, 5))
labels = ["0-20mm", "20-40mm", "40-60mm", "60-100mm\n(팁 근처)"]
values = [4.78, 5.56, 2.88, 9.49]
colors = [GRAY, GRAY, GREEN, RED]
bars = ax.bar(labels, values, color=colors, width=0.6)
for b, v in zip(bars, values):
    ax.annotate(f"{v:.2f}mm", (b.get_x() + b.get_width() / 2, v), textcoords="offset points",
                xytext=(0, 6), ha="center", fontsize=13, fontweight="bold")
ax.set_ylabel("s 예측 MAE (mm)")
ax.set_title("s 오차는 팁 근처(60-100mm)에서 가장 큼\n(실측 홀드아웃 n=99, 2026-08-27)",
             fontweight="bold", fontsize=13)
ax.grid(axis="y", linestyle=":", alpha=0.5)
save(fig, "s_error_by_bin_bar.png")

# ---- 4) L_M=0 하이브리드 개선 (시드별) ----
fig, ax = plt.subplots(figsize=(7, 5))
import numpy as np
seeds = ["시드 42\n(커밋된 버전)", "시드 43\n(진단용)"]
pure = [19.36, 27.05]
hybrid = [7.70, 17.25]
x = np.arange(len(seeds))
w = 0.32
b1 = ax.bar(x - w / 2, pure, w, label="순수 회귀", color=RED)
b2 = ax.bar(x + w / 2, hybrid, w, label="하이브리드(분류+회귀)", color=GREEN)
for bars in (b1, b2):
    for b in bars:
        v = b.get_height()
        ax.annotate(f"{v:.1f}mm", (b.get_x() + b.get_width() / 2, v), textcoords="offset points",
                    xytext=(0, 5), ha="center", fontsize=11, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(seeds)
ax.set_ylabel("L_M=0 구간 MAE (mm)")
ax.set_title("L_M=0 하이브리드 구조 - 방향은 항상 개선, 크기는 시드마다 다름\n(완전 해결은 아님)",
             fontweight="bold", fontsize=13)
ax.legend(fontsize=11)
ax.grid(axis="y", linestyle=":", alpha=0.5)
save(fig, "lm_zero_hybrid_bar.png")
