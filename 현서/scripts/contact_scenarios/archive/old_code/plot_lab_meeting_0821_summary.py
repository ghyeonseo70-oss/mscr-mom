"""8월 3째주 랩미팅용 핵심 결과 차트 2장:
1) 문제 발견 - 합성검증(좋아보임) vs 실측 홀드아웃(폭락)
2) 해결 - 실측 홀드아웃 폭락 vs L_M 조밀화 후 회복
PROJECT_STATUS.md 핵심 결과 요약 표의 실제 수치 그대로 사용."""
import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

HERE = os.path.dirname(__file__)
OUT_DIR = os.path.join(HERE, "..", "..", "data", "contact_scenarios", "fea")

LABEL_A = "Fx_board\n(=로컬Fy, 원래 못 배우던 타겟)"
LABEL_B = "Fy_board\n(=로컬Fx, 원래 잘 배우던 타겟)"

# ---- 차트 1: 문제 발견 (합성검증 vs 실측 홀드아웃) ----
groups1 = ["합성 데이터로 검증\n(좋아 보였음)", "실측 FEA로 검증\n(진짜 성능, 폭락)"]
vals_a1 = [0.799, 0.009]
vals_b1 = [0.884, 0.337]

fig, ax = plt.subplots(figsize=(9, 6))
x = np.arange(len(groups1))
w = 0.32
bars_a = ax.bar(x - w/2, vals_a1, w, label=LABEL_A, color="#4C72B0")
bars_b = ax.bar(x + w/2, vals_b1, w, label=LABEL_B, color="#DD8452")
for bars in (bars_a, bars_b):
    for b in bars:
        h = b.get_height()
        ax.annotate(f"{h:.3f}", (b.get_x() + b.get_width()/2, h), ha="center", va="bottom",
                    fontsize=13, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(groups1, fontsize=13)
ax.set_ylabel("힘 추정 R²", fontsize=13)
ax.set_ylim(0, 1.0)
ax.axhline(0, color="gray", linewidth=0.8)
ax.set_title("합성 데이터로는 R²=0.80~0.88로 좋아 보였지만,\n실측 FEA로 검증하니 R²=0.01~0.34로 폭락함",
             fontsize=15, fontweight="bold")
ax.legend(fontsize=11, loc="upper right")
ax.grid(axis="y", linestyle=":", alpha=0.5)
plt.tight_layout()
out1 = os.path.join(OUT_DIR, "lab0821_r2_collapse.png")
plt.savefig(out1, dpi=150)
print("저장:", out1)

# ---- 차트 2: 해결 (실측 홀드아웃 폭락 vs 조밀화 후 회복) ----
groups2 = ["실측 FEA 244개\n(폭락 상태)", "실측 FEA 402개\nL_M 조밀화 후 (회복)"]
vals_a2 = [0.009, 0.729]
vals_b2 = [0.337, 0.646]

fig, ax = plt.subplots(figsize=(9, 6))
x = np.arange(len(groups2))
bars_a = ax.bar(x - w/2, vals_a2, w, label=LABEL_A, color="#4C72B0")
bars_b = ax.bar(x + w/2, vals_b2, w, label=LABEL_B, color="#DD8452")
for bars in (bars_a, bars_b):
    for b in bars:
        h = b.get_height()
        ax.annotate(f"{h:.3f}", (b.get_x() + b.get_width()/2, h), ha="center", va="bottom",
                    fontsize=13, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(groups2, fontsize=13)
ax.set_ylabel("힘 추정 R² (실측 FEA 홀드아웃 기준)", fontsize=13)
ax.set_ylim(0, 1.0)
ax.axhline(0, color="gray", linewidth=0.8)
ax.set_title("L_M 조밀화로 실측 FEA를 244개→402개로 늘리자\n힘 R²가 실측 기준으로 크게 회복됨",
             fontsize=15, fontweight="bold")
ax.legend(fontsize=11, loc="upper left")
ax.grid(axis="y", linestyle=":", alpha=0.5)
plt.tight_layout()
out2 = os.path.join(OUT_DIR, "lab0821_r2_recovery.png")
plt.savefig(out2, dpi=150)
print("저장:", out2)
