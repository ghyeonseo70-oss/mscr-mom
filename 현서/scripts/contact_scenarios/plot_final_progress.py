"""3000개(5프로브) -> 15000개(편향) -> 15000개(편향제거) 세 단계 개선 과정을 한눈에 정리."""
import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, '..', '..', 'data', 'contact_scenarios')

# 각 단계 R2 (compare_probe_combos.py / train_multiprobe_model.py 출력에서 기록된 값)
STAGES = ["3,000개\n(5프로브)", "15,000개\n(편향 있음)", "15,000개\n(편향 제거)"]
TARGETS = ['s', 'F_mag', 'Fx', 'Fy']
R2 = {
    's':     [0.548, 0.652, 0.750],
    'F_mag': [0.158, 0.317, 0.335],
    'Fx':    [0.450, 0.613, 0.611],
    'Fy':    [0.710, 0.785, 0.834],
}
AVG = [np.mean([R2[t][i] for t in TARGETS]) for i in range(3)]

unbiased = np.load(os.path.join(DATA_DIR, 'multiprobe_train_history_15k_unbiased.npz'))
pred, true, hist_targets = unbiased['val_pred'], unbiased['val_true'], list(unbiased['targets'])

COLORS = {'s': '#8e44ad', 'F_mag': '#2e7d32', 'Fx': '#2a78d6', 'Fy': '#e34948'}
STAGE_COLORS = ['#c9cdd6', '#8fa6c9', '#2451A3']

fig = plt.figure(figsize=(16, 11))
gs = fig.add_gridspec(3, 3, height_ratios=[0.28, 1, 1], hspace=0.42, wspace=0.32)

# ── 헤로 배너 ──────────────────────────────
hero = fig.add_subplot(gs[0, :])
hero.axis("off")
hero.add_patch(plt.Rectangle((0, 0), 1, 1, transform=hero.transAxes, facecolor="#e8eefa",
                              edgecolor="#2451A3", linewidth=1.5))
hero.text(0.03, 0.5, f"{AVG[0]:.3f}  ->  {AVG[-1]:.3f}", fontsize=30, fontweight="bold",
          color="#2451A3", transform=hero.transAxes, va="center", ha="left")
hero.text(0.42, 0.5, "데이터 3,000개 -> 15,000개 + 편향제거로\n평균 R² 35% 향상",
          fontsize=13.5, fontweight="bold", color="#193A7D", transform=hero.transAxes,
          va="center", ha="left", linespacing=1.4)

# ── (A) 타깃별 R2 진행 (그룹 막대) ──────────────────────────
ax = fig.add_subplot(gs[1, :2])
x = np.arange(len(TARGETS))
w = 0.25
for i, stage in enumerate(STAGES):
    vals = [R2[t][i] for t in TARGETS]
    ax.bar(x + (i - 1) * w, vals, w, label=stage, color=STAGE_COLORS[i], edgecolor='white')
ax.set_xticks(x)
ax.set_xticklabels(['접촉위치 s', '힘 크기 F_mag', '힘 Fx', '힘 Fy'])
ax.set_ylabel('R² (검증셋)')
ax.set_ylim(0, 1)
ax.set_title('(A) 단계별 정확도(R²) 변화 - 타깃별', fontweight='bold', fontsize=12)
ax.grid(True, axis='y', linestyle=':', alpha=0.5)
ax.legend(fontsize=9, loc='upper left')

# ── (B) 평균 R2 추이 ──────────────────────────
ax = fig.add_subplot(gs[1, 2])
ax.plot(range(3), AVG, 'o-', color='#2451A3', linewidth=2.5, markersize=10)
for i, v in enumerate(AVG):
    ax.annotate(f"{v:.3f}", (i, v), textcoords="offset points", xytext=(0, 10),
                ha='center', fontweight='bold', fontsize=10)
ax.set_xticks(range(3))
ax.set_xticklabels(['3천개', '1.5만개\n(편향)', '1.5만개\n(편향제거)'], fontsize=9)
ax.set_ylabel('평균 R² (4개 타깃)')
ax.set_ylim(0.4, 0.7)
ax.set_title('(B) 전체 평균 R² 추이', fontweight='bold', fontsize=12)
ax.grid(True, linestyle=':', alpha=0.5)

# ── (C)(D) 최종(편향제거) 모델의 예측 vs 실제 ──────────────────────────
for ax, t, unit in zip([fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])], ['s', 'F_mag'], ['mm', 'N']):
    i = hist_targets.index(t)
    ax.scatter(true[:, i], pred[:, i], s=8, alpha=0.35, color=COLORS[t])
    lo, hi = true[:, i].min(), true[:, i].max()
    ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1.5)
    r2 = R2[t][2]
    ax.set_xlabel(f"실제 {t} ({unit})")
    ax.set_ylabel(f"예측 {t} ({unit})")
    ax.set_title(f"최종모델(편향제거 1.5만개): {t}\nR²={r2:.3f}", fontweight='bold', fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.5)

ax = fig.add_subplot(gs[2, 2])
ax.axis('off')
ax.text(0.0, 0.95, "요약", fontsize=14, fontweight='bold', transform=ax.transAxes)
ax.text(0.0, 0.78,
        "3,000개 -> 15,000개: 평균 R² +0.125\n"
        "  (가장 큰 효과 - 과적합 해소)\n\n"
        "편향 제거: 평균 R² +0.041\n"
        "  (s 위치 추정이 가장 크게 개선,\n"
        "   +0.098)\n\n"
        "최종: 평균 R²=0.633\n"
        "  s 오차 ±9.6mm, F_mag 오차 ±3.5mN",
        fontsize=10.5, transform=ax.transAxes, va='top', linespacing=1.7)

fig.suptitle('접촉(충돌) 위치·힘 추정 - 데이터 개선 3단계', fontweight='bold', fontsize=16, y=1.005)

out_path = os.path.join(DATA_DIR, 'final_progress_summary.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"저장: {out_path}")
