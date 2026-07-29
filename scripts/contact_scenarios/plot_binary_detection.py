"""이진 감지(접촉 유무) 결과 전용 시각화."""
import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, '..', '..', 'data', 'contact_scenarios')

d = np.load(os.path.join(DATA_DIR, 'binary_detection_result.npz'))
val_pred, val_true, val_Fmag = d['val_pred'], d['val_true'], d['val_Fmag']
final_acc, fpr = float(d['final_acc']), float(d['false_positive_rate'])

fig = plt.figure(figsize=(15, 6.6))
gs = fig.add_gridspec(2, 3, height_ratios=[0.3, 1], width_ratios=[1.1, 1.3, 0.9], hspace=0.2, wspace=0.35)

# ── 헤로 배너 ──────────────────────────────
hero = fig.add_subplot(gs[0, :])
hero.axis("off")
hero.add_patch(plt.Rectangle((0, 0), 1, 1, transform=hero.transAxes, facecolor="#e3f3e6",
                              edgecolor="#1E7B34", linewidth=1.5))
hero.text(0.03, 0.5, f"{final_acc*100:.1f}%", fontsize=34, fontweight="bold", color="#1E7B34",
          transform=hero.transAxes, va="center", ha="left")
hero.text(0.24, 0.5, "접촉 유무 감지 정확도\n오탐율 %.1f%% 로 낮음" % (fpr * 100),
          fontsize=13.5, fontweight="bold", color="#175A29", transform=hero.transAxes,
          va="center", ha="left", linespacing=1.4)

# ── (A) 핵심 지표 요약 ──────────────────────────
ax = fig.add_subplot(gs[1, 0])
ax.axis('off')
tpr = (val_pred[val_true == 1] == 1).mean()
tnr = (val_pred[val_true == 0] == 0).mean()
metrics = [("전체 정확도", final_acc, "#2451A3"), ("접촉 탐지율", tpr, "#1E7B34"),
           ("무접촉 정확 판별", tnr, "#1E7B34"), ("오탐율", fpr, "#B23A32")]
for i, (label, val, color) in enumerate(metrics):
    y = 0.85 - i * 0.24
    ax.add_patch(plt.Rectangle((0, y - 0.08), 1, 0.16, transform=ax.transAxes,
                                facecolor=color, alpha=0.12, edgecolor=color, linewidth=1.2))
    ax.text(0.04, y, label, fontsize=12, va='center', transform=ax.transAxes, fontweight='bold')
    ax.text(0.96, y, f"{val*100:.1f}%", fontsize=17, va='center', ha='right',
            transform=ax.transAxes, fontweight='bold', color=color, family='monospace')
ax.set_title("(A) 이진 감지 핵심 지표", fontweight='bold', fontsize=12, loc='left')

# ── (B) 힘 크기 구간별 탐지율 ──────────────────────────
ax = fig.add_subplot(gs[1, 1])
bins = [0, 2, 5, 10, 15, 20]
contact_mask = val_true == 1
f_vals = val_Fmag[contact_mask] * 1000
correct = (val_pred[contact_mask] == 1)

rates, ns, labels = [], [], []
for lo, hi in zip(bins[:-1], bins[1:]):
    m = (f_vals >= lo) & (f_vals < hi)
    if m.sum() > 0:
        rates.append(correct[m].mean() * 100)
        ns.append(m.sum())
        labels.append(f"{lo}-{hi}")

colors = plt.cm.Greens(np.linspace(0.45, 0.9, len(rates)))
bars = ax.bar(labels, rates, color=colors, edgecolor='white')
for bar, n, r in zip(bars, ns, rates):
    ax.text(bar.get_x() + bar.get_width() / 2, r + 2, f"{r:.1f}%\n(n={n})",
            ha='center', fontsize=9.5, fontweight='bold')
ax.axhline(final_acc * 100, color='#B23A32', linestyle='--', linewidth=1.3, label=f'전체 평균 {final_acc*100:.1f}%')
ax.set_ylim(0, 110)
ax.set_xlabel('접촉힘 크기 구간 (mN)')
ax.set_ylabel('탐지율 (%)')
ax.set_title('(B) 힘이 클수록 더 잘 감지됨', fontweight='bold', fontsize=12)
ax.legend(fontsize=9, loc='lower right')
ax.grid(True, axis='y', linestyle=':', alpha=0.5)

# ── (C) 힘 크기별 산점도(탐지/미탐지) ──────────────────────────
ax = fig.add_subplot(gs[1, 2])
jitter = np.random.default_rng(0).uniform(-0.15, 0.15, len(f_vals))
ax.scatter(f_vals[correct], 1 + jitter[correct], s=14, alpha=0.5, color='#1E7B34', label='탐지 성공')
ax.scatter(f_vals[~correct], 0 + jitter[~correct], s=14, alpha=0.5, color='#B23A32', label='탐지 실패')
ax.set_yticks([0, 1])
ax.set_yticklabels(['실패', '성공'])
ax.set_xlabel('접촉힘 크기 (mN)')
ax.set_title('(C) 개별 케이스 분포', fontweight='bold', fontsize=12)
ax.legend(fontsize=9, loc='center right')
ax.grid(True, axis='x', linestyle=':', alpha=0.5)

fig.suptitle('이진 감지(접촉 유무) 모델 결과', fontweight='bold', fontsize=16, y=1.01)

out_path = os.path.join(DATA_DIR, 'binary_detection_result.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"저장: {out_path}")
