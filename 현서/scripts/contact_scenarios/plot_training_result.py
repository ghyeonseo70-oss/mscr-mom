"""train_contact_model.py 결과(loss 곡선 + 예측 vs 실제) 시각화. 피피티에 바로 넣을 수 있게
깔끔한 스타일로 저장."""
import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, '..', '..', 'data', 'contact_scenarios')

d = np.load(os.path.join(DATA_DIR, 'contact_train_history.npz'))
train_losses, val_losses = d['train_losses'], d['val_losses']
pred, true, targets = d['val_pred'], d['val_true'], list(d['targets'])

fig = plt.figure(figsize=(16, 6.3))
gs = fig.add_gridspec(2, 3, height_ratios=[0.32, 1], hspace=0.15, wspace=0.32)

# ── 헤로 배너: 핵심 결과를 첫눈에 ──────────────────────────
hero = fig.add_subplot(gs[0, :])
hero.axis("off")
hero.add_patch(plt.Rectangle((0, 0), 1, 1, transform=hero.transAxes, facecolor="#fbeae8",
                              edgecolor="#B23A32", linewidth=1.5))
hero.text(0.03, 0.5, "R² ~ 0", fontsize=34, fontweight="bold", color="#B23A32",
          transform=hero.transAxes, va="center", ha="left")
hero.text(0.24, 0.5, "단일 스냅샷으로는 사실상 학습 실패\n(평균값 찍는 수준과 동일)",
          fontsize=13.5, fontweight="bold", color="#7a2420", transform=hero.transAxes,
          va="center", ha="left", linespacing=1.4)

axes = [fig.add_subplot(gs[1, i]) for i in range(3)]

# (1) Loss 곡선
ax = axes[0]
ax.plot(train_losses, color="#2a78d6", linewidth=2, label="Train")
ax.plot(val_losses, color="#e34948", linewidth=2, label="Validation")
ax.set_xlabel("Epoch")
ax.set_ylabel("정규화 MSE Loss")
ax.set_title("(A) 학습 곡선", fontweight="bold")
ax.grid(True, linestyle=":", alpha=0.5)
ax.legend()
ax.annotate("검증 loss가 안 줄어듦\n(과적합/학습 실패)", xy=(len(val_losses) * 0.6, val_losses[-1]),
            xytext=(len(val_losses) * 0.35, max(val_losses) * 0.6),
            arrowprops=dict(arrowstyle="->", color="#e34948"), color="#e34948", fontsize=9)

# (2)(3) 예측 vs 실제 산점도 (F_mag, s)
for ax, t, unit, color in zip(axes[1:], ["F_mag", "s"], ["N", "mm"], ["#2e7d32", "#8e44ad"]):
    i = targets.index(t)
    ax.scatter(true[:, i], pred[:, i], s=10, alpha=0.4, color=color)
    lo, hi = true[:, i].min(), true[:, i].max()
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.5, label="완벽한 예측선")
    mae = np.mean(np.abs(pred[:, i] - true[:, i]))
    ax.set_xlabel(f"실제 {t} ({unit})")
    ax.set_ylabel(f"예측 {t} ({unit})")
    ax.set_title(f"({'B' if t=='F_mag' else 'C'}) {t} 예측 vs 실제\nMAE={mae:.4f}{unit}", fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(fontsize=8)

fig.suptitle("1차 접촉 추정 모델 학습 결과 (물리모델 기반 B-필드 3000개, 검증셋 n=%d)" % len(true),
             fontweight="bold", fontsize=14, y=1.02)

out_path = os.path.join(DATA_DIR, "contact_training_result.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"저장: {out_path}")
