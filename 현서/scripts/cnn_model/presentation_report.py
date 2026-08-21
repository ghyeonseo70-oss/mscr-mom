import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import os

plt.rcParams['font.family'] = 'Malgun Gothic'  # 한글 깨짐 방지 (Windows 기본 폰트)
plt.rcParams['axes.unicode_minus'] = False

class CNNPositionEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU()
        )
        self.regressor = nn.Sequential(
            nn.Flatten(), nn.Linear(32 * 5 * 5, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(128, 6)
        )
    def forward(self, x): return self.regressor(self.features(x))

current_dir = os.path.dirname(__file__)
X_path = os.path.join(current_dir, '..', '..', 'data', 'cnn_model', 'mscr_X.npy')
y_path = os.path.join(current_dir, '..', '..', 'data', 'cnn_model', 'mscr_y.npy')
loss_path = os.path.join(current_dir, '..', '..', 'data', 'cnn_model', 'mscr_loss_history.npy')
model_path = os.path.join(current_dir, '..', '..', 'models', 'mscr_cnn_model.pth')
out_dir = os.path.join(current_dir, '..', '..', 'data', 'cnn_model')

X = np.load(X_path)
y = np.load(y_path)
train_losses, val_losses = np.load(loss_path)
split = int(0.8 * len(X))
X_test, y_test = X[split:], y[split:]

model = CNNPositionEstimator()
model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
model.eval()

X_cnn = X_test.reshape(-1, 5, 5, 3).transpose(0, 3, 1, 2)
with torch.no_grad():
    pred = model(torch.tensor(X_cnn, dtype=torch.float32)).numpy()

main_true, main_pred = y_test[:, [0, 1]], pred[:, [0, 1]]
mom_true, mom_pred = y_test[:, [3, 4]], pred[:, [3, 4]]
main_err = np.linalg.norm(main_pred - main_true, axis=1)
mom_err = np.linalg.norm(mom_pred - mom_true, axis=1)

all_true = np.vstack([main_true, mom_true])
all_err = np.concatenate([main_err, mom_err])

# ── 1) 공간별 정확도 히트맵 ──────────────────────────────
N_BINS = 9
edges = np.linspace(0, 180, N_BINS + 1)
grid_err = np.full((N_BINS, N_BINS), np.nan)
grid_count = np.zeros((N_BINS, N_BINS), dtype=int)
xi = np.clip(np.digitize(all_true[:, 0], edges) - 1, 0, N_BINS - 1)
yi = np.clip(np.digitize(all_true[:, 1], edges) - 1, 0, N_BINS - 1)
for i in range(N_BINS):
    for j in range(N_BINS):
        mask = (xi == i) & (yi == j)
        if mask.sum() > 0:
            grid_err[j, i] = all_err[mask].mean()
            grid_count[j, i] = mask.sum()

cmap = LinearSegmentedColormap.from_list('err', ['#cde2fb', '#2a78d6', '#0d366b'])
masked = np.ma.masked_invalid(grid_err)
cmap.set_bad(color='#f0efec')

fig, ax = plt.subplots(figsize=(7, 6.5))
im = ax.imshow(masked, origin='lower', extent=[0, 180, 0, 180], cmap=cmap, aspect='equal')
cbar = fig.colorbar(im, ax=ax, shrink=0.85, label='평균 위치 오차 (mm)')

for i in range(N_BINS):
    for j in range(N_BINS):
        if not np.isnan(grid_err[j, i]):
            cx, cy = edges[i] + 10, edges[j] + 10
            ax.text(cx, cy, f'{grid_err[j, i]:.1f}', ha='center', va='center',
                     fontsize=8, color='white' if grid_err[j, i] > 4.5 else 'black')

ax.plot(90, 0, 'ks', markersize=12, label='로봇 베이스')
ax.set_xlim(0, 180); ax.set_ylim(0, 180)
ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)')
ax.set_title('위치별 예측 정확도 히트맵 (9x9 구역, 테스트셋 %d개)' % len(y_test), fontweight='bold')
ax.legend(loc='upper right')
plt.tight_layout()
heatmap_path = os.path.join(out_dir, 'spatial_accuracy_heatmap.png')
plt.savefig(heatmap_path, dpi=150)
print(f"저장: {heatmap_path}")

# ── 2) 학습 곡선 + 예측 vs 실제 산점도 ──────────────────────
fig2, axes = plt.subplots(1, 2, figsize=(13, 5.5))

axes[0].plot(train_losses, label='Train Loss', color='#2a78d6', linewidth=2)
axes[0].plot(val_losses, label='Val Loss', color='#e34948', linestyle='--', linewidth=2)
axes[0].set_title('CNN 학습 곡선', fontweight='bold')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('MSE Loss')
axes[0].set_yscale('log')
axes[0].grid(True, linestyle=':', alpha=0.6)
axes[0].legend()

axes[1].add_patch(plt.Rectangle((0, 0), 180, 180, fill=False, color='gray', linestyle='--'))
axes[1].plot(90, 0, 'ks', markersize=10, label='로봇 베이스')
idx = np.random.choice(len(y_test), 40, replace=False)
axes[1].scatter(y_test[idx, 0], y_test[idx, 1], c='#e34948', alpha=0.3, label='실제 Main', s=40)
axes[1].scatter(y_test[idx, 3], y_test[idx, 4], c='#2a78d6', alpha=0.3, label='실제 Mom', s=40)
axes[1].scatter(pred[idx, 0], pred[idx, 1], c='#e34948', marker='x', label='예측 Main', s=50)
axes[1].scatter(pred[idx, 3], pred[idx, 4], c='#2a78d6', marker='x', label='예측 Mom', s=50)
axes[1].set_xlim(0, 180); axes[1].set_ylim(0, 180)
axes[1].set_title('예측 vs 실제 위치 (샘플 40개)', fontweight='bold')
axes[1].set_aspect('equal')
axes[1].legend(loc='upper right', fontsize=8)

plt.tight_layout()
summary_path = os.path.join(out_dir, 'training_summary.png')
plt.savefig(summary_path, dpi=150)
print(f"저장: {summary_path}")

# ── 3) 콘솔 요약 ──────────────────────────────
print("\n=== 발표용 핵심 지표 ===")
print(f"테스트 샘플 수: {len(y_test)}")
print(f"Main 평균 오차: {main_err.mean():.2f} mm (P90: {np.percentile(main_err, 90):.2f} mm)")
print(f"Mom  평균 오차: {mom_err.mean():.2f} mm (P90: {np.percentile(mom_err, 90):.2f} mm)")
best_cell = np.unravel_index(np.nanargmin(grid_err), grid_err.shape)
worst_cell = np.unravel_index(np.nanargmax(grid_err), grid_err.shape)
print(f"가장 정확한 구역: X {edges[best_cell[1]]:.0f}-{edges[best_cell[1]+1]:.0f}, "
      f"Y {edges[best_cell[0]]:.0f}-{edges[best_cell[0]+1]:.0f} (오차 {grid_err[best_cell]:.2f}mm)")
print(f"가장 부정확한 구역: X {edges[worst_cell[1]]:.0f}-{edges[worst_cell[1]+1]:.0f}, "
      f"Y {edges[worst_cell[0]]:.0f}-{edges[worst_cell[0]+1]:.0f} (오차 {grid_err[worst_cell]:.2f}mm)")
