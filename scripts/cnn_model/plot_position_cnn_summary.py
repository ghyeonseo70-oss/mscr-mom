"""홀센서 자기장 -> 자석 위치추정 CNN 결과를 깔끔하게 정리한 발표용 시각화."""
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, '..', '..', 'data', 'cnn_model')
MODELS_DIR = os.path.join(HERE, '..', '..', 'models')


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

    def forward(self, x):
        return self.regressor(self.features(x))


X_scaled = np.load(os.path.join(DATA_DIR, 'mscr_X.npy'))
y = np.load(os.path.join(DATA_DIR, 'mscr_y.npy'))
loss_history = np.load(os.path.join(DATA_DIR, 'mscr_loss_history.npy'))
train_losses, val_losses = loss_history[0], loss_history[1]

X_cnn = X_scaled.reshape(-1, 5, 5, 3).transpose(0, 3, 1, 2)
split = int(0.8 * len(X_cnn))
test_X = torch.tensor(X_cnn[split:]).float()
test_y = y[split:]

model = CNNPositionEstimator()
model.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'mscr_cnn_model.pth'),
                                  map_location='cpu', weights_only=True))
model.eval()
with torch.no_grad():
    pred_y = model(test_X).numpy()

main_err = np.linalg.norm(pred_y[:, :2] - test_y[:, :2], axis=1)
mom_err = np.linalg.norm(pred_y[:, 3:5] - test_y[:, 3:5], axis=1)
tip_angle_err = np.abs(pred_y[:, 2] - test_y[:, 2])
mom_angle_err = np.abs(pred_y[:, 5] - test_y[:, 5])

ACCENT_MAIN = "#e34948"
ACCENT_MOM = "#2a78d6"

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05], hspace=0.4, wspace=0.32)

# ── (A) 학습곡선 ──────────────────────────────
ax = fig.add_subplot(gs[0, 0])
ax.plot(train_losses, color="#2a78d6", linewidth=2, label="Train")
ax.plot(val_losses, color="#e34948", linewidth=2, label="Validation")
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE Loss")
ax.set_title("(A) 학습 곡선", fontweight="bold", fontsize=12)
ax.grid(True, linestyle=":", alpha=0.5)
ax.legend(fontsize=9)

# ── (B) 예측 vs 실제 위치 (일부 샘플) ──────────────────────────────
ax = fig.add_subplot(gs[0, 1:])
rng = np.random.default_rng(0)
indices = rng.choice(len(test_y), 40, replace=False)
ax.add_patch(plt.Rectangle((0, 0), 180, 180, fill=False, color="#999999", linestyle="--", linewidth=1))
ax.plot(90, 0, "ks", markersize=10, label="로봇 베이스", zorder=5)
for i, idx in enumerate(indices):
    ax.plot([test_y[idx, 0], pred_y[idx, 0]], [test_y[idx, 1], pred_y[idx, 1]], color=ACCENT_MAIN, alpha=0.25, linewidth=1)
    ax.plot([test_y[idx, 3], pred_y[idx, 3]], [test_y[idx, 4], pred_y[idx, 4]], color=ACCENT_MOM, alpha=0.25, linewidth=1)
    ax.plot(test_y[idx, 0], test_y[idx, 1], "o", color=ACCENT_MAIN, markersize=6, alpha=0.55,
            label="실제 - main(팁)" if i == 0 else "")
    ax.plot(pred_y[idx, 0], pred_y[idx, 1], "x", color=ACCENT_MAIN, markersize=8, mew=1.8,
            label="예측 - main(팁)" if i == 0 else "")
    ax.plot(test_y[idx, 3], test_y[idx, 4], "o", color=ACCENT_MOM, markersize=6, alpha=0.55,
            label="실제 - MOM" if i == 0 else "")
    ax.plot(pred_y[idx, 3], pred_y[idx, 4], "x", color=ACCENT_MOM, markersize=8, mew=1.8,
            label="예측 - MOM" if i == 0 else "")
ax.set_xlim(0, 180)
ax.set_ylim(0, 180)
ax.set_xlabel("x (mm, 보드좌표)")
ax.set_ylabel("y (mm, 보드좌표)")
ax.set_title("(B) 예측 vs 실제 자석 위치 (검증셋 40개 예시)", fontweight="bold", fontsize=12)
ax.set_aspect("equal")
ax.legend(loc="upper right", fontsize=8, ncol=2)
ax.grid(True, linestyle=":", alpha=0.4)

# ── (C)(D) 공간별 오차 히트맵 ──────────────────────────────
sensor_x = np.linspace(0, 180, 5)
sensor_y = np.linspace(180, 0, 5)
sensor_xx, sensor_yy = np.meshgrid(sensor_x, sensor_y)

for col, (label, true_pos, err) in enumerate([("main(팁)", test_y[:, :2], main_err), ("MOM", test_y[:, 3:5], mom_err)]):
    ax = fig.add_subplot(gs[1, col])
    hb = ax.hexbin(true_pos[:, 0], true_pos[:, 1], C=err, gridsize=18, cmap="YlOrRd",
                    reduce_C_function=np.mean, extent=[0, 180, 0, 180])
    ax.scatter(sensor_xx, sensor_yy, c="black", marker="x", s=35, linewidths=1.3, label="홀센서(25개)")
    ax.plot(90, 0, "ks", markersize=9)
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("평균 위치오차 (mm)", fontsize=9)
    ax.set_xlim(0, 180)
    ax.set_ylim(0, 180)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(f"({'C' if col==0 else 'D'}) 공간별 오차 - {label}", fontweight="bold", fontsize=12)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)

# ── 요약 텍스트 ──────────────────────────────
ax = fig.add_subplot(gs[1, 2])
ax.axis("off")
ax.text(0.0, 0.95, "요약", fontsize=15, fontweight="bold", transform=ax.transAxes)
ax.text(0.0, 0.8,
        f"main(팁) 평균 위치오차: {main_err.mean():.2f} mm\n"
        f"MOM 평균 위치오차:     {mom_err.mean():.2f} mm\n\n"
        f"main 평균 각도오차:    {tip_angle_err.mean():.2f}°\n"
        f"MOM 평균 각도오차:     {mom_angle_err.mean():.2f}°\n\n"
        f"검증셋 크기: {len(test_y)}개\n"
        f"입력: 홀센서 25개×3축(75개), 5×5 격자\n"
        f"출력: 자석 2개의 x,y,θ (6개 값)",
        fontsize=12, transform=ax.transAxes, va="top", linespacing=1.9)

fig.suptitle("홀센서 자기장 기반 자석 위치추정 CNN 결과", fontweight="bold", fontsize=17, y=1.02)

out_path = os.path.join(DATA_DIR, "position_cnn_summary.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"저장: {out_path}")
print(f"main 평균오차: {main_err.mean():.2f}mm, MOM 평균오차: {mom_err.mean():.2f}mm")
print(f"main 각도오차: {tip_angle_err.mean():.2f}deg, MOM 각도오차: {mom_angle_err.mean():.2f}deg")
