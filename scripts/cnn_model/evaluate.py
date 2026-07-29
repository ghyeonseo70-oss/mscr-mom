import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import os

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

# 📁 상대 경로 지정
current_dir = os.path.dirname(__file__)
X_path = os.path.join(current_dir, '..', '..', 'data', 'cnn_model', 'mscr_X.npy')
y_path = os.path.join(current_dir, '..', '..', 'data', 'cnn_model', 'mscr_y.npy')
loss_path = os.path.join(current_dir, '..', '..', 'data', 'cnn_model', 'mscr_loss_history.npy')
model_path = os.path.join(current_dir, '..', '..', 'models', 'mscr_cnn_model.pth')

# 파일 로드
X_scaled = np.load(X_path)
y = np.load(y_path)
loss_history = np.load(loss_path)
train_losses, val_losses = loss_history[0], loss_history[1]

X_cnn = X_scaled.reshape(-1, 5, 5, 3).transpose(0, 3, 1, 2)
split = int(0.8 * len(X_cnn))
test_X = torch.tensor(X_cnn[split:]).float()
test_y_np = y[split:]

model = CNNPositionEstimator()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
model.eval()

with torch.no_grad():
    pred_y = model(test_X).numpy()
    score = r2_score(test_y_np[:, [0,1,3,4]], pred_y[:, [0,1,3,4]]) * 100
    main_err = np.mean(np.linalg.norm(pred_y[:, :2] - test_y_np[:, :2], axis=1))
    mom_err = np.mean(np.linalg.norm(pred_y[:, 3:5] - test_y_np[:, 3:5], axis=1))
    tip_angle_err = np.mean(np.abs(pred_y[:, 2] - test_y_np[:, 2]))
    mom_angle_err = np.mean(np.abs(pred_y[:, 5] - test_y_np[:, 5]))

    print(f"\n📊 [최종 평가 점수]")
    print(f"🏆 종합 형태 추정 R2 스코어: {score:.2f} / 100 점")
    print(f"🎯 메인 자석 평균 오차: {main_err:.2f} mm | mom 자석 평균 오차: {mom_err:.2f} mm")
    print(f"📐 tip 각도 평균 오차: {tip_angle_err:.2f}도 | mom 각도 평균 오차: {mom_angle_err:.2f}도")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss', color='blue', linewidth=2)
plt.plot(val_losses, label='Val Loss', color='red', linestyle='--', linewidth=2)
plt.title('CNN Learning Curve', fontsize=14, fontweight='bold')
plt.xlabel('Epochs', fontsize=12); plt.ylabel('MSE Loss', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.7); plt.legend()

plt.subplot(1, 2, 2)
plt.xlim(0, 180); plt.ylim(0, 180)
plt.title('Prediction vs Ground Truth', fontsize=14, fontweight='bold')
plt.gca().add_patch(plt.Rectangle((0, 0), 180, 180, fill=False, color='gray', linestyle='--'))
plt.plot(90, 0, 'ks', markersize=10, label='Robot Base')

indices = np.random.choice(len(test_y_np), 30, replace=False)
for i, idx in enumerate(indices):
    plt.plot(test_y_np[idx, 0], test_y_np[idx, 1], 'ro', markersize=6, alpha=0.3, label='True Main' if i==0 else "")
    plt.plot(test_y_np[idx, 3], test_y_np[idx, 4], 'bo', markersize=6, alpha=0.3, label='True mom' if i==0 else "")
    plt.plot(pred_y[idx, 0], pred_y[idx, 1], 'rx', markersize=8, label='Pred Main' if i==0 else "")
    plt.plot(pred_y[idx, 3], pred_y[idx, 4], 'bx', markersize=8, label='Pred mom' if i==0 else "")
    plt.plot([test_y_np[idx, 0], pred_y[idx, 0]], [test_y_np[idx, 1], pred_y[idx, 1]], 'r-', alpha=0.3)
    plt.plot([test_y_np[idx, 3], pred_y[idx, 3]], [test_y_np[idx, 4], pred_y[idx, 4]], 'b-', alpha=0.3)

plt.legend(loc='upper right', fontsize=8)
plt.gca().set_aspect('equal', adjustable='box')
plt.tight_layout()

fig_path = os.path.join(current_dir, '..', '..', 'data', 'cnn_model', 'training_result.png')
plt.savefig(fig_path, dpi=150)
print(f"🖼️  그래프 저장: {fig_path}")
plt.show()