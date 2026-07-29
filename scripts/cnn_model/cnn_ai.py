import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_DIR, '..', '..', 'data', 'cnn_model')
MODELS_DIR = os.path.join(_DIR, '..', '..', 'models')

# 1. 데이터 로드
X_scaled = np.load(os.path.join(DATA_DIR, 'mscr_X.npy'))
y = np.load(os.path.join(DATA_DIR, 'mscr_y.npy'))

X_cnn = X_scaled.reshape(-1, 5, 5, 3).transpose(0, 3, 1, 2)
split = int(0.8 * len(X_cnn))
train_loader = DataLoader(TensorDataset(torch.tensor(X_cnn[:split]).float(), torch.tensor(y[:split]).float()), batch_size=64, shuffle=True)
test_X, test_y = torch.tensor(X_cnn[split:]).float(), torch.tensor(y[split:]).float()

# 2. 모델 뼈대
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

model = CNNPositionEstimator()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

train_losses = []
val_losses = []

print("🚀 200 Epoch CNN 집중 학습 시작...")
for epoch in range(200):
    model.train()
    batch_losses = []
    for bx, by in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(bx), by)
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())
    
    avg_train_loss = np.mean(batch_losses)
    train_losses.append(avg_train_loss)
    
    model.eval()
    with torch.no_grad():
        val_loss = criterion(model(test_X), test_y).item()
        val_losses.append(val_loss)

    if (epoch + 1) % 20 == 0:
        print(f"🔄 Epoch [{epoch+1:3d}/200] | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")

# 3. 모델 및 Loss 기록 저장 (나중에 그래프 그릴 때 사용)
torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'mscr_cnn_model.pth'))
np.save(os.path.join(DATA_DIR, 'mscr_loss_history.npy'), np.array([train_losses, val_losses]))
print("✅ 모델 학습 및 기록 저장 완료! (이제 평가 코드를 돌려보세요)")

import numpy as np
import torch
import matplotlib.pyplot as plt

# --- (이전에 모델 학습이 완료되고 'mscr_cnn_model.pth'가 저장된 상태라고 가정) ---

# 1. 학습된 모델로 테스트 데이터 추론
model.eval()
with torch.no_grad():
    preds = model(test_X).numpy()
true_y = test_y.numpy()

# 2. Main 자석과 MOM의 위치 오차 계산 (유클리디안 거리, 단위: mm)
# 정답 라벨 y 구조: [tx, ty, ta, mx, my, ma]
# Main 자석: 인덱스 0(tx), 1(ty)
main_error = np.sqrt((preds[:, 0] - true_y[:, 0])**2 + (preds[:, 1] - true_y[:, 1])**2)

# MOM: 인덱스 3(mx), 4(my)
mom_error = np.sqrt((preds[:, 3] - true_y[:, 3])**2 + (preds[:, 4] - true_y[:, 4])**2)

# 3. 공간 오차 히트맵 시각화 (Hexbin Plot)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Spatial Error Heatmap (Sim-to-Real Model Evaluation)', fontsize=16)

# 센서 위치 계산 (X: -90~90, Y: 0~180)
sensor_x = [x for x in np.linspace(-90, 90, 5) for _ in np.linspace(0, 180, 5)]
sensor_y = [y for _ in np.linspace(-90, 90, 5) for y in np.linspace(0, 180, 5)]

# [subplot 1] Main 자석 오차 히트맵
# C=main_error, reduce_C_function=np.mean -> 해당 구역의 '평균 오차'를 색상으로 표현
hb1 = axes[0].hexbin(true_y[:, 0], true_y[:, 1], C=main_error, gridsize=20, 
                     cmap='YlOrRd', reduce_C_function=np.mean, alpha=0.9)
axes[0].scatter(sensor_x, sensor_y, c='black', marker='x', s=50, label='Hall Sensors')
axes[0].set_title('Main Magnet Position Error')
axes[0].set_xlim(-90, 90)
axes[0].set_ylim(0, 180)
axes[0].set_xlabel('X Coordinate (mm)')
axes[0].set_ylabel('Y Coordinate (mm)')
axes[0].legend(loc='upper right')
cb1 = plt.colorbar(hb1, ax=axes[0])
cb1.set_label('Mean Position Error (mm)')

# [subplot 2] MOM 오차 히트맵
hb2 = axes[1].hexbin(true_y[:, 3], true_y[:, 4], C=mom_error, gridsize=20, 
                     cmap='YlOrRd', reduce_C_function=np.mean, alpha=0.9)
axes[1].scatter(sensor_x, sensor_y, c='black', marker='x', s=50, label='Hall Sensors')
axes[1].set_title('MOM Position Error')
axes[1].set_xlim(-90, 90)
axes[1].set_ylim(0, 180)
axes[1].set_xlabel('X Coordinate (mm)')
axes[1].legend(loc='upper right')
cb2 = plt.colorbar(hb2, ax=axes[1])
cb2.set_label('Mean Position Error (mm)')

plt.tight_layout()
plt.show()

# 4. 종합 오차 리포트 출력
print("📊 [Test Dataset Error Report]")
print(f"Main 자석 평균 위치 오차: {np.mean(main_error):.2f} mm (최대: {np.max(main_error):.2f} mm)")
print(f"MOM 평균 위치 오차:      {np.mean(mom_error):.2f} mm (최대: {np.max(mom_error):.2f} mm)")

# 만약 각도 오차도 확인하고 싶다면 (단위: Degree)
main_angle_error = np.abs(preds[:, 2] - true_y[:, 2])
mom_angle_error = np.abs(preds[:, 5] - true_y[:, 5])
print(f"Main 자석 평균 각도 오차: {np.mean(main_angle_error):.2f}°")
print(f"MOM 평균 각도 오차:      {np.mean(mom_angle_error):.2f}°")