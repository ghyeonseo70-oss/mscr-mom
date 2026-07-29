import serial
import torch
import torch.nn as nn
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D
import os
import time

# ⚙️ 설정값
SCALE_FACTOR = 1.0
NOISE_THRESHOLD = 50.0

# 1. 모델 정의
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

# 2. 로드
current_dir = os.path.dirname(__file__)
model = CNNPositionEstimator()
model.load_state_dict(torch.load(os.path.join(current_dir, '..', '..', 'models', 'mscr_cnn_model.pth'), map_location='cpu', weights_only=True))
model.eval()
scaler = pickle.load(open(os.path.join(current_dir, '..', '..', 'models', 'mscr_scaler_cnn.pkl'), 'rb'))

# 3. 그래프 세팅 (Y축 반전 적용 완료)
plt.ion()
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(0, 180)
ax.set_ylim(180, 0)  # 👈 핵심 수정: Y축을 180에서 0으로 설정하여 화면을 위아래로 뒤집음
ax.set_aspect('equal')

MAIN_SIZE = (10, 10)  # main_magnet 지름2mm x 높이2mm 비율(1:1), 시인성 위해 확대
MOM_SIZE = (6, 24)    # mom 지름1mm x 높이8mm 비율(1:8), 시인성 위해 확대

main_patch = Rectangle((-MAIN_SIZE[0]/2, -MAIN_SIZE[1]/2), *MAIN_SIZE, color='red', ec='black')
mom_patch = Rectangle((-MOM_SIZE[0]/2, -MOM_SIZE[1]/2), *MOM_SIZE, color='blue', ec='black')
ax.add_patch(main_patch)
ax.add_patch(mom_patch)

def update_magnet(patch, x, y, angle_deg):
    patch.set_transform(Affine2D().rotate_deg(angle_deg).translate(x, y) + ax.transData)

plt.show(block=False)

# 4. 통신 설정
ser = serial.Serial('COM3', 115200, timeout=1)
time.sleep(2)
ser.reset_input_buffer()

# 5. 영점 측정
print("⏳ 3초간 영점 조절 중...")
offset_list = []
start = time.time()
while time.time() - start < 3:
    if ser.in_waiting > 0:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        parts = line.split(',')
        if len(parts) > 10:
            raw_data = np.array([float(x) for x in parts[1:]][:75])
            if len(raw_data) == 75: offset_list.append(raw_data)
baseline_offset = np.mean(offset_list, axis=0) if offset_list else np.zeros(75)

# 6. 메인 루프
EMA_ALPHA = 0.35  # 낮을수록 더 부드럽지만 반응이 느려짐 (0~1)
smoothed_pred = None

while True:
    try:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            parts = line.split(',')
            if len(parts) > 10:
                raw_data = np.array([float(x) for x in parts[1:]][:75])
                calibrated = raw_data - baseline_offset

                # 데이터가 충분할 때만 예측 실행
                if np.max(np.abs(calibrated)) >= NOISE_THRESHOLD:
                    # AI 추론 파이프라인
                    scaled_down = calibrated / SCALE_FACTOR
                    scaled_data = scaler.transform(scaled_down.reshape(1, -1))
                    cnn_data = scaled_data.reshape(-1, 5, 5, 3).transpose(0, 3, 1, 2)

                    with torch.no_grad():
                        pred = model(torch.tensor(cnn_data, dtype=torch.float32)).numpy()[0]

                    # 프레임 간 지터 완화용 지수이동평균
                    if smoothed_pred is None:
                        smoothed_pred = pred
                    else:
                        smoothed_pred = EMA_ALPHA * pred + (1 - EMA_ALPHA) * smoothed_pred
                    pred = smoothed_pred.copy()

                    # 보정 및 출력 (위치[0,1,3,4]만 0~180 클립, 각도[2,5]는 클립하지 않음)
                    pred[[0, 1, 3, 4]] = np.clip(pred[[0, 1, 3, 4]], 0, 180)

                    # 💡 만약 화면뿐만 아니라 터미널에 찍히는 로그 데이터도 위아래로 
                    # 뒤집고 싶다면 아래 두 줄의 주석(#)을 지우고 실행하세요.
                    # pred[1] = 180.0 - pred[1]
                    # pred[4] = 180.0 - pred[4]

                    update_magnet(main_patch, pred[0], pred[1], pred[2])
                    update_magnet(mom_patch, pred[3], pred[4], pred[5])
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                    print(f"🔥 좌표 추정: Main({pred[0]:.1f}, {pred[1]:.1f}, {pred[2]:.1f}°) | MOM({pred[3]:.1f}, {pred[4]:.1f}, {pred[5]:.1f}°) | 변화량:{np.max(np.abs(calibrated)):.1f}")
                else:
                    smoothed_pred = None  # 자석이 멀어지면 스무딩 상태 초기화
                    print(f"📭 대기 중... (변화량: {np.max(np.abs(calibrated)):.1f})")

    except Exception as e:
        print(f"에러 발생: {e}")
        break
        
ser.close()