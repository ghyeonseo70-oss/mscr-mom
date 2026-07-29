import os
import numpy as np
import magpylib as magpy
from scipy.spatial.transform import Rotation
import pickle
from sklearn.preprocessing import StandardScaler

_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_DIR, '..', '..', 'data', 'cnn_model')
MODELS_DIR = os.path.join(_DIR, '..', '..', 'models')

BASE_X = 90  # 로봇 베이스 x좌표 (센서보드 180x180 중앙)

def get_magnet_positions(L_M, theta_mom_deg, d_theta_deg):
    L, H_M = 100, 8
    t1, dt = np.radians(theta_mom_deg), np.radians(d_theta_deg)
    l1, l2 = L_M - H_M/2, L - L_M - H_M/2

    R1 = l1 / (t1 if abs(t1) > 1e-5 else 1e-5)
    R2 = l2 / (dt if abs(dt) > 1e-5 else 1e-5)

    mom_x = BASE_X + R1 * (np.cos(0) - np.cos(t1)) + (H_M/2) * np.sin(t1)
    mom_y = 0 + R1 * (np.sin(t1) - np.sin(0)) + (H_M/2) * np.cos(t1)
    
    tip_x = mom_x + (H_M/2) * np.sin(t1) + R2 * (np.cos(t1) - np.cos(t1 + dt))
    tip_y = mom_y + (H_M/2) * np.cos(t1) + R2 * (np.sin(t1 + dt) - np.sin(t1))
    
    return mom_x, mom_y, t1, tip_x, tip_y, t1 + dt

print("데이터 생성 ")
# 실제 보드 배치와 동일한 순서: row0(y=180, 주황)~row4(y=0, 진초록), 각 행은 좌(x=0)->우(x=180)
# 센서-자석 간 높이: 30mm에서는 실측 신호(13uT)가 노이즈 바닥(~35uT)보다 작아 감지 불가 -> 15mm로 낮춤 (107uT, 충분한 마진)
SENSOR_HEIGHT_MM = 15
sensor_positions = [(x, y, SENSOR_HEIGHT_MM) for y in np.linspace(180, 0, 5) for x in np.linspace(0, 180, 5)]
sensors = magpy.Collection([magpy.Sensor(position=pos) for pos in sensor_positions])
# magpylib >=5 에서는 magnetization 기본 단위가 mT가 아니라 A/m(SI)이므로,
# 옛 코드의 "magnetization=1000"(1000 mT 의도)은 실제로는 자석 세기가 약 800배 약하게 계산됨.
# polarization은 Tesla 단위로 직접 지정 가능. 실측한 실물 자석 표면 자속밀도 3600G = 0.36T 사용.
MAGNET_BR_TESLA = 0.36
# 실제 자석은 z축(보드 수직)이 아니라 팔의 접선 방향(로컬 y)으로 자화되어 있고,
# main과 mom은 서로 반대 방향(역배치, S극끼리 마주봄)으로 배치되어 있음:
#   - mom의 S극이 main 쪽(앞)을 향함 -> mom의 자화 모멘트는 뒤(베이스쪽, 로컬 -y)를 향함
#   - main의 S극이 mom 쪽(뒤)을 향함 -> main의 자화 모멘트는 앞(팁 바깥쪽, 로컬 +y)을 향함
# 아래 .orientation 회전(각 자석의 굽힘각)이 이 로컬 방향을 실제 방향으로 돌려줌.
main_magnet = magpy.magnet.Cylinder(polarization=(0, MAGNET_BR_TESLA, 0), dimension=(2, 2))
mom = magpy.magnet.Cylinder(polarization=(0, -MAGNET_BR_TESLA, 0), dimension=(1, 8))
mscr_robot = magpy.Collection(main_magnet, mom)

data_X, data_y = [], []

for _ in range(100000):
    L_M = np.random.uniform(20, 80)
    t_mom = np.random.uniform(-90, 90)
    d_t = np.random.uniform(-120, 20) if t_mom > 0 else np.random.uniform(-20, 120)
    
    mx, my, ma, tx, ty, ta = get_magnet_positions(L_M, t_mom, d_t)
    
    mom.position = (mx, my, 0)
    mom.orientation = Rotation.from_euler('z', -np.degrees(ma), degrees=True)
    main_magnet.position = (tx, ty, 0)
    main_magnet.orientation = Rotation.from_euler('z', -np.degrees(ta), degrees=True)
    
    B_field = magpy.getB(mscr_robot, sensors) * 1e6  # Tesla -> uT (MLX90393 실측 단위와 동일하게)
    B_noisy = B_field + np.random.normal(0, np.max(np.abs(B_field)) * 0.05, B_field.shape)
    
    data_X.append(B_noisy.flatten())
    # z자리(항상 0이라 정보 없음) 대신 각도(도) 사용: main/tip=t_mom+d_t, mom=t_mom
    data_y.append([tx, ty, t_mom + d_t, mx, my, t_mom])

X = np.array(data_X)
y = np.array(data_y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

np.save(os.path.join(DATA_DIR, 'mscr_X.npy'), X_scaled)
np.save(os.path.join(DATA_DIR, 'mscr_y.npy'), y)
pickle.dump(scaler, open(os.path.join(MODELS_DIR, 'mscr_scaler_cnn.pkl'), 'wb'))
print("✅ 데이터 및 스케일러 저장 완료!")