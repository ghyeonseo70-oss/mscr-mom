import numpy as np
import matplotlib.pyplot as plt

def calc_pcc_kinematics(L_M, theta_mom_deg, d_theta_deg):
    L, H_M = 100, 8
    t1, dt = np.radians(theta_mom_deg), np.radians(d_theta_deg)
    l1, l2 = L_M - H_M/2, L - L_M - H_M/2
    
    # 0,0에서 시작하지만 X=50으로 이동 (+Y 방향으로 자람)
    def arc(x0, y0, a0, length, theta):
        s = np.linspace(0, length, 30)
        if abs(theta) < 1e-5: return x0 + s * np.sin(a0), y0 + s * np.cos(a0), a0
        R = length / theta
        xs = x0 + R * (np.cos(a0) - np.cos(a0 + theta * (s/length)))
        ys = y0 + R * (np.sin(a0 + theta * (s/length)) - np.sin(a0))
        return xs, ys, a0 + theta

    x1, y1, a1 = arc(50, 0, 0, l1, t1)
    
    # MOM 강체 구간 (직선)
    sm = np.linspace(0, H_M, 10)
    xm = x1[-1] + sm * np.sin(a1)
    ym = y1[-1] + sm * np.cos(a1)
    mom_center_x = x1[-1] + (H_M/2) * np.sin(a1)
    mom_center_y = y1[-1] + (H_M/2) * np.cos(a1)
    
    x2, y2, a2 = arc(xm[-1], ym[-1], a1, l2, dt)
    return x1, y1, xm, ym, x2, y2, mom_center_x, mom_center_y

# 랜덤 샘플 1개 생성 (역배치 특성을 위해 곡률 방향을 엇갈리게 세팅)
L_M = np.random.uniform(20, 80)
t_mom = np.random.uniform(-90, 90)
d_t = np.random.uniform(-120, 20) if t_mom > 0 else np.random.uniform(-20, 120)

x1, y1, xm, ym, x2, y2, mom_x, mom_y = calc_pcc_kinematics(L_M, t_mom, d_t)

plt.figure(figsize=(6, 6))
sx, sy = np.meshgrid(np.linspace(0, 100, 5), np.linspace(0, 100, 5))
plt.scatter(sx, sy, c='lightgray', marker='s', s=40)
plt.plot(50, 0, 'ks', markersize=10) # 베이스

plt.plot(x1, y1, 'k-', linewidth=4)
plt.plot(xm, ym, 'k-', linewidth=4, color='gray') # MOM 강체 부분
plt.plot(x2, y2, 'k-', linewidth=4)

plt.plot(mom_x, mom_y, 'bo', markersize=14)
plt.plot(x2[-1], y2[-1], 'ro', markersize=14)

plt.xlim(0, 100); plt.ylim(0, 100)
plt.gca().set_xticks([]); plt.gca().set_yticks([])
for spine in plt.gca().spines.values(): spine.set_visible(False)
plt.show()