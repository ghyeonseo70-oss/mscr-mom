"""교수님이 주신 MATLAB 코드(code (2).txt)를 Python으로 포팅.

이 MATLAB 코드는 fmincon(비선형 최적화)으로 2-세그먼트 엘라스티카(순수 토크 하중,
constant-curvature 닫힌해)의 경계조건을 맞춰서 팁 위치/각도를 구한다 - force_model.py와
같은 물리(K1/K2 두 구간 굽힘)를 다른 수치기법(ODE 적분+shooting 대신 닫힌해+근찾기)으로 푼다.

MATLAB에 Optimization/Symbolic Toolbox가 없어 실행 불가 -> scipy.optimize.fsolve로 포팅
(fmincon의 목적함수는 사실상 임의값이라 - 등식제약 ceq=0을 만족하는 근을 찾는 것 자체가
목적이므로, root-finding(fsolve)이 더 직접적인 대응).
"""
import os

import numpy as np
from scipy.optimize import brentq

# ── 상수 (MATLAB과 동일) ──────────────────────────
L = 0.1  # m, 전체 길이
E = 850 * 1000  # Pa
mu0 = 4 * np.pi * 1e-7
Br = 0.4  # T
M = Br / mu0
D = 0.002
D2 = 0.001
I = D**4 * np.pi / 64 - D2**4 * np.pi / 64
v11 = 0.006 * np.pi * D**2 / 16
v2 = 0.002 * np.pi * D**2 / 4
EN = 2
H = 40000


def solve_two_segment(L1, thetaB2_deg, x0_guess=(0.0, 0.0)):
    """ceq2(x1,x2)=0은 x1에 대해 명시적으로 풀림: x1 = x2 - (L-L1)*H*M*v2*mu0*sin(th-x2)/(E*I).
    이걸 ceq1에 대입하면 x2 하나짜리 1차원 방정식이 됨 - force_model.py와 동일한 방식
    (넓은 범위 스캔 -> 부호변화 지점마다 brentq)으로 모든 근을 찾고, 그중 fmincon의 목적함수
    (x1^2+x2^2 최소)를 만족하는 근을 선택 - 2D 최적화보다 훨씬 안정적."""
    thetaB2 = np.radians(thetaB2_deg)

    def x1_of_x2(x2):
        return x2 - (L - L1) * H * M * v2 * mu0 * np.sin(thetaB2 - x2) / (E * I)

    def residual(x2):
        x1 = x1_of_x2(x2)
        denom = -((H * M * v11 * mu0 * np.sin(thetaB2 - x1)) - (H * M * v2 * mu0 * np.sin(thetaB2 - x2)))
        if abs(denom) < 1e-30:
            return np.nan
        return EN * E * I * x1 / denom - L1

    scan = np.radians(np.arange(-540, 540.1, 1.0))
    vals = np.array([residual(g) for g in scan])
    roots = []
    for i in range(len(scan) - 1):
        v0, v1 = vals[i], vals[i + 1]
        if not (np.isfinite(v0) and np.isfinite(v1)):
            continue
        if np.sign(v0) != np.sign(v1):
            try:
                r = brentq(residual, scan[i], scan[i + 1], xtol=1e-10)
                x1r = x1_of_x2(r)
                if abs(abs(x1r) - abs(r)) - np.pi <= 1e-6:  # 원 코드의 부등식 제약
                    roots.append((x1r, r))
            except (ValueError, RuntimeError):
                pass
    if not roots:
        raise RuntimeError(f"해 없음 L1={L1}, phi={thetaB2_deg}")
    x1, x2 = min(roots, key=lambda r: r[0] ** 2 + r[1] ** 2)  # fmincon 목적함수: x1^2+x2^2 최소
    return x1, x2


def tip_from_solution(L1, thetaB2_deg, x1, x2):
    thetaB2 = np.radians(thetaB2_deg)
    A1lin = np.linspace(0, x1, 500)
    A2lin = np.linspace(x1, x2, 500)
    denom_a = -(H * M * v11 * mu0 * np.sin(thetaB2 - x1)) + (H * M * v2 * mu0 * np.sin(thetaB2 - x2))
    x1c = EN * E * I * np.sin(A1lin) / denom_a
    y1c = EN * E * I * (1 - np.cos(A1lin)) / denom_a
    x10, y10 = x1c[-1], y1c[-1]
    denom_b = H * M * v2 * mu0 * np.sin(thetaB2 - x2)
    x2c = (E * I * (np.sin(A2lin) - np.sin(x1)) / denom_b) + x10
    y2c = (E * I * (np.cos(x1) - np.cos(A2lin)) / denom_b) + y10
    return y2c[-1]  # r = ytotal(1000) 팁 y위치(m)


def run_L1_sweep(L1, thetaset_deg):
    xs, ys = [], []
    guess = (0.0, 0.0)
    for thetaB2 in thetaset_deg:
        x1, x2 = solve_two_segment(L1, thetaB2, guess)
        guess = (x1, x2)  # continuation - 다음 각도의 초기값으로 재사용(수렴 안정성)
        r = tip_from_solution(L1, thetaB2, x1, x2)
        xs.append(r * 100)
        ys.append(np.degrees(x2))
    return np.array(xs), np.array(ys)


def solve_single_segment(thetaB_deg, x0_guess=0.0):
    thetaB = np.radians(thetaB_deg)

    def residual(x):
        denom = H * M * v11 * mu0 * np.sin(thetaB - x)
        if abs(denom) < 1e-30:
            return np.nan
        return E * I * x / denom - L

    scan = np.radians(np.arange(-179, 180, 1.0))
    vals = np.array([residual(g) for g in scan])
    roots = []
    for i in range(len(scan) - 1):
        v0, v1 = vals[i], vals[i + 1]
        if not (np.isfinite(v0) and np.isfinite(v1)):
            continue
        if np.sign(v0) != np.sign(v1):
            try:
                roots.append(brentq(residual, scan[i], scan[i + 1], xtol=1e-10))
            except (ValueError, RuntimeError):
                pass
    if not roots:
        raise RuntimeError(f"해 없음 thetaB={thetaB_deg}")
    return min(roots, key=abs)


if __name__ == "__main__":
    print(f"상수 확인: I={I:.4e}, M={M:.4e}, v11={v11:.4e}, v2={v2:.4e}")
    print()

    results = {}
    for L1_mm, thetaset in [
        (10, list(range(31, 152, 10))),
        (20, list(range(30, 151, 10))),
        (30, list(range(30, 151, 10))),
        (40, list(range(30, 151, 10))),
        (50, list(range(30, 151, 10))),
    ]:
        L1 = L1_mm / 1000.0
        xs, ys = run_L1_sweep(L1, thetaset)
        results[L1_mm] = (thetaset, xs, ys)
        print(f"L_M={L1_mm}mm:")
        for th, x, y in zip(thetaset, xs, ys):
            print(f"  phi={th:>4}  tip_y*100={x:>8.4f}  tip_theta(deg)={y:>8.3f}")
        print()

    # data0 (L_M -> 0 극한)
    print("L_M->0 (단일 세그먼트):")
    guess = 0.0
    for thetaB in range(10, 151, 10):
        alpha = solve_single_segment(thetaB, guess)
        guess = alpha
        denom = H * M * v11 * mu0 * np.sin(np.radians(thetaB) - alpha)
        y0 = E * I * (1 - np.cos(alpha)) / denom
        print(f"  phi={thetaB:>4}  tip_y*100={y0*100:>8.4f}  tip_theta(deg)={np.degrees(alpha):>8.3f}")

    HERE = os.path.dirname(os.path.abspath(__file__))
    OUT = os.path.join(HERE, "..", "..", "..", "data", "contact_scenarios", "fea", "professor_matlab_port_results.npz")
    np.savez(OUT, **{f"L{k}_theta": v[0] for k, v in results.items()},
             **{f"L{k}_x": v[1] for k, v in results.items()},
             **{f"L{k}_y": v[2] for k, v in results.items()})
    print(f"\n저장: {OUT}")
