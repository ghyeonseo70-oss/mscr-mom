"""
충돌(외력 접촉) 시나리오 데이터셋 생성.

로봇의 구동상태(MOM 위치 L_M, 외부자기장 각도 phi)와 접촉 조건(힘의 크기 |F|, 힘의 방향,
접촉이 발생한 위치 s)을 무작위로 다양하게 조합해서, force_model.solve_shape로 실제 결과
형상(팁/모멘트 위치·각도)을 계산한다. 같은 구동상태에서 "무외력 기준 형상"도 함께 계산해
두 형상의 차이(delta)까지 기록해두는데, 이게 나중에 "관측된 형상이 기준에서 얼마나/어떻게
벗어났는지 보고 접촉력(크기·방향·위치)을 역으로 추정"하는 모델을 만들 때 학습 데이터로 쓰인다.

무외력 기준 형상은 논문 식(7)-(9) 닫힌형(fsolve)으로 빠르게 구하고(이게 hint가 됨),
접촉이 있는 경우만 힘의 위치에 따라 곡률이 변하는 비선형 문제라 force_model의
ODE 슈팅법(solve_shape)을 쓴다 — 이렇게 섞어 써야 5000개 샘플도 몇 분 안에 끝난다.
"""
import os
import sys
import numpy as np
from scipy.optimize import fsolve

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'force_model'))
import force_model as fm

# ── 스윕 범위 ──────────────────────────────────────────────
N_SAMPLES = 3000
L_M_RANGE = (20.0, 80.0)      # mm, MOM 위치
PHI_RANGE = (-150.0, 150.0)   # deg, 외부자기장 방향
S_MARGIN = 5.0                # mm, 접촉위치는 팔 양끝에서 최소 이만큼 떨어짐 (베이스/팁 경계 특이점 회피)
F_MAX = 0.02                  # N (20 mN), 접촉력 크기 상한 (자기토크로 생기는 굽힘과 비슷한 규모 ~ 훨씬 큰 규모까지)

SEED = 42


def free_shape_closed_form(L_M, phi_deg):
    """무외력 기준 형상을 논문 식(7)-(9) 닫힌형으로 빠르게 계산 (ODE 없이 fsolve)."""
    phi = np.radians(phi_deg)
    l1 = (L_M - fm.H_M / 2) / 1000.0
    l2 = (fm.L - L_M - fm.H_M / 2) / 1000.0

    def eqs(vars):
        th_lm, th_l = vars
        tau1 = fm.M1 * fm.B_FIELD * np.sin(phi - th_lm - np.pi)
        tau2 = fm.M2 * fm.B_FIELD * np.sin(phi - th_l)
        return [fm.K1 * th_lm - (tau1 + tau2) * l1, fm.K2 * (th_l - th_lm) - tau2 * l2]

    best = None
    for th0 in np.radians(np.arange(-170, 171, 20)):
        sol, info, ier, msg = fsolve(eqs, [th0, th0], full_output=True)
        if ier == 1:
            dist = abs(((sol[1] - phi + np.pi) % (2 * np.pi)) - np.pi)
            if best is None or dist < best[0]:
                best = (dist, sol)
    if best is None:
        return None
    th_lm, th_l = best[1]

    l1_mm, l2_mm = L_M - fm.H_M / 2, fm.L - L_M - fm.H_M / 2
    R1 = l1_mm / (th_lm if abs(th_lm) > 1e-9 else 1e-9)
    dt = th_l - th_lm
    R2 = l2_mm / (dt if abs(dt) > 1e-9 else 1e-9)
    x_lm = R1 * np.sin(th_lm) + (fm.H_M / 2) * np.cos(th_lm)
    y_lm = R1 * (1 - np.cos(th_lm)) + (fm.H_M / 2) * np.sin(th_lm)
    x_l = x_lm + R2 * (np.sin(th_l) - np.sin(th_lm)) + (fm.H_M / 2) * np.cos(th_lm)
    y_l = y_lm + R2 * (np.cos(th_lm) - np.cos(th_l)) + (fm.H_M / 2) * np.sin(th_lm)
    return {
        "theta_LM_deg": np.degrees(th_lm), "x_LM": x_lm, "y_LM": y_lm,
        "theta_L_deg": np.degrees(th_l), "x_L": x_l, "y_L": y_l,
    }


rng = np.random.default_rng(SEED)
rows = []
n_done, n_fail = 0, 0

while n_done < N_SAMPLES:
    L_M = rng.uniform(*L_M_RANGE)
    phi_deg = rng.uniform(*PHI_RANGE)
    s = rng.uniform(S_MARGIN, fm.L - S_MARGIN)
    F_mag = rng.uniform(0.0, F_MAX)
    F_ang = rng.uniform(0, 2 * np.pi)
    Fx, Fy = F_mag * np.cos(F_ang), F_mag * np.sin(F_ang)

    r_free = free_shape_closed_form(L_M, phi_deg)
    if r_free is None:
        n_fail += 1
        continue
    try:
        r_load = fm.solve_shape(
            L_M=L_M, phi_deg=phi_deg,
            loads=[{"type": "point", "s": s, "Fx": Fx, "Fy": Fy}],
            theta_L_hint_deg=r_free["theta_L_deg"],
        )
    except RuntimeError:
        n_fail += 1
        continue

    rows.append([
        L_M, phi_deg,                                              # 구동상태
        s, F_mag, np.degrees(F_ang), Fx, Fy,                       # 접촉(힘) 파라미터
        r_free["x_L"], r_free["y_L"], r_free["theta_L_deg"],       # 무외력 기준 팁
        r_free["x_LM"], r_free["y_LM"], r_free["theta_LM_deg"],    # 무외력 기준 MOM
        r_load["x_L"], r_load["y_L"], r_load["theta_L_deg"],       # 접촉시 팁
        r_load["x_LM"], r_load["y_LM"], r_load["theta_LM_deg"],    # 접촉시 MOM
    ])
    n_done += 1
    if n_done % 500 == 0:
        print(f"{n_done}/{N_SAMPLES} (실패 {n_fail}건, 스킵됨)", flush=True)

COLUMNS = [
    "L_M", "phi_deg",
    "s", "F_mag", "F_ang_deg", "Fx", "Fy",
    "xL_free", "yL_free", "thL_free", "xLM_free", "yLM_free", "thLM_free",
    "xL_load", "yL_load", "thL_load", "xLM_load", "yLM_load", "thLM_load",
]
data = np.array(rows)

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'contact_scenarios', 'contact_force_scenarios.npz')
np.savez(out_path, data=data, columns=np.array(COLUMNS))
print(f"\n저장: {out_path}  (샘플 {len(data)}개, 실패/스킵 {n_fail}건)")
print(f"컬럼: {COLUMNS}")
