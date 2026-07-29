"""
MSCR-MOM 외력(다중 접촉, 점하중/분포하중 혼합) 포함 정방향 모델
입력: L_M, phi(deg), loads(접촉 리스트)  ->  출력: x_LM,y_LM,theta_LM, x_L,y_L,theta_L

논문(Park et al., IEEE RA-L 2024)의 식(1)~(12)를 외력 항으로 확장.
자기 토크(순수 커플)는 각 구간에서 곡률을 상수로 만들어 닫힌해(원호)가 나오지만,
외력은 모멘트 암이 위치에 따라 달라져서 그 구간은 비선형(엘라스티카)이 되므로
상태공간 ODE(theta,x,y,M,Nx,Ny) + 슈팅법(fsolve)으로 수치적으로 푼다.

loads 형식 (리스트, 여러 개 동시 가능 = 다중 접촉):
  점하중:   {'type': 'point', 's': 80,              'Fx': 0.01, 'Fy': 0.0}
  분포하중: {'type': 'dist',  's_start': 20, 's_end': 40, 'Fx': 0.01, 'Fy': 0.0}
            (Fx,Fy는 그 구간에 걸친 "총합" 힘. 구간 내 균등분포로 가정)

주의: K1,K2,M1은 실물 로봇 실측이 아니라 논문 Fig.3(e)-(h) 산점도를 픽셀 단위로 디지털화(38개
데이터점, scripts/_digitize_full.py)한 뒤 최소자승 피팅(scripts/_fit_k1k2_m.py)한 값이다.
실물 로봇을 만들었다면 그 로봇으로 외팔보 처짐 실험 또는 자기토크 역산을 해서 이 값을 교체할 것.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

# ── 기하/재료 상수 (논문 기준) ──────────────────────────────
L = 100.0          # 전체 길이 (mm)
H_M = 8.0          # MOM 길이 (mm)
MU0 = 4 * np.pi * 1e-7

# ── 굽힘강성 K1(니티놀+실리콘 구간), K2(실리콘 단독 구간) ──────────────────
# 문헌 E값으로 계산(E*I 합산)하면 K1/K2가 1.33배 밖에 안 나오는데, 이 정도로는 두 구간의
# 곡률 부호가 절대 반전되지 않아 논문 Fig.3(b)-(d)의 2중곡률(S자) 형상이 원리적으로 나올 수
# 없다(자세한 원리 진단: scripts/_diag_scurve*.py). 그래서 논문 Fig.3(e)-(h) 산점도(yL,
# thetaL vs phi, LM/L)를 고해상도 원본 이미지에서 직접 픽셀 좌표로 뽑아
# (scripts/_digitize_full.py, 38개점), 식(7)-(8)에 최소자승 피팅(scripts/_fit_k1k2_m.py)해서
# 아래 값을 구했다(정규화 RMS 잔차 0.53). CMSCR 8개점만으로 독립 역산한 K2도 같은 범위
# (~7~8.5e-7)로 나와 상호검증됨(scripts/_check_cmscr.py).
#
# 이 과정에서 논문 원문 표기의 식(5)-(8) 부호("-tau1-tau2","-tau2")를 그대로 구현하면
# CMSCR 데이터에서마저 K2가 항상 음수(물리적으로 불가능)로 나왔고, 부호를 뒤집으니(양수
# "+tau") 모든 LM/L·phi 조합에서 K1,K2가 안정적으로 일관된 값에 수렴했다. 그래서 아래
# shoot() 안의 부호는 논문 원문 표기가 아니라 이 실측(디지털화) 기준으로 확정했다
# (근거: scripts/_check_cmscr.py, scripts/_check_mom.py).
K1 = 3.8248e-06   # N*m^2
K2 = 9.8872e-07   # N*m^2
K_RIGID = 1e6 * max(K1, K2)

BR = 0.36  # T (자석 표면자속밀도 실측값 3600G 기준)
M_MAG = BR / MU0

def magnet_moment(diam_mm, len_mm):
    r, h = diam_mm / 2 * 1e-3, len_mm * 1e-3
    V = np.pi * r**2 * h
    return M_MAG * V

# M1(MOM)/M2(main) 부피비로는 1:1이지만, 위 피팅에서 M1/M2=1.290이 최적으로 나와 반영함
# (제작 공차·자화 불균일 등으로 실제 세기가 부피비와 정확히 같지 않을 수 있음).
M1 = magnet_moment(1.0, 8.0) * 1.290
M2 = magnet_moment(2.0, 2.0)
# 논문 Section II-D "Basic Actuation Tests"(Fig.3, Fig.4 형상 비교용 실험) 조건: 4,000 A/m = 0.05T.
# 0.035T(35mT)는 Section III-D 눈알 팬텀 테스트에서 쓴 별개의 값이라 Fig.3 재현에는 맞지 않음.
B_FIELD = 0.05  # T

DEBUG_RESIDUAL = False

# ── 홀센서 보드 좌표계 변환 ──────────────────────────────
# force_model 내부는 논문 Fig.2(a) 자체 좌표계를 씀: 베이스=(0,0), 로컬 x = 자라나는(feed) 방향.
# 홀센서 보드(CNN 학습에 쓴 cnn_data_generated.py)는: 베이스=(90,0), 전역 y = 자라나는 방향, 0~180 범위.
# 두 결과를 같이 비교하려면(예: 힘 역산) 이 변환이 필요함.
BOARD_BASE_X = 90.0  # mm, 홀센서 보드에서 로봇 베이스의 x좌표


def to_board_frame(x_local, y_local):
    """force_model 로컬좌표(스칼라 또는 배열) -> 홀센서 보드 전역좌표(x,y) 변환."""
    x_local = np.asarray(x_local)
    y_local = np.asarray(y_local)
    x_board = BOARD_BASE_X + y_local
    y_board = x_local
    return x_board, y_board


def solve_shape(L_M, phi_deg, loads=None, s_t=None, Fx=0.0, Fy=0.0, return_curve=False,
                 theta_L_hint_deg=None, window_deg=25.0):
    """
    L_M: MOM 위치 (mm), phi_deg: 외부자기장 방향 (deg)
    loads: 접촉(외력) 리스트, 위 docstring 형식. 여러 개 = 다중 접촉.
    s_t,Fx,Fy: (하위호환용) loads가 없을 때 단일 점하중으로 자동 변환됨.
    return_curve: True면 결과에 "curve_s_mm","curve_x_mm","curve_y_mm" (팔 전체 곡선) 포함.
    theta_L_hint_deg: 주어지면 전체 -180~180 재탐색 대신, 이 값 근처에서만 국소 탐색
                       (연속법(continuation)용 — 힘을 조금씩 늘려갈 때 직전 해를 넘겨주면
                       다른(불안정한) 해로 튀지 않고 같은 물리적 가지를 계속 따라감).
    반환: dict(x_LM,y_LM,theta_LM_deg, x_L,y_L,theta_L_deg)  (좌표는 mm)
    """
    if loads is None:
        loads = [{"type": "point", "s": s_t if s_t is not None else L / 2, "Fx": Fx, "Fy": Fy}]

    phi = np.radians(phi_deg)
    L_m = L / 1000.0
    H_M_m = H_M / 1000.0
    L_M_m = L_M / 1000.0
    a1 = np.clip(L_M_m - H_M_m / 2, 0, L_m)
    a2 = np.clip(L_M_m + H_M_m / 2, 0, L_m)

    # loads를 m 단위로 변환
    loads_m = []
    for ld in loads:
        if ld["type"] == "point":
            loads_m.append({"type": "point", "s": ld["s"] / 1000.0, "Fx": ld["Fx"], "Fy": ld["Fy"]})
        else:
            s0, s1 = ld["s_start"] / 1000.0, ld["s_end"] / 1000.0
            loads_m.append({"type": "dist", "s_start": s0, "s_end": s1, "Fx": ld["Fx"], "Fy": ld["Fy"]})

    # 전체 구간을 쪼갤 breakpoint들: 0, a1, a2, L_m, 각 하중의 경계들
    breakpoints = {0.0, a1, a2, L_m}
    for ld in loads_m:
        if ld["type"] == "point":
            if 0 < ld["s"] < L_m:
                breakpoints.add(ld["s"])
        else:
            for edge in (ld["s_start"], ld["s_end"]):
                if 0 <= edge <= L_m:
                    breakpoints.add(edge)
    breakpoints = sorted(breakpoints)

    def K_of_s(s_mid):
        if s_mid < a1:
            return K1
        elif s_mid < a2:
            return K_RIGID
        return K2

    def q_of_s(s_mid):
        """s_mid 지점에서의 분포하중 밀도 (N/m). 여러 분포하중이 겹치면 합산."""
        qx = qy = 0.0
        for ld in loads_m:
            if ld["type"] == "dist" and ld["s_start"] <= s_mid <= ld["s_end"]:
                length = ld["s_end"] - ld["s_start"]
                if length > 1e-12:
                    qx += ld["Fx"] / length
                    qy += ld["Fy"] / length
        return qx, qy

    def tau2(theta_L):
        return M2 * B_FIELD * np.sin(phi - theta_L)

    def tau1(theta_LM):
        return M1 * B_FIELD * np.sin(phi - theta_LM - np.pi)

    def make_odes(K_seg, qx_seg, qy_seg):
        def odes(s, state):
            theta, x, y, M, Nx, Ny = state
            dtheta = M / K_seg
            dx = np.cos(theta)
            dy = np.sin(theta)
            dM = -(Nx * np.sin(theta) - Ny * np.cos(theta))
            dNx = -qx_seg
            dNy = -qy_seg
            return [dtheta, dx, dy, dM, dNx, dNy]
        return odes

    def shoot(theta_L_guess, collect_curve=False):
        # 부호 확정: 논문 원문 표기(eq.5-8)는 "-tau"이지만, 실제 논문 Fig.3(e)-(h) 산점도를
        # 픽셀단위로 디지털화(scripts/_digitize_full.py)해 역산해보면 "+tau" 쪽이 K1,K2를
        # 물리적으로 타당한(양수, 여러 phi/LM에서 일관된) 값으로 복원함. 좌표계/각도 방향 정의의
        # 차이로 보이며, 실측 데이터 기준으로 부호를 확정함 (scripts/_check_cmscr.py, _check_mom.py 참고).
        M_L = tau2(theta_L_guess)
        # N(L) = 팁(s=L)보다 먼 쪽에 걸린 하중. 보통은 없어서 0이지만,
        # 팁(main자석)에 직접 힘이 걸리는 경우(s_t=L) 그 힘 자체가 초기값이 되어야 함.
        Nx_L = Ny_L = 0.0
        for ld in loads_m:
            if ld["type"] == "point" and abs(ld["s"] - L_m) < 1e-9:
                Nx_L += ld["Fx"]
                Ny_L += ld["Fy"]
        state = np.array([theta_L_guess, 0.0, 0.0, M_L, Nx_L, Ny_L])
        # 곡선을 그릴 때만 촘촘히(0.2mm) 적분하고, 뿌리찾기(residual 계산)용 호출은
        # 최종상태만 필요하므로 훨씬 성긴 스텝으로 충분함. 이걸 항상 0.2mm로 고정해뒀던 게
        # 각도 스캔/브렌트법에서 residual()을 수백~수천 번 부르는 상황을 극도로 느리게 만든 원인.
        max_step = (L_m / 500) if collect_curve else (L_m / 20)
        theta_LM_captured = None
        x_LM_captured = y_LM_captured = None
        curve_s, curve_x, curve_y, curve_theta = [], [], [], []

        # breakpoints를 내림차순(팁->베이스)으로 순회하며 구간별 적분
        for i in range(len(breakpoints) - 1, 0, -1):
            s_hi, s_lo = breakpoints[i], breakpoints[i - 1]
            if s_hi - s_lo < 1e-12:
                continue
            s_mid = (s_hi + s_lo) / 2
            K_seg = K_of_s(s_mid)
            qx_seg, qy_seg = q_of_s(s_mid)
            odes = make_odes(K_seg, qx_seg, qy_seg)
            n_eval = max(2, int((s_hi - s_lo) / max_step))
            t_eval = np.linspace(s_hi, s_lo, n_eval) if collect_curve else None
            sol = solve_ivp(odes, [s_hi, s_lo], state, max_step=max_step, rtol=1e-8, t_eval=t_eval)
            if collect_curve:
                curve_s.extend(sol.t.tolist())
                curve_x.extend(sol.y[1, :].tolist())
                curve_y.extend(sol.y[2, :].tolist())
                curve_theta.extend(sol.y[0, :].tolist())
            state = sol.y[:, -1].copy()

            # s_lo 위치에서 점하중 jump 되돌리기(역적분이므로 더해줌)
            for ld in loads_m:
                if ld["type"] == "point" and abs(ld["s"] - s_lo) < 1e-9:
                    state[4] += ld["Fx"]
                    state[5] += ld["Fy"]

            # s_lo가 a1(=MOM 시작점)이면: MOM 토크 jump 되돌리기 + 위치 캡처
            if abs(s_lo - a1) < 1e-9 and theta_LM_captured is None:
                theta_LM_captured = state[0]
                x_LM_captured = state[1] + (H_M_m / 2) * np.cos(state[0])
                y_LM_captured = state[2] + (H_M_m / 2) * np.sin(state[0])
                state[3] = state[3] + tau1(state[0])

        curve = (np.array(curve_s), np.array(curve_x), np.array(curve_y), np.array(curve_theta)) if collect_curve else None
        return state, theta_LM_captured, x_LM_captured, y_LM_captured, curve

    def residual(theta_L_guess):
        state, *_ = shoot(theta_L_guess)
        return state[0]

    if DEBUG_RESIDUAL:
        for deg in range(-170, 171, 10):
            g = np.radians(deg)
            print(f"  theta_L_guess={deg:5d} deg -> residual={np.degrees(residual(g)):8.2f} deg")
        return None

    if theta_L_hint_deg is not None:
        # 연속법: hint 근처 좁은 구간에서만 부호변화 지점을 찾아 그 가지를 추적
        hint = np.radians(theta_L_hint_deg)
        window = np.radians(np.arange(-window_deg, window_deg + 0.01, 1.0))  # hint가 부정확해도(피팅 초기 등) 전역탐색 폴백을 줄이려 넓게
        scan = hint + window
        vals = np.array([residual(g) for g in scan])
        roots = []
        for i in range(len(scan) - 1):
            if np.sign(vals[i]) != np.sign(vals[i + 1]) and np.isfinite(vals[i]) and np.isfinite(vals[i + 1]):
                try:
                    roots.append(brentq(residual, scan[i], scan[i + 1], xtol=1e-6))
                except ValueError:
                    pass
        if not roots:
            raise RuntimeError(f"theta_L_hint={theta_L_hint_deg}deg 근처에서 해를 못 찾음 (가지가 끊겼을 수 있음)")
        theta_L_sol = min(roots, key=lambda r: abs(r - hint))
    else:
        scan = np.radians(np.arange(-179, 180, 2.0))
        vals = np.array([residual(g) for g in scan])
        roots = []
        for i in range(len(scan) - 1):
            if np.sign(vals[i]) != np.sign(vals[i + 1]) and np.isfinite(vals[i]) and np.isfinite(vals[i + 1]):
                try:
                    roots.append(brentq(residual, scan[i], scan[i + 1], xtol=1e-6))
                except ValueError:
                    pass
        if not roots:
            raise RuntimeError("theta_L 해를 못 찾음 (다시 스캔 필요)")
        theta_L_sol = min(roots, key=lambda r: abs(((r - phi + np.pi) % (2 * np.pi)) - np.pi))

    state_0, theta_LM_rel, x_LM_rel, y_LM_rel, curve = shoot(theta_L_sol, collect_curve=return_curve)
    x0, y0 = state_0[1], state_0[2]

    result = {
        "theta_LM_deg": np.degrees(theta_LM_rel),
        "x_LM": (x_LM_rel - x0) * 1000,
        "y_LM": (y_LM_rel - y0) * 1000,
        "theta_L_deg": np.degrees(theta_L_sol),
        "x_L": (0.0 - x0) * 1000,
        "y_L": (0.0 - y0) * 1000,
    }
    if return_curve:
        cs, cx, cy, ctheta = curve
        result["curve_s_mm"] = cs * 1000
        result["curve_x_mm"] = (cx - x0) * 1000
        result["curve_y_mm"] = (cy - y0) * 1000
        result["curve_theta_deg"] = np.degrees(ctheta)
    return result


def solve_shape_ramped(L_M, phi_deg, loads, theta_L_hint_deg, n_steps=4, return_curve=False, window_deg=25.0):
    """solve_shape의 연속법(continuation) 실전 활용판.

    문제: solve_shape(theta_L_hint_deg=...)는 hint 주변 +-25도 안에서만 해를 찾는데, 접촉
    시나리오 생성 스크립트들이 지금까지 무외력(free) hint에서 목표 힘 전체를 "한 번에" 걸어서
    풀었음. 팁 근처(s가 큼)를 큰 힘으로 누르면 지렛대 효과로 해가 +-25도 밖으로 튀어나가
    RuntimeError("hint 근처에서 해를 못 찾음")가 남 - 실측(600회 시도) 실패율 12.5%, 전부 이
    경로에서 발생함을 확인함.

    해결: 목표 힘을 n_steps 단계로 나눠 0->100%까지 점진적으로 늘리면서, 매 단계 직전 해를
    다음 hint로 넘김 - 각 단계의 각도 변화폭이 작아져서 +-25도 창을 거의 안 벗어남.
    """
    hint = theta_L_hint_deg
    result = None
    for step in range(1, n_steps + 1):
        frac = step / n_steps
        scaled_loads = []
        for ld in loads:
            ld2 = dict(ld)
            ld2["Fx"] = ld["Fx"] * frac
            ld2["Fy"] = ld["Fy"] * frac
            scaled_loads.append(ld2)
        result = solve_shape(L_M=L_M, phi_deg=phi_deg, loads=scaled_loads,
                              theta_L_hint_deg=hint, return_curve=(return_curve and step == n_steps),
                              window_deg=window_deg)
        hint = result["theta_L_deg"]
    return result


def solve_shape_robust(L_M, phi_deg, loads, theta_L_hint_deg, return_curve=False):
    """실패율 0%를 목표로 하는 5단계 캐스케이드. 대부분(87.5%)은 1단계(빠른 단발)에서 바로
    끝나고, 어려운 케이스(팁 근처를 강하게 누르는 등)만 점점 더 비싼 방법으로 넘어감.

    실패를 그냥 버리고 새 무작위 샘플로 넘어가는 기존 방식은 평균적으로는 더 빠르지만,
    실패가 특정 영역(s가 크고 F가 큰 쪽)에 쏠려있어서(실측 확인함) 그 영역이 최종 데이터셋에
    체계적으로 적게 들어가는 편향이 생김. 이 함수는 그 편향을 없애기 위해 "포기하지 않고
    끝까지 푸는" 쪽을 택함 - 대신 평균 2~3배 느림(600회 테스트: 실패율 0%, 138ms/건 평균).
    """
    load_case = loads
    hint = theta_L_hint_deg
    attempts = [
        lambda: solve_shape(L_M=L_M, phi_deg=phi_deg, loads=load_case, theta_L_hint_deg=hint,
                             return_curve=return_curve),
        lambda: solve_shape_ramped(L_M=L_M, phi_deg=phi_deg, loads=load_case, theta_L_hint_deg=hint,
                                    n_steps=4, return_curve=return_curve),
        lambda: solve_shape_ramped(L_M=L_M, phi_deg=phi_deg, loads=load_case, theta_L_hint_deg=hint,
                                    n_steps=8, return_curve=return_curve),
        lambda: solve_shape_ramped(L_M=L_M, phi_deg=phi_deg, loads=load_case, theta_L_hint_deg=hint,
                                    n_steps=16, window_deg=40.0, return_curve=return_curve),
        lambda: solve_shape(L_M=L_M, phi_deg=phi_deg, loads=load_case, theta_L_hint_deg=None,
                             return_curve=return_curve),
    ]
    last_err = None
    for attempt in attempts:
        try:
            return attempt()
        except RuntimeError as e:
            last_err = e
            continue
    raise last_err


if __name__ == "__main__":
    print("=== 외력 없음 ===")
    r = solve_shape(L_M=50, phi_deg=60, loads=[])
    for k, v in r.items():
        print(f"  {k}: {v:.2f}")

    print("\n=== 점하중 1개 (Fx=0.01N at s=80) ===")
    r2 = solve_shape(L_M=50, phi_deg=60, loads=[{"type": "point", "s": 80, "Fx": 0.01, "Fy": 0}])
    for k, v in r2.items():
        print(f"  {k}: {v:.2f}")

    print("\n=== 분포하중 1개 (총 Fx=0.01N, s=60~90 구간 균등분포) ===")
    r3 = solve_shape(L_M=50, phi_deg=60, loads=[{"type": "dist", "s_start": 60, "s_end": 90, "Fx": 0.01, "Fy": 0}])
    for k, v in r3.items():
        print(f"  {k}: {v:.2f}")

    print("\n=== 다중 접촉 (점하중 2개, 위치 다름) ===")
    r4 = solve_shape(L_M=50, phi_deg=60, loads=[
        {"type": "point", "s": 30, "Fx": 0.005, "Fy": 0},
        {"type": "point", "s": 85, "Fx": -0.003, "Fy": 0.002},
    ])
    for k, v in r4.items():
        print(f"  {k}: {v:.2f}")

    print(f"\nK1={K1:.3e} N*m^2, K2={K2:.3e} N*m^2")
