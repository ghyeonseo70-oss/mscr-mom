"""
K1,K2 기준으로 "방향이 안 뒤집히는(=물리적으로 타당한) 안전한 힘의 범위"를 체계적으로 찾는다.
학습데이터 분포와 같은 (L_M, phi, s, F_ang) 조합을 무작위로 많이 뽑아서, 각각 힘을 조금씩
늘려가며 "국소 법선방향 일관성"이 깨지는(반대로 뒤집히는) 지점을 찾는다.
"""
import multiprocessing as mp
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def find_reversal_threshold(seed):
    import sys as _sys
    if HERE not in _sys.path:
        _sys.path.insert(0, HERE)
    import force_model as fm

    rng = np.random.default_rng(seed)
    L_M = rng.uniform(20.0, 80.0)
    phi = rng.choice([-120.0, -60.0, 0.0, 60.0, 120.0])
    s = rng.uniform(5.0, 95.0)
    F_ang = rng.uniform(0, 2 * np.pi)

    try:
        r_free = fm.solve_shape(L_M=L_M, phi_deg=phi, loads=[], return_curve=True)
    except RuntimeError:
        return None
    fcs, fcx, fcy, fcth = r_free["curve_s_mm"], r_free["curve_x_mm"], r_free["curve_y_mm"], r_free["curve_theta_deg"]
    forder = np.argsort(fcs)
    x0 = np.interp(s, fcs[forder], fcx[forder])
    y0 = np.interp(s, fcs[forder], fcy[forder])
    theta_local = np.radians(np.interp(s, fcs[forder], fcth[forder]))
    normal = np.array([-np.sin(theta_local), np.cos(theta_local)])

    F_LEVELS = np.array([0.5, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20]) / 1000.0  # N
    last_good = 0.0
    for F_mag in F_LEVELS:
        Fx, Fy = F_mag * np.cos(F_ang), F_mag * np.sin(F_ang)
        try:
            r_load = fm.solve_shape_robust(L_M=L_M, phi_deg=phi,
                                            loads=[{"type": "point", "s": s, "Fx": Fx, "Fy": Fy}],
                                            theta_L_hint_deg=r_free["theta_L_deg"], return_curve=True)
        except RuntimeError:
            break
        lcs, lcx, lcy = r_load["curve_s_mm"], r_load["curve_x_mm"], r_load["curve_y_mm"]
        lorder = np.argsort(lcs)
        x1 = np.interp(s, lcs[lorder], lcx[lorder])
        y1 = np.interp(s, lcs[lorder], lcy[lorder])
        disp = np.array([x1 - x0, y1 - y0])
        push_normal = np.dot([Fx, Fy], normal)
        disp_normal = np.dot(disp, normal)
        if np.sign(push_normal) != np.sign(disp_normal):
            break
        last_good = F_mag

    return last_good * 1000  # mN


def main():
    N = 400
    n_workers = mp.cpu_count()
    print(f"{N}개 무작위 조합으로 안전한 힘 범위 탐색 (워커 {n_workers}개)...")
    with mp.Pool(processes=n_workers) as pool:
        results = pool.map(find_reversal_threshold, range(N))
    results = [r for r in results if r is not None]
    arr = np.array(results)
    print(f"\n유효 {len(arr)}개")
    print(f"방향이 안 뒤집히는 최대 힘(mN): 평균={arr.mean():.2f}, 중앙값={np.median(arr):.2f}")
    for p in [5, 10, 25, 50, 75, 90]:
        print(f"  {p}퍼센타일: {np.percentile(arr, p):.2f}mN")
    print(f"\n0mN(항상 즉시 뒤집힘) 비율: {(arr < 0.5).mean()*100:.1f}%")

    np.savez(os.path.join(HERE, "..", "..", "data", "force_model", "safe_force_range.npz"), thresholds=arr)


if __name__ == "__main__":
    main()
