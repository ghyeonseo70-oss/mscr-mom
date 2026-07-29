"""
find_confusable_pair.py를 프로브 여러 개로 확장: phi 후보 그리드에서 (s,F) 격자를 전부
계산해두고, 프로브 조합(phi 2~3개)마다 "가장 헷갈리는 쌍이 얼마나 헷갈리는지"를 점수화해서
전수조사 - 어떤 조합이 최악의 경우를 가장 잘 줄이는지 찾는다. 물리모델만 쓰므로 CNN 학습
없이 몇 초~몇 분 안에 수백 개 조합을 비교할 수 있다.
"""
import itertools
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "force_model"))
import force_model as fm

L_M0 = 50.0
S_GRID = np.arange(10.0, 91.0, 10.0)     # 9개
F_GRID = np.array([0.002, 0.006, 0.010, 0.014, 0.018])  # 5개 -> (s,F) 45개 조합
PHI_GRID = np.arange(-150.0, 151.0, 30.0)  # 11개 후보 프로브 각도

# ── 각 phi에서 free 형상 미리 계산(접선각도 함수용) ──────────────────────────
free_cache = {}
for phi in PHI_GRID:
    free_cache[phi] = fm.solve_shape(L_M=L_M0, phi_deg=phi, loads=[], return_curve=True)


def normal_at_s(phi, s):
    r_free = free_cache[phi]
    order = np.argsort(r_free["curve_s_mm"])
    s_arr = r_free["curve_s_mm"][order]
    th_arr = r_free["curve_theta_deg"][order]
    th = np.radians(np.interp(s, s_arr, th_arr))
    return np.array([np.sin(th), -np.cos(th)])


# ── (s,F) x phi 격자 전부 계산: delta 기술자(6개: xL,yL,thL,xLM,yLM,thLM - free 기준 차이) ──
print("격자 계산 중...")
sf_list = [(s, F) for s in S_GRID for F in F_GRID]
descriptors = {}  # (s,F,phi) -> 6D delta vector
for phi in PHI_GRID:
    r_free = free_cache[phi]
    for s, F in sf_list:
        n = normal_at_s(phi, s)
        Fx, Fy = F * n[0], F * n[1]
        try:
            r_load = fm.solve_shape(L_M=L_M0, phi_deg=phi,
                                     loads=[{"type": "point", "s": s, "Fx": Fx, "Fy": Fy}],
                                     theta_L_hint_deg=r_free["theta_L_deg"])
        except RuntimeError:
            descriptors[(s, F, phi)] = None
            continue
        delta = np.array([r_load["x_L"] - r_free["x_L"], r_load["y_L"] - r_free["y_L"],
                           r_load["theta_L_deg"] - r_free["theta_L_deg"],
                           r_load["x_LM"] - r_free["x_LM"], r_load["y_LM"] - r_free["y_LM"],
                           r_load["theta_LM_deg"] - r_free["theta_LM_deg"]])
        descriptors[(s, F, phi)] = delta
print(f"완료: {len(sf_list)}개 (s,F) x {len(PHI_GRID)}개 phi = {len(descriptors)}개 계산")


def score_combo(phi_combo):
    """이 프로브 조합에서 서로 다른 s를 가진 (s,F) 쌍들 중 '가장 헷갈리는(가장 가까운)' 거리.
    클수록 좋음(최악의 경우가 그나마 잘 구별됨)."""
    vecs = []
    valid_sf = []
    for s, F in sf_list:
        parts = [descriptors[(s, F, phi)] for phi in phi_combo]
        if any(p is None for p in parts):
            continue
        vecs.append(np.concatenate(parts))
        valid_sf.append((s, F))
    vecs = np.array(vecs)
    if len(vecs) < 10:
        return -1, None
    vecs_n = (vecs - vecs.mean(0)) / (vecs.std(0) + 1e-9)

    worst = None
    for i, j in itertools.combinations(range(len(valid_sf)), 2):
        if abs(valid_sf[i][0] - valid_sf[j][0]) < 15:  # s가 충분히 다른 쌍만
            continue
        d = np.linalg.norm(vecs_n[i] - vecs_n[j])
        if worst is None or d < worst:
            worst = d
    return worst, len(valid_sf)


# ── 2프로브, 3프로브 조합 전수조사 ──────────────────────────
print("\n=== 2프로브 조합 전수조사 ===")
results_2 = []
for combo in itertools.combinations(PHI_GRID, 2):
    score, n_valid = score_combo(combo)
    if score is not None and score > 0:
        results_2.append((combo, score))
results_2.sort(key=lambda x: -x[1])
print("상위 10개 (최악의 경우 거리가 클수록 좋음):")
for combo, score in results_2[:10]:
    print(f"  phi={combo}  worst-case거리={score:.3f}")
print("하위 5개 (참고용 - 가장 나쁜 조합):")
for combo, score in results_2[-5:]:
    print(f"  phi={combo}  worst-case거리={score:.3f}")

print("\n=== 3프로브 조합 전수조사 ===")
results_3 = []
for combo in itertools.combinations(PHI_GRID, 3):
    score, n_valid = score_combo(combo)
    if score is not None and score > 0:
        results_3.append((combo, score))
results_3.sort(key=lambda x: -x[1])
print("상위 10개:")
for combo, score in results_3[:10]:
    print(f"  phi={combo}  worst-case거리={score:.3f}")

# 기존에 썼던 조합들과 비교
print("\n=== 기존 조합들과 비교 ===")
for combo in [(-90.0, 0.0, 90.0), (-60.0, 60.0), (-120.0, 120.0), (-60.0, 0.0, 60.0),
              (-120.0, 0.0, 120.0), (-120.0, -60.0, 60.0, 120.0)]:
    combo_on_grid = tuple(c for c in combo if c in PHI_GRID)
    if len(combo_on_grid) == len(combo):
        score, _ = score_combo(combo)
        print(f"  phi={combo}  worst-case거리={score:.3f}")
    else:
        print(f"  phi={combo}  (격자에 없는 각도 포함, 스킵)")

import json
out = {
    "top_2probe": [{"phi": list(c), "score": float(s)} for c, s in results_2[:15]],
    "top_3probe": [{"phi": list(c), "score": float(s)} for c, s in results_3[:15]],
}
with open(os.path.join(HERE, "optimal_probe_search.json"), "w") as f:
    json.dump(out, f, indent=2)
print(f"\n저장: {os.path.join(HERE, 'optimal_probe_search.json')}")
