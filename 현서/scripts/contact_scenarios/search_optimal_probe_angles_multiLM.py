"""
search_optimal_probe_angles.py의 개선판. L_M=50 하나만 기준으로 찾았더니 실제(L_M이 20~80
전체로 다양한) 평가에서 오히려 성능이 나빠지는 문제가 있었음 - L_M이 바뀌면 두 구간의 길이비
(l1,l2)가 달라져서 곡률(R1,R2)과 접선방향이 통째로 달라지기 때문.

L_M은 사용자가 직접 정하는(=이미 아는) 값이라, "L_M끼리 헷갈리는 것"은 애초에 문제가 아니다.
문제는 "같은 L_M 안에서 s,F가 헷갈리는 것"이므로, 여러 L_M 각각에 대해 따로 최악의 경우를
구하고, 그중 가장 나쁜 L_M에서도 그럭저럭 버티는(=모든 L_M에 대해 최소 구별거리를 최대화하는)
phi 조합을 찾는다 - min-max 탐색.
"""
import itertools
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "force_model"))
import force_model as fm

L_M_GRID = [25.0, 40.0, 55.0, 70.0, 85.0]   # 실제 사용 범위(20~80mm)를 대표하는 5곳
S_GRID = np.arange(10.0, 91.0, 15.0)        # 6개 (L_M 5개 곱해지니 격자를 좀 줄임)
F_GRID = np.array([0.004, 0.010, 0.016])    # 3개 -> (s,F) 18개 조합
PHI_GRID = np.arange(-150.0, 151.0, 30.0)   # 11개 후보 프로브 각도

print(f"격자 계산 중... (L_M {len(L_M_GRID)} x phi {len(PHI_GRID)} x (s,F) {len(S_GRID)*len(F_GRID)})")

# ── L_M별로 free 형상 + descriptor 계산 ──────────────────────────
descriptors = {}  # (L_M, s, F, phi) -> 6D delta vector
sf_list = [(s, F) for s in S_GRID for F in F_GRID]

for L_M in L_M_GRID:
    free_cache = {}
    for phi in PHI_GRID:
        free_cache[phi] = fm.solve_shape(L_M=L_M, phi_deg=phi, loads=[], return_curve=True)

    def normal_at_s(phi, s, _free_cache=free_cache):
        r_free = _free_cache[phi]
        order = np.argsort(r_free["curve_s_mm"])
        s_arr = r_free["curve_s_mm"][order]
        th_arr = r_free["curve_theta_deg"][order]
        th = np.radians(np.interp(s, s_arr, th_arr))
        return np.array([np.sin(th), -np.cos(th)])

    for phi in PHI_GRID:
        r_free = free_cache[phi]
        for s, F in sf_list:
            n = normal_at_s(phi, s)
            Fx, Fy = F * n[0], F * n[1]
            try:
                r_load = fm.solve_shape(L_M=L_M, phi_deg=phi,
                                         loads=[{"type": "point", "s": s, "Fx": Fx, "Fy": Fy}],
                                         theta_L_hint_deg=r_free["theta_L_deg"])
            except RuntimeError:
                descriptors[(L_M, s, F, phi)] = None
                continue
            delta = np.array([r_load["x_L"] - r_free["x_L"], r_load["y_L"] - r_free["y_L"],
                               r_load["theta_L_deg"] - r_free["theta_L_deg"],
                               r_load["x_LM"] - r_free["x_LM"], r_load["y_LM"] - r_free["y_LM"],
                               r_load["theta_LM_deg"] - r_free["theta_LM_deg"]])
            descriptors[(L_M, s, F, phi)] = delta
    print(f"  L_M={L_M} 완료", flush=True)

print(f"완료: 총 {len(descriptors)}개 계산")


def worst_case_at_LM(L_M, phi_combo):
    """이 L_M 하나에서, 이 프로브 조합으로 s가 다른 (s,F)쌍들을 얼마나 잘 구별하는지
    (최악의 경우, 클수록 좋음)."""
    vecs, valid_sf = [], []
    for s, F in sf_list:
        parts = [descriptors[(L_M, s, F, phi)] for phi in phi_combo]
        if any(p is None for p in parts):
            continue
        vecs.append(np.concatenate(parts))
        valid_sf.append((s, F))
    if len(vecs) < 8:
        return None
    vecs = np.array(vecs)
    vecs_n = (vecs - vecs.mean(0)) / (vecs.std(0) + 1e-9)
    worst = None
    for i, j in itertools.combinations(range(len(valid_sf)), 2):
        if abs(valid_sf[i][0] - valid_sf[j][0]) < 15:
            continue
        d = np.linalg.norm(vecs_n[i] - vecs_n[j])
        if worst is None or d < worst:
            worst = d
    return worst


def score_combo_robust(phi_combo):
    """모든 L_M에 대해 worst_case_at_LM을 구하고, 그중 가장 나쁜(최소) 값을 조합 점수로 사용
    - '어떤 L_M에서 쓰든 이 정도는 구별된다'는 보장(min-max)."""
    scores = [worst_case_at_LM(L_M, phi_combo) for L_M in L_M_GRID]
    if any(s is None for s in scores):
        return None, scores
    return min(scores), scores


print("\n=== 2프로브 조합 전수조사 (모든 L_M에 대해 robust) ===")
results_2 = []
for combo in itertools.combinations(PHI_GRID, 2):
    score, per_lm = score_combo_robust(combo)
    if score is not None:
        results_2.append((combo, score, per_lm))
results_2.sort(key=lambda x: -x[1])
print("상위 8개:")
for combo, score, per_lm in results_2[:8]:
    print(f"  phi={combo}  robust점수(최악L_M)={score:.3f}  (L_M별: {[f'{p:.2f}' for p in per_lm]})")

print("\n=== 3프로브 조합 전수조사 (모든 L_M에 대해 robust) ===")
results_3 = []
for combo in itertools.combinations(PHI_GRID, 3):
    score, per_lm = score_combo_robust(combo)
    if score is not None:
        results_3.append((combo, score, per_lm))
results_3.sort(key=lambda x: -x[1])
print("상위 8개:")
for combo, score, per_lm in results_3[:8]:
    print(f"  phi={combo}  robust점수(최악L_M)={score:.3f}  (L_M별: {[f'{p:.2f}' for p in per_lm]})")

print("\n=== 기존에 시도했던 조합들과 비교 (robust 기준) ===")
for combo in [(-60.0, 60.0), (-150.0, 30.0), (-60.0, 0.0, 60.0), (-150.0, -30.0, 30.0),
              (-120.0, -60.0, 60.0, 120.0), (-120.0, -60.0, 0.0, 60.0, 120.0)]:
    combo_on_grid = tuple(c for c in combo if c in PHI_GRID)
    if len(combo_on_grid) == len(combo):
        score, per_lm = score_combo_robust(combo)
        print(f"  phi={combo}  robust점수={score:.3f}  (L_M별: {[f'{p:.2f}' for p in per_lm]})")

import json
out = {"top_2probe": [{"phi": list(c), "score": float(s)} for c, s, _ in results_2[:15]],
       "top_3probe": [{"phi": list(c), "score": float(s)} for c, s, _ in results_3[:15]]}
with open(os.path.join(HERE, "optimal_probe_search_multiLM.json"), "w") as f:
    json.dump(out, f, indent=2)
print(f"\n저장: {os.path.join(HERE, 'optimal_probe_search_multiLM.json')}")
