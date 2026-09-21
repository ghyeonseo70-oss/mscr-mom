"""2026-09-21: push_depth(충돌 깊이)와 변형/힘의 관계가 선형인지 확인.

이게 중요한 이유: 선형이면 깊이를 몇 개만 확보해도 그 사이를 안전하게 보간/외삽할 수 있어서
추가 FEA가 거의 필요 없음. 비선형(대변형 효과)이면 깊이 값을 촘촘히 모아야 함.

방법: 같은 (L_M,phi,beta,s)에서 깊이만 다른 짝을 찾아, 깊이 비율 대비 변형/힘 비율을 비교.
완전 선형이면 깊이가 2배일 때 변형도 정확히 2배(비율=1.0)가 나와야 함.
"""
import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "..", "data", "contact_scenarios", "fea",
                     "fea_lm_phi_pos_matv2_all.json")

rows = json.load(open(DATA, encoding="utf-8"))
g = defaultdict(dict)
for r in rows:
    g[(r["L_M_mm"], r["phi_deg"], r["beta_deg"], r["contact_s_mm"])][round(r["push_depth_mm"], 3)] = r

QUANTS = [
    ("팁 변위크기", lambda r: np.hypot(r["tip_ux_avg_mm"], r["tip_uy_avg_mm"])),
    ("팁 회전각", lambda r: abs(r["tip_theta_deg_board"])),
    ("접촉힘 F_mag", lambda r: r["F_mag_N"]),
]

for lo, hi in [(0.05, 0.10), (0.10, 0.20)]:
    trips = {k: v for k, v in g.items() if lo in v and hi in v}
    if not trips:
        continue
    ratio_depth = hi / lo
    print(f"\n=== 깊이 {lo}mm -> {hi}mm (깊이 {ratio_depth:.0f}배), 짝 {len(trips)}개 ===")
    print(f"{'물리량':14s} {'변화비율 중앙값':>14s} {'사분위범위':>18s}  판정")
    for name, f in QUANTS:
        ratios = []
        for v in trips.values():
            a, b = f(v[lo]), f(v[hi])
            if abs(a) > 1e-9:
                ratios.append(b / a)
        if not ratios:
            continue
        ratios = np.array(ratios)
        med = np.median(ratios)
        q1, q3 = np.percentile(ratios, [25, 75])
        # 완전 선형이면 변화비율 == 깊이비율
        verdict = "선형에 가까움" if abs(med - ratio_depth) / ratio_depth < 0.15 else "비선형 의심"
        print(f"{name:14s} {med:14.2f}배 {f'[{q1:.2f}, {q3:.2f}]':>18s}  "
              f"{verdict} (선형이면 {ratio_depth:.0f}배)")

# 삼중쌍이 있으면 세 점이 한 직선 위에 있는지도 확인
triples = {k: v for k, v in g.items() if all(d in v for d in (0.05, 0.10, 0.20))}
if triples:
    print(f"\n=== 세 깊이 모두 있는 조합 {len(triples)}개: 원점 통과 직선 적합도 ===")
    for name, f in QUANTS:
        r2s = []
        for v in triples.values():
            x = np.array([0.05, 0.10, 0.20])
            y = np.array([f(v[d]) for d in x])
            if np.abs(y).max() < 1e-9:
                continue
            slope = (x * y).sum() / (x * x).sum()      # y = a*x (원점 통과)
            ss_res = ((y - slope * x) ** 2).sum()
            ss_tot = ((y - y.mean()) ** 2).sum()
            if ss_tot > 1e-18:
                r2s.append(1 - ss_res / ss_tot)
        if r2s:
            print(f"  {name:14s} R²(원점통과 직선) 중앙값={np.median(r2s):.4f}  "
                  f"최소={min(r2s):.4f}  (1.0에 가까울수록 완전 비례)")
