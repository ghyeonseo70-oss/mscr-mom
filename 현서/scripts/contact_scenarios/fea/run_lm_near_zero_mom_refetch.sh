#!/bin/bash
# 2026-08-26 신규: L_M 실측 R^2가 0.565로 낮은 원인을 파봤더니(_diag_lm_holdout_error.py),
# "MOM은 팁 변위의 L_M/100만큼만 움직인다"는 frac 근사 때문에 L_M=0(MOM이 베이스 바로 옆)
# 근처에서 MOM 변위 신호가 거의 사라져(frac~0) 학습 신호 자체가 없었던 게 원인으로 확인됨
# (L_M!=0은 R^2=0.953으로 훌륭함, L_M=0만 MAE=40mm로 사실상 랜덤).
#
# 근본 해결책(코드 수정 완료): make_bent_contact_scene.py가 이제 MOM 강체구간(N_MOM) 절점을
# 직접 뽑고, run_contact.py가 그 실제 변위/회전을 출력하도록 고침. 다만 기존 FEA 결과는
# 전부 옛 메시(N_MOM 없음)로 계산된 거라 재사용 불가(원본 .dat/.frd도 정리 과정에서 삭제됨) -
# 값을 얻으려면 그 (L_M,phi,beta,s) 조합을 다시 풀어야 함.
#
# 범위: 전체 재실행(약 1200케이스, 하루 이상) 대신, 문제가 확인된 L_M=0/12.5mm(가장 작은 두
# 값, frac 근사가 가장 심하게 틀리는 구간)에 해당하는 기존 조합만 골라 다시 품 - 지금
# fea_lm_phi_pos_matv2_all.json에 있는 정확히 그 (phi,beta,s) 조합 그대로 재사용해서 새
# 행이 기존 행을 "대체"하도록 함(중복 방지, merge 단계에서 같은 키는 새 값으로 덮어씀).
set -u
cd "$(dirname "$0")"
date +%s > lm_near_zero_mom_refetch_start_time.txt

python3 - <<'PYEOF' > lm_near_zero_mom_refetch_combos.txt
import json
from collections import defaultdict

rows = json.load(open("../../../data/contact_scenarios/fea/fea_lm_phi_pos_matv2_all.json", encoding="utf-8"))
TARGET_LM = {0.0, 12.5}
groups = defaultdict(list)
for r in rows:
    if r["L_M_mm"] in TARGET_LM:
        groups[(r["L_M_mm"], r["phi_deg"], r["beta_deg"])].append(r["contact_s_mm"])

def tag_of(lm, phi, beta):
    sign = "N" if phi < 0 else ("0" if phi == 0 else "P")
    beta_tag = "b0" if beta == 0 else "b180"
    return f"MOMREFETCH{int(lm*10)}_phi{sign}{int(abs(phi))}_{beta_tag}"

n_combos, n_cases = 0, 0
for (lm, phi, beta), s_list in sorted(groups.items()):
    s_str = ",".join(str(s) for s in sorted(set(s_list)))
    print(f"{lm} {phi} {beta} {s_str} {tag_of(lm, phi, beta)}")
    n_combos += 1
    n_cases += len(set(s_list))

import sys
print(f"# 총 {n_combos}조합, {n_cases}케이스", file=sys.stderr)
PYEOF

N_COMBOS=$(wc -l < lm_near_zero_mom_refetch_combos.txt)
echo "=== L_M=0/12.5mm MOM 재확보 시작 (${N_COMBOS}조합, 동시 9개) ==="
cat lm_near_zero_mom_refetch_combos.txt | xargs -P 9 -L 1 bash -c \
  'python3 -u sweep_lm_phi_position_matv2_worker.py --L_M "$0" --phi "$1" --beta "$2" --tag "$4" --threads 4 --s_list "$3" > "sweep_momrefetch_$4.log" 2>&1'

echo "=== 재확보 완료, 기존 fea_lm_phi_pos_matv2_all.json에서 같은 키는 새 값으로 교체 후 병합 ==="
python3 -c "
import json, glob

out = '../../../data/contact_scenarios/fea/fea_lm_phi_pos_matv2_all.json'
rows = json.load(open(out, encoding='utf-8'))
before = len(rows)

def key(r):
    return (r['L_M_mm'], r['phi_deg'], r['beta_deg'], r['contact_s_mm'])

by_key = {key(r): r for r in rows}
n_replaced, n_added = 0, 0
for f in sorted(glob.glob('../../../data/contact_scenarios/fea/fea_lm_phi_pos_matv2_MOMREFETCH*.json')):
    for r in json.load(open(f, encoding='utf-8')):
        k = key(r)
        if k in by_key:
            n_replaced += 1
        else:
            n_added += 1
        by_key[k] = r  # 새 값(실측 mom_*)으로 교체 또는 신규 추가

new_rows = sorted(by_key.values(), key=lambda r: (r['beta_deg'], r['L_M_mm'], r['phi_deg'], r['contact_s_mm']))
json.dump(new_rows, open(out, 'w'), indent=2, ensure_ascii=False)
print(f'MOM 재확보 병합 완료: {before}개 -> {len(new_rows)}개 (교체 {n_replaced}개, 신규 {n_added}개) ->', out)
"
echo "=== DONE ==="
